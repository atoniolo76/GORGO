"""Convert GLM 5.1 parquet shards into a Mooncake FAST '25 trace JSONL.

Walks the GLM 5.1 ClickHouse export on the ``GORGO-glm5-completions`` volume
(mounted at ``/data``) starting from ``--start-time`` and consumes up to
``--num-requests`` rows. Each row is converted to one Mooncake FAST '25
trace entry::

    {"timestamp": 0,    "input_length": 6955, "output_length": 52, "hash_ids": [0, 1, 2, ...]}
    {"timestamp": 3053, "input_length": 6472, "output_length": 26, "hash_ids": [0, 1, 2, ...]}

Format (per https://github.com/kvcache-ai/Mooncake FAST25-release):

- ``timestamp``       request arrival time in milliseconds, relative to the
                      first emitted row (which is always ``0``).
- ``input_length``    prompt token count.
- ``output_length``   response token count.
- ``hash_ids``        block-level prompt hashes, remapped to consecutive
                      integers. Hashing is prefix-aware -- two requests
                      sharing a K-token prompt prefix get the same first
                      ``ceil(K / block_size)`` ids, which is what makes
                      these traces useful for KV-cache replay simulation.

Walking matches ``proxy/workload.py`` (filename-timestamp file selection
plus a row-level timestamp filter), and tokenization uses tiktoken
``gpt-4o`` for parity with ``data_processing/build_eval_dataset.py``. The
exact tokenizer doesn't matter for replay -- what matters is that input
and output lengths are consistent across the trace.

Usage::

    modal run data_processing/build_mooncake_trace.py \\
        --start-time 2026-04-01T12:00:00 --num-requests 10000

Output JSONL is written to the ``GORGO-glm5-completions`` volume (default
``/data/mooncake_traces/mooncake_<UTC-timestamp>.jsonl``).
"""

from __future__ import annotations

import modal

from app import app, completions_volume

image = (
    modal.Image.debian_slim()
    .pip_install("duckdb", "pyarrow", "tiktoken")
    .add_local_python_source("app")
)

DEFAULT_BLOCK_SIZE = 512
DEFAULT_DATA_DIR = "/data"


@app.function(
    image=image,
    timeout=4 * 60 * 60,
    memory=1024 * 16,
    volumes={"/data": completions_volume},
)
def build_mooncake_trace(
    start_time: str,
    num_requests: int,
    output_path: str | None = None,
    end_time: str | None = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
    data_dir: str = DEFAULT_DATA_DIR,
    include_bodies: bool = False,
    include_raw_bodies: bool = False,
    time_scale: float = 1.0,
    target_duration_ms: int = 0,
    selection_mode: str = "chronological",
    candidate_multiplier: int = 1,
    top_token_hashes: int = 0,
    max_input_tokens: int = 0,
    max_total_tokens: int = 0,
    min_input_tokens: int = 0,
    output_sidecar_path: str | None = None,
) -> dict:
    """Walk GLM 5.1 parquet shards and emit a Mooncake FAST '25 trace JSONL.

    Args:
        start_time: ISO 8601 timestamp -- only rows with ``timestamp >=
            start_time`` are emitted. Required (the GLM dataset is large
            enough that an unbounded scan is rarely what you want).
        num_requests: Cap on emitted rows. Walking stops as soon as this
            many valid (request + response parsed, prompt non-empty) rows
            have been collected.
        output_path: Where to write the JSONL inside the volume. Relative
            paths are resolved under ``/data``. ``None`` (default) ->
            auto-generated ``mooncake_traces/mooncake_<UTC-timestamp>.jsonl``.
        end_time: Optional ISO 8601 upper bound (half-open ``[start, end)``).
        block_size: Token count per KV-cache block when hashing the prompt.
            Mooncake uses 512 in their published traces.
        data_dir: Directory holding the GLM 5.1 ``llm_responses_*.parquet``
            shards. Defaults to the ``/data`` mount.
        include_bodies: Include parsed request/response JSON objects so the
            trace can be replayed against the proxy.
        include_raw_bodies: Include raw request/response strings for exact
            debugging/archival payloads.
        time_scale: Multiply all relative timestamps by this factor.
        target_duration_ms: Alternative to ``time_scale``; when positive,
            compress/expand the selected trace to this total duration.
        selection_mode: ``chronological`` (default), ``top-users``,
            ``high-overlap``, ``mixed-overlap``, or ``token-hash-filter``.
        candidate_multiplier: Scan up to ``num_requests * candidate_multiplier``
            valid candidates before selecting the final trace.
        top_token_hashes: For ``token-hash-filter``, keep requests from the
            top-K token_hash values in the candidate pool. ``0`` picks a small
            default based on candidate diversity.
        max_input_tokens / max_total_tokens / min_input_tokens: Context-length
            filters applied before selection. Positive values enable checks.
        output_sidecar_path: Optional summary JSON path inside the volume.

    Returns:
        Dict with the resolved output path plus a few summary stats. The
        bulk of the result is the JSONL file written to the volume.
    """
    import hashlib
    import json
    import os
    import statistics
    import time
    from collections import Counter
    from datetime import datetime, timezone

    import duckdb
    import tiktoken

    completions_volume.reload()

    start_dt = _parse_iso(start_time)
    if start_dt is None:
        raise SystemExit(f"--start-time is required (got {start_time!r})")
    end_dt = _parse_iso(end_time)
    if num_requests <= 0:
        raise SystemExit(f"--num-requests must be > 0 (got {num_requests})")
    if block_size <= 0:
        raise SystemExit(f"--block-size must be > 0 (got {block_size})")
    if time_scale <= 0:
        raise SystemExit(f"--time-scale must be > 0 (got {time_scale})")
    if target_duration_ms < 0:
        raise SystemExit(f"--target-duration-ms must be >= 0 (got {target_duration_ms})")
    if candidate_multiplier <= 0:
        raise SystemExit(f"--candidate-multiplier must be > 0 (got {candidate_multiplier})")
    supported_modes = {
        "chronological",
        "top-users",
        "high-overlap",
        "mixed-overlap",
        "token-hash-filter",
    }
    if selection_mode not in supported_modes:
        raise SystemExit(f"--selection-mode must be one of {sorted(supported_modes)}")

    files = _select_files(data_dir, start_dt, end_dt)
    if not files:
        raise SystemExit(f"no GLM5 parquet files match the requested time range under {data_dir!r}")

    if output_path is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        resolved_output_path = f"/data/mooncake_traces/mooncake_{ts}.jsonl"
    else:
        resolved_output_path = (
            output_path if os.path.isabs(output_path) else os.path.join("/data", output_path)
        )
    os.makedirs(os.path.dirname(resolved_output_path), exist_ok=True)
    if output_sidecar_path is None:
        resolved_sidecar_path = resolved_output_path + ".summary.json"
    else:
        resolved_sidecar_path = (
            output_sidecar_path
            if os.path.isabs(output_sidecar_path)
            else os.path.join("/data", output_sidecar_path)
        )
    os.makedirs(os.path.dirname(resolved_sidecar_path), exist_ok=True)
    if os.path.exists(resolved_output_path) and os.path.exists(resolved_sidecar_path):
        try:
            with open(resolved_sidecar_path) as f:
                existing = json.load(f)
            print(f"[mooncake] reusing existing trace {resolved_output_path}", flush=True)
            return {
                "output_path": resolved_output_path,
                "sidecar_path": resolved_sidecar_path,
                "rows": existing.get("rows"),
                "candidate_rows": existing.get("candidate_rows"),
                "skipped_rows": existing.get("skipped_rows"),
                "skipped_over_max_input": existing.get("skipped_over_max_input"),
                "skipped_over_max_total": existing.get("skipped_over_max_total"),
                "skipped_under_min_input": existing.get("skipped_under_min_input"),
                "selection_mode": existing.get("selection_mode"),
                "selected_token_hashes": existing.get("selected_token_hashes"),
                "block_size": existing.get("block_size"),
                "unique_blocks": existing.get("unique_blocks"),
                "total_blocks": existing.get("total_blocks"),
                "block_reuse_pct": existing.get("block_reuse_pct"),
                "total_input_tokens": existing.get("total_input_tokens"),
                "total_output_tokens": existing.get("total_output_tokens"),
                "original_duration_ms": existing.get("original_duration_ms"),
                "scaled_duration_ms": existing.get("scaled_duration_ms"),
                "time_scale": existing.get("time_scale"),
                "elapsed_seconds": 0.0,
                "reused_existing": True,
            }
        except Exception:
            pass

    enc = tiktoken.encoding_for_model("gpt-4o")

    # Remap (prefix-aware) block-hash digests to consecutive ints. The
    # first unique block ever seen gets id 0, the second 1, etc., which is
    # exactly what Mooncake calls the "remapped block hash".
    hash_to_id: dict[bytes, int] = {}

    def _block_ids(token_ids: list[int]) -> list[int]:
        if not token_ids:
            return []
        ids: list[int] = []
        prev_digest = b""
        for i in range(0, len(token_ids), block_size):
            block = token_ids[i : i + block_size]
            # Prefix-aware: chain each block's digest into the next so two
            # requests with a shared K-token prefix produce identical
            # ``hash_ids[:ceil(K/B)]``. Token ids are little-endian uint32
            # for a stable, fast-to-hash byte representation.
            h = hashlib.sha256()
            h.update(prev_digest)
            h.update(b"".join(t.to_bytes(4, "little", signed=False) for t in block))
            digest = h.digest()
            mapped = hash_to_id.get(digest)
            if mapped is None:
                mapped = len(hash_to_id)
                hash_to_id[digest] = mapped
            ids.append(mapped)
            prev_digest = digest
        return ids

    def _tokenize_messages(messages: list) -> list[int]:
        ids: list[int] = []
        for msg in messages:
            if isinstance(msg, str):
                ids.extend(enc.encode(msg, disallowed_special=()))
                continue
            if not isinstance(msg, dict):
                continue
            text = _content_to_str(msg.get("content"))
            if text:
                ids.extend(enc.encode(text, disallowed_special=()))
        return ids

    def _response_token_count(response_raw) -> int:
        try:
            resp = json.loads(response_raw) if isinstance(response_raw, str) else response_raw
        except (TypeError, ValueError):
            return 0
        if not isinstance(resp, dict):
            return 0
        # Prefer the upstream usage counter when present (no tokenization
        # cost, and matches whatever the server actually billed); fall
        # back to encoding the message content with our local tokenizer
        # so requests that didn't record usage still get a length.
        usage = resp.get("usage")
        if isinstance(usage, dict):
            ct = usage.get("completion_tokens")
            if isinstance(ct, int) and ct >= 0:
                return ct
        total = 0
        for choice in resp.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            msg = choice.get("message")
            if not isinstance(msg, dict):
                continue
            text = _content_to_str(msg.get("content"))
            if text:
                total += len(enc.encode(text, disallowed_special=()))
        return total

    def _parse_json_maybe(raw):
        try:
            return json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, ValueError):
            return None

    def _candidate_score(candidate: dict, block_freq: Counter, user_freq: Counter) -> float:
        bids = candidate["hash_ids"]
        if not bids:
            overlap = 0.0
        else:
            # Normalize gently so huge prompts do not dominate purely by size,
            # while still rewarding long reusable prefixes.
            overlap = sum(max(0, block_freq[b] - 1) for b in bids) / (len(bids) ** 0.5)
        if selection_mode == "chronological":
            return 0.0
        if selection_mode == "high-overlap":
            return overlap
        user = candidate.get("token_hash") or ""
        user_score = float(user_freq.get(user, 0))
        if selection_mode == "top-users":
            return user_score
        if selection_mode == "token-hash-filter":
            return user_score
        # mixed-overlap: keep the overlap signal but avoid selecting only one
        # mega-user when cross-user system-prompt reuse is present.
        return overlap + min(user_score, 25.0)

    # First pass: walk parquets, tokenize, collect candidates with absolute
    # timestamps. We scan more than we emit for overlap-based curation.
    print(
        f"[mooncake] walking {len(files)} parquet file(s) under {data_dir!r}; "
        f"start={start_time} end={end_time or '-'} target={num_requests} rows "
        f"mode={selection_mode} candidates={num_requests * candidate_multiplier}",
        flush=True,
    )
    t0 = time.perf_counter()
    target_candidates = max(num_requests, num_requests * candidate_multiplier)
    candidates: list[dict] = []
    skipped_rows = 0
    skipped_over_max_input = 0
    skipped_over_max_total = 0
    skipped_under_min_input = 0
    con = duckdb.connect()
    for filename in files:
        if len(candidates) >= target_candidates:
            break
        path = os.path.join(data_dir, filename)
        # ORDER BY timestamp so the in-window scan (and the early-exit on
        # the first row past ``end_dt``) is correct even when the parquet
        # row groups aren't time-sorted -- which they often aren't, since
        # ingestion writes rows in arrival order rather than event order.
        cursor = con.execute(
            """
            SELECT
                uuid,
                timestamp,
                request_metadata.token_hash AS token_hash,
                request,
                response
            FROM read_parquet(?)
            WHERE request NOT LIKE '%keep-alive%'
            ORDER BY timestamp
            """,
            [path],
        )
        while True:
            chunk = cursor.fetchmany(2048)
            if not chunk:
                break
            for uuid, ts, token_hash, request_raw, response_raw in chunk:
                ts_dt = _to_naive_dt(ts)
                if ts_dt is None:
                    skipped_rows += 1
                    continue
                if ts_dt < start_dt:
                    continue
                if end_dt is not None and ts_dt >= end_dt:
                    break
                body = _parse_json_maybe(request_raw)
                if not isinstance(body, dict):
                    skipped_rows += 1
                    continue
                msgs = body.get("messages")
                if not isinstance(msgs, list) or not msgs:
                    skipped_rows += 1
                    continue
                prompt_ids = _tokenize_messages(msgs)
                if not prompt_ids:
                    skipped_rows += 1
                    continue
                output_length = _response_token_count(response_raw)
                input_length = len(prompt_ids)
                if min_input_tokens > 0 and input_length < min_input_tokens:
                    skipped_under_min_input += 1
                    continue
                if max_input_tokens > 0 and input_length > max_input_tokens:
                    skipped_over_max_input += 1
                    continue
                if max_total_tokens > 0 and input_length + output_length > max_total_tokens:
                    skipped_over_max_total += 1
                    continue
                response_body = _parse_json_maybe(response_raw)
                block_ids = _block_ids(prompt_ids)
                candidates.append(
                    {
                        "uuid": uuid,
                        "ts_dt": ts_dt,
                        "token_hash": token_hash or "",
                        "input_length": input_length,
                        "output_length": output_length,
                        "hash_ids": block_ids,
                        "request": body,
                        "response": response_body,
                        "request_raw": request_raw,
                        "response_raw": response_raw,
                    }
                )
                if len(candidates) >= target_candidates:
                    break
            else:
                continue
            break
        print(
            f"[mooncake]   {filename}: collected {len(candidates)}/{target_candidates} candidates "
            f"({time.perf_counter() - t0:.1f}s elapsed)",
            flush=True,
        )
    con.close()

    if not candidates:
        raise SystemExit(
            f"no rows matched start_time={start_time!r} end_time={end_time!r} under {data_dir!r}"
        )

    block_freq = Counter()
    user_freq = Counter()
    for c in candidates:
        block_freq.update(c["hash_ids"])
        user_freq[c.get("token_hash") or ""] += 1
    for c in candidates:
        c["selection_score"] = _candidate_score(c, block_freq, user_freq)

    selected_token_hashes: list[str] = []
    if selection_mode == "chronological":
        selected = sorted(candidates, key=lambda e: e["ts_dt"])[:num_requests]
    elif selection_mode == "token-hash-filter":
        # Pick active users, then preserve original timestamp order. This
        # curates for intra-user KV reuse without fabricating request order or
        # cherry-picking individual requests by overlap score.
        k = top_token_hashes or max(1, min(20, len(user_freq)))
        selected_token_hashes = [th for th, _ in user_freq.most_common(k)]
        allowed = set(selected_token_hashes)
        selected = [
            c for c in sorted(candidates, key=lambda e: e["ts_dt"]) if c["token_hash"] in allowed
        ]
        selected = selected[:num_requests]
    else:
        selected = sorted(candidates, key=lambda e: e["selection_score"], reverse=True)[
            :num_requests
        ]
        selected.sort(key=lambda e: e["ts_dt"])

    if not selected:
        raise SystemExit("selection produced no rows")

    base_ts = selected[0]["ts_dt"]
    original_duration_ms = int((selected[-1]["ts_dt"] - base_ts).total_seconds() * 1000)
    effective_scale = (
        (target_duration_ms / max(original_duration_ms, 1))
        if target_duration_ms > 0
        else time_scale
    )

    total_input = 0
    total_output = 0
    total_blocks = 0
    selected_blocks: set[int] = set()
    token_hash_counts = Counter(c.get("token_hash") or "" for c in selected)
    token_hash_tokens = Counter()
    tmp_path = resolved_output_path + ".tmp"
    with open(tmp_path, "w") as f:
        for idx, c in enumerate(selected):
            relative_ms = int((c["ts_dt"] - base_ts).total_seconds() * 1000)
            delta_ms = int(relative_ms * effective_scale)
            request_id = f"glm5_{base_ts.strftime('%Y%m%dT%H%M%S')}_{idx:06d}"
            row = {
                "timestamp": delta_ms,
                "input_length": c["input_length"],
                "output_length": c["output_length"],
                "hash_ids": c["hash_ids"],
            }
            if include_bodies or include_raw_bodies:
                row.update(
                    {
                        "request_id": request_id,
                        "source_timestamp": c["ts_dt"].isoformat(),
                        "uuid": c["uuid"],
                        "token_hash": c["token_hash"],
                        "selection_score": c["selection_score"],
                    }
                )
            if include_bodies:
                row["request"] = c["request"]
                row["response"] = c["response"]
            if include_raw_bodies:
                row["request_raw"] = c["request_raw"]
                row["response_raw"] = c["response_raw"]
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
            total_input += c["input_length"]
            total_output += c["output_length"]
            total_blocks += len(c["hash_ids"])
            selected_blocks.update(c["hash_ids"])
            token_hash_tokens[c.get("token_hash") or ""] += c["input_length"]
    os.replace(tmp_path, resolved_output_path)

    sidecar = {
        "output_path": resolved_output_path,
        "rows": len(selected),
        "candidate_rows": len(candidates),
        "skipped_rows": skipped_rows,
        "skipped_over_max_input": skipped_over_max_input,
        "skipped_over_max_total": skipped_over_max_total,
        "skipped_under_min_input": skipped_under_min_input,
        "selection_mode": selection_mode,
        "candidate_multiplier": candidate_multiplier,
        "top_token_hashes_requested": top_token_hashes,
        "selected_token_hashes": selected_token_hashes,
        "preserves_source_order": True,
        "curation_note": (
            "token-hash-filter preserves chronological order for top active users. "
            "Future shared-prefix-filter mode could select cross-user repeated "
            "system/harness prefixes by block-id frequency while still emitting "
            "selected rows in source timestamp order."
        ),
        "start_time": start_time,
        "end_time": end_time,
        "base_timestamp": base_ts.isoformat(),
        "original_duration_ms": original_duration_ms,
        "scaled_duration_ms": int(original_duration_ms * effective_scale),
        "time_scale": effective_scale,
        "block_size": block_size,
        "unique_blocks": len(selected_blocks),
        "total_blocks": total_blocks,
        "block_reuse_pct": (
            100.0 * (total_blocks - len(selected_blocks)) / total_blocks if total_blocks else 0.0
        ),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "include_bodies": include_bodies,
        "include_raw_bodies": include_raw_bodies,
        "context_filters": {
            "min_input_tokens": min_input_tokens,
            "max_input_tokens": max_input_tokens,
            "max_total_tokens": max_total_tokens,
        },
        "token_hash_count": len(token_hash_counts),
        "top_token_hashes": [
            {
                "token_hash": th,
                "rows": count,
                "input_tokens": token_hash_tokens[th],
            }
            for th, count in token_hash_counts.most_common(20)
        ],
        "selection_score": {
            "avg": statistics.mean(c["selection_score"] for c in selected),
            "max": max(c["selection_score"] for c in selected),
        },
        "elapsed_seconds": time.perf_counter() - t0,
    }
    sidecar_tmp_path = resolved_sidecar_path + ".tmp"
    with open(sidecar_tmp_path, "w") as f:
        json.dump(sidecar, f, indent=2)
    os.replace(sidecar_tmp_path, resolved_sidecar_path)
    completions_volume.commit()

    summary = {
        "output_path": resolved_output_path,
        "sidecar_path": resolved_sidecar_path,
        "rows": len(selected),
        "candidate_rows": len(candidates),
        "skipped_rows": skipped_rows,
        "skipped_over_max_input": skipped_over_max_input,
        "skipped_over_max_total": skipped_over_max_total,
        "skipped_under_min_input": skipped_under_min_input,
        "selection_mode": selection_mode,
        "selected_token_hashes": selected_token_hashes,
        "block_size": block_size,
        "unique_blocks": len(selected_blocks),
        "total_blocks": total_blocks,
        "block_reuse_pct": sidecar["block_reuse_pct"],
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "original_duration_ms": original_duration_ms,
        "scaled_duration_ms": sidecar["scaled_duration_ms"],
        "time_scale": effective_scale,
        "elapsed_seconds": time.perf_counter() - t0,
    }
    print(
        f"[mooncake] wrote {summary['rows']} rows to {resolved_output_path} "
        f"({summary['total_input_tokens']:,} input / {summary['total_output_tokens']:,} "
        f"output tokens, {summary['unique_blocks']:,} unique {block_size}-token blocks, "
        f"{summary['elapsed_seconds']:.1f}s)"
    )
    return summary


def _parse_iso(s: str | None):
    """Parse an ISO 8601 string into a naive (tz-stripped) datetime.

    Mirrors ``proxy/workload.py::_parse_iso`` so trace builds and replays
    interpret CLI timestamps the same way.
    """
    from datetime import datetime

    if not s:
        return None
    s = s.strip()
    if not s:
        return None
    d = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return d.replace(tzinfo=None) if d.tzinfo else d


def _to_naive_dt(ts):
    """Coerce a parquet ``timestamp`` value (datetime or ISO string) into a
    naive ``datetime``; returns ``None`` for anything we can't parse."""
    from datetime import datetime

    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts.replace(tzinfo=None) if ts.tzinfo else ts
    if isinstance(ts, str):
        try:
            d = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            return None
        return d.replace(tzinfo=None) if d.tzinfo else d
    return None


def _select_files(data_dir, start_dt, end_dt):
    """Pick GLM5 parquets in ``[start_dt, end_dt)`` by filename timestamp,
    plus the file immediately before ``start_dt`` because chunks may
    straddle the boundary -- the row-level filter handles the rest.

    Same logic as ``proxy/workload.py::_select_files``.
    """
    import os
    from datetime import datetime

    names = sorted(
        f for f in os.listdir(data_dir) if f.startswith("llm_responses_") and f.endswith(".parquet")
    )
    dated: list[tuple[datetime, str]] = []
    for name in names:
        stem = name[len("llm_responses_") : -len(".parquet")]
        try:
            ts = datetime.strptime(stem, "%Y%m%d_%H%M%S")
        except ValueError:
            continue
        dated.append((ts, name))
    dated.sort()

    selected: list[str] = []
    last_before_start: str | None = None
    for ts, name in dated:
        if start_dt is not None and ts < start_dt:
            last_before_start = name
            continue
        if end_dt is not None and ts >= end_dt:
            break
        selected.append(name)
    if start_dt is not None and last_before_start is not None:
        selected.insert(0, last_before_start)
    return selected


def _content_to_str(content) -> str:
    """Flatten an OpenAI ``message.content`` value to a plain string.

    Most rows are plain strings; multimodal blocks ``[{"type": "text",
    "text": "..."}, ...]`` are concatenated text-only.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text") or block.get("content") or ""
                if isinstance(text, str) and text:
                    parts.append(text)
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


@app.local_entrypoint()
def main(
    start_time: str,
    num_requests: int,
    output_path: str = "",
    end_time: str = "",
    block_size: int = DEFAULT_BLOCK_SIZE,
    data_dir: str = DEFAULT_DATA_DIR,
    include_bodies: bool = False,
    include_raw_bodies: bool = False,
    time_scale: float = 1.0,
    target_duration_ms: int = 0,
    selection_mode: str = "chronological",
    candidate_multiplier: int = 1,
    top_token_hashes: int = 0,
    max_input_tokens: int = 0,
    max_total_tokens: int = 0,
    min_input_tokens: int = 0,
    output_sidecar_path: str = "",
):
    """CLI wrapper for ``build_mooncake_trace``. Empty-string sentinels map
    to ``None`` because Modal local_entrypoints don't accept ``Optional``
    natively."""
    build_mooncake_trace.remote(
        start_time=start_time,
        num_requests=num_requests,
        output_path=output_path or None,
        end_time=end_time or None,
        block_size=block_size,
        data_dir=data_dir,
        include_bodies=include_bodies,
        include_raw_bodies=include_raw_bodies,
        time_scale=time_scale,
        target_duration_ms=target_duration_ms,
        selection_mode=selection_mode,
        candidate_multiplier=candidate_multiplier,
        top_token_hashes=top_token_hashes,
        max_input_tokens=max_input_tokens,
        max_total_tokens=max_total_tokens,
        min_input_tokens=min_input_tokens,
        output_sidecar_path=output_sidecar_path or None,
    )
