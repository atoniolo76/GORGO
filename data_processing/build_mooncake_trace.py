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

image = modal.Image.debian_slim().pip_install("pyarrow", "tiktoken").add_local_python_source("app")

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

    Returns:
        Dict with the resolved output path plus a few summary stats. The
        bulk of the result is the JSONL file written to the volume.
    """
    import hashlib
    import json
    import os
    import time
    from datetime import datetime, timezone

    import pyarrow.parquet as pq
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

    # First pass: walk parquets, tokenize, collect entries with absolute
    # timestamps. We need everything in memory anyway because (a) the JSONL
    # is sorted by timestamp before being relativized, and (b) ClickHouse
    # rows in adjacent batches can drift slightly out of order.
    print(
        f"[mooncake] walking {len(files)} parquet file(s) under {data_dir!r}; "
        f"start={start_time} end={end_time or '-'} target={num_requests} rows",
        flush=True,
    )
    t0 = time.perf_counter()
    entries: list[tuple[datetime, int, int, list[int]]] = []
    skipped_rows = 0
    for filename in files:
        if len(entries) >= num_requests:
            break
        path = os.path.join(data_dir, filename)
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=2048, columns=["timestamp", "request", "response"]):
            ts_col = batch.column("timestamp").to_pylist()
            req_col = batch.column("request").to_pylist()
            resp_col = batch.column("response").to_pylist()
            for ts, request_raw, response_raw in zip(ts_col, req_col, resp_col):
                ts_dt = _to_naive_dt(ts)
                if ts_dt is None:
                    skipped_rows += 1
                    continue
                if ts_dt < start_dt:
                    continue
                if end_dt is not None and ts_dt >= end_dt:
                    break
                try:
                    body = json.loads(request_raw) if isinstance(request_raw, str) else request_raw
                except (TypeError, ValueError):
                    skipped_rows += 1
                    continue
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
                entries.append((ts_dt, len(prompt_ids), output_length, prompt_ids))
                if len(entries) >= num_requests:
                    break
            else:
                continue
            break
        print(
            f"[mooncake]   {filename}: collected {len(entries)}/{num_requests} "
            f"({time.perf_counter() - t0:.1f}s elapsed)",
            flush=True,
        )

    if not entries:
        raise SystemExit(
            f"no rows matched start_time={start_time!r} end_time={end_time!r} under {data_dir!r}"
        )

    # Adjacent parquet batches can interleave near boundaries; sort by
    # timestamp so the emitted ``timestamp`` deltas are monotonic.
    entries.sort(key=lambda e: e[0])
    base_ts = entries[0][0]

    total_input = 0
    total_output = 0
    tmp_path = resolved_output_path + ".tmp"
    with open(tmp_path, "w") as f:
        for ts_dt, input_length, output_length, prompt_ids in entries:
            delta_ms = int((ts_dt - base_ts).total_seconds() * 1000)
            row = {
                "timestamp": delta_ms,
                "input_length": input_length,
                "output_length": output_length,
                "hash_ids": _block_ids(prompt_ids),
            }
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
            total_input += input_length
            total_output += output_length
    os.replace(tmp_path, resolved_output_path)
    completions_volume.commit()

    summary = {
        "output_path": resolved_output_path,
        "rows": len(entries),
        "skipped_rows": skipped_rows,
        "block_size": block_size,
        "unique_blocks": len(hash_to_id),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "duration_ms": int((entries[-1][0] - base_ts).total_seconds() * 1000),
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
    )
