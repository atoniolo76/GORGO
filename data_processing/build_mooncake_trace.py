"""Convert GLM 5.1 parquet shards or HF chat datasets (LMSYS-Chat-1M /
WildChat-4.8M) into a Mooncake FAST '25 trace JSONL.

Three sources, selected with ``--source``:

* ``glm5`` (default) -- walks the GLM 5.1 ClickHouse export on the
  ``GORGO-glm5-completions`` volume (mounted at ``/data``) starting from
  ``--start-time`` and consumes up to ``--num-requests`` rows. Real
  per-request timestamps come from the parquet ``timestamp`` column.

* ``lmsys`` -- walks the LMSYS-Chat-1M HF ``save_to_disk`` dataset on the
  ``GORGO-lmsys-chat-1m`` volume (mounted at ``/lmsys``). LMSYS has no
  per-request timestamps; arrivals are synthesized via a Poisson process
  at ``--arrival-rate-per-second`` (with ``--arrival-seed`` for
  reproducibility). Each row's conversation is split at its last
  ``assistant`` turn: everything before becomes the replay request; the
  assistant's content becomes the synthetic response used for
  ``output_length`` and (when ``--include-bodies``) the embedded body.

* ``wildchat`` -- walks the allenai/WildChat-4.8M HF ``save_to_disk``
  dataset on the ``GORGO-hf-datasets`` volume (mounted at ``/datasets``).
  WildChat has a ``timestamp`` column, so real arrivals are honored when
  present; rows without a parseable timestamp fall back to the Poisson
  synthesizer (matching the LMSYS code path).

Common output format (per https://github.com/kvcache-ai/Mooncake FAST25-release)::

    {"timestamp": 0,    "input_length": 6955, "output_length": 52, "hash_ids": [0, 1, 2, ...]}
    {"timestamp": 3053, "input_length": 6472, "output_length": 26, "hash_ids": [0, 1, 2, ...]}

- ``timestamp``       request arrival time in milliseconds, relative to the
                      first emitted row (which is always ``0``).
- ``input_length``    prompt token count.
- ``output_length``   response token count.
- ``hash_ids``        block-level prompt hashes, remapped to consecutive
                      integers. Hashing is prefix-aware -- two requests
                      sharing a K-token prompt prefix get the same first
                      ``ceil(K / block_size)`` ids, which is what makes
                      these traces useful for KV-cache replay simulation.

GLM5 walking matches ``proxy/workload.py`` (filename-timestamp file selection
plus a row-level timestamp filter), and tokenization uses tiktoken
``gpt-4o`` for parity with ``data_processing/build_eval_dataset.py``. The
exact tokenizer doesn't matter for replay -- what matters is that input
and output lengths are consistent across the trace.

``--target-input-tokens`` (when > 0) standardizes traces across datasets
to the same total prefill work: collection stops as soon as accumulated
``input_length`` reaches the target. Combined with chronological
selection this yields directly-comparable traces across datasets with
wildly different prompt-length distributions (LMSYS ~500 tok/req,
WildChat ~3k, GLM5 ~17k chronological / ~6k token-hash-filtered).

Usage::

    # GLM5 (existing):
    modal run data_processing/build_mooncake_trace.py \\
        --start-time 2026-04-01T12:00:00 --num-requests 10000

    # LMSYS-Chat-1M with synthetic Poisson at 60 req/s:
    modal run data_processing/build_mooncake_trace.py \\
        --source lmsys --num-requests 60000 \\
        --arrival-rate-per-second 60 --include-bodies

    # WildChat-4.8M, target 30M total input tokens:
    modal run data_processing/build_mooncake_trace.py \\
        --source wildchat --num-requests 50000 \\
        --target-input-tokens 30000000 \\
        --arrival-rate-per-second 11 --include-bodies

Output JSONL is written to the ``GORGO-glm5-completions`` volume (default
``/data/mooncake_traces/mooncake_<UTC-timestamp>.jsonl``).
"""

from __future__ import annotations

import modal

from app import app, completions_volume, hf_datasets_volume, lmsys_chat_1m_volume

image = (
    modal.Image.debian_slim()
    .pip_install("duckdb", "pyarrow", "tiktoken", "datasets>=3.0")
    .add_local_python_source("app")
)

DEFAULT_BLOCK_SIZE = 512
DEFAULT_DATA_DIR = "/data"

# ---- Source selection ----
SOURCE_GLM5 = "glm5"
SOURCE_LMSYS = "lmsys"
SOURCE_WILDCHAT = "wildchat"
SUPPORTED_SOURCES = (SOURCE_GLM5, SOURCE_LMSYS, SOURCE_WILDCHAT)

# Default per-source dataset roots. ``glm5`` lives on the completions
# volume directly; the HF datasets are on their own volumes.
LMSYS_DEFAULT_PATH = "/lmsys/lmsys-chat-1m"
WILDCHAT_DEFAULT_PATH = "/datasets/datasets/allenai__WildChat-4.8M"

# HF column-detection vocab (in priority order).
HF_CONV_COLUMNS = ("conversation", "messages", "conversations")
HF_TIMESTAMP_COLUMNS = ("timestamp", "created_at", "ts")

# Group-by columns valid as a per-row "user" key for the
# token-hash-filter / top-users selection modes on HF sources. Anything
# outside this set is allowed but warned (degraded to per-row key).
HF_GROUP_BY_LMSYS = frozenset({"model", "language", "conversation_id"})
HF_GROUP_BY_WILDCHAT = frozenset({"model", "language", "country", "hashed_ip", "conversation_hash"})

_VALID_ROLES = frozenset({"system", "user", "assistant", "tool", "function"})


@app.function(
    image=image,
    timeout=4 * 60 * 60,
    memory=1024 * 16,
    volumes={
        "/data": completions_volume,
        "/lmsys": lmsys_chat_1m_volume,
        "/datasets": hf_datasets_volume,
    },
)
def build_mooncake_trace(
    num_requests: int,
    source: str = SOURCE_GLM5,
    start_time: str | None = None,
    output_path: str | None = None,
    end_time: str | None = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
    data_dir: str | None = None,
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
    lmsys_group_by: str = "model",
    wildchat_group_by: str = "model",
    arrival_rate_per_second: float = 1.0,
    arrival_seed: int = 0,
    target_input_tokens: int = 0,
    target_unique_input_tokens: int = 0,
    force_synthetic_arrivals: bool = False,
    skip_rows: int = 0,
) -> dict:
    """Walk a chat dataset and emit a Mooncake FAST '25 trace JSONL.

    Args:
        num_requests: Upper bound on emitted rows. Walking stops as soon
            as this many valid candidates have been collected (or as soon
            as ``target_input_tokens`` is reached, whichever is first).
        source: One of ``glm5`` (default; parquet shards) / ``lmsys`` (HF
            ``save_to_disk`` LMSYS-Chat-1M) / ``wildchat`` (HF
            ``save_to_disk`` allenai/WildChat-4.8M).
        start_time: ISO 8601 timestamp. Required for ``glm5``; ignored
            for ``lmsys`` (no row timestamps) and currently ignored for
            ``wildchat`` (full-dataset filter is too expensive on 4.8M
            rows; truncate the trace via ``num_requests`` /
            ``target_input_tokens`` instead).
        output_path: Where to write the JSONL inside the volume. Relative
            paths are resolved under ``/data``. ``None`` (default) ->
            auto-generated ``mooncake_traces/mooncake_<UTC-timestamp>.jsonl``.
        end_time: Optional ISO 8601 upper bound for ``glm5``; ignored for
            HF sources (see ``start_time``).
        block_size: Token count per KV-cache block when hashing the prompt.
            Mooncake uses 512 in their published traces.
        data_dir: Source-specific dataset root. Defaults: ``/data`` for
            ``glm5``, ``/lmsys/lmsys-chat-1m`` for ``lmsys``,
            ``/datasets/datasets/allenai__WildChat-4.8M`` for ``wildchat``.
        include_bodies: Include parsed request/response JSON objects so the
            trace can be replayed against the proxy.
        include_raw_bodies: Include raw request/response strings for exact
            debugging/archival payloads. ``glm5`` only -- HF sources don't
            preserve original API payloads.
        time_scale: Multiply all relative timestamps by this factor.
        target_duration_ms: Alternative to ``time_scale``; when positive,
            compress/expand the selected trace to this total duration.
        selection_mode: ``chronological`` (default), ``top-users``,
            ``high-overlap``, ``mixed-overlap``, or ``token-hash-filter``.
            Required to be ``chronological`` when ``target_input_tokens > 0``.
        candidate_multiplier: Scan up to ``num_requests * candidate_multiplier``
            valid candidates before selecting the final trace.
        top_token_hashes: For ``token-hash-filter``, keep requests from the
            top-K token_hash values in the candidate pool. ``0`` picks a small
            default based on candidate diversity. For HF sources, the
            "token_hash" comes from the source's ``*_group_by`` knob.
        max_input_tokens / max_total_tokens / min_input_tokens: Context-length
            filters applied before selection. Positive values enable checks.
        output_sidecar_path: Optional summary JSON path inside the volume.
        lmsys_group_by: Column to use as the per-row "user" key for the
            ``token-hash-filter`` and ``top-users`` modes when
            ``source=lmsys``. One of ``model``, ``language``,
            ``conversation_id``. ``conversation_id`` is unique per row,
            so it degrades token-hash-filter to chronological.
        wildchat_group_by: Same but for ``source=wildchat``. One of
            ``model``, ``language``, ``country``, ``hashed_ip``,
            ``conversation_hash``.
        arrival_rate_per_second: Synthetic Poisson arrival rate (req/sec)
            applied to the selected rows for HF sources without usable
            real timestamps (always for LMSYS; for WildChat only when the
            row's timestamp column is missing or unparseable).
        arrival_seed: PRNG seed for the Poisson process so the same
            ``(num_requests, selection_mode, arrival_rate, seed)`` always
            produces identical timestamps.
        target_input_tokens: When > 0, treat as a stop criterion during
            collection: stop scanning as soon as accumulated
            ``input_length`` over kept candidates reaches this target.
            Standardizes total prefill work across datasets with
            different prompt-length distributions. Only valid with
            ``selection_mode=chronological`` (other modes oversample
            then truncate, which doesn't compose with a token target).
        target_unique_input_tokens: When > 0, treat as a stop criterion
            using only *unique* prefill work -- tokens belonging to
            block-hash digests not already seen in any earlier accepted
            candidate. This is the "post-cache-reuse" metric: in a
            multi-turn conversation, request N+1 typically encodes
            request N as its prompt prefix, so naively counting
            ``input_length`` double-counts the cached portion. Stopping
            on unique tokens equates fleet load across high-reuse
            (e.g. GLM5 ~82%) and low-reuse (e.g. WildChat ~5%) datasets
            so policies are stress-tested on comparable real prefill
            work. Same chronological-only restriction as
            ``target_input_tokens``. When both are set, collection stops
            on whichever is reached first.
        skip_rows: Number of dataset rows to skip before starting
            collection (HF sources only). ``0`` (default) starts from
            the first row. Use to grab non-overlapping windows from the
            same dataset: window 1 = ``skip_rows=0, num_requests=N``;
            window 2 = ``skip_rows=N, num_requests=N``.
        force_synthetic_arrivals: When True (HF sources only), ignore
            any per-row timestamp column on the dataset and always use
            the synthetic Poisson process. Useful for WildChat -- whose
            real ``timestamp`` column spans days across the dataset
            (~0.02 req/s natural rate, useless for saturation testing)
            -- so the arrival rate matches ``arrival_rate_per_second``
            instead of the dataset's collection cadence.

    Returns:
        Dict with the resolved output path plus a few summary stats. The
        bulk of the result is the JSONL file written to the volume.
    """
    import hashlib
    import json
    import os
    import random
    import statistics
    import time
    from collections import Counter
    from datetime import datetime, timedelta, timezone

    import duckdb
    import tiktoken

    if source not in SUPPORTED_SOURCES:
        raise SystemExit(f"--source must be one of {list(SUPPORTED_SOURCES)} (got {source!r})")

    if data_dir is None:
        data_dir = {
            SOURCE_GLM5: DEFAULT_DATA_DIR,
            SOURCE_LMSYS: LMSYS_DEFAULT_PATH,
            SOURCE_WILDCHAT: WILDCHAT_DEFAULT_PATH,
        }[source]

    # Reload only the volume(s) we read from. Output always lands on
    # completions_volume so reload that one too in case a sibling job
    # just dropped a sidecar there.
    if source == SOURCE_GLM5:
        completions_volume.reload()
    elif source == SOURCE_LMSYS:
        lmsys_chat_1m_volume.reload()
        completions_volume.reload()
    elif source == SOURCE_WILDCHAT:
        hf_datasets_volume.reload()
    completions_volume.reload()

    start_dt = _parse_iso(start_time)
    if source == SOURCE_GLM5 and start_dt is None:
        raise SystemExit(f"--start-time is required for source=glm5 (got {start_time!r})")
    if source != SOURCE_GLM5 and (start_time or end_time):
        # HF sources don't filter on row timestamps in this builder
        # (LMSYS has none; WildChat would require a full-dataset
        # ``ds.filter`` scan over 4.8M rows). Truncate via num_requests
        # / target_input_tokens instead. Warn rather than error so
        # callers retargeting an existing GLM5 invocation just lose the
        # filter cleanly.
        print(
            f"[mooncake] note: --start-time / --end-time ignored for source={source!r} "
            f"(use --num-requests / --target-input-tokens to bound the trace)",
            flush=True,
        )
    end_dt = _parse_iso(end_time) if source == SOURCE_GLM5 else None
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
    if target_input_tokens < 0:
        raise SystemExit(f"--target-input-tokens must be >= 0 (got {target_input_tokens})")
    if target_unique_input_tokens < 0:
        raise SystemExit(
            f"--target-unique-input-tokens must be >= 0 (got {target_unique_input_tokens})"
        )
    if source in (SOURCE_LMSYS, SOURCE_WILDCHAT) and arrival_rate_per_second <= 0:
        raise SystemExit(
            f"--arrival-rate-per-second must be > 0 for HF sources (got {arrival_rate_per_second})"
        )
    if skip_rows < 0:
        raise SystemExit(f"--skip-rows must be >= 0 (got {skip_rows})")
    if skip_rows > 0 and source not in (SOURCE_LMSYS, SOURCE_WILDCHAT):
        raise SystemExit(
            f"--skip-rows only applies to HF sources (lmsys / wildchat); got source={source!r}"
        )
    if force_synthetic_arrivals and source not in (SOURCE_LMSYS, SOURCE_WILDCHAT):
        raise SystemExit(
            f"--force-synthetic-arrivals only applies to HF sources "
            f"(lmsys / wildchat); got source={source!r}"
        )
    supported_modes = {
        "chronological",
        "top-users",
        "high-overlap",
        "mixed-overlap",
        "token-hash-filter",
    }
    if selection_mode not in supported_modes:
        raise SystemExit(f"--selection-mode must be one of {sorted(supported_modes)}")
    if target_input_tokens > 0 and selection_mode != "chronological":
        # Other modes oversample then truncate by overlap/user score,
        # which doesn't compose with a token target -- you'd either
        # under-fill (target reached pre-selection) or over-shoot
        # (selection picks longer prompts). Restrict to keep semantics
        # simple and reproducible.
        raise SystemExit(
            "--target-input-tokens > 0 requires --selection-mode chronological "
            f"(got {selection_mode!r})"
        )
    if target_unique_input_tokens > 0 and selection_mode != "chronological":
        # Same reasoning as ``target_input_tokens`` plus an extra:
        # the "unique" set is order-dependent (the first request to
        # carry a block defines it as cached), so non-chronological
        # selection would re-order the unique attribution.
        raise SystemExit(
            "--target-unique-input-tokens > 0 requires "
            f"--selection-mode chronological (got {selection_mode!r})"
        )

    if source == SOURCE_GLM5:
        files = _select_files(data_dir, start_dt, end_dt)
        if not files:
            raise SystemExit(
                f"no GLM5 parquet files match the requested time range under {data_dir!r}"
            )
    else:
        if not os.path.isdir(data_dir):
            raise SystemExit(f"HF dataset path not found: {data_dir!r}")
        files = []  # not used by HF path; kept for sidecar bookkeeping

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

    # First pass: walk source rows, tokenize, collect candidates with
    # (real or placeholder) timestamps. We scan more than we emit for
    # overlap-based curation; ``target_input_tokens > 0`` overrides the
    # candidate cap with a token-budget stop criterion (chronological
    # only, validated above).
    if source == SOURCE_GLM5:
        print(
            f"[mooncake] walking {len(files)} parquet file(s) under {data_dir!r}; "
            f"start={start_time} end={end_time or '-'} target={num_requests} rows "
            f"mode={selection_mode} candidates={num_requests * candidate_multiplier} "
            f"target_input_tokens={target_input_tokens}",
            flush=True,
        )
    else:
        group_by = lmsys_group_by if source == SOURCE_LMSYS else wildchat_group_by
        print(
            f"[mooncake] walking HF dataset at {data_dir!r}; "
            f"source={source} target={num_requests} rows mode={selection_mode} "
            f"group_by={group_by} arrival_rate={arrival_rate_per_second}/s "
            f"candidates={num_requests * candidate_multiplier} "
            f"target_input_tokens={target_input_tokens}",
            flush=True,
        )
    t0 = time.perf_counter()
    target_candidates = max(num_requests, num_requests * candidate_multiplier)
    candidates: list[dict] = []
    skipped_rows = 0
    skipped_over_max_input = 0
    skipped_over_max_total = 0
    skipped_under_min_input = 0
    accumulated_input_tokens = 0  # for target_input_tokens stop criterion
    accumulated_unique_input_tokens = 0  # for target_unique_input_tokens
    # Block-id set for incremental "unique" accounting. A block id is
    # the prefix-aware sha256-derived integer assigned by ``_block_ids``;
    # the first time a digest is seen across collected candidates it
    # gets a fresh id, and that id stays the same for any subsequent
    # candidate carrying the same prefix block. Tracking the set here
    # (rather than recomputing from the final ``candidates`` list)
    # lets us early-stop on ``target_unique_input_tokens``.
    seen_block_ids: set[int] = set()

    def _token_target_reached() -> bool:
        if target_input_tokens > 0 and accumulated_input_tokens >= target_input_tokens:
            return True
        if (
            target_unique_input_tokens > 0
            and accumulated_unique_input_tokens >= target_unique_input_tokens
        ):
            return True
        return False

    def _record_unique_for(block_ids: list[int], input_length: int) -> int:
        """Return the unique-token contribution of a candidate's block list
        and update ``seen_block_ids`` in place.

        Counts ``block_size`` per new block, but caps the total at
        ``input_length`` so short prompts (< block_size) don't inflate
        the unique-token count beyond their actual token count."""
        new_ids = [b for b in block_ids if b not in seen_block_ids]
        seen_block_ids.update(block_ids)
        return min(len(new_ids) * block_size, input_length)

    if source == SOURCE_GLM5:
        con = duckdb.connect()
        for filename in files:
            if len(candidates) >= target_candidates or _token_target_reached():
                break
            path = os.path.join(data_dir, filename)
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
                    unique_input_tokens = _record_unique_for(block_ids, input_length)
                    candidates.append(
                        {
                            "uuid": uuid,
                            "ts_dt": ts_dt,
                            "token_hash": token_hash or "",
                            "input_length": input_length,
                            "output_length": output_length,
                            "unique_input_tokens": unique_input_tokens,
                            "hash_ids": block_ids,
                            "request": body,
                            "response": response_body,
                            "request_raw": request_raw,
                            "response_raw": response_raw,
                        }
                    )
                    accumulated_input_tokens += input_length
                    accumulated_unique_input_tokens += unique_input_tokens
                    if len(candidates) >= target_candidates or _token_target_reached():
                        break
                else:
                    continue
                break
            print(
                f"[mooncake]   {filename}: collected {len(candidates)}/{target_candidates} "
                f"candidates ({accumulated_input_tokens:,} input tokens, "
                f"{accumulated_unique_input_tokens:,} unique, "
                f"{time.perf_counter() - t0:.1f}s elapsed)",
                flush=True,
            )
        con.close()
    else:
        # ---- HF source (LMSYS / WildChat) ----
        from datasets import Dataset, DatasetDict, load_from_disk

        dsd = load_from_disk(data_dir)
        if isinstance(dsd, DatasetDict):
            ds = dsd["train"] if "train" in dsd else dsd[next(iter(dsd))]
        elif isinstance(dsd, Dataset):
            ds = dsd
        else:
            raise SystemExit(f"unsupported HF dataset object at {data_dir!r}: {type(dsd)!r}")
        columns = set(ds.column_names)
        conv_col: str | None = None
        for k in HF_CONV_COLUMNS:
            if k in columns:
                conv_col = k
                break
        if conv_col is None:
            raise SystemExit(
                f"HF dataset at {data_dir!r} has none of {HF_CONV_COLUMNS}; "
                f"available columns: {sorted(columns)}"
            )
        ts_col: str | None = None
        if source == SOURCE_WILDCHAT:
            for k in HF_TIMESTAMP_COLUMNS:
                if k in columns:
                    ts_col = k
                    break
            if ts_col is None:
                print(
                    f"[mooncake] note: WildChat dataset at {data_dir!r} has no "
                    f"timestamp column from {HF_TIMESTAMP_COLUMNS}; falling back "
                    f"to synthetic Poisson arrivals",
                    flush=True,
                )
        group_by_col = lmsys_group_by if source == SOURCE_LMSYS else wildchat_group_by
        if group_by_col not in columns:
            print(
                f"[mooncake] warning: --{source}-group-by={group_by_col!r} not in "
                f"dataset columns {sorted(columns)}; falling back to per-row key "
                f"(token-hash-filter / top-users will degrade to chronological)",
                flush=True,
            )
        print(
            f"[mooncake]   dataset has {len(ds)} rows; conv_col={conv_col!r} "
            f"ts_col={ts_col!r} group_by={group_by_col!r}",
            flush=True,
        )
        # Placeholder timestamps preserve dataset row order under
        # chronological selection; real (or synthetic) timestamps are
        # assigned post-selection.
        placeholder_origin = datetime(2026, 1, 1)
        for row_idx, row in enumerate(ds):
            if row_idx < skip_rows:
                continue
            if len(candidates) >= target_candidates or _token_target_reached():
                break
            raw_conv = row[conv_col]
            msgs_full = _normalize_chat_messages(raw_conv)
            if not msgs_full:
                skipped_rows += 1
                continue
            request_msgs, response_text = _split_at_last_assistant(msgs_full)
            if not request_msgs:
                skipped_rows += 1
                continue
            prompt_ids = _tokenize_messages(request_msgs)
            if not prompt_ids:
                skipped_rows += 1
                continue
            output_length = (
                len(enc.encode(response_text, disallowed_special=())) if response_text else 0
            )
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
            block_ids = _block_ids(prompt_ids)
            unique_input_tokens = _record_unique_for(block_ids, input_length)
            uuid = str(
                row.get("conversation_id")
                or row.get("conversation_hash")
                or f"{source}_row{row_idx}"
            )
            if group_by_col in columns:
                v = row.get(group_by_col)
                group_key = "" if v is None else str(v)
            else:
                # No grouping column -> per-row key so token-hash-filter
                # still works (degraded to chronological).
                group_key = uuid
            # Real timestamp from WildChat ``timestamp`` column when
            # available; else placeholder (overwritten post-selection
            # with Poisson). We tag rows here with whether the ts is
            # real so the post-selection step knows which to overwrite.
            real_ts = False
            if ts_col is not None:
                ts_dt = _to_naive_dt(row.get(ts_col))
                if ts_dt is not None:
                    real_ts = True
            if not real_ts:
                ts_dt = placeholder_origin + timedelta(seconds=row_idx)
            request_body = {"messages": request_msgs}
            response_body = (
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": response_text,
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": input_length,
                        "completion_tokens": output_length,
                        "total_tokens": input_length + output_length,
                    },
                }
                if response_text
                else None
            )
            candidates.append(
                {
                    "uuid": uuid,
                    "ts_dt": ts_dt,
                    "token_hash": group_key,
                    "input_length": input_length,
                    "output_length": output_length,
                    "unique_input_tokens": unique_input_tokens,
                    "hash_ids": block_ids,
                    "request": request_body,
                    "response": response_body,
                    "request_raw": "",
                    "response_raw": "",
                    "real_ts": real_ts,
                }
            )
            accumulated_input_tokens += input_length
            accumulated_unique_input_tokens += unique_input_tokens
            if (row_idx + 1) % 5000 == 0:
                print(
                    f"[mooncake]   scanned {row_idx + 1} rows; "
                    f"collected {len(candidates)}/{target_candidates} candidates "
                    f"({accumulated_input_tokens:,} input tokens, "
                    f"{accumulated_unique_input_tokens:,} unique, "
                    f"{time.perf_counter() - t0:.1f}s elapsed)",
                    flush=True,
                )

    if not candidates:
        raise SystemExit(
            f"no rows matched source={source!r} data_dir={data_dir!r} "
            f"start_time={start_time!r} end_time={end_time!r}"
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
        # When ``target_input_tokens`` or ``target_unique_input_tokens``
        # drove collection we already stopped at the right total; take
        # all candidates in the natural sort order rather than
        # truncating to ``num_requests`` (which might under-fill the
        # token target). When neither target is set the historical
        # truncate-to-num_requests behavior is preserved.
        if target_input_tokens > 0 or target_unique_input_tokens > 0:
            selected = sorted(candidates, key=lambda e: e["ts_dt"])
        else:
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

    # Post-selection Poisson timestamp synthesis for HF sources without
    # real per-row timestamps. LMSYS always synthesizes (no source ts);
    # WildChat synthesizes only for rows missing a parseable timestamp
    # *unless* ``force_synthetic_arrivals`` is set, in which case the
    # real timestamps are dropped and Poisson is used unconditionally.
    # The placeholder ``ts_dt`` (datetime(2026,1,1) + row-index seconds)
    # gets overwritten in selection order so earlier-selected rows get
    # earlier wall times.
    if force_synthetic_arrivals and source in (SOURCE_LMSYS, SOURCE_WILDCHAT):
        # Override per-row ``real_ts`` so the loop below rewrites every
        # selected row's timestamp from the Poisson process. Also clear
        # the placeholder ts_dt to make the override obvious in any
        # debugger output.
        for c in selected:
            c["real_ts"] = False
    needs_synthetic_ts = source in (SOURCE_LMSYS, SOURCE_WILDCHAT) and any(
        not c.get("real_ts", False) for c in selected
    )
    if needs_synthetic_ts:
        rng = random.Random(arrival_seed)
        synthetic_base = datetime(2026, 1, 1)
        elapsed_ms = 0.0
        for c in selected:
            if not c.get("real_ts", False):
                c["ts_dt"] = synthetic_base + timedelta(milliseconds=elapsed_ms)
                gap_s = rng.expovariate(arrival_rate_per_second)
                elapsed_ms += gap_s * 1000.0
        # Re-sort after overwriting in case any real-ts rows were
        # interleaved (defensive; shouldn't happen with current sources).
        selected.sort(key=lambda e: e["ts_dt"])

    base_ts = selected[0]["ts_dt"]
    original_duration_ms = int((selected[-1]["ts_dt"] - base_ts).total_seconds() * 1000)
    effective_scale = (
        (target_duration_ms / max(original_duration_ms, 1))
        if target_duration_ms > 0
        else time_scale
    )

    total_input = 0
    total_output = 0
    total_unique_input_tokens = 0
    total_blocks = 0
    selected_blocks: set[int] = set()
    # Re-accumulate unique tokens in EMIT order: per-candidate
    # ``unique_input_tokens`` was computed in COLLECTION order, which
    # only matches the emit set when chronological selection picks all
    # collected rows. Recomputing here keeps the JSONL row's
    # ``unique_input_tokens`` and the sidecar aggregate canonical for
    # whatever selection path actually ran.
    emit_seen_block_ids: set[int] = set()
    token_hash_counts = Counter(c.get("token_hash") or "" for c in selected)
    token_hash_tokens = Counter()
    tmp_path = resolved_output_path + ".tmp"
    with open(tmp_path, "w") as f:
        for idx, c in enumerate(selected):
            relative_ms = int((c["ts_dt"] - base_ts).total_seconds() * 1000)
            delta_ms = int(relative_ms * effective_scale)
            request_id = f"{source}_{base_ts.strftime('%Y%m%dT%H%M%S')}_{idx:06d}"
            new_block_count = sum(1 for b in c["hash_ids"] if b not in emit_seen_block_ids)
            unique_input_tokens = min(new_block_count * block_size, c["input_length"])
            emit_seen_block_ids.update(c["hash_ids"])
            row = {
                "timestamp": delta_ms,
                "input_length": c["input_length"],
                "output_length": c["output_length"],
                "unique_input_tokens": unique_input_tokens,
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
            total_unique_input_tokens += unique_input_tokens
            total_blocks += len(c["hash_ids"])
            selected_blocks.update(c["hash_ids"])
            token_hash_tokens[c.get("token_hash") or ""] += c["input_length"]
    os.replace(tmp_path, resolved_output_path)

    sidecar = {
        "output_path": resolved_output_path,
        "source": source,
        "data_dir": data_dir,
        "lmsys_group_by": lmsys_group_by if source == SOURCE_LMSYS else None,
        "wildchat_group_by": wildchat_group_by if source == SOURCE_WILDCHAT else None,
        "arrival_rate_per_second": (
            arrival_rate_per_second if source in (SOURCE_LMSYS, SOURCE_WILDCHAT) else None
        ),
        "arrival_seed": (arrival_seed if source in (SOURCE_LMSYS, SOURCE_WILDCHAT) else None),
        "synthetic_arrivals_used": needs_synthetic_ts,
        "force_synthetic_arrivals": force_synthetic_arrivals,
        "skip_rows": skip_rows or None,
        "target_input_tokens": target_input_tokens or None,
        "target_unique_input_tokens": target_unique_input_tokens or None,
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
        "start_time": start_time if source == SOURCE_GLM5 else None,
        "end_time": end_time if source == SOURCE_GLM5 else None,
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
        "total_unique_input_tokens": total_unique_input_tokens,
        "unique_token_share_pct": (
            100.0 * total_unique_input_tokens / total_input if total_input else 0.0
        ),
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
        "total_unique_input_tokens": total_unique_input_tokens,
        "unique_token_share_pct": sidecar["unique_token_share_pct"],
        "original_duration_ms": original_duration_ms,
        "scaled_duration_ms": sidecar["scaled_duration_ms"],
        "time_scale": effective_scale,
        "elapsed_seconds": time.perf_counter() - t0,
    }
    print(
        f"[mooncake] wrote {summary['rows']} rows to {resolved_output_path} "
        f"({summary['total_input_tokens']:,} input / "
        f"{summary['total_output_tokens']:,} output tokens, "
        f"{summary['total_unique_input_tokens']:,} unique input "
        f"({summary['unique_token_share_pct']:.1f}% of input), "
        f"{summary['unique_blocks']:,} unique {block_size}-token blocks, "
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


def _normalize_chat_messages(raw) -> list[dict]:
    """Normalize a HF row's conversation column into a chat-completions
    ``messages`` list. Mirrors ``proxy.workload_core._hf_normalize_messages``
    so traces built here replay identically to live HF reads.

    Drops empty messages; coerces unrecognized roles to ``user``.
    Accepts JSON-string columns (parses them) and lists of dicts/strings.
    """
    import json as _json

    if isinstance(raw, str):
        try:
            raw = _json.loads(raw)
        except _json.JSONDecodeError:
            return []
    if not isinstance(raw, list):
        return []
    out: list[dict] = []
    for msg in raw:
        if isinstance(msg, str):
            text = msg
            role = "user"
        elif isinstance(msg, dict):
            role = str(msg.get("role") or "user").lower()
            text = _content_to_str(msg.get("content"))
        else:
            continue
        if not text:
            continue
        if role not in _VALID_ROLES:
            role = "user"
        out.append({"role": role, "content": text})
    return out


def _split_at_last_assistant(msgs: list[dict]) -> tuple[list[dict], str]:
    """Split a normalized message list at its last ``assistant`` turn.

    Returns ``(request_messages, response_text)``. The request is
    everything *before* the last assistant turn (so ``messages`` ends
    with a ``user`` or ``system`` turn, ready for chat-completions).
    The response is the assistant's content (used for ``output_length``
    and the embedded synthetic response body).

    If there's no assistant turn, returns the full conversation as the
    request and an empty response. If the only assistant turn is the
    very first message, also returns the full conversation -- an empty
    "before" prefix isn't replayable.
    """
    last_assistant_idx: int | None = None
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i].get("role") == "assistant":
            last_assistant_idx = i
            break
    if last_assistant_idx is None or last_assistant_idx == 0:
        return msgs, ""
    return msgs[:last_assistant_idx], msgs[last_assistant_idx].get("content", "")


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
    num_requests: int,
    source: str = SOURCE_GLM5,
    start_time: str = "",
    output_path: str = "",
    end_time: str = "",
    block_size: int = DEFAULT_BLOCK_SIZE,
    data_dir: str = "",
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
    lmsys_group_by: str = "model",
    wildchat_group_by: str = "model",
    arrival_rate_per_second: float = 1.0,
    arrival_seed: int = 0,
    target_input_tokens: int = 0,
    target_unique_input_tokens: int = 0,
    force_synthetic_arrivals: bool = False,
    skip_rows: int = 0,
):
    """CLI wrapper for ``build_mooncake_trace``. Empty-string sentinels map
    to ``None`` because Modal local_entrypoints don't accept ``Optional``
    natively. ``--source glm5`` (default) keeps existing GLM5 behavior;
    ``--source lmsys`` and ``--source wildchat`` read HF datasets and
    synthesize Poisson arrivals where needed.

    ``--target-unique-input-tokens`` standardizes traces by post-cache-reuse
    prefill work (avoids double-counting cached prefixes in multi-turn
    conversations); ``--force-synthetic-arrivals`` overrides any real
    timestamp column on HF sources so the trace's arrival rate matches
    ``--arrival-rate-per-second`` (useful for WildChat, whose real
    timestamps span days)."""
    build_mooncake_trace.remote(
        num_requests=num_requests,
        source=source,
        start_time=start_time or None,
        output_path=output_path or None,
        end_time=end_time or None,
        block_size=block_size,
        data_dir=data_dir or None,
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
        lmsys_group_by=lmsys_group_by,
        wildchat_group_by=wildchat_group_by,
        arrival_rate_per_second=arrival_rate_per_second,
        arrival_seed=arrival_seed,
        target_input_tokens=target_input_tokens,
        target_unique_input_tokens=target_unique_input_tokens,
        force_synthetic_arrivals=force_synthetic_arrivals,
        skip_rows=skip_rows,
    )


analyze_image = modal.Image.debian_slim().add_local_python_source("app")


@app.function(
    image=analyze_image,
    volumes={"/data": completions_volume},
    timeout=30 * 60,
    memory=1024 * 8,
)
def analyze_trace_reuse(trace_paths_csv: str, max_input_tokens: int = 24000, max_tokens: int = 128):
    """Analyze KV cache reuse (global, intra-user, cross-user) for traces on the volume.

    Prints a JSON summary per trace with block reuse decomposed by user session.
    """
    import json
    from collections import defaultdict

    completions_volume.reload()
    results = []
    trace_paths = [p.strip() for p in trace_paths_csv.split(",") if p.strip()]

    for path in trace_paths:
        rows = []
        with open(path) as f:
            for line in f:
                rows.append(json.loads(line))

        effective_cap = max_input_tokens - max_tokens
        kept = [r for r in rows if r.get("input_length", 0) <= effective_cap]
        filtered = len(rows) - len(kept)

        global_blocks: set[int] = set()
        global_total = 0

        user_blocks: dict[str, set[int]] = defaultdict(set)
        user_total: dict[str, int] = defaultdict(int)
        user_rows: dict[str, int] = defaultdict(int)

        input_lengths = []
        output_lengths = []

        for r in kept:
            bids = r.get("hash_ids", [])
            user = r.get("token_hash", "unknown")
            input_lengths.append(r.get("input_length", 0))
            output_lengths.append(r.get("output_length", 0))

            global_total += len(bids)
            global_blocks.update(bids)

            user_total[user] += len(bids)
            user_blocks[user].update(bids)
            user_rows[user] += 1

        global_unique = len(global_blocks)
        global_reuse_pct = (
            100.0 * (global_total - global_unique) / global_total if global_total else 0.0
        )

        intra_user_reused = 0
        intra_user_total = 0
        for user in user_total:
            ut = user_total[user]
            uu = len(user_blocks[user])
            intra_user_total += ut
            intra_user_reused += ut - uu

        intra_user_reuse_pct = (
            100.0 * intra_user_reused / intra_user_total if intra_user_total else 0.0
        )

        all_user_unique = sum(len(bs) for bs in user_blocks.values())
        cross_user_reused = all_user_unique - global_unique
        cross_user_reuse_pct = 100.0 * cross_user_reused / global_total if global_total else 0.0

        sorted_input = sorted(input_lengths)
        n = len(sorted_input)
        p50_input = sorted_input[n // 2] if n else 0
        p95_input = sorted_input[int(n * 0.95)] if n else 0
        p99_input = sorted_input[int(n * 0.99)] if n else 0
        avg_input = sum(input_lengths) / n if n else 0
        avg_output = sum(output_lengths) / n if n else 0

        num_users = len(user_total)
        top_users = sorted(user_rows.items(), key=lambda x: -x[1])[:5]

        result = {
            "path": path,
            "name": path.split("/")[-1].replace(".jsonl", ""),
            "total_rows": len(rows),
            "kept_after_filter": len(kept),
            "filtered_out": filtered,
            "filter_cap": effective_cap,
            "num_users": num_users,
            "input_tokens": {
                "avg": round(avg_input),
                "p50": p50_input,
                "p95": p95_input,
                "p99": p99_input,
                "total": sum(input_lengths),
            },
            "output_tokens": {
                "avg": round(avg_output),
                "total": sum(output_lengths),
            },
            "blocks": {
                "total": global_total,
                "global_unique": global_unique,
                "global_reuse_pct": round(global_reuse_pct, 1),
                "intra_user_reuse_pct": round(intra_user_reuse_pct, 1),
                "cross_user_reuse_pct": round(cross_user_reuse_pct, 1),
            },
            "top_users_by_rows": [
                {"token_hash": th[:12] + "...", "rows": cnt} for th, cnt in top_users
            ],
        }
        results.append(result)
        print(json.dumps(result, indent=2))
        print(flush=True)

    return results


@app.function(
    image=analyze_image,
    volumes={"/data": completions_volume},
    timeout=60 * 60,
    memory=1024 * 16,
)
def simulate_pareto_sweep(
    trace_path: str,
    replicas_json: str,
    prefill_weights_csv: str = "0.01,0.05,0.1,0.2,0.5,1.0,1.5,2.0,3.0,5.0",
    rtt_weights_csv: str = "0.01,0.05,0.1,0.2,0.5,1.0,2.0,3.0,5.0,10.0,20.0",
    queue_weights_csv: str = "",
    concurrency: int = 64,
    max_tokens: int = 128,
    max_input_tokens: int = 24000,
    block_size: int = 512,
    decode_tok_per_s: float = 120.0,
):
    """Simulate GORGO routing for a grid of (prefill_weight, rtt_weight)
    and produce per-point TTFT/E2E percentiles for Pareto frontier analysis.

    Replays the trace through the scoring function with simulated queue
    dynamics and prefix cache tracking. No GPUs needed.

    ``replicas_json`` is a JSON list of objects, each with:
      - ``region``: human label
      - ``rtt_ms``: measured RTT in milliseconds
      - ``prefill_rate``: calibrated ms/tok
      - ``queue_rate``: fitted ms/tok (from live traffic)

    Example::

        [
          {"region": "us-ashburn-1", "rtt_ms": 32, "prefill_rate": 0.093, "queue_rate": 0.01},
          {"region": "eu-frankfurt-1", "rtt_ms": 364, "prefill_rate": 0.073, "queue_rate": 0.01},
          {"region": "ap-seoul-1", "rtt_ms": 602, "prefill_rate": 0.130, "queue_rate": 0.01}
        ]
    """
    import heapq
    import json

    completions_volume.reload()

    replicas = json.loads(replicas_json)
    prefill_weights = [float(x) for x in prefill_weights_csv.split(",")]
    rtt_weights = [float(x) for x in rtt_weights_csv.split(",")]

    effective_cap = max_input_tokens - max_tokens

    print(f"[pareto] loading trace {trace_path}", flush=True)
    rows = []
    with open(trace_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("input_length", 0) > effective_cap:
                continue
            rows.append(
                {
                    "ts_ms": r["timestamp"],
                    "input_length": r["input_length"],
                    "output_length": r.get("output_length", max_tokens),
                    "hash_ids": r.get("hash_ids", []),
                }
            )

    print(f"[pareto] loaded {len(rows)} requests, {len(replicas)} replicas", flush=True)

    decode_ms_per_tok = 1000.0 / decode_tok_per_s

    queue_weights = (
        [float(x) for x in queue_weights_csv.split(",") if x.strip()]
        if queue_weights_csv
        else [0.0]
    )

    total_points = len(prefill_weights) * len(rtt_weights) * len(queue_weights)
    print(
        f"[pareto] grid: {len(prefill_weights)} x {len(rtt_weights)} x {len(queue_weights)} = {total_points} points",
        flush=True,
    )

    from collections import Counter

    results = []
    for pw in prefill_weights:
        for rw in rtt_weights:
            for qw in queue_weights:
                ttfts, e2es, targets = _simulate_one(
                    rows=rows,
                    replicas=replicas,
                    prefill_weight=pw,
                    rtt_weight=rw,
                    queue_weight=qw,
                    concurrency=concurrency,
                    block_size=block_size,
                    decode_ms_per_tok=decode_ms_per_tok,
                )
                n = len(ttfts)
                if n == 0:
                    continue
                ttfts_sorted = sorted(ttfts)
                e2es_sorted = sorted(e2es)

                target_dist = Counter(targets)

                point = {
                    "prefill_weight": pw,
                    "rtt_weight": rw,
                    "queue_weight": qw,
                    "n": n,
                    "ttft_ms": {
                        "avg": sum(ttfts) / n,
                        "p50": ttfts_sorted[n // 2],
                        "p95": ttfts_sorted[int(n * 0.95)],
                        "p99": ttfts_sorted[int(n * 0.99)],
                    },
                    "e2e_ms": {
                        "avg": sum(e2es) / n,
                        "p50": e2es_sorted[n // 2],
                        "p95": e2es_sorted[int(n * 0.95)],
                        "p99": e2es_sorted[int(n * 0.99)],
                    },
                    "routing_distribution": {
                        replicas[i]["region"]: target_dist.get(i, 0) for i in range(len(replicas))
                    },
                }
                results.append(point)
                qw_str = f" qw={qw:<6}" if len(queue_weights) > 1 else ""
                print(
                    f"[pareto] pw={pw:<5} rw={rw:<5}{qw_str}  "
                    f"TTFT p50={point['ttft_ms']['p50']:>7.1f}  p95={point['ttft_ms']['p95']:>7.1f}  "
                    f"E2E  p50={point['e2e_ms']['p50']:>7.1f}  p95={point['e2e_ms']['p95']:>7.1f}  "
                    f"dist={dict(target_dist)}",
                    flush=True,
                )

    print(f"\n[pareto] completed {len(results)} grid points", flush=True)
    output_path = trace_path.rsplit(".", 1)[0] + "_pareto_sweep.json"
    with open(output_path, "w") as f:
        json.dump({"trace_path": trace_path, "replicas": replicas, "results": results}, f, indent=2)
    completions_volume.commit()
    print(f"[pareto] saved to {output_path}", flush=True)
    return results


def _simulate_one(
    *,
    rows: list[dict],
    replicas: list[dict],
    prefill_weight: float,
    rtt_weight: float,
    queue_weight: float = 0.0,
    concurrency: int,
    block_size: int,
    decode_ms_per_tok: float,
) -> tuple[list[float], list[float], list[int]]:
    """Event-driven simulation of GORGO routing for one weight pair.

    Queue dynamics emerge from serialized prefill scheduling — no
    queue_rate constant.  Each replica processes prefills one at a time;
    a request waits for all prefills ahead of it to complete before its
    own prefill starts.  Decode runs concurrently (batched) and does not
    block the next prefill.

    The scoring function uses ``queue_weight * queued_tokens`` as a
    routing signal (how the policy sees load), but the physical TTFT is
    determined by the actual prefill queue depth in milliseconds.

    Returns (ttft_list_ms, e2e_list_ms, target_indices).
    """
    import heapq

    n_replicas = len(replicas)

    # Per-replica prefix cache (hash_id based, simulating radix trie)
    prefix_cache: list[set[int]] = [set() for _ in range(n_replicas)]
    # Per-replica queued token counter (for the scoring function)
    queued_tokens = [0] * n_replicas
    # Per-replica: earliest time the GPU can start the next prefill (ms)
    prefill_available_at = [0.0] * n_replicas
    # Per-replica completion heap: (completion_time_ms, input_tokens)
    completion_heaps: list[list[tuple[float, int]]] = [[] for _ in range(n_replicas)]

    ttfts: list[float] = []
    e2es: list[float] = []
    targets: list[int] = []

    for row in rows:
        arrive_ms = float(row["ts_ms"])
        input_len = row["input_length"]
        output_len = row["output_length"]
        hash_ids = row["hash_ids"]

        # Drain completed requests from all replicas up to arrival time
        for i in range(n_replicas):
            heap = completion_heaps[i]
            while heap and heap[0][0] <= arrive_ms:
                _, completed_tokens = heapq.heappop(heap)
                queued_tokens[i] = max(0, queued_tokens[i] - completed_tokens)

        # Compute cached prefix length per replica (shared hash_id prefix)
        cached_tokens = []
        for i in range(n_replicas):
            cached_blocks = 0
            for bid in hash_ids:
                if bid in prefix_cache[i]:
                    cached_blocks += 1
                else:
                    break
            cached_tokens.append(cached_blocks * block_size)

        # Score each replica (routing decision)
        best_idx = 0
        best_score = float("inf")
        for i in range(n_replicas):
            rep = replicas[i]
            uncached = max(0, input_len - cached_tokens[i])
            prefill_cost = prefill_weight * rep["prefill_rate"] * uncached
            queue_cost = queue_weight * queued_tokens[i]
            rtt_cost = rtt_weight * rep["rtt_ms"]
            score = rtt_cost + prefill_cost + queue_cost
            if score < best_score:
                best_score = score
                best_idx = i

        # Physical TTFT: RTT + wait for GPU + own prefill
        rep = replicas[best_idx]
        uncached = max(0, input_len - cached_tokens[best_idx])
        own_prefill_ms = rep["prefill_rate"] * uncached
        prefill_start = max(arrive_ms, prefill_available_at[best_idx])
        wait_ms = prefill_start - arrive_ms
        actual_ttft_ms = rep["rtt_ms"] + wait_ms + own_prefill_ms

        # GPU is busy with this prefill until it finishes
        prefill_available_at[best_idx] = prefill_start + own_prefill_ms

        decode_ms = output_len * decode_ms_per_tok
        e2e_ms = actual_ttft_ms + decode_ms

        ttfts.append(actual_ttft_ms)
        e2es.append(e2e_ms)
        targets.append(best_idx)

        # Update queued token counter and completion heap
        queued_tokens[best_idx] += input_len
        completion_time = arrive_ms + e2e_ms
        heapq.heappush(completion_heaps[best_idx], (completion_time, input_len))

        # Update prefix cache
        prefix_cache[best_idx].update(hash_ids)

    return ttfts, e2es, targets
