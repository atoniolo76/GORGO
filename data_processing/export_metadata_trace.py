"""Export privacy-safe metadata traces from production parquets.

Produces a JSONL where each row contains per-request routing metadata
with no message content:

    {
        "timestamp": 0,
        "token_hash": "abc123...",
        "input_length": 13200,
        "output_length": 77,
        "messages": [
            {"role": "system", "tokens": 482},
            {"role": "user", "tokens": 1200},
            {"role": "assistant", "tokens": 800},
            {"role": "user", "tokens": 10718}
        ],
        "hash_ids": [0, 1, 2, ...]
    }

The ``hash_ids`` are prefix-aware block hashes computed from the real
token sequence — they preserve all prefix reuse (intra-user and
cross-user) without revealing token values. ``messages`` preserves
per-message roles and token counts for synthetic trace reconstruction.

A downstream script (``build_decoded_trace.py``) converts metadata
traces into Mooncake-format decoded traces with gibberish Unicode
bodies that tokenize to exact token counts.

Usage::

    modal run --env=alessio-dev data_processing/export_metadata_trace.py::main \\
        --start-time 2026-04-02T00:30:00 --end-time 2026-04-02T01:00:00 \\
        --output-path /data/mooncake_traces/metadata/glm5_metadata_apr2_0030_to_0100.jsonl
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime

import modal

from app import app, completions_volume

DEFAULT_BLOCK_SIZE = 256
NO_CAP = 10**12  # sentinel for "no length cap" (include the full >24k tail)

# Production export covering Apr 1-7 2026 (the 411K corpus).
WEEK_FILE_PREFIX = "llm_responses_202604"
WEEK_FILE_CUTOFF = "llm_responses_20260408"

image = (
    modal.Image.debian_slim()
    .pip_install("duckdb", "tiktoken", "pyarrow")
    .add_local_python_source("app")
)


def _content_to_str(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", "") or "")
            elif isinstance(block, str):
                parts.append(block)
        return " ".join(parts)
    return ""


def _to_naive_dt(ts):
    if isinstance(ts, datetime):
        return ts.replace(tzinfo=None) if ts.tzinfo else ts
    if isinstance(ts, str):
        for fmt in (
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
        ):
            try:
                return datetime.strptime(ts[:26], fmt)
            except ValueError:
                continue
    return None


@app.function(
    image=image,
    memory=1024 * 16,
    timeout=14400,
    volumes={"/data": completions_volume},
)
def export_metadata(
    start_time: str,
    end_time: str,
    output_path: str,
    max_input_tokens: int = 24000,
    block_size: int = DEFAULT_BLOCK_SIZE,
):
    import duckdb
    import tiktoken

    enc = tiktoken.encoding_for_model("gpt-4o")
    start_dt = datetime.fromisoformat(start_time)
    end_dt = datetime.fromisoformat(end_time)

    FILE_PREFIX = "llm_responses_202604"
    FILE_CUTOFF = "llm_responses_20260408"

    all_files = sorted(
        f
        for f in os.listdir("/data")
        if f.endswith(".parquet") and f.startswith(FILE_PREFIX) and f < FILE_CUTOFF + ".parquet"
    )

    # Filter files by name-encoded timestamp to avoid scanning the entire
    # volume for narrow time windows.  Each file covers a 30-min shard
    # named like llm_responses_20260402_003000.parquet (= Apr 2 00:30).
    # We keep files whose shard start is within [start - 30min, end]
    # to account for requests that span shard boundaries.
    def _file_ts(name: str) -> datetime | None:
        stem = name.replace(".parquet", "")[len(FILE_PREFIX) :]
        try:
            return datetime.strptime(f"202604{stem}", "%Y%m%d_%H%M%S")
        except ValueError:
            return None

    from datetime import timedelta

    margin = timedelta(seconds=1800)
    files = [
        f for f in all_files if (fts := _file_ts(f)) is None or (start_dt - margin <= fts <= end_dt)
    ]

    print(
        f"[metadata] {len(files)}/{len(all_files)} parquet files for window {start_time} -> {end_time}"
    )

    # Prefix-aware block hashing (same as build_mooncake_trace.py)
    hash_to_id: dict[bytes, int] = {}

    def _block_ids(token_ids: list[int]) -> list[int]:
        if not token_ids:
            return []
        ids: list[int] = []
        prev_digest = b""
        for i in range(0, len(token_ids), block_size):
            block = token_ids[i : i + block_size]
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

    t0 = time.perf_counter()
    rows_out: list[dict] = []
    first_ts: datetime | None = None
    skipped = 0

    con = duckdb.connect()
    for filename in files:
        path = os.path.join("/data", filename)
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
                    skipped += 1
                    continue
                if ts_dt < start_dt:
                    continue
                if ts_dt >= end_dt:
                    break

                try:
                    req = json.loads(request_raw) if isinstance(request_raw, str) else request_raw
                except (json.JSONDecodeError, TypeError):
                    skipped += 1
                    continue
                if not isinstance(req, dict):
                    skipped += 1
                    continue

                msgs = req.get("messages", [])
                if not isinstance(msgs, list) or not msgs:
                    skipped += 1
                    continue

                # Tokenize with tiktoken (same as build_mooncake_trace.py)
                # to match the original trace's filtering and token counts.
                per_msg: list[dict] = []
                all_token_ids: list[int] = []
                system_prompt_hash = None
                for msg in msgs:
                    if isinstance(msg, str):
                        ids = enc.encode(msg, disallowed_special=())
                        per_msg.append({"role": "raw", "tokens": len(ids)})
                        all_token_ids.extend(ids)
                    elif isinstance(msg, dict):
                        role = msg.get("role", "unknown")
                        text = _content_to_str(msg.get("content"))
                        ids = enc.encode(text, disallowed_special=()) if text else []
                        per_msg.append({"role": role, "tokens": len(ids)})
                        all_token_ids.extend(ids)
                        if role == "system" and system_prompt_hash is None and text:
                            system_prompt_hash = hashlib.sha256(text.encode()).hexdigest()[:16]

                input_length = len(all_token_ids)
                if input_length == 0 or input_length > max_input_tokens:
                    skipped += 1
                    continue

                # Compute prefix-aware block hashes
                block_ids = _block_ids(all_token_ids)

                # Get output length from response usage
                output_length = 0
                try:
                    resp = (
                        json.loads(response_raw) if isinstance(response_raw, str) else response_raw
                    )
                    if isinstance(resp, dict):
                        usage = resp.get("usage")
                        if isinstance(usage, dict):
                            ct = usage.get("completion_tokens")
                            if isinstance(ct, int) and ct >= 0:
                                output_length = ct
                except (json.JSONDecodeError, TypeError):
                    pass

                if first_ts is None:
                    first_ts = ts_dt
                delta_ms = int((ts_dt - first_ts).total_seconds() * 1000)

                rows_out.append(
                    {
                        "timestamp": delta_ms,
                        "token_hash": token_hash or "",
                        "system_prompt_hash": system_prompt_hash,
                        "input_length": input_length,
                        "output_length": output_length,
                        "messages": per_msg,
                        "hash_ids": block_ids,
                    }
                )

            if rows_out and rows_out[-1].get("_break"):
                break

        print(
            f"[metadata]   {filename}: {len(rows_out)} rows ({time.perf_counter() - t0:.1f}s)",
            flush=True,
        )
    con.close()

    # Write metadata trace
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w") as f:
        for row in rows_out:
            f.write(json.dumps(row) + "\n")
    os.replace(tmp_path, output_path)
    completions_volume.commit()

    total_input = sum(r["input_length"] for r in rows_out)
    unique_blocks = len(hash_to_id)
    total_blocks = sum(len(r["hash_ids"]) for r in rows_out)
    users = len(set(r["token_hash"] for r in rows_out))
    duration_ms = rows_out[-1]["timestamp"] if rows_out else 0

    print(f"\n[metadata] wrote {output_path}")
    print(f"  requests: {len(rows_out):,}")
    print(f"  users: {users:,}")
    print(f"  skipped: {skipped:,}")
    print(f"  total input tokens: {total_input:,}")
    print(f"  unique blocks: {unique_blocks:,} / {total_blocks:,}")
    print(f"  block reuse: {100 * (1 - unique_blocks / max(total_blocks, 1)):.1f}%")
    print(f"  duration: {duration_ms / 1000:.0f}s")
    print(f"  elapsed: {time.perf_counter() - t0:.1f}s")

    return {
        "output_path": output_path,
        "requests": len(rows_out),
        "users": users,
        "total_input_tokens": total_input,
        "unique_blocks": unique_blocks,
        "total_blocks": total_blocks,
        "duration_ms": duration_ms,
    }


@app.local_entrypoint()
def main(
    start_time: str = "2026-04-02T00:30:00",
    end_time: str = "2026-04-02T01:00:00",
    output_path: str = "",
    max_input_tokens: int = 24000,
    block_size: int = DEFAULT_BLOCK_SIZE,
):
    if not output_path:
        st = start_time.replace(":", "").replace("-", "")
        et = end_time.replace(":", "").replace("-", "")
        output_path = f"/data/mooncake_traces/metadata/glm5_metadata_{st}_to_{et}.jsonl"

    result = export_metadata.remote(
        start_time=start_time,
        end_time=end_time,
        output_path=output_path,
        max_input_tokens=max_input_tokens,
        block_size=block_size,
    )
    print(json.dumps(result, indent=2))


# ---------------------------------------------------------------------------
# Per-file parallel week export (CPU fan-out, one container per parquet shard)
# ---------------------------------------------------------------------------


def _block_digests(token_ids: list[int], block_size: int) -> list[str]:
    """Prefix-cumulative SHA-256 block hashes, content-only so they are
    identical across independently-run shards (enables per-file parallel
    export with globally consistent prefix reuse). Returns 12-byte (96-bit)
    hex digests; a downstream step may remap these to compact ints."""
    if not token_ids:
        return []
    out: list[str] = []
    prev = b""
    for i in range(0, len(token_ids), block_size):
        block = token_ids[i : i + block_size]
        h = hashlib.sha256()
        h.update(prev)
        h.update(b"".join(t.to_bytes(4, "little", signed=False) for t in block))
        digest = h.digest()
        out.append(digest[:12].hex())
        prev = digest
    return out


@app.function(
    image=image,
    memory=1024 * 16,
    timeout=7200,
    retries=2,
    cpu=4.0,
    volumes={"/data": completions_volume},
)
def export_metadata_file(
    filename: str,
    output_dir: str,
    max_input_tokens: int = NO_CAP,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> dict:
    """Export content-free metadata for ONE raw parquet shard.

    Idempotent (skips if the per-shard output exists). Block ids are content
    digests (globally consistent across workers); timestamps are absolute
    epoch ms, converted to relative at concat time.
    """
    import json
    import os

    import duckdb
    import tiktoken

    stem = filename.replace(".parquet", "")
    out_path = os.path.join(output_dir, f"{stem}.jsonl")
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(out_path):
        n = sum(1 for line in open(out_path) if line.strip())
        return {"filename": filename, "skipped": True, "requests": n}

    enc = tiktoken.encoding_for_model("gpt-4o")
    epoch = datetime(1970, 1, 1)
    con = duckdb.connect()
    path = os.path.join("/data", filename)

    rows_out: list[dict] = []
    skipped = 0
    users: set[str] = set()

    cursor = con.execute(
        """
        SELECT uuid, timestamp,
               request_metadata.token_hash AS token_hash,
               request, response
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
                skipped += 1
                continue
            try:
                req = json.loads(request_raw) if isinstance(request_raw, str) else request_raw
            except (json.JSONDecodeError, TypeError):
                skipped += 1
                continue
            if not isinstance(req, dict):
                skipped += 1
                continue
            msgs = req.get("messages", [])
            if not isinstance(msgs, list) or not msgs:
                skipped += 1
                continue

            per_msg: list[dict] = []
            all_token_ids: list[int] = []
            system_prompt_hash = None
            for msg in msgs:
                if isinstance(msg, str):
                    ids = enc.encode(msg, disallowed_special=())
                    per_msg.append({"role": "raw", "tokens": len(ids)})
                    all_token_ids.extend(ids)
                elif isinstance(msg, dict):
                    role = msg.get("role", "unknown")
                    text = _content_to_str(msg.get("content"))
                    ids = enc.encode(text, disallowed_special=()) if text else []
                    per_msg.append({"role": role, "tokens": len(ids)})
                    all_token_ids.extend(ids)
                    if role == "system" and system_prompt_hash is None and text:
                        system_prompt_hash = hashlib.sha256(text.encode()).hexdigest()[:16]

            input_length = len(all_token_ids)
            if input_length == 0 or input_length > max_input_tokens:
                skipped += 1
                continue

            block_ids = _block_digests(all_token_ids, block_size)

            output_length = 0
            try:
                resp = json.loads(response_raw) if isinstance(response_raw, str) else response_raw
                if isinstance(resp, dict):
                    usage = resp.get("usage")
                    if isinstance(usage, dict):
                        ct = usage.get("completion_tokens")
                        if isinstance(ct, int) and ct >= 0:
                            output_length = ct
            except (json.JSONDecodeError, TypeError):
                pass

            abs_ms = int((ts_dt - epoch).total_seconds() * 1000)
            users.add(token_hash or "")
            rows_out.append(
                {
                    "abs_timestamp_ms": abs_ms,
                    "token_hash": token_hash or "",
                    "system_prompt_hash": system_prompt_hash,
                    "input_length": input_length,
                    "output_length": output_length,
                    "messages": per_msg,
                    "hash_ids": block_ids,
                }
            )
    con.close()

    rows_out.sort(key=lambda r: r["abs_timestamp_ms"])
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w") as f:
        for row in rows_out:
            f.write(json.dumps(row) + "\n")
    os.replace(tmp_path, out_path)
    completions_volume.commit()

    return {
        "filename": filename,
        "skipped": False,
        "requests": len(rows_out),
        "users": len(users),
        "total_input_tokens": sum(r["input_length"] for r in rows_out),
    }


@app.function(image=image, timeout=14400, volumes={"/data": completions_volume})
def export_week(
    output_dir: str = "/data/mooncake_traces/metadata_week",
    max_input_tokens: int = NO_CAP,
    block_size: int = DEFAULT_BLOCK_SIZE,
    limit: int = 0,
) -> dict:
    """Fan ``export_metadata_file`` over every Apr 1-7 raw parquet shard."""
    import os
    import time

    files = sorted(
        f
        for f in os.listdir("/data")
        if f.endswith(".parquet")
        and f.startswith(WEEK_FILE_PREFIX)
        and f < WEEK_FILE_CUTOFF + ".parquet"
    )
    if limit:
        files = files[:limit]
    print(f"[week] {len(files)} parquet shards -> {output_dir}")

    args = [(f, output_dir, max_input_tokens, block_size) for f in files]
    t0 = time.time()
    total_requests = 0
    total_tokens = 0
    skipped_files = 0
    for i, r in enumerate(export_metadata_file.starmap(args), start=1):
        total_requests += r.get("requests", 0)
        total_tokens += r.get("total_input_tokens", 0) or 0
        if r.get("skipped"):
            skipped_files += 1
        if i % 20 == 0 or i == len(files):
            print(
                f"  {i}/{len(files)} shards | {total_requests:,} requests | "
                f"{total_tokens:,} tokens | {skipped_files} cached | "
                f"{time.time() - t0:.0f}s",
                flush=True,
            )
    completions_volume.commit()
    return {
        "output_dir": output_dir,
        "shards": len(files),
        "requests": total_requests,
        "total_input_tokens": total_tokens,
        "skipped_files": skipped_files,
        "elapsed_seconds": time.time() - t0,
    }


@app.local_entrypoint()
def week(
    output_dir: str = "/data/mooncake_traces/metadata_week",
    max_input_tokens: int = NO_CAP,
    limit: int = 0,
):
    result = export_week.remote(
        output_dir=output_dir,
        max_input_tokens=max_input_tokens,
        limit=limit,
    )
    print(json.dumps(result, indent=2))
