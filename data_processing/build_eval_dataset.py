"""Build an evaluation dataset for the first week of April 2026.

For each unique session (token_hash + hash of system+first message), finds
the request with the longest conversation (most messages), computes
per-message token counts, and outputs a dataset representing the maximum
KV cache footprint per session.
"""

import itertools

from app import app, completions_volume
import modal

image = (
    modal.Image.debian_slim()
    .pip_install("duckdb", "tiktoken", "pyarrow")
    .add_local_python_source("app")
)

FILE_PREFIX = "llm_responses_202604"
FILE_CUTOFF = "llm_responses_20260408"


def tokenized_dir(file_prefix: str = FILE_PREFIX) -> str:
    return f"/data/tokenized_{file_prefix}"


def tokenized_path_for(filename: str, file_prefix: str = FILE_PREFIX) -> str:
    import os

    stem = filename[: -len(".parquet")] if filename.endswith(".parquet") else filename
    return os.path.join(tokenized_dir(file_prefix), f"{stem}.tokenized.parquet")


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


def _session_fingerprint(token_hash: str, messages: list) -> str:
    """Hash token_hash + system prompt + first user message to identify a session."""
    import hashlib

    parts = [token_hash]
    for msg in messages[:2]:
        if isinstance(msg, dict):
            parts.append(msg.get("role", ""))
            parts.append(_content_to_str(msg.get("content", "")))
        elif isinstance(msg, str):
            parts.append(msg)
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def _read_rows(filename: str):
    """Pull relevant columns out of one raw parquet on the /data volume."""
    import os

    import duckdb

    path = os.path.join("/data", filename)
    con = duckdb.connect()
    try:
        rows = con.execute(
            """
            SELECT
                uuid,
                timestamp,
                request_metadata.token_hash AS token_hash,
                request,
                response
            FROM read_parquet(?)
            WHERE request NOT LIKE '%keep-alive%'
            """,
            [path],
        ).fetchall()
    finally:
        con.close()
    return rows


def _rows_to_session_entries(rows, enc, include_token_ids: bool) -> list[dict]:
    """Reduce raw (uuid, ts, token_hash, request, response) rows down to one
    entry per session (the longest conversation wins)."""
    import json

    best_by_session: dict[str, dict] = {}

    for uuid, timestamp, token_hash, request_raw, response_raw in rows:
        try:
            req = json.loads(request_raw) if isinstance(request_raw, str) else request_raw
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(req, dict):
            continue

        messages = req.get("messages", [])
        if not isinstance(messages, list):
            continue

        session_id = _session_fingerprint(token_hash, messages)
        msg_count = len(messages)
        prev = best_by_session.get(session_id)
        if prev and prev["message_count"] >= msg_count:
            continue

        per_message = []
        total_tokens = 0
        prompt_token_ids: list[int] = []
        for msg in messages:
            role = "unknown"
            ids: list[int] = []
            if isinstance(msg, str):
                role = "raw"
                ids = enc.encode(msg, disallowed_special=())
            elif isinstance(msg, dict):
                role = msg.get("role", "unknown")
                content = msg.get("content")
                text = _content_to_str(content)
                if text:
                    ids = enc.encode(text, disallowed_special=())
            tokens = len(ids)
            per_message.append({"role": role, "tokens": tokens})
            total_tokens += tokens
            if include_token_ids and ids:
                prompt_token_ids.extend(ids)

        resp_tokens = 0
        response_token_ids: list[int] = []
        try:
            resp = json.loads(response_raw) if isinstance(response_raw, str) else response_raw
            if isinstance(resp, dict):
                for choice in resp.get("choices", []):
                    if isinstance(choice, dict):
                        msg = choice.get("message", {})
                        if isinstance(msg, dict):
                            c = msg.get("content")
                            if isinstance(c, str):
                                ids = enc.encode(c, disallowed_special=())
                                resp_tokens += len(ids)
                                if include_token_ids:
                                    response_token_ids.extend(ids)
        except (json.JSONDecodeError, TypeError):
            pass

        entry = {
            "session_id": session_id,
            "uuid": uuid,
            "timestamp": str(timestamp),
            "token_hash": token_hash,
            "message_count": msg_count,
            "user_messages": sum(1 for m in per_message if m["role"] == "user"),
            "assistant_messages": sum(1 for m in per_message if m["role"] == "assistant"),
            "total_prompt_tokens": total_tokens,
            "response_tokens": resp_tokens,
            "kv_cache_tokens": total_tokens + resp_tokens,
            "per_message": per_message,
        }
        if include_token_ids:
            entry["prompt_token_ids"] = prompt_token_ids
            entry["response_token_ids"] = response_token_ids
        best_by_session[session_id] = entry

    return list(best_by_session.values())


@app.function(
    image=image,
    volumes={"/data": completions_volume},
    timeout=3600,
    retries=2,
    memory=1024 * 16,
)
def process_file(filename: str, include_token_ids: bool = False) -> list[dict]:
    """For each session in the file, return the request with the most messages
    along with per-message token counts.

    If ``include_token_ids`` is True, each returned entry also includes
    ``prompt_token_ids`` (flat list of tiktoken gpt-4o ids for the longest
    conversation's messages, concatenated in order) and
    ``response_token_ids`` (ids for the assistant response text).
    """
    import tiktoken

    enc = tiktoken.encoding_for_model("gpt-4o")
    rows = _read_rows(filename)
    return _rows_to_session_entries(rows, enc, include_token_ids=include_token_ids)


@app.function(
    image=image,
    volumes={"/data": completions_volume},
    timeout=3600,
    retries=2,
    memory=1024 * 16,
    cpu=4.0,
)
def tokenize_file(filename: str, file_prefix: str = FILE_PREFIX) -> dict:
    """Tokenize one raw parquet and write per-session token ids to
    ``/data/tokenized_<file_prefix>/<stem>.tokenized.parquet``.

    Idempotent: if the output already exists, it is left alone and the
    function returns quickly with ``skipped=True``.
    """
    import os

    import pyarrow as pa
    import pyarrow.parquet as pq
    import tiktoken

    out_path = tokenized_path_for(filename, file_prefix=file_prefix)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if os.path.exists(out_path):
        try:
            md = pq.read_metadata(out_path)
            return {
                "filename": filename,
                "out_path": out_path,
                "skipped": True,
                "num_sessions": md.num_rows,
                "bytes": os.path.getsize(out_path),
            }
        except Exception:
            os.remove(out_path)

    enc = tiktoken.encoding_for_model("gpt-4o")
    rows = _read_rows(filename)
    entries = _rows_to_session_entries(rows, enc, include_token_ids=True)

    session_ids = [e["session_id"] for e in entries]
    token_hashes = [e["token_hash"] for e in entries]
    prompt_ids = [e["prompt_token_ids"] for e in entries]
    prompt_token_counts = [len(ids) for ids in prompt_ids]
    message_counts = [e["message_count"] for e in entries]

    table = pa.table(
        {
            "session_id": pa.array(session_ids, type=pa.string()),
            "token_hash": pa.array(token_hashes, type=pa.string()),
            "message_count": pa.array(message_counts, type=pa.int32()),
            "prompt_token_count": pa.array(prompt_token_counts, type=pa.int32()),
            "prompt_ids": pa.array(prompt_ids, type=pa.list_(pa.uint32())),
        }
    )

    tmp_path = out_path + ".tmp"
    pq.write_table(table, tmp_path, compression="zstd")
    os.replace(tmp_path, out_path)
    completions_volume.commit()

    return {
        "filename": filename,
        "out_path": out_path,
        "skipped": False,
        "num_sessions": len(entries),
        "num_tokens": sum(prompt_token_counts),
        "bytes": os.path.getsize(out_path),
    }


@app.function(image=image, volumes={"/data": completions_volume}, timeout=7200)
def tokenize_dataset(file_prefix: str = FILE_PREFIX, file_cutoff: str = FILE_CUTOFF) -> dict:
    """Fan ``tokenize_file`` out over every raw parquet in the date window.

    Each worker writes its own output parquet under
    ``/data/tokenized_<file_prefix>/``; this driver just aggregates stats.
    Re-runs are cheap because per-file outputs are skipped if already written.
    """
    import os
    import time

    parquet_dir = "/data"
    files = sorted(
        f
        for f in os.listdir(parquet_dir)
        if f.endswith(".parquet") and file_prefix in f and f < file_cutoff
    )
    print(f"Tokenizing {len(files)} files -> {tokenized_dir(file_prefix)}")

    t0 = time.time()
    total_sessions = 0
    total_tokens = 0
    total_bytes = 0
    skipped_files = 0

    args = [(f, file_prefix) for f in files]
    for i, result in enumerate(tokenize_file.starmap(args), start=1):
        total_sessions += result.get("num_sessions", 0)
        total_tokens += result.get("num_tokens", 0) or 0
        total_bytes += result.get("bytes", 0)
        if result.get("skipped"):
            skipped_files += 1
        if i % 20 == 0 or i == len(files):
            elapsed = time.time() - t0
            print(
                f"  {i}/{len(files)} files | "
                f"{total_sessions:,} sessions | "
                f"{total_tokens:,} prompt tokens | "
                f"{total_bytes / 1e9:,.2f} GB on disk | "
                f"{skipped_files:,} pre-existing | "
                f"elapsed {elapsed:,.0f}s"
            )

    completions_volume.commit()
    print(
        f"\nDone. {total_sessions:,} sessions across {len(files)} files "
        f"({skipped_files:,} already cached)."
    )
    return {
        "file_prefix": file_prefix,
        "file_cutoff": file_cutoff,
        "num_files": len(files),
        "num_sessions": total_sessions,
        "num_tokens": total_tokens,
        "bytes": total_bytes,
        "skipped_files": skipped_files,
        "elapsed_seconds": time.time() - t0,
    }


@app.function(image=image, volumes={"/data": completions_volume}, timeout=7200)
def build_dataset(batch_size: int = 50):
    import csv
    import json
    import os

    parquet_dir = "/data"
    files = sorted(
        f
        for f in os.listdir(parquet_dir)
        if f.endswith(".parquet") and FILE_PREFIX in f and f < FILE_CUTOFF
    )
    batches = list(itertools.batched(files, batch_size))
    print(f"April 1-7: {len(files)} files in {len(batches)} batch(es)")

    global_best: dict[str, dict] = {}

    for batch_idx, batch in enumerate(batches):
        for file_results in process_file.map(batch):
            for entry in file_results:
                sid = entry["session_id"]
                prev = global_best.get(sid)
                if not prev or entry["message_count"] > prev["message_count"]:
                    global_best[sid] = entry

        print(f"  Batch {batch_idx + 1}/{len(batches)}: {len(global_best)} unique sessions so far")

    records = sorted(global_best.values(), key=lambda r: r["kv_cache_tokens"], reverse=True)

    csv_path = "/data/eval_kv_cache_april_w1.csv"
    csv_fields = [
        "session_id",
        "token_hash",
        "uuid",
        "timestamp",
        "message_count",
        "user_messages",
        "assistant_messages",
        "total_prompt_tokens",
        "response_tokens",
        "kv_cache_tokens",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    json_path = "/data/eval_kv_cache_april_w1.json"
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2, default=str)

    completions_volume.commit()

    total_sessions = len(records)
    unique_hashes = len({r["token_hash"] for r in records})
    avg_prompt = (
        sum(r["total_prompt_tokens"] for r in records) / total_sessions if total_sessions else 0
    )
    avg_kv = sum(r["kv_cache_tokens"] for r in records) / total_sessions if total_sessions else 0
    max_kv = records[0]["kv_cache_tokens"] if records else 0
    max_msgs = max(r["message_count"] for r in records) if records else 0
    avg_msgs = sum(r["message_count"] for r in records) / total_sessions if total_sessions else 0

    print(f"\n{'=' * 60}")
    print(f"Unique sessions:               {total_sessions:,}")
    print(f"Unique token_hashes:           {unique_hashes:,}")
    print(f"Avg prompt tokens/conversation:{avg_prompt:>12,.1f}")
    print(f"Avg KV cache tokens:           {avg_kv:>12,.1f}")
    print(f"Max KV cache tokens:           {max_kv:>12,}")
    print(f"Max conversation messages:     {max_msgs:>12,}")
    print(f"Avg conversation messages:     {avg_msgs:>12,.1f}")
    print(f"\nSaved: {csv_path}")
    print(f"       {json_path} (includes per_message breakdown)")


@app.local_entrypoint()
def main(batch_size: int = 50):
    build_dataset.remote(batch_size=batch_size)


@app.local_entrypoint()
def tokenize_main(file_prefix: str = FILE_PREFIX, file_cutoff: str = FILE_CUTOFF):
    tokenize_dataset.remote(file_prefix=file_prefix, file_cutoff=file_cutoff)
