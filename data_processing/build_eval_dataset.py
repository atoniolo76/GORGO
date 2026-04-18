"""Build an evaluation dataset for the first week of April 2026.

For each unique session (token_hash + hash of system+first message), finds
the request with the longest conversation (most messages), computes
per-message token counts, and outputs a dataset representing the maximum
KV cache footprint per session.
"""

import itertools

from app import app, completions_volume
import modal

image = modal.Image.debian_slim().pip_install("duckdb", "tiktoken").add_local_python_source("app")

FILE_PREFIX = "llm_responses_202604"
FILE_CUTOFF = "llm_responses_20260408"


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


@app.function(image=image, volumes={"/data": completions_volume}, timeout=3600, retries=2)
def process_file(filename: str, include_token_ids: bool = False) -> list[dict]:
    """For each session in the file, return the request with the most messages
    along with per-message token counts.

    If ``include_token_ids`` is True, each returned entry also includes
    ``prompt_token_ids`` (flat list of tiktoken gpt-4o ids for the longest
    conversation's messages, concatenated in order) and
    ``response_token_ids`` (ids for the assistant response text).
    """
    import json
    import os

    import duckdb
    import tiktoken

    enc = tiktoken.encoding_for_model("gpt-4o")
    path = os.path.join("/data", filename)

    con = duckdb.connect()
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
    con.close()

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
