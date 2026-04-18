import itertools

from app import app, completions_volume
import modal

image = modal.Image.debian_slim().pip_install("duckdb", "tiktoken").add_local_python_source("app")


@app.function(image=image, volumes={"/data": completions_volume}, timeout=3600)
def count_file_tokens(filename: str) -> dict:
    import json
    import os

    import duckdb
    import tiktoken

    enc = tiktoken.encoding_for_model("gpt-4o")
    path = os.path.join("/data", filename)

    con = duckdb.connect()
    rows = con.execute("SELECT request FROM read_parquet(?)", [path]).fetchall()
    con.close()

    total_user_tokens = 0
    total_user_messages = 0
    total_all_tokens = 0
    valid_requests = 0

    for (request_raw,) in rows:
        try:
            req = json.loads(request_raw) if isinstance(request_raw, str) else request_raw
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(req, dict):
            continue

        messages = req.get("messages", [])
        if not isinstance(messages, list):
            continue

        valid_requests += 1
        for msg in messages:
            if isinstance(msg, str):
                n = len(enc.encode(msg, disallowed_special=()))
                total_all_tokens += n
                total_user_tokens += n
                total_user_messages += 1
            elif isinstance(msg, dict):
                content = msg.get("content")
                if not isinstance(content, str):
                    continue
                n = len(enc.encode(content, disallowed_special=()))
                total_all_tokens += n
                if msg.get("role") == "user":
                    total_user_tokens += n
                    total_user_messages += 1

    return {
        "file": filename,
        "rows": len(rows),
        "valid_requests": valid_requests,
        "total_all_tokens": total_all_tokens,
        "total_user_tokens": total_user_tokens,
        "total_user_messages": total_user_messages,
    }


FIELDNAMES = [
    "file",
    "rows",
    "valid_requests",
    "total_all_tokens",
    "total_user_tokens",
    "total_user_messages",
    "avg_all_tokens_per_request",
    "avg_user_tokens_per_message",
]


@app.function(image=image, volumes={"/data": completions_volume}, timeout=7200)
def count_request_tokens(batch_size: int = 100):
    import csv
    import os

    files = sorted(f for f in os.listdir("/data") if f.endswith(".parquet"))
    batches = list(itertools.batched(files, batch_size))
    print(
        f"Found {len(files)} parquet file(s), processing in {len(batches)} batch(es) of <={batch_size}"
    )

    per_file_stats = []
    agg = {
        k: 0
        for k in [
            "rows",
            "valid_requests",
            "total_all_tokens",
            "total_user_tokens",
            "total_user_messages",
        ]
    }

    for batch_idx, batch in enumerate(batches):
        batch_results = []
        for stat in count_file_tokens.map(batch):
            for k in agg:
                agg[k] += stat[k]
            stat["avg_all_tokens_per_request"] = (
                round(stat["total_all_tokens"] / stat["valid_requests"], 1)
                if stat["valid_requests"]
                else 0
            )
            stat["avg_user_tokens_per_message"] = (
                round(stat["total_user_tokens"] / stat["total_user_messages"], 1)
                if stat["total_user_messages"]
                else 0
            )
            batch_results.append(stat)

        per_file_stats.extend(batch_results)
        done = sum(len(b) for b in batches[: batch_idx + 1])
        print(
            f"  Batch {batch_idx + 1}/{len(batches)} done ({done}/{len(files)} files, {agg['total_user_messages']:,} user messages so far)"
        )

    per_file_stats.sort(key=lambda s: s["file"])

    avg_all = (
        round(agg["total_all_tokens"] / agg["valid_requests"], 1) if agg["valid_requests"] else 0
    )
    avg_user = (
        round(agg["total_user_tokens"] / agg["total_user_messages"], 1)
        if agg["total_user_messages"]
        else 0
    )

    csv_path = "/data/token_counts.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(per_file_stats)
        writer.writerow(
            {
                "file": "TOTAL",
                "rows": agg["rows"],
                "valid_requests": agg["valid_requests"],
                "total_all_tokens": agg["total_all_tokens"],
                "total_user_tokens": agg["total_user_tokens"],
                "total_user_messages": agg["total_user_messages"],
                "avg_all_tokens_per_request": avg_all,
                "avg_user_tokens_per_message": avg_user,
            }
        )
    completions_volume.commit()

    print(f"\n{'=' * 60}")
    print(f"Files:           {len(files):,}")
    print(f"Rows:            {agg['rows']:,}")
    print(f"Valid requests:  {agg['valid_requests']:,}")
    print(f"User messages:   {agg['total_user_messages']:,}")
    print(f"All tokens:      {agg['total_all_tokens']:,}")
    print(f"User tokens:     {agg['total_user_tokens']:,}")
    print(f"Avg all tokens/request:      {avg_all:,.1f}")
    print(f"Avg user tokens/message:     {avg_user:,.1f}")
    print(f"Results saved to {csv_path}")


@app.local_entrypoint()
def main(batch_size: int = 100):
    count_request_tokens.remote(batch_size=batch_size)
