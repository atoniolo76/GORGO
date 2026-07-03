"""Package the per-shard decoded week trace into per-day release files.

Reads the decoded shards produced by ``build_decoded_trace.py::decode_week``
(``/data/mooncake_traces/decoded_week/llm_responses_YYYYMMDD_HHMMSS.jsonl``,
each row carrying ``abs_timestamp_ms``), and writes, per calendar day:

- a Mooncake-format JSONL with relative ``timestamp`` (replay-ready), and
- a zstd parquet with a documented schema (HuggingFace-friendly).

Fans out one worker per day (CPU). Timestamps are made relative to the global
first request of the week so the trace starts at t=0.

Usage::

    modal run --env=alessio-dev data_processing/build_week_release.py::main
"""

from __future__ import annotations

import json

import modal

from app import app, completions_volume

image = modal.Image.debian_slim().pip_install("pyarrow").add_local_python_source("app")

DECODED_DIR = "/data/mooncake_traces/decoded_week"
JSONL_OUT_DIR = "/data/mooncake_traces/week_release/jsonl"
PARQUET_OUT_DIR = "/data/mooncake_traces/week_release/parquet"

PARQUET_BATCH_ROWS = 2000


def _day_of(shard: str) -> str:
    # llm_responses_20260401_003000.jsonl -> "20260401"
    return shard.replace("llm_responses_", "").split("_")[0]


@app.function(
    image=image,
    memory=1024 * 8,
    timeout=10800,
    retries=1,
    cpu=4.0,
    volumes={"/data": completions_volume},
)
def package_day(day: str, shards: list[str], global_min_ms: int) -> dict:
    """Concatenate one day's decoded shards -> relative-ts JSONL + parquet."""
    import os

    import pyarrow as pa
    import pyarrow.parquet as pq

    os.makedirs(JSONL_OUT_DIR, exist_ok=True)
    os.makedirs(PARQUET_OUT_DIR, exist_ok=True)
    jsonl_path = os.path.join(JSONL_OUT_DIR, f"glm5_artchat_week_{day}.jsonl")
    parquet_path = os.path.join(PARQUET_OUT_DIR, f"{day}.parquet")

    if os.path.exists(jsonl_path) and os.path.exists(parquet_path):
        n = sum(1 for line in open(jsonl_path) if line.strip())
        return {"day": day, "skipped": True, "requests": n}

    schema = pa.schema(
        [
            ("request_id", pa.string()),
            ("token_hash", pa.string()),
            ("system_prompt_hash", pa.string()),
            ("timestamp_ms", pa.int64()),
            ("input_length", pa.int32()),
            ("output_length", pa.int32()),
            ("hash_ids", pa.list_(pa.string())),
            ("messages", pa.string()),  # JSON-encoded [{role, content}, ...]
        ]
    )
    writer = pq.ParquetWriter(parquet_path + ".tmp", schema, compression="zstd")

    batch: list[dict] = []

    def _flush():
        if not batch:
            return
        table = pa.Table.from_pylist(batch, schema=schema)
        writer.write_table(table)
        batch.clear()

    jsonl_tmp = jsonl_path + ".tmp"
    count = 0
    with open(jsonl_tmp, "w") as fout:
        for shard in shards:
            in_path = os.path.join(DECODED_DIR, shard)
            with open(in_path) as fin:
                for line in fin:
                    if not line.strip():
                        continue
                    e = json.loads(line)
                    rel_ts = int(e["abs_timestamp_ms"]) - global_min_ms
                    messages = e["request"]["messages"]
                    # Mooncake replay row (relative timestamp)
                    out = {
                        "timestamp": rel_ts,
                        "input_length": e["input_length"],
                        "output_length": e["output_length"],
                        "unique_input_tokens": e["input_length"],
                        "hash_ids": e["hash_ids"],
                        "request": e["request"],
                        "response": None,
                        "request_id": e["request_id"],
                        "token_hash": e.get("token_hash", ""),
                        "system_prompt_hash": e.get("system_prompt_hash"),
                    }
                    fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                    batch.append(
                        {
                            "request_id": e["request_id"],
                            "token_hash": e.get("token_hash", ""),
                            "system_prompt_hash": e.get("system_prompt_hash"),
                            "timestamp_ms": rel_ts,
                            "input_length": e["input_length"],
                            "output_length": e["output_length"],
                            "hash_ids": e["hash_ids"],
                            "messages": json.dumps(messages, ensure_ascii=False),
                        }
                    )
                    if len(batch) >= PARQUET_BATCH_ROWS:
                        _flush()
                    count += 1
    _flush()
    writer.close()
    os.replace(jsonl_tmp, jsonl_path)
    os.replace(parquet_path + ".tmp", parquet_path)
    completions_volume.commit()
    return {
        "day": day,
        "skipped": False,
        "requests": count,
        "jsonl": jsonl_path,
        "parquet": parquet_path,
    }


@app.function(image=image, timeout=14400, volumes={"/data": completions_volume})
def package_week() -> dict:
    import os
    import time

    shards = sorted(f for f in os.listdir(DECODED_DIR) if f.endswith(".jsonl"))
    # Global first timestamp = first row of the earliest (sorted) shard,
    # whose rows are already sorted by abs_timestamp_ms.
    with open(os.path.join(DECODED_DIR, shards[0])) as f:
        global_min_ms = int(json.loads(f.readline())["abs_timestamp_ms"])

    by_day: dict[str, list[str]] = {}
    for s in shards:
        by_day.setdefault(_day_of(s), []).append(s)
    days = sorted(by_day)
    print(f"[package_week] {len(shards)} shards -> {len(days)} days; global_min_ms={global_min_ms}")

    args = [(day, by_day[day], global_min_ms) for day in days]
    t0 = time.time()
    total = 0
    out = []
    for r in package_day.starmap(args):
        total += r.get("requests", 0)
        out.append({k: r[k] for k in ("day", "requests", "skipped") if k in r})
        print(
            f"  {r['day']}: {r.get('requests', 0):,} requests "
            f"({'cached' if r.get('skipped') else 'written'})",
            flush=True,
        )
    completions_volume.commit()
    return {
        "days": len(days),
        "requests": total,
        "global_min_ms": global_min_ms,
        "elapsed_seconds": time.time() - t0,
        "per_day": out,
    }


@app.local_entrypoint()
def main():
    print(json.dumps(package_week.remote(), indent=2))
