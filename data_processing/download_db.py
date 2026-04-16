from datetime import datetime, timedelta
import itertools
import os
import time

import modal

from app import app, completions_volume

query_clickhouse_fun = modal.Function.from_name(
    "db-wrappers", "query_clickhouse", environment_name="modal-etl"
)


image = modal.Image.debian_slim().pip_install("pyarrow").add_local_python_source("app")


@app.function(
    image=image,
    volumes={"/data": completions_volume},
    timeout=86400,
    memory=64 * 1024,
    retries=10,
)
def download_responses(start: datetime, end: datetime, chunk_minutes: int = 60):
    import pyarrow as pa
    import pyarrow.parquet as pq

    existing = {f for f in os.listdir("/data") if f.endswith(".parquet")}

    chunks = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + timedelta(minutes=chunk_minutes), end)
        label = cursor.strftime("%Y%m%d_%H%M%S")
        filename = f"llm_responses_{label}.parquet"
        if filename not in existing:
            chunks.append(
                (
                    f"SELECT * FROM flash_llm_responses"
                    f" WHERE timestamp >= '{cursor.strftime('%Y-%m-%d %H:%M:%S')}'"
                    f" AND timestamp < '{chunk_end.strftime('%Y-%m-%d %H:%M:%S')}'",
                    label,
                )
            )
        cursor = chunk_end

    skipped = (end - start) // timedelta(minutes=chunk_minutes) - len(chunks)
    print(f"Skipping {skipped} already-downloaded chunk(s)")

    if not chunks:
        print("All chunks already downloaded, nothing to do.")
        return 0

    queries = [q for q, _ in chunks]
    labels = [l for _, l in chunks]

    max_concurrent = int(os.environ.get("MAX_CONCURRENT_CONNECTIONS", 20))
    batches = list(itertools.batched(zip(queries, labels), max_concurrent))
    print(
        f"Spawning {len(chunks)} function(s) in {len(batches)} batch(es) of ≤{max_concurrent} to cover {start} → {end}"
    )

    total = 0
    chunk_times = []
    for i, batch in enumerate(batches):
        batch_queries, batch_labels = zip(*batch)
        print(f"Batch {i + 1}/{len(batches)} ({len(batch_queries)} queries)")
        for label, results in zip(batch_labels, query_clickhouse_fun.map(batch_queries)):
            if not results:
                continue
            chunk_start = time.time()
            table = pa.Table.from_pylist(results)
            pq.write_table(table, f"/data/llm_responses_{label}.parquet", compression="zstd")
            chunk_times.append(time.time() - chunk_start)
            total += len(results)

        commit_start = time.time()
        print(f"Committing volume after batch {i + 1}/{len(batches)}...")
        completions_volume.commit()
        print(f"Committed in {time.time() - commit_start:.2f}s ({total} rows so far)")

    if chunk_times:
        avg_chunk_s = sum(chunk_times) / len(chunk_times)
        print(f"Avg per-chunk runtime: {avg_chunk_s:.3f}s over {len(chunk_times)} chunk(s)")

    print(f"Done. Downloaded {total} rows across {len(chunks)} chunk(s)")
    return total


@app.local_entrypoint()
def main(
    start: str = "2026-03-12T12:00:00",
    end: str = "2026-04-12T12:00:00",
    chunk_minutes: int = 30,
):
    download_responses.remote(
        start=datetime.fromisoformat(start),
        end=datetime.fromisoformat(end),
        chunk_minutes=chunk_minutes,
    )
