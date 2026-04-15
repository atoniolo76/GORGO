from datetime import datetime, timedelta
import time

import modal

from app import app, completions_volume

query_clickhouse_fun = modal.Function.from_name("db-wrappers", "query_clickhouse", environment_name="modal-etl")


image = modal.Image.debian_slim().pip_install("pyarrow").add_local_python_source("app")


@app.function(image=image, volumes={"/data": completions_volume}, timeout=86400)
def download_responses(start: datetime, end: datetime, chunk_minutes: int = 60):
    import pyarrow as pa
    import pyarrow.parquet as pq

    chunks = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + timedelta(minutes=chunk_minutes), end)
        chunks.append((
            f"SELECT * FROM flash_llm_responses"
            f" WHERE timestamp >= '{cursor.strftime('%Y-%m-%d %H:%M:%S')}'"
            f" AND timestamp < '{chunk_end.strftime('%Y-%m-%d %H:%M:%S')}'",
            cursor.strftime("%Y%m%d_%H%M%S"),
        ))
        cursor = chunk_end

    queries = [q for q, _ in chunks]
    labels  = [l for _, l in chunks]

    total = 0
    chunk_times = []
    for label, results in zip(labels, query_clickhouse_fun.map(queries)):
        if not results:
            continue
        chunk_start = time.time()
        table = pa.Table.from_pylist(results)
        pq.write_table(table, f"/data/llm_responses_{label}.parquet", compression="zstd")
        chunk_times.append(time.time() - chunk_start)
        total += len(results)

    if chunk_times:
        avg_chunk_s = sum(chunk_times) / len(chunk_times)
        print(f"Avg per-chunk runtime: {avg_chunk_s:.3f}s over {len(chunk_times)} chunk(s)")

    start = time.time()
    print(f"Writing to volume at {start}")
    completions_volume.commit()
    print(f"Downloaded {total} rows across {len(chunks)} chunk(s) and committed to volume in {time.time() - start} seconds")
    return total


@app.local_entrypoint()
def main():
    download_responses.remote(
        start=datetime(2026, 4, 12, 11, 0, 0),
        end=datetime(2026, 4, 12, 11, 15, 0),
   
        chunk_minutes=15
    )
