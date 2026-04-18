from app import app, completions_volume
import modal

image = modal.Image.debian_slim().pip_install("duckdb").add_local_python_source("app")


@app.function(image=image, volumes={"/data": completions_volume}, timeout=300)
def dump_sample_requests(n: int = 20):
    """Dump a sample of raw `request` values from the first parquet file to stdout as JSON lines."""
    import json
    import os

    import duckdb

    parquet_dir = "/data"
    files = sorted(f for f in os.listdir(parquet_dir) if f.endswith(".parquet"))
    if not files:
        print("No parquet files found")
        return

    path = os.path.join(parquet_dir, files[0])
    print(f"Sampling {n} rows from {files[0]}")

    con = duckdb.connect()

    schema = con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [path]).fetchall()
    print("=== SCHEMA ===")
    for col_name, col_type, *_ in schema:
        print(f"  {col_name}: {col_type}")
    print()

    rows = con.execute(
        f"SELECT * FROM read_parquet(?) WHERE request NOT LIKE '%keep-alive%' USING SAMPLE {int(n)}",
        [path],
    ).fetchall()
    columns = [desc[0] for desc in con.description]
    con.close()

    samples = []
    for i, row in enumerate(rows):
        sample = {"index": i}
        for col, val in zip(columns, row):
            sample[col] = {"type": type(val).__name__, "value": val}
        samples.append(sample)

    print(json.dumps(samples, indent=2, default=str))


@app.local_entrypoint()
def main(n: int = 20):
    dump_sample_requests.remote(n=n)
