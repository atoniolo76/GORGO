from app import app, completions_volume
import modal

image = (
    modal.Image.debian_slim()
    .pip_install("duckdb", "numpy", "pandas")
    .add_local_python_source("app")
)


@app.function(image=image, volumes={"/data": completions_volume})
def load_completions():
    import duckdb

    con = duckdb.connect()

    df = con.execute("""
        SELECT *
        FROM '/data/llm_responses_*.parquet'
    """).df()

    print(df)
