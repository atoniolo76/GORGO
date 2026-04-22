"""Find ``conversation_id`` values that appear more than once in WildChat ``train``.

Runs on the ``GORGO-hf-datasets`` volume (``--env=alessio-dev``). Expects a
Hugging Face ``save_to_disk`` tree (volume mounted at ``/datasets``; same layout as
``download_hf_dataset``: ``datasets/<org>__<name>`` on the volume)::

    /datasets/datasets/allenai__WildChat-4.8M/
        dataset_dict.json
        train/data-*.arrow

Example::

    modal run --env=alessio-dev data_processing/query_wildchat_duplicate_conversations.py::main
"""

from __future__ import annotations

import os

import modal

from app import app, hf_datasets_volume

DATASET_ROOT = "/datasets/datasets/allenai__WildChat-4.8M"

image = (
    modal.Image.debian_slim()
    .pip_install("datasets>=3.0", "duckdb", "pyarrow")
    .add_local_python_source("app")
)


def _id_column(train) -> str:
    """Pick a stable per-conversation key (WildChat on disk uses ``conversation_hash``)."""
    names = train.column_names
    for c in ("conversation_id", "conversation_hash"):
        if c in names:
            return c
    raise RuntimeError(f"need conversation_id or conversation_hash; have {names[:40]!r}")


def _load_train_id_table(dataset_root: str, id_column: str | None):
    """Return a single-column PyArrow table for the chosen id column (aliased to ``conversation_id`` for SQL)."""
    from datasets import Dataset, DatasetDict, load_from_disk

    if not os.path.isdir(dataset_root):
        raise RuntimeError(f"dataset root missing: {dataset_root!r}")

    dsd = load_from_disk(dataset_root)
    if isinstance(dsd, DatasetDict):
        if "train" not in dsd:
            raise RuntimeError(f"no train split under {dataset_root!r}")
        train = dsd["train"]
    elif isinstance(dsd, Dataset):
        train = dsd
    else:
        raise RuntimeError(f"unexpected load_from_disk type: {type(dsd)!r}")

    col = id_column if id_column is not None else _id_column(train)
    if col not in train.column_names:
        raise RuntimeError(f"column {col!r} not in dataset; have {train.column_names[:40]!r}")

    narrow = train.select_columns([col])
    t = narrow.data.table
    if col != "conversation_id":
        t = t.rename_columns(["conversation_id"])
    return t, col


@app.function(
    image=image,
    volumes={"/datasets": hf_datasets_volume},
    memory=1024 * 32,
    timeout=3600,
)
def duplicate_conversation_ids(
    *,
    dataset_root: str = DATASET_ROOT,
    id_column: str | None = None,
    limit_preview: int = 50,
):
    import duckdb

    table, source_col = _load_train_id_table(dataset_root, id_column)
    print(f"grouping column: {source_col} (exposed to SQL as conversation_id)")

    con = duckdb.connect()
    try:
        con.register("train", table)
        dupes = con.execute(
            """
            SELECT conversation_id, COUNT(*) AS occurrence_count
            FROM train
            GROUP BY conversation_id
            HAVING COUNT(*) > 1
            ORDER BY occurrence_count DESC, conversation_id
            """
        ).fetchdf()
    finally:
        con.close()

    n_dup_ids = len(dupes)
    extra_rows = int(dupes["occurrence_count"].sum() - n_dup_ids) if n_dup_ids else 0
    print(f"train rows scanned: {table.num_rows}")
    print(f"conversation_id values with count > 1: {n_dup_ids}")
    print(f"extra duplicate rows (sum(count-1)): {extra_rows}")
    if n_dup_ids:
        print("\n--- preview (head) ---")
        print(dupes.head(limit_preview).to_string(index=False))
    return {
        "dataset_root": dataset_root,
        "source_column": source_col,
        "rows_scanned": int(table.num_rows),
        "duplicate_id_count": int(n_dup_ids),
        "extra_duplicate_rows": int(extra_rows),
    }


@app.local_entrypoint()
def main():
    duplicate_conversation_ids.remote()
