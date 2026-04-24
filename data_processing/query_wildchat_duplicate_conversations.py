"""WildChat ``train`` conversation token stats (tiktoken ``gpt-4o``).

For each conversation key (``conversation_id`` or ``conversation_hash``), picks the
**single train row with the most messages** (tie: earliest row) — same rule as
``build_eval_dataset`` / ``build_hf_prefix_trie`` — then summarizes the distribution
of token counts for those rows (mean, min, max, percentiles). Row counts still reflect
how many train rows share each key.

Runs on the ``GORGO-hf-datasets`` volume (``--env=GORGO``). Layout::

    /datasets/datasets/allenai__WildChat-4.8M/
        dataset_dict.json
        train/data-*.arrow

Example::

    modal run --env=GORGO data_processing/query_wildchat_duplicate_conversations.py::main
"""

from __future__ import annotations

import os

import modal

from app import app, hf_datasets_volume

DATASET_ROOT = "/datasets/datasets/allenai__WildChat-4.8M"

MESSAGE_COLUMNS = ("conversation", "messages", "conversations")

image = (
    modal.Image.debian_slim()
    .pip_install("datasets>=3.0", "tiktoken")
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


def _messages_from_row(row: dict) -> list:
    import json

    for key in MESSAGE_COLUMNS:
        if key not in row:
            continue
        conv = row[key]
        if conv is None:
            continue
        if isinstance(conv, str):
            try:
                conv = json.loads(conv)
            except json.JSONDecodeError:
                continue
        if isinstance(conv, list):
            return conv
    return []


def _row_token_count(row: dict, enc) -> int:
    n = 0
    for msg in _messages_from_row(row):
        if isinstance(msg, dict):
            text = _content_to_str(msg.get("content"))
        elif isinstance(msg, str):
            text = msg
        else:
            text = ""
        if text:
            n += len(enc.encode(text, disallowed_special=()))
    return n


def _conversation_winners(
    train_narrow,
    *,
    conv_col: str,
    limit: int,
    row_batch_size: int,
) -> dict[str, int]:
    """conv key -> row index with max len(messages); tie: smallest index."""
    winners: dict[str, int] = {}
    best_n: dict[str, int] = {}
    for start in range(0, limit, row_batch_size):
        end = min(start + row_batch_size, limit)
        batch = train_narrow[start:end]
        keys = list(batch.keys())
        batch_len = len(batch[keys[0]]) if keys else 0
        for j in range(batch_len):
            row = {k: batch[k][j] for k in keys}
            row_index = start + j
            raw = row.get(conv_col)
            if raw is None or str(raw) == "":
                continue
            ckey = str(raw)
            mc = len(_messages_from_row(row))
            prev = best_n.get(ckey)
            if prev is None or mc > prev:
                best_n[ckey] = mc
                winners[ckey] = row_index
    return winners


def _conversation_key_column(column_names: list[str]) -> str:
    for c in ("conversation_id", "conversation_hash"):
        if c in column_names:
            return c
    raise RuntimeError(f"need conversation_id or conversation_hash; have {column_names[:40]!r}")


def _summarize(values: list[int]) -> dict:
    import statistics

    if not values:
        return {
            "n": 0,
            "mean": None,
            "min": None,
            "max": None,
            "stdev": None,
            "p50": None,
            "p90": None,
            "p99": None,
        }
    v = sorted(values)
    n = len(v)

    def pct(p: float) -> int:
        if n == 1:
            return v[0]
        i = min(n - 1, max(0, int(round(p * (n - 1)))))
        return v[i]

    return {
        "n": n,
        "mean": statistics.fmean(v),
        "min": v[0],
        "max": v[-1],
        "stdev": statistics.stdev(v) if n > 1 else 0.0,
        "p50": pct(0.5),
        "p90": pct(0.9),
        "p99": pct(0.99),
    }


@app.function(
    image=image,
    volumes={"/datasets": hf_datasets_volume},
    memory=1024 * 32,
    timeout=86400,
)
def conversation_token_stats(
    *,
    dataset_root: str = DATASET_ROOT,
    id_column: str | None = None,
    row_batch_size: int = 512,
):
    import time

    import tiktoken
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

    conv_col = id_column if id_column is not None else _conversation_key_column(train.column_names)
    if conv_col not in train.column_names:
        raise RuntimeError(f"column {conv_col!r} not in dataset; have {train.column_names[:40]!r}")
    if not any(k in train.column_names for k in MESSAGE_COLUMNS):
        raise RuntimeError(
            f"need one of {MESSAGE_COLUMNS} for tokenization; have {train.column_names!r}"
        )

    need_cols = [conv_col] + [k for k in MESSAGE_COLUMNS if k in train.column_names]
    train_narrow = train.select_columns(need_cols)

    enc = tiktoken.encoding_for_model("gpt-4o")
    token_by_conv: dict[str, int] = {}
    row_count_by_conv: dict[str, int] = {}
    rows_empty_key = 0
    limit = len(train_narrow)
    t0 = time.time()

    print("Resolving longest row per conversation (by message count)...", flush=True)
    winners = _conversation_winners(
        train_narrow,
        conv_col=conv_col,
        limit=limit,
        row_batch_size=row_batch_size,
    )

    for start in range(0, limit, row_batch_size):
        end = min(start + row_batch_size, limit)
        batch = train_narrow[start:end]
        keys = list(batch.keys())
        batch_len = len(batch[keys[0]]) if keys else 0
        for j in range(batch_len):
            row = {k: batch[k][j] for k in keys}
            row_index = start + j
            raw = row.get(conv_col)
            if raw is None or str(raw) == "":
                rows_empty_key += 1
                continue
            ckey = str(raw)
            row_count_by_conv[ckey] = row_count_by_conv.get(ckey, 0) + 1
            if row_index != winners.get(ckey):
                continue
            token_by_conv[ckey] = _row_token_count(row, enc)
        elapsed = time.time() - t0
        print(
            f"  rows {end:,}/{limit:,} | {len(row_count_by_conv):,} distinct conversations | "
            f"{elapsed:,.0f}s",
            flush=True,
        )

    per_conv_totals = list(token_by_conv.values())
    dist = _summarize(per_conv_totals)
    rows_per_conv_values = list(row_count_by_conv.values())
    row_dist = _summarize(rows_per_conv_values)
    total_tokens = sum(per_conv_totals)
    n_multi_row = sum(1 for c in rows_per_conv_values if c > 1)

    print(f"\n--- Conversation key: {conv_col!r} (tiktoken gpt-4o) ---")
    print(f"train rows:                        {limit:>14,}")
    print(f"rows with non-empty key:           {limit - rows_empty_key:>14,}")
    print(f"rows with empty / missing key:     {rows_empty_key:>14,}")
    print(f"distinct conversations:          {dist['n']:>14,}")
    print(f"conversations with >1 train row: {n_multi_row:>14,}")
    print(f"total tokens (longest row per key): {total_tokens:>14,}")
    print()
    print("Tokens **per conversation** (longest row by message count for that key):")
    print(
        f"  mean={dist['mean']:.2f}  stdev={dist['stdev']:.2f}  "
        f"min={dist['min']:,}  max={dist['max']:,}  "
        f"p50={dist['p50']:,}  p90={dist['p90']:,}  p99={dist['p99']:,}"
    )
    print()
    print("Train **rows per conversation** (how many rows share the key):")
    print(
        f"  mean={row_dist['mean']:.2f}  stdev={row_dist['stdev']:.2f}  "
        f"min={row_dist['min']:,}  max={row_dist['max']:,}  "
        f"p50={row_dist['p50']:,}  p90={row_dist['p90']:,}  p99={row_dist['p99']:,}"
    )

    return {
        "dataset_root": dataset_root,
        "conversation_key_column": conv_col,
        "train_rows": limit,
        "rows_with_empty_conversation_key": rows_empty_key,
        "distinct_conversations": dist["n"],
        "conversations_with_multiple_train_rows": n_multi_row,
        "tokens_per_conversation_semantic": "longest_row_by_message_count",
        "total_tokens_longest_row_per_key": total_tokens,
        "tokens_per_conversation": dist,
        "train_rows_per_conversation": row_dist,
    }


@app.local_entrypoint()
def main():
    conversation_token_stats.remote()
