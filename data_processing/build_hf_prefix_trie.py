"""Build radix tries over Hugging Face ``save_to_disk`` datasets (LMSYS-Chat-1M,
WildChat-4.8M, etc.) and report KV-style prefix overlap (intra-user vs global),
same metrics as ``build_prefix_trie.py``.

Expects::

    <root>/dataset_dict.json
    <root>/train/data-*.arrow

Tokenization matches ``build_eval_dataset`` (tiktoken ``gpt-4o``; message ``content``
concatenated in order). Supports ``conversation`` / ``messages`` / ``conversations``.

**User key:** tries ``user_id``, ``hashed_user_id``, ``hashed_ip`` (WildChat),
``user_hash``, ``ip_hash``, ``conversation_id``, ``conversation_hash``. Override with
``--user-key-column``. If nothing matches, falls back to ``row:<index>`` (intra-user
overlap ~0).

**Conversation stats:** if the split has ``conversation_id`` or ``conversation_hash``,
also reports duplicate conversation keys (rows with count > 1), rows-per-conversation
distribution, and tokens-per-conversation (tiktoken ``gpt-4o``, summed across rows in
the processed range). Aggregates are included in checkpoints; old checkpoints without
them skip this block until you delete ``prefix_trie_checkpoints/`` and rerun from row 0.

After a full pass, per-conversation aggregates are written to
``prefix_trie_conversation_aggregates.pkl`` and
``prefix_trie_checkpoints/conversation_aggregates.pkl`` (small files, no tries).
``--conversation-stats-only`` tries the **newest** ``prefix_trie_checkpoints/rows_*.pkl``
first (heavy unpickle), then small aggregate pickles. Partial checkpoints
(``next_row`` < ``limit``) still yield conversation stats with a warning; small caches
are only written after a **full**-run checkpoint load.

Volumes: ``GORGO-lmsys-chat-1m`` at ``/lmsys``, ``GORGO-hf-datasets`` at ``/datasets``.

Examples::

    modal run --env=alessio-dev data_processing/build_hf_prefix_trie.py::prefix_trie
    modal run --detach --env=alessio-dev data_processing/build_hf_prefix_trie.py::prefix_trie --preset wildchat
    modal run data_processing/build_hf_prefix_trie.py::prefix_trie --preset lmsys \\
        --dataset-disk-path /lmsys/lmsys-chat-1m
    # ``--preset auto`` tries LMSYS paths first, then WildChat. Override search order with
    # ``--extra-candidates /path/one,/path/two`` (tried before preset list).
"""

from __future__ import annotations

import os
import re
import sys
from array import array

import modal

from app import app, hf_datasets_volume, lmsys_chat_1m_volume
from build_eval_dataset import _content_to_str
from utils.radix_trie import RadixTrie

image = (
    modal.Image.debian_slim()
    .pip_install("datasets>=3.0", "pyarrow", "tiktoken")
    .add_local_python_source("app", "build_eval_dataset", "utils")
)

USER_KEY_CANDIDATES = (
    "user_id",
    "hashed_user_id",
    "hashed_ip",
    "user_hash",
    "ip_hash",
    "conversation_id",
    "conversation_hash",
)

# For duplicate / per-conversation token stats (not the same ordering as USER_KEY_CANDIDATES).
CONVERSATION_KEY_CANDIDATES = ("conversation_id", "conversation_hash")

MESSAGE_COLUMNS = ("conversation", "messages", "conversations")

# Tried in order until ``dataset_dict.json`` is found (first match wins).
LMSYS_DISK_CANDIDATES: list[tuple[str, str]] = [
    ("lmsys", "/lmsys/lmsys-chat-1m"),
    ("lmsys", "/lmsys/datasets/lmsys__lmsys-chat-1m"),
    # ``download_hf_dataset`` writes ``datasets/<hub_id>`` on the volume → ``/datasets/datasets/...``.
    ("hf", "/datasets/datasets/lmsys__lmsys-chat-1m"),
    ("hf", "/datasets/lmsys-chat-1m"),
]

# Hub download layout on ``GORGO-hf-datasets``: ``datasets/allenai__WildChat-4.8M/`` (checkpoints
# under ``.../prefix_trie_checkpoints/``). Mount volume at ``/datasets`` → use ``/datasets/datasets/...``.
WILDCHAT_DISK_CANDIDATES: list[tuple[str, str]] = [
    ("hf", "/datasets/datasets/allenai__WildChat-4.8M"),
    ("hf", "/datasets/allenai__WildChat-4.8M"),
]

CONVERSATION_AGGREGATES_FILENAME = "prefix_trie_conversation_aggregates.pkl"
# Small pickle next to ``rows_*.pkl`` (no tries); fast path for ``--conversation-stats-only``.
CHECKPOINT_CONVERSATION_AGGREGATES_FILENAME = "conversation_aggregates.pkl"
CONVERSATION_AGGREGATES_VERSION = 1
# Progress logs while computing overlap stats over per-user tries (can be millions of keys).
OVERLAP_STATS_LOG_EVERY_USERS = 50_000
# Log sort progress inside ``_summarize_int_distribution`` when this many values or more.
DISTRIBUTION_SORT_LOG_MIN = 300_000


def _fmt_pct(num: float, den: float) -> str:
    if den <= 0:
        return "  n/a"
    return f"{100.0 * num / den:6.2f}%"


def _is_hf_save_dir(path: str) -> bool:
    """True if ``path`` looks like ``datasets`` ``save_to_disk`` output."""
    if not os.path.isdir(path):
        return False
    if os.path.isfile(os.path.join(path, "dataset_dict.json")):
        return True
    # Single-split ``Dataset.save_to_disk`` (no dataset_dict.json).
    if os.path.isfile(os.path.join(path, "dataset_info.json")) and os.path.isfile(
        os.path.join(path, "state.json")
    ):
        return True
    return False


def _normalize_hf_disk_root(path: str) -> str:
    """Resolve path to a loadable HF on-disk dataset; fix ``.../train`` → parent when needed."""
    path = os.path.normpath(path)
    if not os.path.isdir(path):
        raise RuntimeError(
            f"Dataset path is not a directory: {path!r}. "
            "Check Modal volume mounts: LMSYS is often /lmsys/lmsys-chat-1m; "
            "HF hub downloads on this volume use /datasets/datasets/<org>__<name> "
            "(volume-relative datasets/<org>__<name>)."
        )
    if _is_hf_save_dir(path):
        return path
    parent = os.path.dirname(path)
    if parent and _is_hf_save_dir(parent):
        print(f"[prefix_trie] Using parent with dataset metadata: {parent}")
        return parent
    hint = ""
    if path.startswith("/datasets/") and "lmsys" in path.lower():
        hint = (
            " For LMSYS-Chat-1M, data is usually on volume GORGO-lmsys-chat-1m "
            "at /lmsys/lmsys-chat-1m (not on GORGO-hf-datasets unless you ran "
            "download_hf_cli for that dataset)."
        )
    try:
        listing = sorted(os.listdir(path))[:40]
    except OSError:
        listing = []
    raise RuntimeError(
        f"Not a Hugging Face datasets save_to_disk directory: {path!r}.{hint} "
        "Expected dataset_dict.json (DatasetDict) or dataset_info.json + state.json (Dataset). "
        f"Listing: {listing}"
    )


def _find_disk_root(candidates: list[tuple[str, str]]) -> tuple[str, str]:
    for tag, path in candidates:
        if os.path.isdir(path) and _is_hf_save_dir(path):
            return tag, path
    paths = [p for _, p in candidates]
    raise RuntimeError("Could not find HF save_to_disk root. Tried: " + ", ".join(paths))


def _candidates_for_preset(preset: str) -> list[tuple[str, str]]:
    p = preset.lower().strip().replace("-", "_")
    if p in ("auto", "all", "combined"):
        return list(LMSYS_DISK_CANDIDATES) + list(WILDCHAT_DISK_CANDIDATES)
    if p in ("lmsys", "lmsys_chat", "lmsys_chat_1m"):
        return list(LMSYS_DISK_CANDIDATES)
    if p in ("wildchat", "wild_chat", "allenai_wildchat"):
        return list(WILDCHAT_DISK_CANDIDATES)
    raise ValueError(f"unknown --preset {preset!r}; use auto, lmsys, or wildchat")


def _merge_disk_candidates(
    preset: str,
    extra_candidates: str | None,
) -> list[tuple[str, str]] | None:
    """Paths from ``extra_candidates`` (comma-separated) are tried first, then ``preset`` list."""
    merged: list[tuple[str, str]] = []
    if extra_candidates:
        for i, raw in enumerate(extra_candidates.split(",")):
            path = raw.strip()
            if path:
                merged.append((f"extra{i}", path))
    merged.extend(_candidates_for_preset(preset))
    return merged


def _stats_basename(root: str) -> str:
    base = os.path.basename(os.path.normpath(root))
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", base)
    return safe or "dataset"


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


def _conversation_key_column(columns: set[str]) -> str | None:
    for k in CONVERSATION_KEY_CANDIDATES:
        if k in columns:
            return k
    return None


def _conversation_key_value(row: dict, conv_col: str | None) -> str | None:
    if not conv_col:
        return None
    v = row.get(conv_col)
    if v is None or str(v) == "":
        return None
    return str(v)


def _conversation_aggregates_path(root: str) -> str:
    return os.path.join(root, CONVERSATION_AGGREGATES_FILENAME)


def _conversation_aggregates_checkpoint_path(root: str) -> str:
    return os.path.join(
        root, "prefix_trie_checkpoints", CHECKPOINT_CONVERSATION_AGGREGATES_FILENAME
    )


def _try_load_conversation_aggregates(
    path: str,
    *,
    limit: int,
    n_total: int,
    min_sequence_len: int,
    conv_col: str,
) -> tuple[dict[str, int], dict[str, int]] | None:
    import pickle

    if not os.path.isfile(path):
        return None
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
    except (OSError, pickle.UnpicklingError):
        return None
    if payload.get("version") != CONVERSATION_AGGREGATES_VERSION:
        return None
    if payload.get("limit") != limit:
        return None
    if payload.get("n_total") != n_total:
        return None
    if payload.get("min_sequence_len") != min_sequence_len:
        return None
    if payload.get("conversation_key_column") != conv_col:
        return None
    done_row = payload.get("completed_next_row", limit)
    if done_row != limit:
        return None
    crc = payload.get("conversation_row_counts")
    ctt = payload.get("conversation_token_totals")
    if not isinstance(crc, dict) or not isinstance(ctt, dict):
        return None
    return crc, ctt


def _save_conversation_aggregates(
    path: str,
    *,
    limit: int,
    n_total: int,
    min_sequence_len: int,
    conv_col: str,
    conversation_row_counts: dict[str, int],
    conversation_token_totals: dict[str, int],
    completed_next_row: int | None = None,
) -> None:
    import pickle

    done = limit if completed_next_row is None else completed_next_row
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(
            {
                "version": CONVERSATION_AGGREGATES_VERSION,
                "limit": limit,
                "n_total": n_total,
                "min_sequence_len": min_sequence_len,
                "conversation_key_column": conv_col,
                "completed_next_row": done,
                "conversation_row_counts": conversation_row_counts,
                "conversation_token_totals": conversation_token_totals,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    os.replace(tmp_path, path)


def _try_load_conversation_aggregates_from_latest_rows_checkpoint(
    tries_dir: str,
    *,
    limit: int,
) -> tuple[dict[str, int], dict[str, int], str, bool, int | None] | None:
    """Load aggregates from the **single newest** ``rows_*.pkl`` (by row suffix; unpickle is heavy).

    Returns ``(row_counts, token_totals, path, full_run, checkpoint_next_row)`` where
    ``full_run`` means ``payload[\"next_row\"] == limit``.
    """
    import pickle

    if not os.path.isdir(tries_dir):
        return None

    numbered: list[tuple[int, str]] = []
    for name in os.listdir(tries_dir):
        m = re.fullmatch(r"rows_(\d+)\.pkl", name)
        if m:
            numbered.append((int(m.group(1)), name))
    if not numbered:
        return None
    _row_end, name = max(numbered, key=lambda x: x[0])
    path = os.path.join(tries_dir, name)
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
    except (OSError, pickle.UnpicklingError, EOFError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("limit") != limit:
        return None
    crc = payload.get("conversation_row_counts")
    ctt = payload.get("conversation_token_totals")
    if not isinstance(crc, dict) or not isinstance(ctt, dict):
        return None
    next_row = payload.get("next_row")
    full_run = next_row == limit
    ckpt_next = int(next_row) if isinstance(next_row, int) else None
    return crc, ctt, path, full_run, ckpt_next


def _resolve_conversation_aggregates_stats_only(
    root: str,
    *,
    limit: int,
    n_total: int,
    min_sequence_len: int,
    conv_col: str,
) -> tuple[dict[str, int], dict[str, int], str, bool, int | None] | None:
    """Return (row_counts, token_totals, source_label, full_run, checkpoint_next_row)."""

    tries_dir = os.path.join(root, "prefix_trie_checkpoints")

    from_ckpt = _try_load_conversation_aggregates_from_latest_rows_checkpoint(
        tries_dir, limit=limit
    )
    if from_ckpt is not None:
        crc, ctt, ck_path, full_run, ckpt_next = from_ckpt
        return (
            crc,
            ctt,
            f"latest rows checkpoint ({ck_path!r})",
            full_run,
            ckpt_next,
        )

    for path, label in (
        (_conversation_aggregates_path(root), "dataset root aggregate file"),
        (
            _conversation_aggregates_checkpoint_path(root),
            "prefix_trie_checkpoints/conversation_aggregates.pkl",
        ),
    ):
        got = _try_load_conversation_aggregates(
            path,
            limit=limit,
            n_total=n_total,
            min_sequence_len=min_sequence_len,
            conv_col=conv_col,
        )
        if got is not None:
            return got[0], got[1], f"{label} ({path!r})", True, None

    return None


def _emit_conversation_stats(
    conv_col: str | None,
    conv_stats_incomplete: bool,
    conv_row_counts: dict[str, int],
    conv_token_totals: dict[str, int],
) -> dict | None:
    """Print the conversation block; return the JSON-able ``conversation_stats`` dict."""
    if conv_col and not conv_stats_incomplete and conv_row_counts:
        import time

        t_conv = time.time()
        n_distinct = len(conv_row_counts)
        print(
            f"\n[stats] conversation-level metrics: {n_distinct:,} distinct keys | "
            "counts, duplicates, distributions...",
            flush=True,
        )
        dup_conv_keys = sum(1 for c in conv_row_counts.values() if c > 1)
        extra_dup_rows = sum(c - 1 for c in conv_row_counts.values() if c > 1)
        rows_attributed = sum(conv_row_counts.values())
        print(
            f"  [stats] building rows-per-conversation and tokens-per-conversation vectors "
            f"({n_distinct:,} entries)...",
            flush=True,
        )
        rows_per_conv = list(conv_row_counts.values())
        tok_per_conv = [conv_token_totals.get(k, 0) for k in conv_row_counts]
        print(f"  [stats] summarizing row-count distribution...", flush=True)
        row_dist = _summarize_int_distribution(rows_per_conv, log_label="rows per conversation")
        print(f"  [stats] summarizing token-count distribution...", flush=True)
        tok_dist = _summarize_int_distribution(tok_per_conv, log_label="tokens per conversation")
        print(
            f"  [stats] ranking conversations for top-10 display ({n_distinct:,} keys)...",
            flush=True,
        )
        top_by_rows = sorted(((c, k) for k, c in conv_row_counts.items()), reverse=True)[:10]

        print(f"\n--- Conversation stats (column {conv_col!r}) ---")
        print(f"Rows with non-empty {conv_col}:     {rows_attributed:>14,}")
        print(f"Distinct conversations:             {n_distinct:>14,}")
        print(f"Conversations with >1 row:          {dup_conv_keys:>14,}")
        print(f"Extra duplicate rows Σ(count−1):    {extra_dup_rows:>14,}")
        print(
            f"Rows / conversation:  min={row_dist['min']:,}  max={row_dist['max']:,}  "
            f"mean={row_dist['mean']:.2f}  stdev={row_dist['stdev']:.2f}  "
            f"p50={row_dist['p50']:,}  p90={row_dist['p90']:,}  p99={row_dist['p99']:,}"
        )
        print(
            "Tokens / conversation (tiktoken gpt-4o, summed over rows in split):  "
            f"min={tok_dist['min']:,}  max={tok_dist['max']:,}  "
            f"mean={tok_dist['mean']:.2f}  stdev={tok_dist['stdev']:.2f}  "
            f"p50={tok_dist['p50']:,}  p90={tok_dist['p90']:,}  p99={tok_dist['p99']:,}"
        )
        print("Top 10 conversations by row count:")
        for cnt, key in top_by_rows:
            print(f"  {cnt:>10,}  rows  {key!s}")

        print(
            f"  [stats] conversation block finished in {time.time() - t_conv:,.1f}s",
            flush=True,
        )

        return {
            "conversation_key_column": conv_col,
            "rows_with_conversation_key": rows_attributed,
            "distinct_conversations": n_distinct,
            "conversations_with_duplicate_rows": dup_conv_keys,
            "extra_duplicate_rows": extra_dup_rows,
            "rows_per_conversation": row_dist,
            "tokens_per_conversation": tok_dist,
            "top_conversations_by_row_count": [
                {"conversation_key": k, "row_count": c} for c, k in top_by_rows
            ],
        }
    if conv_col and conv_stats_incomplete:
        print(
            "\n--- Conversation stats ---\n"
            "Skipped (resume checkpoint without conversation aggregates). "
            "Delete prefix_trie_checkpoints/ and rerun from row 0 for these metrics."
        )
        return {
            "skipped": True,
            "reason": "checkpoint_missing_conversation_aggregates",
            "conversation_key_column": conv_col,
        }
    if conv_col and not conv_stats_incomplete and not conv_row_counts:
        print(
            f"\n--- Conversation stats (column {conv_col!r}) ---\n"
            "No rows had a non-empty conversation key in the processed range."
        )
        return {
            "skipped": True,
            "reason": "no_non_empty_conversation_keys_in_processed_rows",
            "conversation_key_column": conv_col,
        }
    if not conv_col:
        return {"skipped": True, "reason": "no_conversation_column"}
    return None


def _summarize_int_distribution(
    values: list[int],
    *,
    log_label: str = "",
) -> dict:
    """Mean, stdev, min, max, and percentiles for a non-empty list of ints."""
    import statistics
    import time

    if not values:
        return {
            "n": 0,
            "mean": None,
            "stdev": None,
            "min": None,
            "max": None,
            "p50": None,
            "p90": None,
            "p99": None,
        }
    n = len(values)
    if n >= DISTRIBUTION_SORT_LOG_MIN:
        t0 = time.time()
        suffix = f" ({log_label})" if log_label else ""
        print(
            f"  [stats] sorting {n:,} values{suffix} for percentiles...",
            flush=True,
        )
    vals = sorted(values)
    if n >= DISTRIBUTION_SORT_LOG_MIN:
        suffix = f" ({log_label})" if log_label else ""
        print(
            f"  [stats] sort done{suffix} in {time.time() - t0:,.1f}s",
            flush=True,
        )

    def pct(p: float) -> int:
        if n == 1:
            return vals[0]
        idx = min(n - 1, max(0, int(round(p * (n - 1)))))
        return vals[idx]

    return {
        "n": n,
        "mean": statistics.fmean(vals),
        "stdev": statistics.stdev(vals) if n > 1 else 0.0,
        "min": vals[0],
        "max": vals[-1],
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p99": pct(0.99),
    }


def _user_key(
    row: dict,
    columns: set[str],
    row_index: int,
    user_key_column: str | None,
) -> str:
    if user_key_column:
        v = row.get(user_key_column)
        if v is not None and str(v) != "":
            return str(v)
        return f"missing:{user_key_column}:{row_index}"
    for k in USER_KEY_CANDIDATES:
        if k in columns:
            v = row.get(k)
            if v is not None and str(v) != "":
                return str(v)
    return f"row:{row_index}"


def _row_to_token_ids(row: dict, enc) -> list[int]:
    messages = _messages_from_row(row)
    out: list[int] = []
    for msg in messages:
        if isinstance(msg, dict):
            text = _content_to_str(msg.get("content"))
        elif isinstance(msg, str):
            text = msg
        else:
            text = ""
        if text:
            out.extend(enc.encode(text, disallowed_special=()))
    return out


def _commit_for_root(root: str) -> None:
    if root.startswith("/lmsys"):
        lmsys_chat_1m_volume.commit()
    elif root.startswith("/datasets"):
        hf_datasets_volume.commit()


@app.function(
    image=image,
    memory=1024 * 64,
    timeout=86400,
    volumes={
        "/lmsys": lmsys_chat_1m_volume,
        "/datasets": hf_datasets_volume,
    },
)
def build_hf_disk_prefix_tries(
    *,
    dataset_disk_path: str | None = None,
    disk_candidates: list[tuple[str, str]] | None = None,
    run_preset: str | None = None,
    extra_candidates_logged: str | None = None,
    user_key_column: str | None = None,
    row_batch_size: int = 512,
    checkpoint_every_rows: int = 50_000,
    min_sequence_len: int = 1,
    max_rows: int | None = None,
    stats_tag: str | None = None,
    conversation_stats_only: bool = False,
):
    import json
    import pickle
    import time

    import tiktoken
    from datasets import Dataset, DatasetDict, load_from_disk

    if dataset_disk_path:
        root = _normalize_hf_disk_root(dataset_disk_path)
    else:
        cands = (
            disk_candidates
            if disk_candidates is not None
            else (LMSYS_DISK_CANDIDATES + WILDCHAT_DISK_CANDIDATES)
        )
        _, root = _find_disk_root(cands)

    sys.setrecursionlimit(1_000_000)

    dsd = load_from_disk(root)
    if isinstance(dsd, DatasetDict):
        if "train" in dsd:
            dset = dsd["train"]
        else:
            key = next(iter(dsd))
            dset = dsd[key]
    elif isinstance(dsd, Dataset):
        dset = dsd
    else:
        raise RuntimeError(f"Unexpected load_from_disk result: {type(dsd)!r}")

    n_total = len(dset)
    limit = n_total if max_rows is None else min(n_total, max_rows)
    columns = set(dset.column_names)
    conv_col = _conversation_key_column(columns)
    print(
        f"Loaded {root!r} split len={n_total:,} (processing {limit:,}); "
        f"columns={sorted(columns)[:24]}{'…' if len(columns) > 24 else ''}"
    )
    if conv_col:
        print(f"[prefix_trie] Per-conversation stats use column {conv_col!r}")
    else:
        print(
            "[prefix_trie] No conversation_id / conversation_hash column; "
            "skipping duplicate-conversation and per-conversation token stats."
        )

    tag = stats_tag if stats_tag else _stats_basename(root)

    if conversation_stats_only:
        if not conv_col:
            print(
                "[prefix_trie] --conversation-stats-only needs conversation_id or "
                "conversation_hash on this split."
            )
            return {}
        resolved = _resolve_conversation_aggregates_stats_only(
            root,
            limit=limit,
            n_total=n_total,
            min_sequence_len=min_sequence_len,
            conv_col=conv_col,
        )
        if resolved is None:
            tries_dir = os.path.join(root, "prefix_trie_checkpoints")
            print(
                f"[prefix_trie] No usable conversation aggregates (tried latest rows_*.pkl "
                f"under {tries_dir!r}, then dataset root {CONVERSATION_AGGREGATES_FILENAME!r}, "
                f"then prefix_trie_checkpoints/{CHECKPOINT_CONVERSATION_AGGREGATES_FILENAME!r}). "
                f"Latest checkpoint must match limit={limit:,} and include "
                "conversation_row_counts / conversation_token_totals."
            )
            return {}
        conv_row_counts, conv_token_totals, agg_source, aggregates_full_run, ckpt_next = resolved
        print(
            f"[prefix_trie] Loaded conversation aggregates from {agg_source} "
            "(skipping trie build and per-row tokenization)."
        )
        if not aggregates_full_run and ckpt_next is not None:
            print(
                f"[prefix_trie] WARNING: partial checkpoint (next_row={ckpt_next:,} "
                f"vs limit={limit:,}); conversation stats are only for rows processed "
                "before that checkpoint."
            )
        if "rows checkpoint" in agg_source and aggregates_full_run:
            os.makedirs(os.path.join(root, "prefix_trie_checkpoints"), exist_ok=True)
            for save_path in (
                _conversation_aggregates_path(root),
                _conversation_aggregates_checkpoint_path(root),
            ):
                _save_conversation_aggregates(
                    save_path,
                    limit=limit,
                    n_total=n_total,
                    min_sequence_len=min_sequence_len,
                    conv_col=conv_col,
                    conversation_row_counts=conv_row_counts,
                    conversation_token_totals=conv_token_totals,
                    completed_next_row=limit,
                )
            print(
                "[prefix_trie] Wrote small aggregate caches "
                f"({CONVERSATION_AGGREGATES_FILENAME!r} and "
                f"prefix_trie_checkpoints/{CHECKPOINT_CONVERSATION_AGGREGATES_FILENAME!r}) "
                "for faster loads next time."
            )
        conversation_stats = _emit_conversation_stats(
            conv_col, False, conv_row_counts, conv_token_totals
        )
        stats = {
            "dataset_disk_path": root,
            "stats_tag": tag,
            "preset": run_preset,
            "extra_candidates": extra_candidates_logged,
            "conversation_stats_only": True,
            "conversation_aggregates_source": agg_source,
            "conversation_aggregates_full_run": aggregates_full_run,
            "checkpoint_next_row": ckpt_next,
            "conversation_stats": conversation_stats,
        }
        stats_path = os.path.join(root, f"prefix_trie_stats_{tag}.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        if _is_hf_save_dir(root):
            _commit_for_root(root)
        print(f"\nStats written to {stats_path}")
        return stats

    enc = tiktoken.encoding_for_model("gpt-4o")
    tries_dir = os.path.join(root, "prefix_trie_checkpoints")
    os.makedirs(tries_dir, exist_ok=True)

    global_trie = RadixTrie()
    per_user_tries: dict[str, RadixTrie] = {}
    total_tokens = 0
    total_sequences = 0
    skipped_empty = 0
    resume_row = 0
    conv_row_counts: dict[str, int] = {}
    conv_token_totals: dict[str, int] = {}
    conv_stats_incomplete = False

    ckpt_files = sorted(
        f for f in os.listdir(tries_dir) if f.startswith("rows_") and f.endswith(".pkl")
    )
    if ckpt_files:
        latest = os.path.join(tries_dir, ckpt_files[-1])
        print(f"Resuming from checkpoint: {latest}")
        with open(latest, "rb") as f:
            payload = pickle.load(f)
        global_trie = payload["global_trie"]
        per_user_tries = payload["per_user_tries"]
        total_tokens = payload["total_tokens"]
        total_sequences = payload["total_sequences"]
        skipped_empty = payload["skipped_empty"]
        resume_row = payload["next_row"]
        if conv_col and payload.get("conversation_row_counts") is not None:
            conv_row_counts = payload["conversation_row_counts"]
            conv_token_totals = payload.get("conversation_token_totals") or {}
        elif conv_col:
            conv_stats_incomplete = True
            print(
                "  WARNING: checkpoint has no conversation_row_counts; "
                "duplicate / per-conversation token stats omitted this run."
            )
        if payload.get("limit") != limit:
            print(
                f"  WARNING: checkpoint limit={payload.get('limit')} vs current {limit}; "
                "delete prefix_trie_checkpoints/ for a clean run."
            )
        print(
            f"  restored at row {resume_row:,} | {total_sequences:,} seqs, "
            f"{total_tokens:,} toks, {len(per_user_tries):,} user keys"
        )

    t0 = time.time()
    warned_user_key = False

    for start in range(resume_row, limit, row_batch_size):
        end = min(start + row_batch_size, limit)
        batch = dset[start:end]
        keys = list(batch.keys())
        batch_len = len(batch[keys[0]]) if keys else 0
        for j in range(batch_len):
            row = {k: batch[k][j] for k in keys}
            row_index = start + j

            ckey = _conversation_key_value(row, conv_col)
            if ckey is not None and not conv_stats_incomplete:
                conv_row_counts[ckey] = conv_row_counts.get(ckey, 0) + 1

            uid = _user_key(row, columns, row_index, user_key_column)
            if uid.startswith("row:") and not warned_user_key and not user_key_column:
                warned_user_key = True
                print(
                    "[warn] No user-id column found; using row index as key — "
                    "intra-user overlap (A) will be ~0. Set --user-key-column if "
                    "your schema has a grouping column."
                )

            token_ids = _row_to_token_ids(row, enc)
            if ckey is not None and not conv_stats_incomplete:
                conv_token_totals[ckey] = conv_token_totals.get(ckey, 0) + len(token_ids)

            if len(token_ids) < min_sequence_len:
                skipped_empty += 1
                continue
            seq = array("I", token_ids)

            global_trie.insert(seq)
            user_trie = per_user_tries.get(uid)
            if user_trie is None:
                user_trie = RadixTrie()
                per_user_tries[uid] = user_trie
            user_trie.insert(seq)

            total_tokens += len(seq)
            total_sequences += 1

        elapsed = time.time() - t0
        print(
            f"  rows {end:,}/{limit:,} | {total_sequences:,} seqs, {total_tokens:,} toks | "
            f"{elapsed:,.0f}s elapsed"
        )

        crossed = (end // checkpoint_every_rows) > (start // checkpoint_every_rows)
        if end == limit or crossed:
            ckpt_path = os.path.join(tries_dir, f"rows_{end:08d}.pkl")
            tmp_path = ckpt_path + ".tmp"
            payload = {
                "next_row": end,
                "limit": limit,
                "total_sequences": total_sequences,
                "total_tokens": total_tokens,
                "skipped_empty": skipped_empty,
                "global_trie": global_trie,
                "per_user_tries": per_user_tries,
                "conversation_row_counts": conv_row_counts,
                "conversation_token_totals": conv_token_totals,
            }
            with open(tmp_path, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, ckpt_path)
            if end == limit and conv_col and conv_row_counts and not conv_stats_incomplete:
                _save_conversation_aggregates(
                    _conversation_aggregates_checkpoint_path(root),
                    limit=limit,
                    n_total=n_total,
                    min_sequence_len=min_sequence_len,
                    conv_col=conv_col,
                    conversation_row_counts=conv_row_counts,
                    conversation_token_totals=conv_token_totals,
                    completed_next_row=end,
                )
            if _is_hf_save_dir(root):
                _commit_for_root(root)
            print(f"    checkpoint: {ckpt_path} ({os.path.getsize(ckpt_path) / 1e6:,.1f} MB)")

    if total_sequences == 0:
        print("No sequences ingested; nothing to report.")
        return {}

    print("\nComputing overlap stats...", flush=True)
    t_overlap = time.time()
    n_user_keys = len(per_user_tries)
    print(
        f"  [stats] {n_user_keys:,} user/group tries | global trie unique_token_count()...",
        flush=True,
    )
    global_unique = global_trie.unique_token_count()
    print(
        f"  [stats] global U(all)={global_unique:,} ({time.time() - t_overlap:,.1f}s)",
        flush=True,
    )

    sum_intra_unique = 0
    user_count = 0
    top_users = []
    if n_user_keys <= 0:
        log_every = 1
    else:
        log_every = max(1, n_user_keys // 25)
        log_every = min(log_every, OVERLAP_STATS_LOG_EVERY_USERS)
    for uid, user_trie in per_user_tries.items():
        u = user_trie.unique_token_count()
        sum_intra_unique += u
        user_count += 1
        top_users.append((user_trie.total_tokens_inserted, u, uid))
        if user_count % log_every == 0:
            print(
                f"  [stats] per-user unique: {user_count:,}/{n_user_keys:,} keys | "
                f"Σ_u U(R_u) so far={sum_intra_unique:,} | {time.time() - t_overlap:,.1f}s",
                flush=True,
            )

    print(
        f"  [stats] per-user unique counts done ({n_user_keys:,} keys) in "
        f"{time.time() - t_overlap:,.1f}s",
        flush=True,
    )

    intra_savings = total_tokens - sum_intra_unique
    global_savings = total_tokens - global_unique
    cross_user_extra = sum_intra_unique - global_unique

    print(f"\n{'=' * 72}")
    print(f"Dataset root:                    {root}")
    print(f"Sequences inserted:            {total_sequences:>14,}")
    print(f"User / group keys:             {user_count:>14,}")
    print(f"Empty / short skipped:         {skipped_empty:>14,}")
    print(f"Total tokens T:                {total_tokens:>14,}")
    print(f"{'-' * 72}")
    print("(A) INTRA-USER overlap   -- prefixes shared within the same user/group key")
    print(
        f"      (T - sum_u U(R_u)) / T   "
        f"= {intra_savings:>14,} / {total_tokens:,}  "
        f"= {_fmt_pct(intra_savings, total_tokens)}"
    )
    print()
    print("(C) CROSS-USER overlap   -- extra sharing from pooling across keys")
    print(
        f"      (sum_u U(R_u) - U(all)) / T   "
        f"= {cross_user_extra:>14,} / {total_tokens:,}  "
        f"= {_fmt_pct(cross_user_extra, total_tokens)}"
    )
    print(f"{'-' * 72}")
    print("(B) GLOBAL overlap")
    print(
        f"      (T - U(all)) / T         "
        f"= {global_savings:>14,} / {total_tokens:,}  "
        f"= {_fmt_pct(global_savings, total_tokens)}"
    )
    print(
        f"      A + C                     "
        f"= {_fmt_pct(intra_savings + cross_user_extra, total_tokens)}"
    )
    print(f"{'=' * 72}")

    conversation_stats = _emit_conversation_stats(
        conv_col, conv_stats_incomplete, conv_row_counts, conv_token_totals
    )

    if (
        conv_col
        and not conv_stats_incomplete
        and conv_row_counts
        and conversation_stats
        and not conversation_stats.get("skipped")
    ):
        agg_path = _conversation_aggregates_path(root)
        _save_conversation_aggregates(
            agg_path,
            limit=limit,
            n_total=n_total,
            min_sequence_len=min_sequence_len,
            conv_col=conv_col,
            conversation_row_counts=conv_row_counts,
            conversation_token_totals=conv_token_totals,
            completed_next_row=limit,
        )
        print(f"[prefix_trie] Wrote conversation aggregates cache {agg_path!r}")

    top_users.sort(reverse=True)
    print("\nTop 10 groups by tokens inserted:")
    print(f"  {'tokens':>14}  {'unique':>14}  {'savings':>8}  key")
    for total_t, uniq_t, th in top_users[:10]:
        saved = total_t - uniq_t
        print(f"  {total_t:>14,}  {uniq_t:>14,}  {_fmt_pct(saved, total_t)}  {th!s}")

    stats = {
        "dataset_disk_path": root,
        "stats_tag": tag,
        "preset": run_preset,
        "extra_candidates": extra_candidates_logged,
        "user_key_column": user_key_column,
        "row_batch_size": row_batch_size,
        "min_sequence_len": min_sequence_len,
        "max_rows": max_rows,
        "total_sequences": total_sequences,
        "total_tokens": total_tokens,
        "skipped_empty": skipped_empty,
        "user_count": user_count,
        "global_unique_tokens": global_unique,
        "sum_intra_unique_tokens": sum_intra_unique,
        "intra_user_savings": intra_savings,
        "cross_user_extra_savings": cross_user_extra,
        "global_savings": global_savings,
        "intra_user_savings_pct": (
            100.0 * intra_savings / total_tokens if total_tokens > 0 else None
        ),
        "cross_user_extra_pct": (
            100.0 * cross_user_extra / total_tokens if total_tokens > 0 else None
        ),
        "global_savings_pct": (100.0 * global_savings / total_tokens if total_tokens > 0 else None),
        "elapsed_seconds": time.time() - t0,
        "top_groups": [
            {"key": k, "tokens": tt, "unique_tokens": uu} for tt, uu, k in top_users[:10]
        ],
        "conversation_stats": conversation_stats,
    }

    stats_path = os.path.join(root, f"prefix_trie_stats_{tag}.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    _commit_for_root(root)
    print(f"\nStats written to {stats_path}")
    return stats


@app.local_entrypoint()
def prefix_trie(
    preset: str = "auto",
    dataset_disk_path: str | None = None,
    extra_candidates: str | None = None,
    user_key_column: str | None = None,
    row_batch_size: int = 512,
    checkpoint_every_rows: int = 50_000,
    min_sequence_len: int = 1,
    max_rows: int | None = None,
    stats_tag: str | None = None,
    conversation_stats_only: bool = False,
):
    """KV prefix-trie overlap stats for HF ``save_to_disk`` data on ``/lmsys`` or ``/datasets`` mounts.

    * ``--preset auto`` — search LMSYS default paths, then WildChat (first with valid metadata wins).
    * ``--preset lmsys`` / ``wildchat`` — only that dataset’s default paths.
    * ``--dataset-disk-path /abs/root`` — use this tree; ``preset`` / ``extra_candidates`` ignored for resolution.
    * ``--extra-candidates /path/a,/path/b`` — try these directories first (comma-separated), then ``preset`` paths.
    * ``--conversation-stats-only`` — load the **newest** ``rows_*.pkl`` first, then small
      aggregate pickles; materialize small caches only after a full-run checkpoint load.
    """
    disk_candidates: list[tuple[str, str]] | None = None
    if dataset_disk_path is None:
        disk_candidates = _merge_disk_candidates(preset, extra_candidates)

    build_hf_disk_prefix_tries.remote(
        dataset_disk_path=dataset_disk_path,
        disk_candidates=disk_candidates,
        run_preset=preset,
        extra_candidates_logged=extra_candidates,
        user_key_column=user_key_column,
        row_batch_size=row_batch_size,
        checkpoint_every_rows=checkpoint_every_rows,
        min_sequence_len=min_sequence_len,
        max_rows=max_rows,
        stats_tag=stats_tag,
        conversation_stats_only=conversation_stats_only,
    )
