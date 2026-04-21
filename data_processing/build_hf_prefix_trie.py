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

Volumes: ``GORGO-lmsys-chat-1m`` at ``/lmsys``, ``GORGO-hf-datasets`` at ``/hf``.

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

MESSAGE_COLUMNS = ("conversation", "messages", "conversations")

# Tried in order until ``dataset_dict.json`` is found (first match wins).
LMSYS_DISK_CANDIDATES: list[tuple[str, str]] = [
    ("lmsys", "/lmsys/lmsys-chat-1m"),
    ("lmsys", "/lmsys/datasets/lmsys__lmsys-chat-1m"),
    ("hf", "/hf/lmsys-chat-1m"),
    ("hf", "/hf/datasets/lmsys__lmsys-chat-1m"),
]

WILDCHAT_DISK_CANDIDATES: list[tuple[str, str]] = [
    ("hf", "/hf/datasets/allenai__WildChat-4.8M"),
]


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
            "HF hub downloads use /hf/datasets/<org>__<name>."
        )
    if _is_hf_save_dir(path):
        return path
    parent = os.path.dirname(path)
    if parent and _is_hf_save_dir(parent):
        print(f"[prefix_trie] Using parent with dataset metadata: {parent}")
        return parent
    hint = ""
    if path.startswith("/hf/") and "lmsys" in path.lower():
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
    elif root.startswith("/hf"):
        hf_datasets_volume.commit()


@app.function(
    image=image,
    memory=1024 * 32,
    timeout=86400,
    volumes={
        "/lmsys": lmsys_chat_1m_volume,
        "/hf": hf_datasets_volume,
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
    print(
        f"Loaded {root!r} split len={n_total:,} (processing {limit:,}); "
        f"columns={sorted(columns)[:24]}{'…' if len(columns) > 24 else ''}"
    )

    enc = tiktoken.encoding_for_model("gpt-4o")
    tries_dir = os.path.join(root, "prefix_trie_checkpoints")
    os.makedirs(tries_dir, exist_ok=True)

    global_trie = RadixTrie()
    per_user_tries: dict[str, RadixTrie] = {}
    total_tokens = 0
    total_sequences = 0
    skipped_empty = 0
    resume_row = 0

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

            uid = _user_key(row, columns, row_index, user_key_column)
            if uid.startswith("row:") and not warned_user_key and not user_key_column:
                warned_user_key = True
                print(
                    "[warn] No user-id column found; using row index as key — "
                    "intra-user overlap (A) will be ~0. Set --user-key-column if "
                    "your schema has a grouping column."
                )

            token_ids = _row_to_token_ids(row, enc)
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
            }
            with open(tmp_path, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, ckpt_path)
            if _is_hf_save_dir(root):
                _commit_for_root(root)
            print(f"    checkpoint: {ckpt_path} ({os.path.getsize(ckpt_path) / 1e6:,.1f} MB)")

    if total_sequences == 0:
        print("No sequences ingested; nothing to report.")
        return {}

    print("\nComputing overlap stats...")
    global_unique = global_trie.unique_token_count()

    sum_intra_unique = 0
    user_count = 0
    top_users = []
    for uid, user_trie in per_user_tries.items():
        u = user_trie.unique_token_count()
        sum_intra_unique += u
        user_count += 1
        top_users.append((user_trie.total_tokens_inserted, u, uid))

    intra_savings = total_tokens - sum_intra_unique
    global_savings = total_tokens - global_unique
    cross_user_extra = sum_intra_unique - global_unique

    tag = stats_tag if stats_tag else _stats_basename(root)

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
):
    """KV prefix-trie overlap stats for HF ``save_to_disk`` data on ``/lmsys`` or ``/hf`` mounts.

    * ``--preset auto`` — search LMSYS default paths, then WildChat (first with valid metadata wins).
    * ``--preset lmsys`` / ``wildchat`` — only that dataset’s default paths.
    * ``--dataset-disk-path /abs/root`` — use this tree; ``preset`` / ``extra_candidates`` ignored for resolution.
    * ``--extra-candidates /path/a,/path/b`` — try these directories first (comma-separated), then ``preset`` paths.
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
    )
