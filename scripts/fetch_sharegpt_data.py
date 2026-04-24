"""Materialize a JSONL ShareGPT dataset for the sharegpt workload adapter.

Output: ``data/sharegpt/sharegpt.jsonl`` (gitignored). The sharegpt
``workload`` adapter (``src/routing_harness/workload/sharegpt.py``)
expects this exact format: one JSON object per line with ``id`` and
``conversations`` (a list of ``{from, value}``).

Source preference (the script picks the first one that works):

1. ``liyucheng/ShareGPT90K`` — parquet-backed, streamable via the
   ``datasets`` library, ungated. Stores the ~90k Vicuna-cleaned
   ShareGPT dump with ``conversations`` as a struct-of-arrays
   (``{from: [...], value: [...]}``) which we flatten on write. This
   is the default because streaming avoids downloading the full
   ~700MB canonical JSON.
2. ``anon8231489123/ShareGPT_Vicuna_unfiltered`` (file:
   ``HTML_cleaned_raw_dataset/sg_90k_part1.json``) — the canonical
   raw dump, fetched via ``hf_hub_download``. Used as a fallback in
   case the parquet mirror disappears. ~150MB, single-shot download.

The actual source is recorded in ``data/sharegpt/SOURCE.txt`` so the
metrics report can label it accurately.

Re-runnable: overwrites existing files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "sharegpt"
OUT_PATH = OUT_DIR / "sharegpt.jsonl"
SOURCE_PATH = OUT_DIR / "SOURCE.txt"


def _normalize_parquet_record(rec: dict) -> dict | None:
    """Convert liyucheng/ShareGPT90K's struct-of-arrays row to the adapter schema.

    Parquet rows look like ``{id, conversations: {from: [...], value: [...]}}``;
    the adapter wants ``conversations`` as a list of ``{from, value}`` dicts.
    """
    cid = str(rec.get("id", ""))
    conv = rec.get("conversations")
    if not isinstance(conv, dict):
        return None
    froms = conv.get("from") or []
    values = conv.get("value") or []
    if not froms or len(froms) != len(values):
        return None
    turns = [
        {"from": f, "value": v}
        for f, v in zip(froms, values)
        if isinstance(v, str) and v
    ]
    if not turns:
        return None
    return {"id": cid, "conversations": turns}


def _normalize_raw_record(rec: dict) -> dict | None:
    """Coerce a canonical sg_90k_part1.json record to the adapter schema.

    Raw records already have ``{id, conversations: [{from, value}]}`` —
    this just filters out malformed rows and strips unexpected fields.
    """
    cid = str(rec.get("id", ""))
    conv = rec.get("conversations")
    if not isinstance(conv, list):
        return None
    turns = [
        {"from": t.get("from", ""), "value": t.get("value", "")}
        for t in conv
        if isinstance(t, dict) and isinstance(t.get("value"), str) and t.get("value")
    ]
    if not turns:
        return None
    return {"id": cid, "conversations": turns}


def _fetch_parquet(max_conversations: int) -> int | None:
    try:
        from datasets import load_dataset
    except ImportError as e:
        print(f"  datasets unavailable: {e}", file=sys.stderr)
        return None
    try:
        ds = load_dataset("liyucheng/ShareGPT90K", split="train", streaming=True)
    except Exception as e:
        print(f"  parquet source unavailable: {type(e).__name__}: {str(e)[:160]}", file=sys.stderr)
        return None
    n_written = 0
    with OUT_PATH.open("w", encoding="utf-8") as out:
        for rec in ds:
            norm = _normalize_parquet_record(rec)
            if norm is None:
                continue
            out.write(json.dumps(norm, ensure_ascii=False) + "\n")
            n_written += 1
            if n_written >= max_conversations:
                break
    return n_written


def _fetch_raw(max_conversations: int) -> int | None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        print(f"  huggingface_hub unavailable: {e}", file=sys.stderr)
        return None
    try:
        path = hf_hub_download(
            repo_id="anon8231489123/ShareGPT_Vicuna_unfiltered",
            filename="HTML_cleaned_raw_dataset/sg_90k_part1.json",
            repo_type="dataset",
        )
    except Exception as e:
        print(f"  raw source unavailable: {type(e).__name__}: {str(e)[:160]}", file=sys.stderr)
        return None
    with open(path, "r", encoding="utf-8") as fh:
        blob = json.load(fh)
    if not isinstance(blob, list):
        print("  raw source had unexpected shape (not a JSON array)", file=sys.stderr)
        return None
    n_written = 0
    with OUT_PATH.open("w", encoding="utf-8") as out:
        for rec in blob:
            norm = _normalize_raw_record(rec)
            if norm is None:
                continue
            out.write(json.dumps(norm, ensure_ascii=False) + "\n")
            n_written += 1
            if n_written >= max_conversations:
                break
    return n_written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument(
        "--max-conversations",
        type=int,
        default=10000,
        help="Cap the number of conversations written (default 10000).",
    )
    args = ap.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"materializing ShareGPT JSONL into {OUT_PATH}")

    for label, fetcher in (
        ("sharegpt90k-parquet", _fetch_parquet),
        ("sharegpt-vicuna-raw-part1", _fetch_raw),
    ):
        print(f"trying source: {label}")
        n = fetcher(args.max_conversations)
        if n is not None and n > 0:
            SOURCE_PATH.write_text(f"hf_dataset: {label}\nn_conversations: {n}\n")
            print(f"wrote {OUT_PATH} ({n:,} conversations) — source={label}")
            return 0

    print(
        "ERROR: no source available. Check network access to huggingface.co "
        "or set HF_TOKEN if rate-limited.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
