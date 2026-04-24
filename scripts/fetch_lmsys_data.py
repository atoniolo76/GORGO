"""Materialize a JSONL chat dataset for the lmsys workload adapter.

Output: ``data/lmsys/lmsys-chat.jsonl`` (gitignored). The lmsys ``workload``
adapter (``src/routing_harness/workload/lmsys.py``) expects this exact
format: one JSON object per line with at least ``conversation_id``,
``language``, and ``conversation`` (a list of ``{role, content}``).

Source preference (the script picks the first one that works):

1. ``lmsys/lmsys-chat-1m`` — the canonical dataset. Gated on Hugging Face;
   requires ``HF_TOKEN`` environment variable. Set it and re-run to use
   the real lmsys data.
2. ``allenai/WildChat-1M`` — ungated, structurally identical schema
   (Allen AI's release of the WildChat conversations). Used as a
   fallback so the metrics pipeline isn't blocked on dataset access.
   This is what most environments will get.

The actual source is recorded in ``data/lmsys/SOURCE.txt`` so the
metrics report can label it accurately.

Re-runnable: overwrites existing files.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "lmsys"
OUT_PATH = OUT_DIR / "lmsys-chat.jsonl"
SOURCE_PATH = OUT_DIR / "SOURCE.txt"

# WildChat reports language as the full English name; the lmsys adapter
# filters on ISO 639-1 codes by default. Map the most common ones; pass
# unknowns through unchanged so the user sees them in the metrics report.
_LANG_TO_ISO = {
    "English": "en",
    "Chinese": "zh",
    "Spanish": "es",
    "Russian": "ru",
    "French": "fr",
    "German": "de",
    "Portuguese": "pt",
    "Japanese": "ja",
    "Korean": "ko",
    "Italian": "it",
    "Vietnamese": "vi",
    "Polish": "pl",
    "Indonesian": "id",
    "Turkish": "tr",
    "Arabic": "ar",
    "Dutch": "nl",
    "Ukrainian": "uk",
    "Thai": "th",
    "Hindi": "hi",
    "Czech": "cs",
}


def _normalize_lang(s: str | None) -> str:
    if not s:
        return "und"
    return _LANG_TO_ISO.get(s, s if len(s) <= 5 else s[:5])


def _normalize_record(rec: dict, source: str) -> dict | None:
    """Coerce a HF record into the lmsys adapter's JSONL schema.

    Returns None if the record is missing the conversation field — those
    are useless and would just inflate the file with no usable turns.
    """
    convo = rec.get("conversation")
    if not convo:
        return None
    if source == "lmsys-chat-1m":
        cid = str(rec.get("conversation_id", ""))
        lang = rec.get("language", "en")
    else:  # wildchat
        cid = str(rec.get("conversation_hash", rec.get("conversation_id", "")))
        lang = _normalize_lang(rec.get("language"))
    # Trim each turn down to the two fields the adapter actually reads
    # — otherwise WildChat's per-turn moderation payload bloats the
    # file by 5-10x with data we never look at.
    trimmed = [
        {"role": t.get("role"), "content": t.get("content", "")}
        for t in convo
        if isinstance(t, dict) and t.get("role") in ("user", "assistant")
    ]
    if not trimmed:
        return None
    return {
        "conversation_id": cid,
        "language": lang,
        "conversation": trimmed,
    }


def _stream_dataset(name: str, *, use_token: bool):
    from datasets import load_dataset

    kwargs: dict = {"split": "train", "streaming": True}
    if use_token and os.environ.get("HF_TOKEN"):
        kwargs["token"] = os.environ["HF_TOKEN"]
    return load_dataset(name, **kwargs)


def _try_source(name: str, label: str, *, use_token: bool):
    try:
        ds = _stream_dataset(name, use_token=use_token)
        # Pull one record to confirm access before we commit to it.
        first = next(iter(ds))
        return ds, first, label
    except Exception as e:
        print(f"  unavailable ({label}): {type(e).__name__}: {str(e)[:140]}", file=sys.stderr)
        return None


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
    print(f"materializing lmsys-shaped JSONL into {OUT_PATH}")

    chosen = None
    for name, label in (
        ("lmsys/lmsys-chat-1m", "lmsys-chat-1m"),
        ("allenai/WildChat-1M", "wildchat-1m"),
    ):
        print(f"trying source: {name}")
        chosen = _try_source(name, label, use_token=True)
        if chosen is not None:
            break

    if chosen is None:
        print(
            "ERROR: no source available. Either set HF_TOKEN with access to "
            "lmsys/lmsys-chat-1m, or check network access to huggingface.co.",
            file=sys.stderr,
        )
        return 1

    ds, first, source_label = chosen
    print(f"  using: {source_label}")

    n_written = 0
    with OUT_PATH.open("w", encoding="utf-8") as out:
        # `first` was already consumed by the probe; re-stream from the start
        # so the cap counts records consistently regardless of which source.
        for rec in _stream_dataset(
            "lmsys/lmsys-chat-1m" if source_label == "lmsys-chat-1m" else "allenai/WildChat-1M",
            use_token=True,
        ):
            norm = _normalize_record(rec, source_label)
            if norm is None:
                continue
            out.write(json.dumps(norm, ensure_ascii=False) + "\n")
            n_written += 1
            if n_written >= args.max_conversations:
                break

    SOURCE_PATH.write_text(f"hf_dataset: {source_label}\nn_conversations: {n_written}\n")
    print(f"wrote {OUT_PATH} ({n_written:,} conversations) — source={source_label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
