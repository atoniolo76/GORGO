"""Materialize a JSONL chat dataset for the lmsys workload adapter.

Output: ``data/lmsys/lmsys-chat.jsonl`` (gitignored). The lmsys ``workload``
adapter (``src/routing_harness/workload/lmsys.py``) expects this exact
format: one JSON object per line with at least ``conversation_id``,
``language``, and ``conversation`` (a list of ``{role, content}``).

Source: ``lmsys/lmsys-chat-1m`` — the canonical dataset. Gated on Hugging
Face; requires ``HF_TOKEN`` to be set to a token that has accepted the
dataset's license. There is intentionally NO fallback dataset: silently
substituting a different corpus would invalidate the metrics. If access
fails, the script exits non-zero.

To run on Modal (the supported path on devices without HF_TOKEN locally),
use ``scripts/fetch_lmsys_modal.py`` which reads ``HF_TOKEN_ROME`` from
the Modal secret and maps it to ``HF_TOKEN`` inside the container.

The actual source is recorded in ``data/lmsys/SOURCE.txt`` so the
metrics report can label it accurately.

Re-runnable: overwrites existing files.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "lmsys"
OUT_PATH = OUT_DIR / "lmsys-chat.jsonl"
SOURCE_PATH = OUT_DIR / "SOURCE.txt"

HF_DATASET = "lmsys/lmsys-chat-1m"
SOURCE_LABEL = "lmsys-chat-1m"


def _normalize_record(rec: dict) -> dict | None:
    """Coerce a HF record into the lmsys adapter's JSONL schema.

    Returns None if the record is missing the conversation field — those
    are useless and would just inflate the file with no usable turns.
    """
    convo = rec.get("conversation")
    if not convo:
        return None
    cid = str(rec.get("conversation_id", ""))
    lang = rec.get("language", "en")
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


def _stream_dataset():
    from datasets import load_dataset

    return load_dataset(
        HF_DATASET,
        split="train",
        streaming=True,
        token=os.environ["HF_TOKEN"],
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument(
        "--max-conversations",
        type=int,
        default=10000,
        help="Cap the number of conversations written (default 10000).",
    )
    args = ap.parse_args(argv)

    if not os.environ.get("HF_TOKEN"):
        raise SystemExit(
            "HF_TOKEN is not set. lmsys/lmsys-chat-1m is gated; export a "
            "token that has accepted the dataset license, or run via "
            "scripts/fetch_lmsys_modal.py on the arcadia-research Modal "
            "workspace (HF_TOKEN_ROME secret). Refusing to substitute a "
            "different dataset — that would invalidate the metrics."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"materializing lmsys-shaped JSONL into {OUT_PATH}")
    print(f"source: {HF_DATASET} (canonical, gated)")

    try:
        ds = _stream_dataset()
        # Probe one record to surface auth/license errors before we open
        # the output file.
        probe_iter = iter(ds)
        first = next(probe_iter)
    except Exception as e:
        raise SystemExit(
            f"FATAL: could not access {HF_DATASET}: "
            f"{type(e).__name__}: {str(e)[:300]}\n"
            "Check that HF_TOKEN belongs to an account that has accepted "
            "the lmsys/lmsys-chat-1m license at "
            "https://huggingface.co/datasets/lmsys/lmsys-chat-1m"
        ) from e

    n_written = 0
    with OUT_PATH.open("w", encoding="utf-8") as out:
        # `first` was consumed by the probe — write it before re-streaming
        # the rest so we don't lose a record.
        norm = _normalize_record(first)
        if norm is not None:
            out.write(json.dumps(norm, ensure_ascii=False) + "\n")
            n_written += 1
        for rec in probe_iter:
            norm = _normalize_record(rec)
            if norm is None:
                continue
            out.write(json.dumps(norm, ensure_ascii=False) + "\n")
            n_written += 1
            if n_written >= args.max_conversations:
                break

    SOURCE_PATH.write_text(f"hf_dataset: {SOURCE_LABEL}\nn_conversations: {n_written}\n")
    print(f"wrote {OUT_PATH} ({n_written:,} conversations) — source={SOURCE_LABEL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
