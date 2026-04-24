"""Download HumanEval + MBPP and merge into a single code-completion JSONL.

Output: ``data/code_completion/combined.jsonl`` (1138 tasks). ``data/`` is
gitignored — this script is the canonical way to re-materialize the
dataset ``scripts/dataset_metrics.py --dataset code_completion`` expects.

Re-runnable: overwrites existing files.
"""

from __future__ import annotations

import gzip
import json
import shutil
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "code_completion"

HUMANEVAL_URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
MBPP_URL = (
    "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl"
)


def _fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp, dest.open("wb") as out:
        shutil.copyfileobj(resp, out)


def main() -> int:
    gz = OUT_DIR / "HumanEval.jsonl.gz"
    he = OUT_DIR / "HumanEval.jsonl"
    mbpp = OUT_DIR / "mbpp.jsonl"
    combined = OUT_DIR / "combined.jsonl"

    print(f"fetching {HUMANEVAL_URL}")
    _fetch(HUMANEVAL_URL, gz)
    with gzip.open(gz, "rb") as src, he.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    gz.unlink()

    print(f"fetching {MBPP_URL}")
    _fetch(MBPP_URL, mbpp)

    n = 0
    with combined.open("w") as out:
        for line in he.open():
            r = json.loads(line)
            out.write(
                json.dumps({"task_id": r["task_id"], "prompt": r["prompt"], "language": "python"})
                + "\n"
            )
            n += 1
        for line in mbpp.open():
            r = json.loads(line)
            out.write(
                json.dumps(
                    {
                        "task_id": f"MBPP/{r['task_id']}",
                        "prompt": r["text"],
                        "language": "python",
                    }
                )
                + "\n"
            )
            n += 1
    print(f"wrote {combined} ({n} tasks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
