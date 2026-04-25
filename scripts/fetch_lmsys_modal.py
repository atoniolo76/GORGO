"""Fetch lmsys/lmsys-chat-1m + run dataset_metrics on Modal CPU.

The host device cannot fetch ``lmsys/lmsys-chat-1m`` directly because
the dataset is gated and there is no ``HF_TOKEN`` available locally.
The arcadia-research Modal workspace has the ``hf_token_rome`` secret
which exposes ``HF_TOKEN_ROME`` inside the container; we map that to
``HF_TOKEN`` and run the same fetch + metrics pipeline that would run
locally.

Run:

    MODAL_PROFILE=arcadia-research modal run --env=GORGO \\
        scripts/fetch_lmsys_modal.py [--max-conversations 10000]

Outputs (committed back to the host worktree):

    research/data/dataset_metrics/lmsys.json
    research/reports/dataset_metrics_lmsys.md
    research/reports/figures/lmsys_*.png
    research/data/lmsys/SOURCE.txt   # canonical-source provenance

The raw ``data/lmsys/lmsys-chat.jsonl`` produced inside the container is
intentionally NOT pulled back: the host's ``data/`` directory is
gitignored and the canonical numbers in ``research/`` are what we
actually need.
"""

from __future__ import annotations

from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent

app = modal.App("gorgo-lmsys-fetch")

# Mirror the host repo into the container so the existing
# fetch + metrics scripts can run unmodified. Skip caches and large
# generated artifacts that would just slow upload.
_image_root = modal.Image.debian_slim(python_version="3.12").pip_install(
    "datasets>=2.14",
    "huggingface_hub>=0.20",
    "tiktoken>=0.7",
    "numpy>=1.26",
    "matplotlib>=3.8",
    "PyYAML>=6.0",
)
image = _image_root.add_local_dir(
    local_path=str(REPO_ROOT),
    remote_path="/repo",
    copy=True,
    ignore=[
        ".git/**",
        ".venv/**",
        "**/__pycache__/**",
        "**/*.pyc",
        "data/**",
        "research/data/dataset_metrics/**",
        "research/reports/dataset_metrics_*.md",
        "research/reports/figures/**",
        ".beads/**",
        "node_modules/**",
    ],
)


@app.function(
    image=image,
    timeout=3600,
    cpu=4.0,
    memory=8192,
    secrets=[modal.Secret.from_name("hf_token_rome")],
)
def fetch_and_profile(max_conversations: int = 10000) -> dict[str, bytes]:
    import os
    import subprocess
    import sys

    repo = Path("/repo")
    os.chdir(repo)

    token = os.environ.get("HF_TOKEN_ROME") or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "Neither HF_TOKEN_ROME nor HF_TOKEN is set inside the container. "
            "Confirm the hf_token_rome Modal secret is attached and exposes "
            "HF_TOKEN_ROME."
        )
    os.environ["HF_TOKEN"] = token

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo / "src")

    print(f"=== fetching lmsys/lmsys-chat-1m (max={max_conversations:,}) ===", flush=True)
    subprocess.check_call(
        [
            sys.executable,
            str(repo / "scripts" / "fetch_lmsys_data.py"),
            "--max-conversations",
            str(max_conversations),
        ],
        env=env,
    )

    print("=== running dataset_metrics --dataset lmsys ===", flush=True)
    subprocess.check_call(
        [sys.executable, str(repo / "scripts" / "dataset_metrics.py"), "--dataset", "lmsys"],
        env=env,
    )

    outputs: dict[str, bytes] = {}
    rels = [
        "research/data/dataset_metrics/lmsys.json",
        "research/reports/dataset_metrics_lmsys.md",
    ]
    for rel in rels:
        p = repo / rel
        if not p.exists():
            raise RuntimeError(f"expected output not produced: {rel}")
        outputs[rel] = p.read_bytes()

    fig_dir = repo / "research" / "reports" / "figures"
    figs = sorted(fig_dir.glob("lmsys_*.png")) if fig_dir.exists() else []
    if not figs:
        raise RuntimeError("no lmsys_*.png figures produced")
    for f in figs:
        outputs[f"research/reports/figures/{f.name}"] = f.read_bytes()

    src_file = repo / "data" / "lmsys" / "SOURCE.txt"
    if src_file.exists():
        outputs["research/data/lmsys/SOURCE.txt"] = src_file.read_bytes()

    total = sum(len(v) for v in outputs.values())
    print(f"=== returning {len(outputs)} files, {total:,} bytes total ===", flush=True)
    return outputs


@app.local_entrypoint()
def main(max_conversations: int = 10000):
    print(f"submitting fetch_and_profile(max_conversations={max_conversations:,}) to Modal …")
    outputs = fetch_and_profile.remote(max_conversations=max_conversations)
    for rel, content in outputs.items():
        out = REPO_ROOT / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(content)
        print(f"wrote {out} ({len(content):,} bytes)")
