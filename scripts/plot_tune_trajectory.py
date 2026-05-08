"""Plot hyperparameter trajectory with tested mutations as backdrop.

Usage:
    python scripts/plot_tune_trajectory.py \
        --tune-jsonl results/proxy_traces/<run_id>_gorgo-hillclimb/tune.jsonl \
        --out results/analysis/tune_trajectory.png \
        --title "Hillclimb Convergence"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tune-jsonl", required=True)
    parser.add_argument("--out", default="results/analysis/tune_trajectory.png")
    parser.add_argument("--title", default="Hillclimb Hyperparameter Convergence")
    args = parser.parse_args()

    events = [json.loads(l) for l in open(args.tune_jsonl)]
    if not events:
        print("No events")
        return

    samples = [e.get("total_samples", i) for i, e in enumerate(events)]

    best_tp = [e["best_params"]["t_prefill"] for e in events]
    best_qw = [e["best_params"]["queued_tokens_weight"] for e in events]

    # Candidate trajectory: the value that was actually active (being evaluated)
    # at each step. This is the candidate if present, otherwise the incumbent.
    active_tp, active_qw = [], []
    for e in events:
        c = e.get("candidate")
        if c:
            active_tp.append(c["t_prefill"])
            active_qw.append(c["queued_tokens_weight"])
        else:
            active_tp.append(e["best_params"]["t_prefill"])
            active_qw.append(e["best_params"]["queued_tokens_weight"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(args.title, fontsize=13, fontweight="bold")

    ax1.plot(
        samples,
        active_tp,
        color="tab:blue",
        alpha=0.2,
        linewidth=1.2,
        zorder=1,
        label="Tested candidate",
    )
    ax1.plot(samples, best_tp, color="tab:blue", linewidth=2.5, zorder=2, label="Incumbent")
    ax1.set_yscale("log")
    ax1.set_ylabel("$t_{\\mathrm{prefill}}$", fontsize=12)
    ax1.legend(fontsize=9, loc="lower right")
    ax1.grid(True, alpha=0.2)

    ax2.plot(
        samples,
        active_qw,
        color="tab:orange",
        alpha=0.2,
        linewidth=1.2,
        zorder=1,
        label="Tested candidate",
    )
    ax2.plot(samples, best_qw, color="tab:orange", linewidth=2.5, zorder=2, label="Incumbent")
    ax2.set_yscale("log")
    ax2.set_ylabel("$\\alpha$ (queued tokens weight)", fontsize=12)
    ax2.set_xlabel("Total request samples", fontsize=11)
    ax2.legend(fontsize=9, loc="lower right")
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
