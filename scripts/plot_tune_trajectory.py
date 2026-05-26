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

    best_tp = [e["best_params"]["prefill_weight"] for e in events]
    best_qw = [e["best_params"]["load_weight"] for e in events]
    has_rtt = "rtt_weight" in events[0].get("best_params", {})
    best_rtt = [e["best_params"].get("rtt_weight", 1.0) for e in events] if has_rtt else []

    # Candidate trajectory: the value that was actually active (being evaluated)
    # at each step. This is the candidate if present, otherwise the incumbent.
    active_tp, active_qw, active_rtt = [], [], []
    for e in events:
        c = e.get("candidate")
        if c:
            active_tp.append(c.get("prefill_weight", e["best_params"]["prefill_weight"]))
            active_qw.append(c.get("load_weight", e["best_params"]["load_weight"]))
            if has_rtt:
                active_rtt.append(c.get("rtt_weight", e["best_params"].get("rtt_weight", 1.0)))
        else:
            active_tp.append(e["best_params"]["prefill_weight"])
            active_qw.append(e["best_params"]["load_weight"])
            if has_rtt:
                active_rtt.append(e["best_params"].get("rtt_weight", 1.0))

    n_panels = 3 if has_rtt else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 3.5 * n_panels), sharex=True)
    fig.suptitle(args.title, fontsize=13, fontweight="bold")

    ax1 = axes[0]
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
    ax1.set_ylabel("prefill_weight", fontsize=12)
    ax1.legend(fontsize=9, loc="lower right")
    ax1.grid(True, alpha=0.2)

    ax2 = axes[1]
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
    ax2.set_ylabel("load_weight", fontsize=12)
    ax2.legend(fontsize=9, loc="lower right")
    ax2.grid(True, alpha=0.2)

    if has_rtt:
        ax3 = axes[2]
        ax3.plot(
            samples,
            active_rtt,
            color="tab:green",
            alpha=0.2,
            linewidth=1.2,
            zorder=1,
            label="Tested candidate",
        )
        ax3.plot(samples, best_rtt, color="tab:green", linewidth=2.5, zorder=2, label="Incumbent")
        ax3.set_yscale("log")
        ax3.set_ylabel("rtt_weight", fontsize=12)
        ax3.legend(fontsize=9, loc="lower right")
        ax3.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Total request samples", fontsize=11)

    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
