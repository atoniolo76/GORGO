"""Parallel-coordinates TTFT comparison across policies.

Three axes (p50, p95, p99) with one line per policy. GORGO variants
are bold dark blue; baselines are muted gray. Rank shifts from p50
to p99 are immediately visible.

Usage:
    python scripts/plot_ttft_parallel.py \
        --w1 results/analysis/glm5_w1.csv \
        --w2 results/analysis/glm5_w2.csv \
        --out paper/figures/ttft_parallel.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


GORGO_POLICIES = {"gorgo-hillclimb", "gorgo-static", "gorgo-autotune"}
PCTS = ["p50", "p95", "p99"]
PCT_COLS = [f"ttft_{p}" for p in PCTS]

POLICY_COLORS = {
    "gorgo-hillclimb": "#1b3a5c",
    "gorgo-static": "#2d6a9f",
    "gorgo-autotune": "#4a86c7",
}
BASELINE_COLOR = "#b0b0b0"


def _plot_panel(ax: plt.Axes, df: pd.DataFrame, title: str) -> None:
    df = df.sort_values("ttft_p95", ascending=True).reset_index(drop=True)
    x = np.arange(len(PCTS))

    for _, row in df.iterrows():
        policy = row["policy"]
        is_gorgo = policy in GORGO_POLICIES
        vals = [row[c] for c in PCT_COLS]
        color = POLICY_COLORS.get(policy, BASELINE_COLOR)
        lw = 2.4 if is_gorgo else 1.2
        alpha = 1.0 if is_gorgo else 0.45
        zorder = 10 if is_gorgo else 2
        marker = "o" if is_gorgo else "."
        ms = 7 if is_gorgo else 4

        ax.plot(
            x,
            vals,
            color=color,
            linewidth=lw,
            alpha=alpha,
            zorder=zorder,
            marker=marker,
            markersize=ms,
            markeredgecolor="white" if is_gorgo else "none",
            markeredgewidth=0.6,
            label=policy if is_gorgo else None,
        )

    for _, row in df.iterrows():
        policy = row["policy"]
        if policy in GORGO_POLICIES:
            continue
        vals = [row[c] for c in PCT_COLS]
        ax.plot(
            [],
            [],
            color=BASELINE_COLOR,
            linewidth=1.2,
            alpha=0.45,
            marker=".",
            markersize=4,
            label=policy,
        )

    best_p95 = df.loc[df["ttft_p95"].idxmin(), "policy"]
    for _, row in df.iterrows():
        policy = row["policy"]
        is_gorgo = policy in GORGO_POLICIES
        val_p99 = row["ttft_p99"]
        color = POLICY_COLORS.get(policy, "#777777")
        fw = "bold" if is_gorgo else "normal"
        fs = 8.5 if is_gorgo else 7.5
        ax.annotate(
            policy,
            xy=(2, val_p99),
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=fs,
            fontweight=fw,
            color=color,
            va="center",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(PCTS, fontsize=11, fontweight="bold")
    ax.set_ylabel("TTFT (s)")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.set_xlim(-0.3, 2.3 + 1.8)
    ax.grid(axis="y", alpha=0.2)
    ax.grid(axis="x", alpha=0.15)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--w1", type=Path, required=True)
    parser.add_argument("--w2", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    df_w1 = pd.read_csv(args.w1)
    df_w2 = pd.read_csv(args.w2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    _plot_panel(ax1, df_w1, "W1 — Tuning Window")
    _plot_panel(ax2, df_w2, "W2 — Held-Out Evaluation")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
