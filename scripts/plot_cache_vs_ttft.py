"""Scatter plot: cache hit rate vs p95 TTFT per policy.

Shows the fundamental tradeoff — high cache utilization doesn't guarantee
low TTFT when load and network are ignored. GORGO variants achieve both.

Cache hit rates from the W1 analysis run (abstract_night_w2_000 plots).
TTFT from the W1/W2 analysis CSVs.

Usage:
    python scripts/plot_cache_vs_ttft.py \
        --w1 results/analysis/glm5_w1.csv \
        --w2 results/analysis/glm5_w2.csv \
        --out paper/figures/cache_vs_ttft.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

GORGO_POLICIES = {"gorgo-hillclimb", "gorgo-static", "gorgo-autotune"}

CACHE_HIT_W1 = {
    "gorgo-hillclimb": 67.0,
    "gorgo-static": 67.7,
    "prefix-cache": 67.7,
    "gorgo-autotune": 49.5,
    "least-load": 47.6,
    "least-request": 47.5,
    "random": 45.8,
}

CACHE_HIT_W2 = {
    "simple-session-affinity": 68.7,
    "gorgo-static": 68.6,
    "prefix-cache": 67.0,
    "gorgo-hillclimb": 65.2,
    "gorgo-autotune": 50.6,
    "least-load": 48.1,
    "random": 47.1,
    "least-request": 45.9,
}

POLICY_COLORS = {
    "gorgo-hillclimb": "#1b3a5c",
    "gorgo-static": "#2d6a9f",
    "gorgo-autotune": "#4a86c7",
}
BASELINE_COLOR = "#a0b8cc"


def _plot_panel(ax: plt.Axes, df: pd.DataFrame, cache_map: dict, title: str) -> None:
    for _, row in df.iterrows():
        policy = row["policy"]
        if policy not in cache_map:
            continue
        cache = cache_map[policy]
        ttft = row["ttft_p95"]
        is_gorgo = policy in GORGO_POLICIES
        color = POLICY_COLORS.get(policy, BASELINE_COLOR)
        marker = "D" if is_gorgo else "o"
        size = 120 if is_gorgo else 80
        edge = "white" if is_gorgo else "none"
        zorder = 10 if is_gorgo else 5

        ax.scatter(
            cache,
            ttft,
            color=color,
            marker=marker,
            s=size,
            edgecolors=edge,
            linewidths=1.0,
            zorder=zorder,
        )

        offset_x = 1.2
        offset_y = 0.03
        ha = "left"
        if policy == "gorgo-static":
            offset_y = -0.8
        if policy == "prefix-cache":
            offset_y = 0.5
        if policy == "simple-session-affinity":
            offset_x = -1.0
            ha = "right"
        if policy == "random":
            offset_y = 0.5
        if policy == "least-request":
            offset_y = -0.7

        ax.annotate(
            policy,
            xy=(cache, ttft),
            xytext=(offset_x, offset_y),
            textcoords="offset fontsize",
            fontsize=8,
            fontweight="bold" if is_gorgo else "normal",
            color=color if is_gorgo else "#555555",
            va="center",
            ha=ha,
        )

    ax.set_xlabel("KV-cache hit rate (%)")
    ax.set_ylabel("p95 TTFT (s)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(alpha=0.2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--w1", type=Path, required=True)
    parser.add_argument("--w2", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    df_w1 = pd.read_csv(args.w1)
    df_w2 = pd.read_csv(args.w2)

    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    _plot_panel(ax, df_w2, CACHE_HIT_W2, "Cache Hit Rate vs p95 TTFT (W2)")

    sns.despine()
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
