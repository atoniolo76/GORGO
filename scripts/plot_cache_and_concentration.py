"""Two-panel figure: cache hit rate vs p95 TTFT scatter + routing concentration.

Left: scatter showing cache-TTFT tradeoff (bottom-right = best).
Right: bar showing % of requests sent to the most-used replica per policy.
       Higher = more concentrated / less balanced. Uniform across 3 replicas = 33%.

Usage:
    python scripts/plot_cache_and_concentration.py \
        --w2 results/analysis/glm5_w2.csv \
        --out paper/figures/cache_and_concentration.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

GORGO_POLICIES = {"gorgo-hillclimb", "gorgo-static", "gorgo-autotune"}

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

CONCENTRATION_W2 = {
    "simple-session-affinity": 59,
    "gorgo-static": 52,
    "prefix-cache": 48,
    "gorgo-hillclimb": 49,
    "gorgo-autotune": 42,
    "least-load": 38,
    "random": 35,
    "least-request": 36,
}

POLICY_COLORS = {
    "gorgo-hillclimb": "#1b3a5c",
    "gorgo-static": "#2d6a9f",
    "gorgo-autotune": "#4a86c7",
}
BASELINE_COLOR = "#a0b8cc"


def _color(policy: str) -> str:
    return POLICY_COLORS.get(policy, BASELINE_COLOR)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--w2", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.w2)
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.3, 1]})

    # --- Left: cache vs TTFT scatter ---
    for _, row in df.iterrows():
        policy = row["policy"]
        if policy not in CACHE_HIT_W2:
            continue
        cache = CACHE_HIT_W2[policy]
        ttft = row["ttft_p95"]
        is_gorgo = policy in GORGO_POLICIES
        color = _color(policy)
        marker = "D" if is_gorgo else "o"
        size = 120 if is_gorgo else 80

        ax1.scatter(
            cache,
            ttft,
            color=color,
            marker=marker,
            s=size,
            edgecolors="white" if is_gorgo else "none",
            linewidths=1.0,
            zorder=10 if is_gorgo else 5,
        )

        ha = "left"
        ox, oy = 1.2, 0.03
        if policy == "gorgo-static":
            oy = -0.8
        elif policy == "simple-session-affinity":
            ox, ha = -1.0, "right"
        elif policy == "random":
            oy = 0.5
        elif policy == "least-request":
            oy = -0.7
        elif policy == "prefix-cache":
            oy = 0.5

        ax1.annotate(
            policy,
            xy=(cache, ttft),
            xytext=(ox, oy),
            textcoords="offset fontsize",
            fontsize=9 if is_gorgo else 8.5,
            fontweight="bold" if is_gorgo else "normal",
            color=color if is_gorgo else "#444444",
            va="center",
            ha=ha,
        )

    ax1.set_xlabel("KV-cache hit rate (%)")
    ax1.set_ylabel("p95 TTFT (s)")
    ax1.set_title("Cache utilization vs latency", fontsize=11, fontweight="bold")
    ax1.grid(alpha=0.2)

    # --- Right: routing concentration bars ---
    # Match the TTFT bar chart ordering: sorted by W1 p95 descending (worst at top)
    w1_p95_order = [
        "random",
        "least-load",
        "gorgo-autotune",
        "prefix-cache",
        "least-request",
        "gorgo-static",
        "simple-session-affinity",
        "gorgo-hillclimb",
    ]
    order = [p for p in w1_p95_order if p in CONCENTRATION_W2]
    y = np.arange(len(order))
    vals = [CONCENTRATION_W2[p] for p in order]
    colors = [_color(p) for p in order]

    ax2.barh(y, vals, color=colors, edgecolor="white", linewidth=0.5)
    ax2.axvline(
        100 / 3, color="#888888", linestyle="--", linewidth=1.0, alpha=0.6, label="Uniform (33%)"
    )

    for i, (p, v) in enumerate(zip(order, vals)):
        ax2.text(v + 0.8, i, f"{v}%", va="center", fontsize=8, color="#333")

    ax2.set_yticks(y)
    ax2.set_yticklabels(order, fontsize=9)
    for tick in ax2.get_yticklabels():
        if tick.get_text() in GORGO_POLICIES:
            tick.set_fontweight("bold")
            tick.set_color("#1b3a5c")
    ax2.set_xlabel("Requests on most-used replica (%)")
    ax2.set_title("Routing concentration", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8, loc="lower right")
    ax2.grid(axis="x", alpha=0.2)
    ax2.grid(axis="y", visible=False)
    ax2.set_xlim(0, max(vals) * 1.25)

    sns.despine()
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
