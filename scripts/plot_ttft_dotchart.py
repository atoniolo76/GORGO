"""Horizontal dot-chart comparing TTFT percentiles across policies.

Each policy is a row; p50, p95, p99 are dots on the same x-axis connected
by a thin line so the reader can immediately scan vertically to compare
policies at any percentile.

Produces side-by-side W1/W2 panels from the analysis CSVs.

Usage:
    python scripts/plot_ttft_dotchart.py \
        --w1 results/analysis/glm5_w1.csv \
        --w2 results/analysis/glm5_w2.csv \
        --out paper/figures/ttft_dotchart.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


GORGO_POLICIES = {"gorgo-hillclimb", "gorgo-static", "gorgo-autotune"}
MARKER_STYLE = {
    "p50": {"marker": "o", "color": "#a8cce8", "zorder": 3, "s": 50},
    "p95": {"marker": "D", "color": "#4a86c7", "zorder": 4, "s": 55},
    "p99": {"marker": "s", "color": "#1b3a5c", "zorder": 5, "s": 55},
}


def _plot_panel(ax: plt.Axes, df: pd.DataFrame, title: str) -> None:
    df = df.sort_values("ttft_p95", ascending=True).reset_index(drop=True)
    y = list(range(len(df)))
    labels = df["policy"].tolist()

    for i, row in df.iterrows():
        is_gorgo = row["policy"] in GORGO_POLICIES
        lw = 1.8 if is_gorgo else 0.8
        alpha = 1.0 if is_gorgo else 0.6
        ax.plot(
            [row["ttft_p50"], row["ttft_p99"]],
            [i, i],
            color="#1b3a5c" if is_gorgo else "#999999",
            linewidth=lw,
            alpha=alpha,
            zorder=1,
        )

    for pct, style in MARKER_STYLE.items():
        col = f"ttft_{pct}"
        for i, row in df.iterrows():
            is_gorgo = row["policy"] in GORGO_POLICIES
            edgecolor = "white" if is_gorgo else "none"
            alpha = 1.0 if is_gorgo else 0.5
            ax.scatter(
                row[col],
                i,
                alpha=alpha,
                edgecolors=edgecolor,
                linewidths=0.8,
                **style,
            )

    for pct, style in MARKER_STYLE.items():
        ax.scatter([], [], label=pct, **style)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    for tick in ax.get_yticklabels():
        if tick.get_text() in GORGO_POLICIES:
            tick.set_fontweight("bold")
            tick.set_color("#1b3a5c")
    ax.set_xlabel("TTFT (s)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    ax.set_xlim(left=0)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--w1", type=Path, required=True)
    parser.add_argument("--w2", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    df_w1 = pd.read_csv(args.w1)
    df_w2 = pd.read_csv(args.w2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    _plot_panel(ax1, df_w1, "W1 — Tuning Window (c=32)")
    _plot_panel(ax2, df_w2, "W2 — Held-Out Evaluation (c=32)")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
