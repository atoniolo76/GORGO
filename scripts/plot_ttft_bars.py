"""Horizontal bar chart: one panel per percentile, side-by-side W1/W2.

Six panels total (3 percentiles x 2 windows). Policies are in the same
vertical order across all panels (sorted by W1 p95) so the reader can
scan left-to-right and top-to-bottom.

Usage:
    python scripts/plot_ttft_bars.py \
        --w1 results/analysis/glm5_w1.csv \
        --w2 results/analysis/glm5_w2.csv \
        --out paper/figures/ttft_bars.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GORGO_POLICIES = {"gorgo-hillclimb", "gorgo-static", "gorgo-autotune"}

POLICY_COLORS = {
    "gorgo-hillclimb": "#1b3a5c",
    "gorgo-static": "#2d6a9f",
    "gorgo-autotune": "#4a86c7",
}
BASELINE_COLOR = "#b0c4d8"


def _color(policy: str) -> str:
    return POLICY_COLORS.get(policy, BASELINE_COLOR)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--w1", type=Path, required=True)
    parser.add_argument("--w2", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    df_w1 = pd.read_csv(args.w1)
    df_w2 = pd.read_csv(args.w2)

    order = df_w1.sort_values("ttft_p95", ascending=False)["policy"].tolist()
    df_w1 = df_w1.set_index("policy").loc[order].reset_index()
    df_w2 = df_w2.set_index("policy").loc[order].reset_index()

    pcts = ["p50", "p95", "p99"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharey=True)

    for col_idx, (window_label, df) in enumerate([("W2", df_w2)]):
        for pct_idx, pct in enumerate(pcts):
            ax = axes[pct_idx]
            col = f"ttft_{pct}"
            y = np.arange(len(df))
            vals = df[col].values
            colors = [_color(p) for p in df["policy"]]

            bars = ax.barh(y, vals, color=colors, edgecolor="white", linewidth=0.5)

            best_idx = np.argmin(vals)
            bars[best_idx].set_edgecolor("#e8c840")
            bars[best_idx].set_linewidth(2.0)

            for i, v in enumerate(vals):
                ax.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=7, color="#333333")

            ax.set_xlabel("TTFT (s)", fontsize=8)
            ax.set_title(f"{window_label} — {pct}", fontsize=10, fontweight="bold")
            ax.grid(axis="x", alpha=0.2)
            ax.set_xlim(0, max(vals) * 1.25)

            if col_idx == 0 and pct_idx == 0:
                ax.set_yticks(y)
                labels = df["policy"].tolist()
                ax.set_yticklabels(labels, fontsize=9)
                for tick in ax.get_yticklabels():
                    if tick.get_text() in GORGO_POLICIES:
                        tick.set_fontweight("bold")
                        tick.set_color("#1b3a5c")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
