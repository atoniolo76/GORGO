"""Combined TTFT p95 multibar for the 2D-model results (paper Figure).

Visualizes Table tab:results: TTFT p95 across the three decoded load-sweep
windows (Apr 5 full / Apr 6 half / Apr 7 third), grouped by policy, GORGO
highlighted. Numbers are taken verbatim from tab:results so the figure and the
table cannot drift.

Usage:
    python scripts/plot_ttft_multibar.py --out research/figures/ttft_multibar.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

WINDOWS = ["Apr 5\n(full load, saturated)", "Apr 6\n(half load)", "Apr 7\n(third load)"]
POLICIES = ["GORGO", "simple-session-affinity", "least-load", "least-request", "prefix-cache"]

# TTFT p95 (ms) per window, from tab:results (2D weights w_rtt=0.276, w_queue=0.5).
P95_MS = {
    "GORGO":                   [2514, 1584, 1377],
    "simple-session-affinity": [2428, 1875, 1495],
    "least-load":              [2447, 1818, 1637],
    "least-request":           [3970, 1852, 1724],
    "prefix-cache":            [6784, 1798, 1830],
}
COLORS = {
    "GORGO": "#d4a017",
    "simple-session-affinity": "#6a9fd8",
    "least-load": "#b0b0b0",
    "least-request": "#4a6d8c",
    "prefix-cache": "#9bbcd6",
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    n_win, n_pol = len(WINDOWS), len(POLICIES)
    x = np.arange(n_win)
    width = 0.8 / n_pol
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for i, pol in enumerate(POLICIES):
        offs = x + (i - (n_pol - 1) / 2) * width
        vals = [v / 1000.0 for v in P95_MS[pol]]  # ms -> s
        bars = ax.bar(
            offs, vals, width, label=pol, color=COLORS[pol],
            edgecolor="black" if pol == "GORGO" else "white",
            linewidth=1.3 if pol == "GORGO" else 0.4, zorder=3,
        )
        if pol == "GORGO":
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#8a6d10")

    ax.set_xticks(x)
    ax.set_xticklabels(WINDOWS, fontsize=9)
    ax.set_ylabel("TTFT p95 (s)")
    ax.set_title("TTFT p95 across load regimes (2D GORGO vs. baselines)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=7.5, ncol=3, loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(y=0.18)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
