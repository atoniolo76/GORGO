"""Combined TTFT multibar (proposal for go-xfr).

Consolidates the per-window TTFT bar figures (ttft_bars_w1 / _eval0 / _w2b)
into a single grouped-bar chart: x = evaluation window, grouped bars = policy,
y = TTFT p95 (seconds). GORGO is highlighted. Numbers are taken directly from
the p95-objective result tables (tab:w1, tab:w2, tab:w2b) so the figure and the
tables can never drift.

Usage:
    python scripts/plot_ttft_multibar.py --out research/ttft_multibar.png [--metric p95]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# TTFT (seconds) from the p95-objective tables. GORGO = gorgo-hillclimb in W1
# (tuning) and gorgo-static in W2a/W2b (held-out, frozen W1 weights).
WINDOWS = ["W1\n(tuning, night)", "W2a\n(held-out, night)", "W2b\n(held-out, midday)"]
POLICIES = [
    "GORGO",
    "simple-session-affinity",
    "least-request",
    "prefix-cache",
    "least-load",
    "random",
]
# metric -> per-window dict of policy -> value
DATA = {
    "p50": {
        "GORGO": [0.180, 0.186, 0.195],
        "simple-session-affinity": [0.194, 0.201, 0.234],
        "least-request": [0.197, 0.260, 0.294],
        "prefix-cache": [0.283, 0.190, 0.280],
        "least-load": [0.271, 0.276, 0.297],
        "random": [0.292, 0.292, 0.292],
    },
    "p95": {
        "GORGO": [1.010, 0.896, 1.136],
        "simple-session-affinity": [1.185, 0.981, 1.519],
        "least-request": [1.179, 1.131, 1.297],
        "prefix-cache": [1.305, 1.041, 1.558],
        "least-load": [2.158, 1.144, 2.030],
        "random": [1.192, 1.280, 1.518],
    },
    "p99": {
        "GORGO": [2.660, 1.844, 2.060],
        "simple-session-affinity": [6.757, 1.785, 2.539],
        "least-request": [2.062, 1.758, 2.064],
        "prefix-cache": [2.210, 2.011, 2.491],
        "least-load": [8.750, 1.808, 3.765],
        "random": [1.996, 2.481, 2.457],
    },
}

COLORS = {
    "GORGO": "#d4a017",
    "simple-session-affinity": "#6a9fd8",
    "least-request": "#4a6d8c",
    "prefix-cache": "#9bbcd6",
    "least-load": "#b0b0b0",
    "random": "#d8d8d8",
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--metric", choices=["p50", "p95", "p99"], default="p95")
    args = p.parse_args()

    data = DATA[args.metric]
    n_win = len(WINDOWS)
    n_pol = len(POLICIES)
    x = np.arange(n_win)
    width = 0.8 / n_pol

    fig, ax = plt.subplots(figsize=(9, 4.2))
    for i, pol in enumerate(POLICIES):
        offsets = x + (i - (n_pol - 1) / 2) * width
        vals = data[pol]
        bars = ax.bar(
            offsets,
            vals,
            width,
            label=pol,
            color=COLORS[pol],
            edgecolor="black" if pol == "GORGO" else "white",
            linewidth=1.3 if pol == "GORGO" else 0.4,
            zorder=3,
        )
        if pol == "GORGO":
            for b, v in zip(bars, vals):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    v + 0.02,
                    f"{v:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    fontweight="bold",
                    color="#8a6d10",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(WINDOWS, fontsize=9)
    ax.set_ylabel(f"TTFT {args.metric} (s)")
    ax.set_title(
        f"TTFT {args.metric} across windows (p95-objective GORGO vs. baselines)",
        fontsize=11,
        fontweight="bold",
    )
    ax.legend(fontsize=7.5, ncol=3, loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(y=0.18)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out} (metric={args.metric})")


if __name__ == "__main__":
    main()
