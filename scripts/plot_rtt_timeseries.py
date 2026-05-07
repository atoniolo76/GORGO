"""RTT time series showing proxy-to-replica latency for the three regions.

Produces a clean seaborn-styled figure with one line per region, mean RTT
annotated, and shaded spread. Suitable for a paper figure.

Usage:
    python scripts/plot_rtt_timeseries.py \
        --csv results/analysis/rtt_timeseries_glm5_w1.csv \
        --summary results/analysis/rtt_summary_glm5_w1.csv \
        --out paper/figures/rtt_timeseries.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

REPLICA_ORDER = [
    ("ta-01kqtxa49nxjyzdskbpysv0rsy-", "Seoul (ap-seoul)", "#1b3a5c"),
    ("ta-01kqtxa4e6xe1qrny42znj8255-", "Frankfurt (eu-frankfurt)", "#4a86c7"),
    ("ta-01kqtxa4z59a8vh6nfj5dg9k5r-", "Ashburn (us-east)", "#a8cce8"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    summary = pd.read_csv(args.summary)

    mean_map = dict(zip(summary["replica"], summary["rtt_mean_ms"]))
    std_map = dict(zip(summary["replica"], summary["rtt_std_ms"]))

    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(10, 4))

    EWMA_ALPHA = 0.3

    for replica_id, label, color in REPLICA_ORDER:
        sub = df[df["replica"] == replica_id].sort_values("elapsed_min")
        raw = sub["network_rtt_ms"].values
        ewma = []
        prev = None
        for v in raw:
            if prev is None:
                prev = v
            else:
                prev = EWMA_ALPHA * v + (1 - EWMA_ALPHA) * prev
            ewma.append(prev)
        mean_rtt = mean_map[replica_id]
        std_rtt = std_map[replica_id]

        ax.plot(
            sub["elapsed_min"].values,
            raw,
            color=color,
            linewidth=0.7,
            alpha=0.4,
        )
        ax.plot(
            sub["elapsed_min"].values,
            ewma,
            color=color,
            linewidth=2.2,
            alpha=0.9,
            label=f"{label}  (mean {mean_rtt:.0f} ± {std_rtt:.0f} ms)",
        )

        ax.axhline(
            mean_rtt,
            color=color,
            linewidth=1.0,
            linestyle="--",
            alpha=0.4,
        )

    ax.set_xlabel("Elapsed time (minutes)")
    ax.set_ylabel("RTT (ms)")
    ax.set_title("Proxy → Replica Round-Trip Time (W1 Tuning Window)", fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_xlim(0, df["elapsed_min"].max())
    ax.set_ylim(bottom=0)

    sns.despine()
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
