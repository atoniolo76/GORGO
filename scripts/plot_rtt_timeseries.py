"""Generate RTT timeseries figure from proxy metrics.jsonl.

Usage:
    python scripts/plot_rtt_timeseries.py \
        --metrics-jsonl /tmp/gorgo_dl/proxy_traces/tune_gorgo-hillclimb-p95/metrics.jsonl \
        --out figures/rtt_timeseries.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paper_style import REGION_COLORS, classify_region, apply_paper_style


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--metrics-jsonl", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ewma-alpha", type=float, default=0.3)
    args = parser.parse_args()

    metrics = []
    with open(args.metrics_jsonl) as f:
        for line in f:
            if line.strip():
                try:
                    metrics.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    metrics = [m for m in metrics if m.get("ok") and m.get("network_rtt_seconds") is not None]
    if not metrics:
        raise SystemExit("No valid metrics with network_rtt_seconds")

    t0 = min(m["monotonic_s"] for m in metrics)

    replicas: dict[str, list[dict]] = {}
    for m in metrics:
        replicas.setdefault(m["replica_url"], []).append(m)

    region_map = {}
    for url, ms in replicas.items():
        med = np.median([m["network_rtt_seconds"] * 1000 for m in ms])
        region_map[url] = (classify_region(med), med)

    fig, ax = plt.subplots(figsize=(10, 4))

    for url in sorted(replicas, key=lambda u: region_map[u][1], reverse=True):
        region, med = region_map[url]
        color = REGION_COLORS[region]
        ms = sorted(replicas[url], key=lambda m: m["monotonic_s"])
        times = [(m["monotonic_s"] - t0) / 60.0 for m in ms]
        rtts = [m["network_rtt_seconds"] * 1000 for m in ms]
        std = np.std(rtts)

        ax.plot(times, rtts, "-", linewidth=0.5, alpha=0.3, color=color)
        ewma = [rtts[0]]
        for v in rtts[1:]:
            ewma.append(args.ewma_alpha * v + (1 - args.ewma_alpha) * ewma[-1])
        ax.plot(
            times,
            ewma,
            linewidth=2.5,
            color=color,
            label=f"{region} (ap-{region.lower()})  (mean {med:.0f} \u00b1 {std:.0f} ms)",
        )
        ax.axhline(med, color=color, linestyle="--", linewidth=0.8, alpha=0.4)

    ax.set_xlabel("Elapsed time (minutes)", fontsize=10)
    ax.set_ylabel("RTT (ms)", fontsize=10)
    ax.set_title(
        "Proxy \u2192 Replica Round-Trip Time (W1 Tuning Window)", fontsize=11, fontweight="bold"
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, None)
    apply_paper_style(ax)

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
