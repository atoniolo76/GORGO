"""Generate TTFT horizontal bar charts for paper figures.

Usage:
    python scripts/plot_ttft_bars.py \
        --workload-dir /tmp/gorgo_dl/workload_runs \
        --prefix tune_ --label W1 --out figures/ttft_bars_w1.png

    python scripts/plot_ttft_bars.py \
        --workload-dir /tmp/gorgo_dl/workload_runs \
        --prefix eval_ --label W2 --out figures/ttft_bars.png
"""

from __future__ import annotations

import argparse
import json
import glob
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from paper_style import (
    get_color,
    is_gorgo,
    display_name,
    apply_paper_style,
    WINNER_OUTLINE,
    WINNER_LW,
)


def _load_stats(workload_dir: str, prefix: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(workload_dir, f"{prefix}*.json"))):
        with open(path) as f:
            data = json.load(f)
        pol = os.path.basename(path).replace(prefix, "").replace(".json", "")
        s = data["stats"]
        ttft = s["ttft_seconds"]
        rows.append(
            {
                "label": pol,
                "display": display_name(pol),
                "p50": ttft["p50"],
                "p95": ttft["p95"],
                "p99": ttft["p99"],
                "is_gorgo": is_gorgo(pol),
            }
        )
    rows.sort(key=lambda r: r["p95"])
    return rows


def plot(policies: list[dict], window_label: str, out_path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharey=True)
    labels = [p["display"] for p in policies]
    y = np.arange(len(policies))

    for ax, metric, pct in zip(axes, ["p50", "p95", "p99"], ["p50", "p95", "p99"]):
        vals = [p[metric] for p in policies]
        colors = [get_color(p["label"]) for p in policies]

        bars = ax.barh(y, vals, color=colors, height=0.7, edgecolor="white", linewidth=0.5)

        best_idx = int(np.argmin(vals))
        bar = bars[best_idx]
        rect = mpatches.FancyBboxPatch(
            (bar.get_x(), bar.get_y()),
            bar.get_width(),
            bar.get_height(),
            boxstyle="square,pad=0",
            linewidth=WINNER_LW,
            edgecolor=WINNER_OUTLINE,
            facecolor="none",
            zorder=10,
        )
        ax.add_patch(rect)

        for i, v in enumerate(vals):
            ax.text(v + max(vals) * 0.02, y[i], f"{v:.2f}", va="center", fontsize=8, color="#333")

        ax.set_yticks(y)
        tick_labels = ax.set_yticklabels(labels, fontsize=9)
        for tl, p in zip(tick_labels, policies):
            if p["is_gorgo"]:
                tl.set_fontweight("bold")
                tl.set_color("#1b3a5c")

        ax.set_xlabel("TTFT (s)", fontsize=9)
        ax.set_title(f"{window_label} — {pct}", fontsize=11, fontweight="bold")
        ax.set_xlim(0, max(vals) * 1.25)
        ax.invert_yaxis()
        apply_paper_style(ax)
        ax.grid(axis="x", alpha=0.2, linewidth=0.5)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--workload-dir", required=True)
    parser.add_argument("--prefix", required=True, help="File prefix, e.g. 'tune_' or 'eval_'")
    parser.add_argument("--label", required=True, help="Window label, e.g. 'W1' or 'W2'")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    policies = _load_stats(args.workload_dir, args.prefix)
    if not policies:
        raise SystemExit(f"No workload JSONs matching {args.workload_dir}/{args.prefix}*.json")
    plot(policies, args.label, args.out)


if __name__ == "__main__":
    main()
