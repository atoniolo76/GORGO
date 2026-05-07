"""Geographic diagram showing proxy-to-replica RTT on a world map outline.

Produces a clean schematic with three replica locations and RTT annotations
from the US East proxy, using a mid-window snapshot from the W1 timeseries.

Usage:
    python scripts/plot_geo_diagram.py --out paper/figures/geo_rtt.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


PROXY = {"label": "Proxy (US East)", "xy": (-95, 48)}

REPLICAS = [
    {
        "label": "US East\n(Ashburn)",
        "xy": (-75, 35),
        "rtt_ms": 25,
        "color": "#1b3a5c",
        "label_x": -87,
        "label_y": 37,
    },
    {
        "label": "Europe\n(Frankfurt)",
        "xy": (12, 48),
        "rtt_ms": 147,
        "color": "#2d6a9f",
        "label_x": -32,
        "label_y": 58,
    },
    {
        "label": "Asia\n(Seoul)",
        "xy": (127, 35),
        "rtt_ms": 437,
        "color": "#4a86c7",
        "label_x": 75,
        "label_y": 60,
    },
]

CONTINENTS_OUTLINE = [
    [(-130, 22), (-55, 22), (-55, 58), (-130, 58)],
    [(-12, 33), (42, 33), (42, 62), (-12, 62)],
    [(98, 18), (148, 18), (148, 52), (98, 52)],
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(12, 4.5))

    for outline in CONTINENTS_OUTLINE:
        xs = [p[0] for p in outline] + [outline[0][0]]
        ys = [p[1] for p in outline] + [outline[0][1]]
        ax.fill(xs, ys, color="#e8eff5", edgecolor="#c0d0e0", linewidth=1.0, zorder=1)

    px, py = PROXY["xy"]

    for rep in REPLICAS:
        rx, ry = rep["xy"]
        rtt = rep["rtt_ms"]
        color = rep["color"]

        ax.annotate(
            "",
            xy=(rx, ry),
            xytext=(px, py),
            arrowprops=dict(
                arrowstyle="-",
                color=color,
                linewidth=2.0 + rtt / 200,
                alpha=0.7,
                connectionstyle="arc3,rad=0.15",
            ),
            zorder=2,
        )

        bbox = dict(
            boxstyle="round,pad=0.3", facecolor="white", edgecolor=color, alpha=0.95, linewidth=1.5
        )
        ax.text(
            rep.get("label_x", (px + rx) / 2),
            rep.get("label_y", (py + ry) / 2 + 6),
            f"{rtt} ms",
            fontsize=11,
            fontweight="bold",
            color=color,
            ha="center",
            va="center",
            bbox=bbox,
            zorder=5,
        )

        ax.scatter(
            rx, ry, s=180, color=color, edgecolors="white", linewidths=1.5, zorder=4, marker="s"
        )
        ax.text(
            rx,
            ry - 5,
            rep["label"],
            fontsize=9,
            fontweight="bold",
            color=color,
            ha="center",
            va="top",
            zorder=5,
        )

    ax.scatter(
        px, py, s=250, color="#c0392b", edgecolors="white", linewidths=2.0, zorder=4, marker="*"
    )
    bbox_proxy = dict(
        boxstyle="round,pad=0.3",
        facecolor="#fdecea",
        edgecolor="#c0392b",
        alpha=0.95,
        linewidth=1.5,
    )
    ax.text(
        px,
        py + 4,
        PROXY["label"],
        fontsize=9,
        fontweight="bold",
        color="#c0392b",
        ha="center",
        va="bottom",
        bbox=bbox_proxy,
        zorder=5,
    )

    gpu_label = "2×L40S per replica\nQwen3.5-35B-A3B-FP8"
    ax.text(
        135,
        55,
        gpu_label,
        fontsize=8,
        color="#555555",
        ha="right",
        va="top",
        style="italic",
        zorder=5,
    )

    ax.set_xlim(-145, 160)
    ax.set_ylim(8, 68)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
