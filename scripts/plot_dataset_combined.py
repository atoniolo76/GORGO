"""Regenerate Figure 1 (``dataset_combined.png``).

Left panel: prefix-reuse token composition (intra/cross/unique) per dataset.
Right panel: radar chart of the six routing-relevant axes, normalized to the
per-axis maximum across datasets.

All metrics are request-row, block-level (256-token) reuse, computed on the
same basis for every dataset: ART-Chat-2.5M via
``data_processing/week_reuse_stats.py`` and the public sets via
``data_processing/block_reuse_public.py``. The production trace is labelled
``ART-Chat-2.5M``.

Usage:
    python scripts/plot_dataset_combined.py --out figures/dataset_combined.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

DATASETS = {
    "ART-Chat-2.5M": "results/decoded_v9/week_reuse_stats.json",
    "LMSYS-Chat-1M": "results/decoded_v9/block_reuse_lmsys.json",
    "WildChat-4.8M": "results/decoded_v9/block_reuse_wildchat.json",
}

# Blue palette: production trace darkest, public sets lighter.
COLORS = {
    "ART-Chat-2.5M": "#1f3b57",
    "LMSYS-Chat-1M": "#4f8fc0",
    "WildChat-4.8M": "#a9d0ec",
}
UNIQUE_COLOR = "#dce6f0"


def _load_stats() -> dict[str, dict]:
    out = {}
    for name, path in DATASETS.items():
        s = json.loads(Path(path).read_text())
        out[name] = {
            "avg_tokens": s["avg_input_tokens"],
            "requests_per_user": s["requests_per_user"],
            "intra_user_reuse": s["block_intra_user_reuse_pct"],
            "cross_user_reuse": s["block_cross_user_reuse_pct"],
            "global_reuse": s["block_global_reuse_pct"],
            "top10_concentration": s["top10_user_concentration_pct"],
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("figures/dataset_combined.png"))
    args = parser.parse_args()

    stats = _load_stats()
    names = list(stats.keys())

    fig = plt.figure(figsize=(15, 6))
    ax_bar = fig.add_subplot(1, 2, 1)
    ax_radar = fig.add_subplot(1, 2, 2, polar=True)

    # ---- Left: token composition stacked bars ----
    x = np.arange(len(names))
    for i, n in enumerate(names):
        s = stats[n]
        intra = s["intra_user_reuse"]
        cross = s["cross_user_reuse"]
        unique = 100 - s["global_reuse"]
        c = COLORS[n]
        ax_bar.bar(i, intra, color=c)
        ax_bar.bar(i, cross, bottom=intra, color=c, alpha=0.55, hatch="//")
        ax_bar.bar(i, unique, bottom=intra + cross, color=UNIQUE_COLOR)
        if intra > 2:
            ax_bar.text(
                i,
                intra / 2,
                f"{intra:.0f}%",
                ha="center",
                va="center",
                color="white",
                fontsize=9,
                fontweight="bold",
            )
        if cross > 2:
            ax_bar.text(
                i,
                intra + cross / 2,
                f"{cross:.0f}%",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
            )
        ax_bar.text(
            i,
            intra + cross + unique / 2,
            f"{unique:.0f}%",
            ha="center",
            va="center",
            color="#555",
            fontsize=9,
        )

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(names, fontsize=9)
    ax_bar.set_ylabel("Token composition (%)")
    ax_bar.set_ylim(0, 100)
    ax_bar.set_title("Prefix reuse breakdown", fontsize=12, fontweight="bold", pad=30)
    legend_handles = [
        Patch(facecolor="#1f3b57", label="Intra-user reuse"),
        Patch(facecolor="#1f3b57", alpha=0.55, hatch="//", label="Cross-user reuse"),
        Patch(facecolor=UNIQUE_COLOR, label="Unique (no reuse)"),
    ]
    ax_bar.legend(handles=legend_handles, fontsize=8, loc="upper right")
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    # ---- Right: radar, normalized per-axis to the max across datasets ----
    axes_labels = [
        "Avg tokens\nper request",
        "Requests\nper user",
        "Intra-user\nreuse",
        "Cross-user\nreuse",
        "Global\nreuse",
        "Top-10 user\nconcentration",
    ]
    keys = [
        "avg_tokens",
        "requests_per_user",
        "intra_user_reuse",
        "cross_user_reuse",
        "global_reuse",
        "top10_concentration",
    ]
    maxes = {k: max(stats[n][k] for n in names) or 1.0 for k in keys}
    angles = np.linspace(0, 2 * np.pi, len(keys), endpoint=False).tolist()
    angles += angles[:1]

    for n in names:
        vals = [stats[n][k] / maxes[k] for k in keys]
        vals += vals[:1]
        c = COLORS[n]
        ax_radar.plot(angles, vals, color=c, linewidth=2.2, label=n)
        ax_radar.fill(angles, vals, color=c, alpha=0.12)

    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(axes_labels, fontsize=9)
    # Radial headroom: data is normalized to max=1.0, but we extend the axis to
    # 1.35 so the data polygon sits well inside the outer circle, leaving a clear
    # ring of space between the vertices and the axis labels.
    ax_radar.set_ylim(0, 1.13)
    ax_radar.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax_radar.set_yticklabels([])
    ax_radar.tick_params(axis="x", pad=6)
    # Remove the heavy default outer ring; keep only light reference circles.
    ax_radar.spines["polar"].set_visible(False)
    ax_radar.grid(color="#d9d9d9", linewidth=0.8, alpha=0.9)
    ax_radar.set_title("Dataset characterization", fontsize=12, fontweight="bold", pad=30)
    ax_radar.legend(loc="lower right", bbox_to_anchor=(1.32, -0.10), fontsize=8, frameon=True)

    fig.tight_layout(w_pad=5)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {args.out}")
    for n in names:
        print(n, {k: round(stats[n][k], 2) for k in keys})


if __name__ == "__main__":
    main()
