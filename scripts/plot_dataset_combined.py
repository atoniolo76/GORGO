"""Combined figure (paper Figure 1): vertical stacked reuse bars + radar chart.

Left: Vertical stacked bars showing intra-user, cross-user, and unique
      token composition per dataset.
Right: Radar chart with 6 axes comparing dataset characteristics.

The production dataset is labelled ``ArtChat-411K`` (411,169 requests). Stats
JSON inputs live under ``research/data/dataset_stats/`` so this figure is
reproducible from the rome branch without the proprietary trace.

Usage:
    python scripts/plot_dataset_combined.py --out research/dataset_combined.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
STATS_DIR = REPO_ROOT / "research" / "data" / "dataset_stats"

DATASETS_CFG = [
    ("ArtChat-411K", STATS_DIR / "artchat-411k.json"),
    ("LMSYS-Chat-1M", STATS_DIR / "lmsys-chat-1m.json"),
    ("WildChat-4.8M", STATS_DIR / "wildchat.json"),
]

COLORS = {
    "ArtChat-411K": "#1b3a5c",
    "LMSYS-Chat-1M": "#6a9fd8",
    "WildChat-4.8M": "#a8cce8",
}

RADAR_AXES = [
    "Intra-user\nreuse",
    "Requests\nper user",
    "Avg tokens\nper request",
    "Top-10 user\nconcentration",
    "Global\nreuse",
    "Cross-user\nreuse",
]


def _load() -> list[tuple[str, dict]]:
    out = []
    for name, path in DATASETS_CFG:
        s = json.loads(Path(path).read_text())
        total_seqs = s["total_sequences"]
        total_tokens = s["total_tokens"]
        users = s["user_count"]
        top_key = "top_users" if "top_users" in s else "top_groups"
        top10 = sum(e["tokens"] for e in s.get(top_key, [])[:10])
        out.append(
            (
                name,
                {
                    "intra": s["intra_user_savings_pct"],
                    "cross": s["cross_user_extra_pct"],
                    "global_reuse": s["global_savings_pct"],
                    "avg_tokens": total_tokens / total_seqs,
                    "requests_per_user": total_seqs / users,
                    "top10_pct": 100.0 * top10 / total_tokens,
                },
            )
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    data = _load()
    names = [d[0] for d in data]

    fig = plt.figure(figsize=(11, 5))
    ax_bar = fig.add_axes([0.05, 0.12, 0.38, 0.82])
    ax_radar = fig.add_axes([0.52, 0.05, 0.48, 0.9], polar=True)

    # --- Left: vertical stacked bars ---
    x = np.arange(len(names))
    intra = [d[1]["intra"] for d in data]
    cross = [d[1]["cross"] for d in data]
    unique = [100 - d[1]["intra"] - d[1]["cross"] for d in data]

    bar_colors_intra = [COLORS[n] for n in names]
    bar_colors_cross = ["#4a6d8c", "#89b4d8", "#c0daea"]
    bar_colors_unique = ["#d5e4f0", "#e8f0f8", "#f0f5fa"]

    b1 = ax_bar.bar(
        x, intra, color=bar_colors_intra, edgecolor="white", linewidth=0.5, label="Intra-user reuse"
    )
    b2 = ax_bar.bar(
        x,
        cross,
        bottom=intra,
        color=bar_colors_cross,
        edgecolor="white",
        linewidth=0.5,
        hatch="//",
        label="Cross-user reuse",
    )
    b3 = ax_bar.bar(
        x,
        unique,
        bottom=[i + c for i, c in zip(intra, cross)],
        color=bar_colors_unique,
        edgecolor="white",
        linewidth=0.5,
        label="Unique (no reuse)",
    )

    for i, (iv, cv, uv) in enumerate(zip(intra, cross, unique)):
        total_reuse = iv + cv
        # Intra-user
        if iv > 8:
            ax_bar.text(
                i,
                iv / 2,
                f"{iv:.0f}%",
                ha="center",
                va="center",
                fontsize=9,
                color="white",
                fontweight="bold",
            )
        elif iv > 0.5:
            ax_bar.text(
                i,
                iv / 2,
                f"{iv:.0f}%",
                ha="center",
                va="center",
                fontsize=7,
                color="white",
                fontweight="bold",
            )
        # Cross-user — inside the segment, bolded
        if cv > 5:
            ax_bar.text(
                i,
                iv + cv / 2,
                f"{cv:.0f}%",
                ha="center",
                va="center",
                fontsize=8,
                color="#1b3a5c",
                fontweight="bold",
            )
        elif cv > 0.5:
            ax_bar.text(
                i,
                iv + cv + 1.5,
                f"{cv:.0f}%",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#1b3a5c",
                fontweight="bold",
            )
        # Unique
        ax_bar.text(
            i,
            total_reuse + uv / 2,
            f"{uv:.0f}%",
            ha="center",
            va="center",
            fontsize=8,
            color="#666",
        )

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(names, fontsize=9)
    ax_bar.set_ylabel("Token composition (%)")
    ax_bar.set_title("Prefix reuse breakdown", fontsize=11, fontweight="bold")
    ax_bar.set_ylim(0, 100)
    ax_bar.legend(fontsize=7.5, loc="upper right")
    ax_bar.grid(visible=False)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    # --- Right: radar chart ---
    radar_keys = ["intra", "requests_per_user", "avg_tokens", "top10_pct", "global_reuse", "cross"]
    raw = {n: [d[1][k] for k in radar_keys] for n, d in zip(names, data)}
    all_vals = list(raw.values())
    num_dims = len(RADAR_AXES)
    maxes = [max(v[i] for v in all_vals) for i in range(num_dims)]
    normed = {}
    for n, v in raw.items():
        normed[n] = [v[i] / maxes[i] if maxes[i] > 0 else 0 for i in range(num_dims)]

    angles = np.linspace(0, 2 * np.pi, num_dims, endpoint=False).tolist()
    angles += angles[:1]

    for name in names:
        vals = normed[name] + normed[name][:1]
        color = COLORS[name]
        ax_radar.fill(angles, vals, alpha=0.15, color=color)
        ax_radar.plot(angles, vals, linewidth=2.2, color=color, label=name)
        for angle, nv in zip(angles[:-1], normed[name]):
            ax_radar.scatter(
                angle, nv, s=25, color=color, zorder=5, edgecolors="white", linewidths=0.5
            )

    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(RADAR_AXES, fontsize=8.5)
    ax_radar.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax_radar.set_yticklabels([])
    ax_radar.set_ylim(0, 1.15)
    ax_radar.spines["polar"].set_visible(False)
    ax_radar.grid(alpha=0.25)
    ax_radar.set_title("Dataset characterization", fontsize=11, fontweight="bold", pad=20)
    ax_radar.legend(loc="lower right", bbox_to_anchor=(1.25, -0.05), fontsize=8, framealpha=0.9)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
