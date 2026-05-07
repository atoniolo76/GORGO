"""Regenerate paper/figures/dataset_comparison.png with corrected LMSYS user count.

LMSYS-Chat-1M's HF dataset card reports 210,479 unique IPs across 1M
conversations. Our local prefix-trie analysis used conversation-id
partitioning (since the on-disk schema lacks the IP column) and produced
user_count=1,000,000 as an artifact. This script regenerates the
3-panel comparison figure with the HF-canonical LMSYS user count.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Inlined stats: numbers from the prefix-trie analysis on origin/main
# (data_processing/prefix_trie_results/<dataset>/stats.json), with the
# LMSYS user_count overridden per the HF dataset card.
STATS = {
    "GLM-5.1": {
        "total_sequences": 411_169,
        "users": 4_984,
        "avg_tokens": 8_652_547_293 / 411_169,
        "intra_user_reuse": 53.67,
        "cross_user_reuse": 1.63,
        "global_reuse": 55.30,
    },
    "LMSYS-Chat-1M": {
        "total_sequences": 1_000_000,
        "users": 210_479,  # HF dataset card; overrides on-disk user_count=1M (conv-id artifact)
        "avg_tokens": 466_833_462 / 1_000_000,
        "intra_user_reuse": 0.0,  # not measurable on the on-disk schema (no IP column)
        "cross_user_reuse": 8.95,
        "global_reuse": 8.95,
    },
    "WildChat-4.8M": {
        "total_sequences": 3_199_860,
        "users": 1_833_730,
        "avg_tokens": 9_361_080_375 / 3_199_860,
        "intra_user_reuse": 5.30,
        "cross_user_reuse": 29.06,
        "global_reuse": 34.35,
    },
}

COLORS = {
    "GLM-5.1": "#1f3a5f",  # navy
    "LMSYS-Chat-1M": "#3d8a3d",  # green
    "WildChat-4.8M": "#5a3a7a",  # purple
}

NAMES = ["GLM-5.1", "LMSYS-Chat-1M", "WildChat-4.8M"]


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "paper" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(NAMES))
    bar_colors = [COLORS[n] for n in NAMES]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: avg tokens / request (log)
    vals = [STATS[n]["avg_tokens"] for n in NAMES]
    bars = axes[0].bar(x, vals, color=bar_colors, edgecolor="black", linewidth=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(NAMES, fontsize=10)
    axes[0].set_ylabel("Avg tokens / request")
    axes[0].set_title("Prompt Length")
    axes[0].set_yscale("log")
    axes[0].grid(axis="y", alpha=0.3)
    for b, v in zip(bars, vals):
        axes[0].text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{v:,.0f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Panel 2: requests / user (log)
    vals = [STATS[n]["total_sequences"] / STATS[n]["users"] for n in NAMES]
    bars = axes[1].bar(x, vals, color=bar_colors, edgecolor="black", linewidth=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(NAMES, fontsize=10)
    axes[1].set_ylabel("Requests / user")
    axes[1].set_title("Multi-turn Density")
    axes[1].set_yscale("log")
    axes[1].grid(axis="y", alpha=0.3)
    for b, v in zip(bars, vals):
        label = f"{v:.1f}" if v < 10 else f"{v:.0f}"
        axes[1].text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            label,
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Panel 3: intra-user + cross-user reuse stacked
    intra = [STATS[n]["intra_user_reuse"] for n in NAMES]
    cross = [STATS[n]["cross_user_reuse"] for n in NAMES]
    bars1 = axes[2].bar(
        x, intra, color=bar_colors, edgecolor="black", linewidth=0.5, alpha=0.95, label="Intra-user"
    )
    bars2 = axes[2].bar(
        x,
        cross,
        bottom=intra,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.45,
        hatch="//",
        label="Cross-user",
    )
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(NAMES, fontsize=10)
    axes[2].set_ylabel("KV-cache reuse (%)")
    axes[2].set_title("Prefix Reuse Breakdown")
    axes[2].legend(fontsize=9, loc="upper right")
    axes[2].set_ylim(0, 70)
    axes[2].grid(axis="y", alpha=0.3)
    for i, (iv, cv) in enumerate(zip(intra, cross)):
        total = iv + cv
        axes[2].text(
            i, total + 1, f"{total:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold"
        )
        if iv > 2:
            axes[2].text(
                i, iv / 2, f"{iv:.0f}%", ha="center", va="center", fontsize=9, color="white"
            )
        if cv > 5:
            axes[2].text(i, iv + cv / 2, f"{cv:.0f}%", ha="center", va="center", fontsize=9)

    fig.suptitle("Dataset Characteristics: GLM-5.1 vs Public Chat Datasets", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out_path = out_dir / "dataset_comparison.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    print()
    print("Reqs/user values used:")
    for n in NAMES:
        print(f"  {n:18s}: {STATS[n]['total_sequences'] / STATS[n]['users']:6.1f}")


if __name__ == "__main__":
    main()
