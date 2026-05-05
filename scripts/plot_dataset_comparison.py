"""Generate paper-grade figures comparing dataset characteristics.

Produces:
  1. Dataset comparison bar chart (tokens/request, reuse %, requests/user)
  2. Intra vs cross-user reuse breakdown (stacked bars)
  3. Multi-panel summary with prompt length distributions from traces

Usage:
    python scripts/plot_dataset_comparison.py --out-dir results/analysis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATASETS = {
    "GLM-5.1": "data_processing/prefix_trie_results/glm-5.1-completions/stats.json",
    "LMSYS-Chat-1M": "data_processing/prefix_trie_results/lmsys-chat-1m/stats.json",
    "WildChat-4.8M": "data_processing/prefix_trie_results/wildchat/stats.json",
}

COLORS = {
    "GLM-5.1": "#d62728",
    "LMSYS-Chat-1M": "#1f77b4",
    "WildChat-4.8M": "#2ca02c",
}


def _load_stats() -> dict[str, dict]:
    out = {}
    for name, path in DATASETS.items():
        s = json.loads(Path(path).read_text())
        total_seqs = s["total_sequences"]
        total_tokens = s["total_tokens"]
        users = s["user_count"]
        out[name] = {
            "total_sequences": total_seqs,
            "total_tokens": total_tokens,
            "users": users,
            "avg_tokens": total_tokens / total_seqs,
            "requests_per_user": total_seqs / users,
            "intra_user_reuse": s["intra_user_savings_pct"],
            "cross_user_reuse": s["cross_user_extra_pct"],
            "global_reuse": s["global_savings_pct"],
            "global_unique_tokens": s["global_unique_tokens"],
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("results/analysis"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    stats = _load_stats()
    names = list(stats.keys())
    x = np.arange(len(names))

    # ---- Figure 1: 3-panel dataset overview ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Avg tokens per request
    vals = [stats[n]["avg_tokens"] for n in names]
    bars = axes[0].bar(x, vals, color=[COLORS[n] for n in names])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, fontsize=9)
    axes[0].set_ylabel("Avg tokens / request")
    axes[0].set_title("Prompt Length")
    for b, v in zip(bars, vals):
        axes[0].text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{v:,.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes[0].set_yscale("log")
    axes[0].grid(axis="y", alpha=0.3)

    # Panel 2: Requests per user
    vals = [stats[n]["requests_per_user"] for n in names]
    bars = axes[1].bar(x, vals, color=[COLORS[n] for n in names])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, fontsize=9)
    axes[1].set_ylabel("Requests / user")
    axes[1].set_title("Multi-turn Density")
    for b, v in zip(bars, vals):
        label = f"{v:.0f}" if v >= 2 else f"{v:.1f}"
        axes[1].text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes[1].set_yscale("log")
    axes[1].grid(axis="y", alpha=0.3)

    # Panel 3: Reuse breakdown (stacked: intra + cross)
    intra = [stats[n]["intra_user_reuse"] for n in names]
    cross = [stats[n]["cross_user_reuse"] for n in names]
    bars1 = axes[2].bar(x, intra, color=[COLORS[n] for n in names], alpha=0.9, label="Intra-user")
    bars2 = axes[2].bar(
        x,
        cross,
        bottom=intra,
        color=[COLORS[n] for n in names],
        alpha=0.5,
        label="Cross-user",
        hatch="//",
    )
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, fontsize=9)
    axes[2].set_ylabel("KV-cache reuse (%)")
    axes[2].set_title("Prefix Reuse Breakdown")
    for b, iv, cv in zip(bars1, intra, cross):
        total = iv + cv
        axes[2].text(
            b.get_x() + b.get_width() / 2,
            total + 1,
            f"{total:.0f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
        if iv > 2:
            axes[2].text(
                b.get_x() + b.get_width() / 2,
                iv / 2,
                f"{iv:.0f}%",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
            )
        if cv > 5:
            axes[2].text(
                b.get_x() + b.get_width() / 2,
                iv + cv / 2,
                f"{cv:.0f}%",
                ha="center",
                va="center",
                fontsize=8,
            )
    axes[2].legend(fontsize=8, loc="upper right")
    axes[2].set_ylim(0, 70)
    axes[2].grid(axis="y", alpha=0.3)

    fig.suptitle("Dataset Characteristics: GLM-5.1 vs Public Chat Datasets", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path1 = args.out_dir / "dataset_comparison.png"
    fig.savefig(path1, dpi=180)
    plt.close(fig)
    print(f"wrote {path1}")

    # ---- Figure 2: Reuse composition (horizontal stacked bars) ----
    fig, ax = plt.subplots(figsize=(10, 4))
    y = np.arange(len(names))
    unique_pct = [100 - stats[n]["global_reuse"] for n in names]
    intra_pct = [stats[n]["intra_user_reuse"] for n in names]
    cross_pct = [stats[n]["cross_user_reuse"] for n in names]

    ax.barh(y, unique_pct, color="#cccccc", label="Unique (no reuse)")
    ax.barh(y, intra_pct, left=unique_pct, color="#d62728", alpha=0.8, label="Intra-user reuse")
    left2 = [u + i for u, i in zip(unique_pct, intra_pct)]
    ax.barh(y, cross_pct, left=left2, color="#1f77b4", alpha=0.8, label="Cross-user reuse")

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel("Token composition (%)")
    ax.set_title("Where do tokens come from? (Unique vs Reusable)")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 100)
    ax.grid(axis="x", alpha=0.3)

    for i, n in enumerate(names):
        s = stats[n]
        ax.text(
            2, i, f"{100 - s['global_reuse']:.0f}% unique", va="center", fontsize=8, color="#333"
        )
        if s["intra_user_reuse"] > 5:
            ax.text(
                unique_pct[i] + intra_pct[i] / 2,
                i,
                f"{s['intra_user_reuse']:.0f}%",
                va="center",
                ha="center",
                fontsize=8,
                color="white",
            )
        if s["cross_user_reuse"] > 5:
            ax.text(
                left2[i] + cross_pct[i] / 2,
                i,
                f"{s['cross_user_reuse']:.0f}%",
                va="center",
                ha="center",
                fontsize=8,
                color="white",
            )

    fig.tight_layout()
    path2 = args.out_dir / "dataset_reuse_composition.png"
    fig.savefig(path2, dpi=180)
    plt.close(fig)
    print(f"wrote {path2}")

    # ---- Figure 3: Summary table as text ----
    print(
        f"\n{'Dataset':<18} {'Sequences':>12} {'Users':>12} {'Avg tok':>10} {'Req/user':>10} {'Intra':>8} {'Cross':>8} {'Global':>8}"
    )
    for n in names:
        s = stats[n]
        print(
            f"{n:<18} {s['total_sequences']:>12,} {s['users']:>12,} {s['avg_tokens']:>10,.0f} "
            f"{s['requests_per_user']:>10.1f} {s['intra_user_reuse']:>7.1f}% {s['cross_user_reuse']:>7.1f}% {s['global_reuse']:>7.1f}%"
        )


if __name__ == "__main__":
    main()
