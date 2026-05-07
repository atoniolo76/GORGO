"""Side-by-side dataset profile: one row per metric, grouped bars.

Usage:
    python scripts/plot_dataset_profile.py --out paper/figures/dataset_profile.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATASETS_CFG = [
    ("GLM-5.1", "data_processing/prefix_trie_results/glm-5.1-completions/stats.json"),
    ("LMSYS-Chat-1M", "data_processing/prefix_trie_results/lmsys-chat-1m/stats.json"),
    ("WildChat-4.8M", "data_processing/prefix_trie_results/wildchat/stats.json"),
]

COLORS = {
    "GLM-5.1": "#1b3a5c",
    "LMSYS-Chat-1M": "#6a9fd8",
    "WildChat-4.8M": "#a8cce8",
}


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
                    "Total requests": total_seqs,
                    "Users": users,
                    "Avg tokens / request": total_tokens / total_seqs,
                    "Requests / user": total_seqs / users,
                    "Intra-user reuse": s["intra_user_savings_pct"],
                    "Cross-user reuse": s["cross_user_extra_pct"],
                    "Global reuse": s["global_savings_pct"],
                    "Top-10 user concentration": 100.0 * top10 / total_tokens,
                },
            )
        )
    return out


def _fmt(metric: str, val: float) -> str:
    if "reuse" in metric.lower() or "concentration" in metric.lower():
        return f"{val:.1f}%"
    if val >= 1_000_000:
        return f"{val / 1e6:.1f}M"
    if val >= 1_000:
        return f"{val / 1e3:.0f}K" if val >= 10_000 else f"{val:,.0f}"
    if val < 10:
        return f"{val:.1f}"
    return f"{val:,.0f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    data = _load()
    names = [d[0] for d in data]
    metrics = list(data[0][1].keys())

    fig, ax = plt.subplots(figsize=(10, 5))

    n_metrics = len(metrics)
    n_datasets = len(names)
    bar_h = 0.22
    y_base = np.arange(n_metrics)

    for di, (name, vals) in enumerate(data):
        color = COLORS[name]
        raw = [vals[m] for m in metrics]
        maxes = [max(d[1][m] for d in data) for m in metrics]
        normed = [r / mx if mx > 0 else 0 for r, mx in zip(raw, maxes)]

        y_pos = y_base + (di - (n_datasets - 1) / 2) * bar_h
        bars = ax.barh(
            y_pos, normed, height=bar_h, color=color, edgecolor="white", linewidth=0.5, label=name
        )

        for i, (b, rv) in enumerate(zip(bars, raw)):
            label = _fmt(metrics[i], rv)
            ax.text(
                b.get_width() + 0.02,
                b.get_y() + b.get_height() / 2,
                label,
                va="center",
                fontsize=7.5,
                color="#333",
            )

    ax.set_yticks(y_base)
    ax.set_yticklabels(metrics, fontsize=9)
    ax.set_xlim(0, 1.35)
    ax.set_xlabel("Normalized (max = 1.0)", fontsize=9, color="#888")
    ax.invert_yaxis()
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
    ax.grid(visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks([])

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
