"""Radar/spider chart comparing dataset characteristics.

Three axes: prompt length, multi-turn density, user concentration.
One polygon per dataset. GLM-5.1 fills most of the chart; public
datasets are tiny.

Usage:
    python scripts/plot_dataset_radar.py --out paper/figures/dataset_radar.png
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
    "GLM-5.1": "#1b3a5c",
    "LMSYS-Chat-1M": "#6a9fd8",
    "WildChat-4.8M": "#a8cce8",
}

AXES = [
    "Avg tokens\nper request",
    "Requests\nper user",
    "Intra-user\nreuse (%)",
    "Cross-user\nreuse (%)",
    "Global\nreuse (%)",
    "Top-10 user\nconcentration (%)",
]


def _load() -> dict[str, dict]:
    out = {}
    for name, path in DATASETS.items():
        s = json.loads(Path(path).read_text())
        total_seqs = s["total_sequences"]
        total_tokens = s["total_tokens"]
        users = s["user_count"]
        top_key = "top_users" if "top_users" in s else "top_groups"
        top_entries = s.get(top_key, [])
        top10_tokens = sum(e["tokens"] for e in top_entries[:10])
        out[name] = {
            "avg_tokens": total_tokens / total_seqs,
            "requests_per_user": total_seqs / users,
            "intra_reuse": s["intra_user_savings_pct"],
            "cross_reuse": s["cross_user_extra_pct"],
            "global_reuse": s["global_savings_pct"],
            "top10_pct": 100.0 * top10_tokens / total_tokens,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    stats = _load()
    names = list(stats.keys())

    raw = {
        n: [
            stats[n]["avg_tokens"],
            stats[n]["requests_per_user"],
            stats[n]["intra_reuse"],
            stats[n]["cross_reuse"],
            stats[n]["global_reuse"],
            stats[n]["top10_pct"],
        ]
        for n in names
    }

    all_vals = list(raw.values())
    num_dims = len(AXES)
    maxes = [max(v[i] for v in all_vals) for i in range(num_dims)]
    normed = {
        n: [v[i] / maxes[i] if maxes[i] > 0 else 0 for i in range(num_dims)] for n, v in raw.items()
    }

    num_axes = len(AXES)
    angles = np.linspace(0, 2 * np.pi, num_axes, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw=dict(polar=True))

    for name in names:
        vals = normed[name] + normed[name][:1]
        raw_vals = raw[name]
        color = COLORS[name]
        ax.fill(angles, vals, alpha=0.15, color=color)
        ax.plot(angles, vals, linewidth=2.2, color=color, label=name)

        for i, (angle, nv) in enumerate(zip(angles[:-1], normed[name])):
            ax.scatter(angle, nv, s=25, color=color, zorder=5, edgecolors="white", linewidths=0.5)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(AXES, fontsize=10)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([])
    ax.set_ylim(0, 1.15)
    ax.spines["polar"].set_visible(False)
    ax.grid(alpha=0.25)

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9, framealpha=0.9)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
