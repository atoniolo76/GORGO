"""Generate cache utilization vs latency scatter + routing concentration bars.

Usage:
    python scripts/plot_paper_cache_and_concentration.py \
        --workload-dir /tmp/gorgo_dl/workload_runs \
        --trace-dir /tmp/gorgo_dl/proxy_traces \
        --prefix tune_ --out figures/cache_and_concentration.png
"""

from __future__ import annotations

import argparse
import json
import glob
import os
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paper_style import (
    get_color,
    is_gorgo,
    display_name,
    apply_paper_style,
)


def _load(workload_dir: str, trace_dir: str, prefix: str) -> list[dict]:
    policies = []
    for path in sorted(glob.glob(os.path.join(workload_dir, f"{prefix}*.json"))):
        pol = os.path.basename(path).replace(prefix, "").replace(".json", "")
        with open(path) as f:
            stats = json.load(f)["stats"]

        trace_path = os.path.join(trace_dir, f"{prefix}{pol}", "requests.jsonl")
        reqs = []
        if os.path.exists(trace_path):
            with open(trace_path) as f:
                for line in f:
                    if line.strip():
                        try:
                            reqs.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass

        ok = [r for r in reqs if r.get("status") == 200]
        cache_hits = []
        for r in ok:
            cached = r.get("cached_tokens_at_dispatch", 0) or 0
            total = r.get("request_tokens", 0) or r.get("prompt_tokens", 0) or 1
            if total > 0:
                cache_hits.append(cached / total)

        targets = Counter(r.get("target", "") for r in ok if r.get("target"))
        total_routed = sum(targets.values())
        most_used = (max(targets.values()) / total_routed * 100) if total_routed else 0

        policies.append(
            {
                "label": pol,
                "display": display_name(pol),
                "is_gorgo": is_gorgo(pol),
                "p95": stats["ttft_seconds"]["p95"],
                "cache_hit": np.mean(cache_hits) * 100 if cache_hits else 0,
                "concentration": most_used,
            }
        )
    return policies


def plot(policies: list[dict], out_path: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.3, 1]})

    for p in policies:
        color = get_color(p["label"])
        marker = "D" if p["is_gorgo"] else "o"
        size = 120 if p["is_gorgo"] else 80
        edgecolor = "white" if p["is_gorgo"] else "none"
        ax1.scatter(
            p["cache_hit"],
            p["p95"],
            c=color,
            s=size,
            marker=marker,
            edgecolors=edgecolor,
            linewidths=1.0,
            zorder=10 if p["is_gorgo"] else 5,
        )
        ax1.annotate(
            p["display"],
            (p["cache_hit"], p["p95"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7.5,
            ha="left",
            va="bottom",
            color="#1b3a5c" if p["is_gorgo"] else "#555",
            fontweight="bold" if p["is_gorgo"] else "normal",
        )

    ax1.set_xlabel("KV-cache hit rate (%)", fontsize=10)
    ax1.set_ylabel("p95 TTFT (s)", fontsize=10)
    ax1.set_title("Cache utilization vs latency", fontsize=11, fontweight="bold")
    apply_paper_style(ax1)

    policies_sorted = sorted(policies, key=lambda p: p["concentration"], reverse=True)
    labels = [p["display"] for p in policies_sorted]
    concentrations = [p["concentration"] for p in policies_sorted]
    colors = [get_color(p["label"]) for p in policies_sorted]
    y = np.arange(len(policies_sorted))

    ax2.barh(y, concentrations, color=colors, height=0.65, edgecolor="white", linewidth=0.5)
    ax2.axvline(
        100 / 3, color="#888888", linestyle="--", linewidth=1.0, alpha=0.6, label="Uniform (33%)"
    )

    ax2.set_yticks(y)
    tick_labels = ax2.set_yticklabels(labels, fontsize=9)
    for tl, p in zip(tick_labels, policies_sorted):
        if p["is_gorgo"]:
            tl.set_fontweight("bold")
            tl.set_color("#1b3a5c")

    for i, v in enumerate(concentrations):
        ax2.text(v + 0.8, y[i], f"{v:.0f}%", va="center", fontsize=8, color="#333")

    ax2.set_xlabel("Requests on most-used replica (%)", fontsize=10)
    ax2.set_title("Routing concentration", fontsize=11, fontweight="bold")
    ax2.set_xlim(0, max(concentrations) * 1.15)
    apply_paper_style(ax2)
    ax2.grid(axis="x", alpha=0.2)

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
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    policies = _load(args.workload_dir, args.trace_dir, args.prefix)
    if not policies:
        raise SystemExit("No data found")
    plot(policies, args.out)


if __name__ == "__main__":
    main()
