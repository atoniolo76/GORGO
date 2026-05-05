"""Analyze cache hit rates per policy and plot convergence-related metrics.

Produces:
  1. Per-policy cache hit rate comparison (bar chart)
  2. Cache hit rate over time per policy (smooth curves)
  3. Per-policy achieved cache tokens vs total tokens

Usage:
    python scripts/plot_cache_and_convergence.py \
        --run-prefix abstract_night_000_glm5_0030_to_0100 \
        --results-dir results \
        --out-dir results/analysis
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

HIGHLIGHT_COLOR = "#0d3b66"
PALETTE_NAME = "Blues_d"


def _load_trace(trace_dir: Path, run_prefix: str, policy: str) -> list[dict]:
    path = trace_dir / f"{run_prefix}_{policy}" / "requests.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.open():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("kind") != "request":
            continue
        rows.append(r)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/analysis"))
    parser.add_argument("--highlight", default="gorgo-hillclimb")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    trace_dir = args.results_dir / "proxy_traces"
    workload_dir = args.results_dir / "workload_runs"

    policies = []
    for f in sorted(workload_dir.glob(f"{args.run_prefix}_*.json")):
        label = f.stem[len(args.run_prefix) + 1 :]
        policies.append(label)

    if not policies:
        print(f"No workload results found for prefix {args.run_prefix}")
        return

    print(f"Found {len(policies)} policies: {policies}")

    policy_stats = {}
    for policy in policies:
        rows = _load_trace(trace_dir, args.run_prefix, policy)
        if not rows:
            print(f"  {policy}: no trace data")
            continue

        ok_rows = [r for r in rows if r.get("status") == 200]
        total_request_tokens = 0
        total_cached_at_dispatch = 0
        total_cached_on_target = 0
        cache_over_time: list[tuple[float, float]] = []
        min_mono: float | None = None

        for r in ok_rows:
            req_tok = r.get("request_tokens") or r.get("prompt_tokens") or 0
            cached_dispatch = r.get("cached_tokens_at_dispatch") or 0
            total_request_tokens += req_tok
            total_cached_at_dispatch += cached_dispatch

            target = r.get("target")
            snap = (r.get("candidate_snapshot") or {}).get(target) or {}
            cached_target = snap.get("cached_prefix_tokens") or 0
            total_cached_on_target += cached_target

            mono = r.get("monotonic_s")
            if mono is not None:
                if min_mono is None:
                    min_mono = mono
                if req_tok > 0:
                    cache_over_time.append((mono - min_mono, cached_target / req_tok))

        hit_rate_dispatch = (
            total_cached_at_dispatch / total_request_tokens if total_request_tokens else 0
        )
        hit_rate_target = (
            total_cached_on_target / total_request_tokens if total_request_tokens else 0
        )

        policy_stats[policy] = {
            "n": len(ok_rows),
            "total_request_tokens": total_request_tokens,
            "total_cached_dispatch": total_cached_at_dispatch,
            "total_cached_target": total_cached_on_target,
            "hit_rate_dispatch": hit_rate_dispatch,
            "hit_rate_target": hit_rate_target,
            "cache_over_time": cache_over_time,
        }
        print(
            f"  {policy}: {len(ok_rows)} ok, cache_hit_rate={hit_rate_target:.1%} "
            f"(dispatch={hit_rate_dispatch:.1%})"
        )

    if not policy_stats:
        print("No trace data available for any policy")
        return

    sorted_policies = sorted(
        policy_stats.keys(), key=lambda p: policy_stats[p]["hit_rate_target"], reverse=True
    )

    n_policies = len(sorted_policies)
    palette = sns.color_palette(PALETTE_NAME, n_colors=max(n_policies, 4))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    x = np.arange(n_policies)
    hit_rates = [policy_stats[p]["hit_rate_target"] * 100 for p in sorted_policies]
    colors = [
        HIGHLIGHT_COLOR if p == args.highlight else palette[i]
        for i, p in enumerate(sorted_policies)
    ]

    bars = ax1.bar(x, hit_rates, color=colors, edgecolor="white", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(sorted_policies, rotation=30, ha="right", fontsize=9)
    ax1.set_ylabel("Cache hit rate (%)")
    ax1.set_title("Achieved KV-cache hit rate per policy")
    for b, v in zip(bars, hit_rates):
        ax1.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.5,
            f"{v:.1f}%",
            ha="center",
            fontsize=8,
        )
    for tick, p in zip(ax1.get_xticklabels(), sorted_policies):
        if p == args.highlight:
            tick.set_color(HIGHLIGHT_COLOR)
            tick.set_fontweight("bold")

    # Panel 2: Cache hit rate over time (smoothed)
    WINDOW_S = 30
    EVAL_POINTS = 500
    line_palette = sns.color_palette("mako", n_colors=n_policies)
    for idx, policy in enumerate(sorted_policies):
        data = policy_stats[policy]["cache_over_time"]
        if len(data) < 10:
            continue
        times = np.array([t for t, _ in data])
        vals = np.array([v * 100 for _, v in data])
        t_min, t_max = times.min(), times.max()
        eval_t = np.linspace(t_min, t_max, EVAL_POINTS)
        smoothed = np.full_like(eval_t, np.nan)
        for i, t in enumerate(eval_t):
            mask = np.abs(times - t) <= WINDOW_S / 2
            if mask.sum() >= 5:
                smoothed[i] = np.mean(vals[mask])
        valid = ~np.isnan(smoothed)
        is_highlight = policy == args.highlight
        color = HIGHLIGHT_COLOR if is_highlight else line_palette[idx]
        lw = 2.5 if is_highlight else 1.3
        alpha = 1.0 if is_highlight else 0.7
        ax2.plot(
            eval_t[valid] / 60.0,
            smoothed[valid],
            label=policy,
            color=color,
            linewidth=lw,
            alpha=alpha,
        )

    ax2.set_xlabel("Elapsed time (minutes)")
    ax2.set_ylabel("Cache hit rate (%)")
    ax2.set_title(f"Cache hit rate over time ({WINDOW_S}s sliding window)")
    ax2.legend(fontsize=7, loc="lower right", ncols=2)

    fig.suptitle(args.run_prefix.replace("_", " "), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path1 = args.out_dir / f"cache_hit_rate_{args.run_prefix.split('_')[-1]}.png"
    fig.savefig(path1, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {path1}")

    # Print summary table
    print(f"\n{'policy':<28} {'cache hit %':>12} {'cached tok':>12} {'total tok':>12} {'n':>6}")
    for p in sorted_policies:
        s = policy_stats[p]
        print(
            f"{p:<28} {s['hit_rate_target'] * 100:>11.1f}% {s['total_cached_target']:>12,} "
            f"{s['total_request_tokens']:>12,} {s['n']:>6}"
        )


if __name__ == "__main__":
    main()
