"""Paper-grade per-policy comparison from a downloaded results tree.

Reads the ``workload_runs/`` JSONs and ``proxy_traces/<run>/requests.jsonl``
artifacts produced by ``experiments/policy_matrix_app.py`` and renders four
panels:

  1. TTFT distribution per policy (p50/p95/p99 bars, sorted by p95).
  2. Per-policy routing concentration (share of requests on most-used replica).
  3. TTFT vs request index per policy (warm-up curve - shows the cache
     ramp).
  4. Bucketed p95 TTFT per policy (rolling p95 over 50-request buckets).

Usage:
    python scripts/plot_policy_summary.py \\
        --results-dir results \\
        --run-prefix moon_neurips_main_001_00028_20260401T140000Z_token_hash_filter_top20 \\
        --out results/policy-summary.png
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))]


def _load_policies(results_dir: Path, run_prefix: str) -> list[dict]:
    workload_dir = results_dir / "workload_runs"
    trace_dir = results_dir / "proxy_traces"
    out: list[dict] = []
    for path in sorted(workload_dir.glob(f"{run_prefix}_*.json")):
        label = path.stem[len(run_prefix) + 1 :]
        d = json.loads(path.read_text())
        reqs = d.get("requests") or []
        ok = [r for r in reqs if 200 <= r["status"] < 300]
        ttfts = [r["ttft_ns"] / 1e9 for r in ok if r.get("ttft_ns")]
        e2e = [r["total_ns"] / 1e9 for r in ok if r.get("total_ns")]
        ttfts_indexed = [
            (r.get("request_id") or "", r["ttft_ns"] / 1e9)
            for r in ok
            if r.get("ttft_ns") and r.get("request_id")
        ]
        targets = Counter()
        ttft_over_time: list[tuple[float, float]] = []
        rt_path = trace_dir / f"{run_prefix}_{label}" / "requests.jsonl"
        min_monotonic: float | None = None
        if rt_path.exists():
            for line in rt_path.open():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = row.get("target")
                if t:
                    targets[t] += 1
                mono = row.get("monotonic_s")
                ttft_ns = row.get("ttft_ns")
                if mono is not None and ttft_ns is not None and row.get("status") == 200:
                    if min_monotonic is None:
                        min_monotonic = mono
                    ttft_over_time.append((mono - min_monotonic, ttft_ns / 1e9))
        ttft_over_time.sort()
        out.append(
            {
                "label": label,
                "n": len(reqs),
                "ok": len(ok),
                "success_pct": 100.0 * len(ok) / max(1, len(reqs)),
                "ttft_p50": _percentile(ttfts, 0.50),
                "ttft_p95": _percentile(ttfts, 0.95),
                "ttft_p99": _percentile(ttfts, 0.99),
                "ttft_max": max(ttfts) if ttfts else 0.0,
                "e2e_p95": _percentile(e2e, 0.95),
                "ttfts_indexed": ttfts_indexed,
                "ttft_over_time": ttft_over_time,
                "targets": targets,
            }
        )
    return out


def _bucket_p95(rows: list[tuple[int, float]], bucket: int) -> tuple[list[int], list[float]]:
    by_bucket: dict[int, list[float]] = {}
    for idx, ttft in rows:
        by_bucket.setdefault(idx // bucket, []).append(ttft)
    xs = [b * bucket for b in sorted(by_bucket)]
    ys = [_percentile(by_bucket[b], 0.95) for b in sorted(by_bucket)]
    return xs, ys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--highlight",
        default="gorgo-autotuned",
        help="Policy label to render in bold/accent color.",
    )
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

    HIGHLIGHT_COLOR = "#0d3b66"
    ACCENT_LIGHT = "#a8dadc"

    policies = _load_policies(args.results_dir, args.run_prefix)
    if not policies:
        raise SystemExit(
            f"no workload result JSONs found under {args.results_dir}/workload_runs "
            f"matching prefix {args.run_prefix}_*"
        )

    policies.sort(key=lambda p: p["ttft_p95"])

    rid_set: set[str] = set()
    for p in policies:
        rid_set.update(rid for rid, _ in p["ttfts_indexed"])
    rid_index = {rid: i for i, rid in enumerate(sorted(rid_set))}
    for p in policies:
        p["ttft_by_index"] = sorted(
            [(rid_index[r], t) for r, t in p["ttfts_indexed"] if r in rid_index]
        )

    labels = [p["label"] for p in policies]
    n_policies = len(policies)
    bar_palette = sns.color_palette("Blues_d", n_colors=max(n_policies, 4))
    colors = [
        HIGHLIGHT_COLOR if lab == args.highlight else bar_palette[i] for i, lab in enumerate(labels)
    ]
    text_colors = [HIGHLIGHT_COLOR if lab == args.highlight else "#333333" for lab in labels]

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    ax_bars, ax_route, ax_curve, ax_bucket = axes.flat

    # Panel 1: TTFT p50/p95/p99 bars per policy
    x = np.arange(n_policies)
    w = 0.27
    p50 = [p["ttft_p50"] for p in policies]
    p95 = [p["ttft_p95"] for p in policies]
    p99 = [p["ttft_p99"] for p in policies]

    blue_tints = sns.color_palette("Blues", 5)
    ax_bars.bar(x - w, p50, w, label="p50", color=blue_tints[1], edgecolor="white", linewidth=0.5)
    ax_bars.bar(x, p95, w, label="p95", color=blue_tints[3], edgecolor="white", linewidth=0.5)
    bar99 = ax_bars.bar(x + w, p99, w, label="p99", color=colors, edgecolor="white", linewidth=0.5)
    for i, lab in enumerate(labels):
        if lab == args.highlight:
            for container in ax_bars.containers[:2]:
                container[i].set_edgecolor(HIGHLIGHT_COLOR)
                container[i].set_linewidth(1.6)
            bar99[i].set_edgecolor(HIGHLIGHT_COLOR)
            bar99[i].set_linewidth(1.6)
    ax_bars.set_xticks(x)
    ax_bars.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    for tick, c in zip(ax_bars.get_xticklabels(), text_colors):
        tick.set_color(c)
        if c == HIGHLIGHT_COLOR:
            tick.set_fontweight("bold")
    ax_bars.set_ylabel("TTFT (s)")
    ax_bars.set_title("TTFT per policy (sorted by p95, lower = better)")
    ax_bars.legend(loc="upper left")

    # Panel 2: routing concentration
    most_used = []
    n_replicas_used = []
    for p in policies:
        total = sum(p["targets"].values())
        if total == 0:
            most_used.append(0.0)
            n_replicas_used.append(0)
            continue
        top_count = max(p["targets"].values())
        most_used.append(100.0 * top_count / total)
        n_replicas_used.append(sum(1 for c in p["targets"].values() if c > 0))
    bars = ax_route.bar(x, most_used, color=colors, edgecolor="white", linewidth=0.5)
    ax_route.axhline(33.3, color="#6c757d", linestyle="--", linewidth=1, alpha=0.7)
    ax_route.text(
        len(policies) - 0.5,
        34.5,
        "uniform (3 replicas)",
        color="#6c757d",
        fontsize=8,
        ha="right",
    )
    ax_route.set_xticks(x)
    ax_route.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    for tick, c in zip(ax_route.get_xticklabels(), text_colors):
        tick.set_color(c)
        if c == HIGHLIGHT_COLOR:
            tick.set_fontweight("bold")
    for i, b in enumerate(bars):
        ax_route.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 1,
            f"{most_used[i]:.0f}%",
            ha="center",
            fontsize=8,
        )
    ax_route.set_ylabel("Share of requests on most-used replica")
    ax_route.set_title("Routing concentration (lower = more balanced)")
    ax_route.set_ylim(0, max(100, max(most_used) * 1.12))

    # Panel 3: smoothed median TTFT over elapsed time
    WINDOW_S = 30
    EVAL_POINTS = 1000
    line_palette = sns.color_palette("mako", n_colors=n_policies)
    for idx, p in enumerate(policies):
        data = p.get("ttft_over_time") or []
        if len(data) < 10:
            continue
        times = np.array([t for t, _ in data])
        vals = np.array([v for _, v in data])
        t_min, t_max = times.min(), times.max()
        eval_t = np.linspace(t_min, t_max, EVAL_POINTS)
        smoothed = np.full_like(eval_t, np.nan)
        for i, t in enumerate(eval_t):
            mask = np.abs(times - t) <= WINDOW_S / 2
            if mask.sum() >= 5:
                smoothed[i] = np.median(vals[mask])
        valid = ~np.isnan(smoothed)
        is_highlight = p["label"] == args.highlight
        color = HIGHLIGHT_COLOR if is_highlight else line_palette[idx]
        lw = 2.5 if is_highlight else 1.3
        alpha = 1.0 if is_highlight else 0.7
        ax_curve.plot(
            eval_t[valid] / 60.0,
            smoothed[valid],
            label=p["label"],
            color=color,
            linewidth=lw,
            alpha=alpha,
        )
    ax_curve.set_xlabel("Elapsed time (minutes)")
    ax_curve.set_ylabel("Median TTFT (s)")
    ax_curve.set_title(f"Smoothed median TTFT over time ({WINDOW_S}s sliding window)")
    ax_curve.legend(loc="upper right", ncols=2, fontsize=8)
    all_p99 = [p["ttft_p99"] for p in policies if p["ttft_p99"] > 0]
    if all_p99:
        clip = sorted(all_p99)[len(all_p99) // 2] * 2.0
        ax_curve.set_ylim(0, clip)

    # Panel 4: smoothed p95 TTFT over time
    for idx, p in enumerate(policies):
        data = p.get("ttft_over_time") or []
        if len(data) < 10:
            continue
        times = np.array([t for t, _ in data])
        vals = np.array([v for _, v in data])
        t_min, t_max = times.min(), times.max()
        eval_t = np.linspace(t_min, t_max, EVAL_POINTS)
        smoothed = np.full_like(eval_t, np.nan)
        for i, t in enumerate(eval_t):
            mask = np.abs(times - t) <= WINDOW_S / 2
            if mask.sum() >= 5:
                window_vals = np.sort(vals[mask])
                smoothed[i] = window_vals[int(len(window_vals) * 0.95)]
        valid = ~np.isnan(smoothed)
        is_highlight = p["label"] == args.highlight
        color = HIGHLIGHT_COLOR if is_highlight else line_palette[idx]
        lw = 2.5 if is_highlight else 1.3
        alpha = 1.0 if is_highlight else 0.7
        ax_bucket.plot(
            eval_t[valid] / 60.0,
            smoothed[valid],
            label=p["label"],
            color=color,
            linewidth=lw,
            alpha=alpha,
        )
    ax_bucket.set_xlabel("Elapsed time (minutes)")
    ax_bucket.set_ylabel("p95 TTFT (s)")
    ax_bucket.set_title(f"Smoothed p95 TTFT over time ({WINDOW_S}s sliding window)")
    ax_bucket.legend(loc="upper right", ncols=2, fontsize=8)
    if all_p99:
        clip_p95 = sorted(all_p99)[-1] * 1.5
        ax_bucket.set_ylim(0, clip_p95)

    fig.suptitle(
        f"{args.run_prefix}\n"
        f"{len(policies)} policies, {policies[0]['n']} requests/policy, "
        f"{policies[0]['ok']} succeeded",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180, bbox_inches="tight")
    print(f"wrote {args.out}")

    print("\n=== Headline numbers ===")
    print(
        f"{'policy':<28} {'success':>7}  {'p50':>6} {'p95':>6} {'p99':>6} "
        f"{'max':>6}  {'most-used%':>10} {'replicas':>8}"
    )
    for i, p in enumerate(policies):
        marker = "*" if p["label"] == args.highlight else " "
        print(
            f"{marker} {p['label']:<26} {p['success_pct']:>6.1f}% "
            f"{p['ttft_p50']:>6.2f} {p['ttft_p95']:>6.2f} {p['ttft_p99']:>6.2f} "
            f"{p['ttft_max']:>6.2f}  {most_used[i]:>9.1f}% {n_replicas_used[i]:>8}"
        )


if __name__ == "__main__":
    main()
