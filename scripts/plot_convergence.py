"""Plot online optimizer convergence from proxy request traces.

Shows how the GORGO hillclimb (online-ES) evolves its routing behavior
over time by tracking:
  1. % of requests routed to the best-cache replica (convergence signal)
  2. Per-replica traffic share over time (balance signal)
  3. Cache hit rate over time (reuse signal)

Usage:
    python scripts/plot_convergence.py \
        --run-prefix abstract_night_000_glm5_0030_to_0100 \
        --results-dir results \
        --out-dir results/analysis
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_requests(trace_dir: Path, run_prefix: str, policy: str) -> list[dict]:
    path = trace_dir / f"{run_prefix}_{policy}" / "requests.jsonl"
    if not path.exists():
        return []
    rows = []
    min_mono: float | None = None
    for line in path.open():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("kind") != "request" or r.get("status") != 200:
            continue
        mono = r.get("monotonic_s")
        if mono is None:
            continue
        if min_mono is None:
            min_mono = mono

        target = r.get("target", "")
        snap = r.get("candidate_snapshot") or {}
        req_tokens = r.get("request_tokens", 0)
        target_cache = (snap.get(target) or {}).get("cached_prefix_tokens", 0)
        all_caches = {url: (s.get("cached_prefix_tokens") or 0) for url, s in snap.items()}
        best_cache = max(all_caches.values()) if all_caches else 0

        rows.append(
            {
                "elapsed_min": (mono - min_mono) / 60.0,
                "elapsed_s": mono - min_mono,
                "req_tokens": req_tokens,
                "target_cache": target_cache,
                "best_cache": best_cache,
                "picked_best_cache": target_cache >= best_cache if best_cache > 0 else True,
                "cache_ratio": target_cache / req_tokens if req_tokens > 0 else 0,
                "target": target,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--policy", default="gorgo-hillclimb")
    parser.add_argument(
        "--compare", nargs="*", default=["random", "prefix-cache", "simple-session-affinity"]
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/analysis"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    trace_dir = args.results_dir / "proxy_traces"
    all_policies = [args.policy] + [p for p in args.compare if p != args.policy]

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    WINDOW = 50
    STEP = WINDOW // 4

    policy_data = {}
    for policy in all_policies:
        rows = _load_requests(trace_dir, args.run_prefix, policy)
        if not rows:
            print(f"  {policy}: no trace data")
            continue
        policy_data[policy] = rows
        print(f"  {policy}: {len(rows)} requests")

    if not policy_data:
        print("No data found")
        return

    # ---- Panel 1: % picking best-cache replica over time ----
    for policy, rows in policy_data.items():
        ts, vals = [], []
        for i in range(0, len(rows) - WINDOW, STEP):
            chunk = rows[i : i + WINDOW]
            rate = sum(1 for s in chunk if s["picked_best_cache"]) / len(chunk) * 100
            ts.append(chunk[len(chunk) // 2]["elapsed_min"])
            vals.append(rate)
        is_main = policy == args.policy
        axes[0].plot(
            ts,
            vals,
            label=policy,
            color="#d62728" if is_main else None,
            linewidth=2.5 if is_main else 1.2,
            alpha=1.0 if is_main else 0.6,
        )

    axes[0].set_ylabel("Best-cache pick rate (%)")
    axes[0].set_title(
        "Convergence: how often each policy routes to the replica with the most cached prefix"
    )
    axes[0].legend(fontsize=8, loc="lower right")
    axes[0].set_ylim(0, 105)
    axes[0].grid(alpha=0.3)
    axes[0].axhline(100, color="#888", linestyle="--", linewidth=0.8, alpha=0.5)

    # ---- Panel 2: Per-replica traffic share over time ----
    rows = policy_data.get(args.policy, [])
    if rows:
        all_targets = sorted(set(r["target"] for r in rows))
        target_labels = {
            t: t.split("//", 1)[-1].split(".", 1)[0][:15] if "//" in t else t[:15]
            for t in all_targets
        }

        for target in all_targets:
            ts, vals = [], []
            for i in range(0, len(rows) - WINDOW, STEP):
                chunk = rows[i : i + WINDOW]
                share = sum(1 for s in chunk if s["target"] == target) / len(chunk) * 100
                ts.append(chunk[len(chunk) // 2]["elapsed_min"])
                vals.append(share)
            axes[1].plot(ts, vals, label=target_labels[target], linewidth=1.5)

        axes[1].axhline(
            100 / len(all_targets),
            color="#888",
            linestyle="--",
            linewidth=0.8,
            alpha=0.5,
            label=f"uniform ({100 / len(all_targets):.0f}%)",
        )
        axes[1].set_ylabel("Traffic share (%)")
        axes[1].set_title(f"{args.policy}: per-replica traffic share over time")
        axes[1].legend(fontsize=7, loc="upper right")
        axes[1].set_ylim(0, 60)
        axes[1].grid(alpha=0.3)

    # ---- Panel 3: Cache hit rate over time ----
    SMOOTH_WINDOW_S = 30
    EVAL_POINTS = 1000
    for policy, rows in policy_data.items():
        times = np.array([r["elapsed_s"] for r in rows])
        vals = np.array([r["cache_ratio"] * 100 for r in rows])
        t_min, t_max = times.min(), times.max()
        eval_t = np.linspace(t_min, t_max, EVAL_POINTS)
        smoothed = np.full_like(eval_t, np.nan)
        for i, t in enumerate(eval_t):
            mask = np.abs(times - t) <= SMOOTH_WINDOW_S / 2
            if mask.sum() >= 5:
                smoothed[i] = np.mean(vals[mask])
        valid = ~np.isnan(smoothed)
        is_main = policy == args.policy
        axes[2].plot(
            eval_t[valid] / 60.0,
            smoothed[valid],
            label=policy,
            color="#d62728" if is_main else None,
            linewidth=2.5 if is_main else 1.2,
            alpha=1.0 if is_main else 0.6,
        )

    axes[2].set_xlabel("Elapsed time (minutes)")
    axes[2].set_ylabel("Cache hit rate (%)")
    axes[2].set_title(f"Achieved cache hit rate over time ({SMOOTH_WINDOW_S}s sliding window)")
    axes[2].legend(fontsize=8, loc="lower right")
    axes[2].grid(alpha=0.3)

    fig.suptitle(f"Online Optimizer Convergence: {args.policy}\n{args.run_prefix}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    slug = args.run_prefix.split("_")[-1]
    out_path = args.out_dir / f"convergence_{slug}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
