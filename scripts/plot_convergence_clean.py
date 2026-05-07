"""Clean 2-panel convergence figure for the paper.

Panel 1: Best-cache-pick rate over time (% of requests routed to the
         replica with the most cached prefix for that request).
Panel 2: Achieved cache hit rate over time (sliding window).

Uses the same proxy trace data as plot_convergence.py but with a
paper-grade blue palette and seaborn styling.

Usage:
    python scripts/plot_convergence_clean.py \
        --run-prefix abstract_night_000_glm5_0030_to_0100 \
        --results-dir results \
        --out paper/figures/convergence.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

GORGO_POLICIES = {"gorgo-hillclimb", "gorgo-static", "gorgo-autotune"}

POLICY_STYLE = {
    "gorgo-hillclimb": {"color": "#1b3a5c", "lw": 2.5, "alpha": 1.0, "zorder": 10},
    "gorgo-static": {"color": "#2d6a9f", "lw": 2.0, "alpha": 0.9, "zorder": 9},
    "prefix-cache": {"color": "#7ab5e0", "lw": 1.3, "alpha": 0.7, "zorder": 5},
    "simple-session-affinity": {"color": "#aaa", "lw": 1.3, "alpha": 0.6, "zorder": 4},
    "random": {"color": "#ccc", "lw": 1.3, "alpha": 0.5, "zorder": 3},
}

WINDOW = 50
STEP = WINDOW // 4


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
                "picked_best_cache": target_cache >= best_cache if best_cache > 0 else True,
                "cache_ratio": target_cache / req_tokens if req_tokens > 0 else 0,
            }
        )
    return rows


def _rolling(rows: list[dict], key: str, as_pct: bool = True) -> tuple[list[float], list[float]]:
    ts, vals = [], []
    for i in range(0, len(rows) - WINDOW, STEP):
        chunk = rows[i : i + WINDOW]
        if as_pct:
            val = sum(1 for s in chunk if s[key]) / len(chunk) * 100
        else:
            val = np.mean([s[key] for s in chunk]) * 100
        ts.append(chunk[len(chunk) // 2]["elapsed_min"])
        vals.append(val)
    return ts, vals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    trace_dir = args.results_dir / "proxy_traces"
    policies = [
        "gorgo-hillclimb",
        "gorgo-static",
        "prefix-cache",
        "simple-session-affinity",
        "random",
    ]

    policy_data = {}
    for p in policies:
        rows = _load_requests(trace_dir, args.run_prefix, p)
        if rows:
            policy_data[p] = rows
            print(f"  {p}: {len(rows)} requests")

    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    for policy, rows in policy_data.items():
        style = POLICY_STYLE.get(policy, {"color": "#ccc", "lw": 1.0, "alpha": 0.5, "zorder": 2})
        is_gorgo = policy in GORGO_POLICIES

        # Panel 1: best-cache-pick rate
        ts, vals = _rolling(rows, "picked_best_cache", as_pct=True)
        ax1.plot(ts, vals, label=policy, **style)

        # Panel 2: cache hit rate
        ts2, vals2 = _rolling(rows, "cache_ratio", as_pct=False)
        ax2.plot(ts2, vals2, label=policy, **style)

    ax1.set_ylabel("Best-cache pick rate (%)")
    ax1.set_xlabel("Elapsed time (minutes)")
    ax1.set_title("Routing to best-cached replica", fontsize=11, fontweight="bold")
    ax1.set_ylim(50, 105)
    ax1.legend(fontsize=8, loc="lower right", framealpha=0.9)

    ax2.set_ylabel("Cache hit rate (%)")
    ax2.set_xlabel("Elapsed time (minutes)")
    ax2.set_title("Achieved KV-cache utilization", fontsize=11, fontweight="bold")
    ax2.set_ylim(30, 105)
    ax2.legend(fontsize=8, loc="lower right", framealpha=0.9)

    sns.despine()
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
