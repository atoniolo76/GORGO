"""Cache hit rate convergence: EWMA-smoothed, single panel.

Shows gorgo-hillclimb's cache utilization ramping up and stabilizing
as the ES tunes, vs baselines that stay flat.

Usage:
    python scripts/plot_cache_convergence.py \
        --run-prefix abstract_night_000_glm5_0030_to_0100 \
        --results-dir results \
        --out paper/figures/cache_convergence.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

POLICY_STYLE = {
    "gorgo-hillclimb": {"color": "#1b3a5c", "lw": 2.5, "zorder": 10},
    "gorgo-static": {"color": "#2d6a9f", "lw": 1.8, "zorder": 9},
    "prefix-cache": {"color": "#7ab5e0", "lw": 1.4, "zorder": 5},
    "simple-session-affinity": {"color": "#b0b0b0", "lw": 1.2, "zorder": 4},
    "random": {"color": "#d0d0d0", "lw": 1.2, "zorder": 3},
}

WINDOW = 64
STEP = 16
EWMA_ALPHA = 0.06


def _load_cache_ratios(trace_dir: Path, run_prefix: str, policy: str) -> list[dict]:
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
        ratio = target_cache / req_tokens if req_tokens > 0 else 0

        rows.append(
            {
                "elapsed_min": (mono - min_mono) / 60.0,
                "cache_ratio": ratio,
            }
        )
    return rows


def _rolling_raw(rows: list[dict]) -> tuple[list[float], list[float]]:
    ts, vals = [], []
    for i in range(0, len(rows) - WINDOW, STEP):
        chunk = rows[i : i + WINDOW]
        raw = np.mean([s["cache_ratio"] for s in chunk]) * 100
        ts.append(chunk[len(chunk) // 2]["elapsed_min"])
        vals.append(raw)
    return ts, vals


def _ewma_direct(rows: list[dict]) -> tuple[list[float], list[float]]:
    ts, vals = [], []
    prev = None
    for s in rows:
        v = s["cache_ratio"] * 100
        if prev is None:
            prev = v
        else:
            prev = EWMA_ALPHA * v + (1 - EWMA_ALPHA) * prev
        ts.append(s["elapsed_min"])
        vals.append(prev)
    return ts, vals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    trace_dir = args.results_dir / "proxy_traces"
    policies = ["gorgo-hillclimb", "gorgo-static"]

    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(6, 4.5))

    for policy in policies:
        rows = _load_cache_ratios(trace_dir, args.run_prefix, policy)
        if not rows:
            continue
        style = POLICY_STYLE[policy]

        ts_raw, vals_raw = _rolling_raw(rows)

        ts_smooth = ts_raw
        vals_smooth = []
        prev = None
        for v in vals_raw:
            if prev is None:
                prev = v
            else:
                prev = EWMA_ALPHA * v + (1 - EWMA_ALPHA) * prev
            vals_smooth.append(prev)
        ax.plot(ts_smooth, vals_smooth, label=policy, **style)

    ax.set_ylabel("Achieved prefix hit rate (%)")
    ax.set_xlabel("Elapsed time (minutes)")
    ax.set_title("Achieved prefix hit rate over W1 tuning window", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    ax.set_xlim(0)
    ax.set_ylim(0, 100)

    sns.despine()
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
