"""Rolling p95 TTFT over the W1 tuning window.

Shows the ES objective (negative p95 TTFT) improving over time for
gorgo-hillclimb while baselines stay flat. Single panel, clean.

Usage:
    python scripts/plot_rolling_p95.py \
        --run-prefix abstract_night_000_glm5_0030_to_0100 \
        --results-dir results \
        --out paper/figures/rolling_p95.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

POLICY_STYLE = {
    "gorgo-hillclimb": {"color": "#1b3a5c", "lw": 2.5, "alpha": 1.0, "zorder": 10},
    "gorgo-static": {"color": "#2d6a9f", "lw": 2.0, "alpha": 0.85, "zorder": 9},
    "prefix-cache": {"color": "#7ab5e0", "lw": 1.5, "alpha": 0.7, "zorder": 5},
    "simple-session-affinity": {"color": "#b0b0b0", "lw": 1.3, "alpha": 0.6, "zorder": 4},
    "random": {"color": "#d0d0d0", "lw": 1.3, "alpha": 0.5, "zorder": 3},
}

WINDOW = 64
STEP = 16


def _load_ttfts(trace_dir: Path, run_prefix: str, policy: str) -> list[dict]:
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
        ttft_ns = r.get("ttft_ns")
        if mono is None or ttft_ns is None:
            continue
        if min_mono is None:
            min_mono = mono
        rows.append(
            {
                "elapsed_min": (mono - min_mono) / 60.0,
                "ttft_s": ttft_ns / 1e9,
            }
        )
    return rows


def _rolling_p95(rows: list[dict]) -> tuple[list[float], list[float]]:
    ts, vals = [], []
    for i in range(0, len(rows) - WINDOW, STEP):
        chunk = rows[i : i + WINDOW]
        ttfts = sorted(s["ttft_s"] for s in chunk)
        p95_idx = min(len(ttfts) - 1, int(0.95 * len(ttfts)))
        ts.append(chunk[len(chunk) // 2]["elapsed_min"])
        vals.append(ttfts[p95_idx])
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

    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(8, 4))

    for policy in policies:
        rows = _load_ttfts(trace_dir, args.run_prefix, policy)
        if not rows:
            continue
        style = POLICY_STYLE[policy]
        ts, vals = _rolling_p95(rows)
        ax.plot(ts, vals, label=policy, **style)

    ax.set_ylabel("Rolling p95 TTFT (s)")
    ax.set_xlabel("Elapsed time (minutes)")
    ax.set_title("ES objective (p95 TTFT) over W1 tuning window", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax.set_xlim(0)
    ax.set_ylim(bottom=0)

    ax.annotate(
        "ES converges\n(~2 min)",
        xy=(2.5, 1.2),
        fontsize=8,
        color="#1b3a5c",
        ha="center",
        style="italic",
    )

    sns.despine()
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
