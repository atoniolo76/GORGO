"""Plot hyperparameter convergence from tune.jsonl traces.

Shows how gorgo-hillclimb (online-ES) and gorgo-autotune (fit) evolve
their hyperparameters over time:
  1. prefill_weight and load_weight trajectories
  2. Sigma decay (hillclimb only)
  3. Score / best_score evolution (hillclimb only)
  4. Acceptance rate (hillclimb only)

Usage:
    # From proxy trace directory (after pulling from Modal):
    python scripts/plot_tune_convergence.py \
        --tune-jsonl results/proxy_traces/<run_id>_gorgo-hillclimb/tune.jsonl \
        --out-dir results/analysis \
        --title "GLM5 W1 Stress Hillclimb Convergence"

    # Compare hillclimb and autotune side by side:
    python scripts/plot_tune_convergence.py \
        --tune-jsonl results/proxy_traces/<run_id>_gorgo-hillclimb/tune.jsonl \
        --tune-jsonl2 results/proxy_traces/<run_id>_gorgo-autotune/tune.jsonl \
        --out-dir results/analysis \
        --title "Hillclimb vs Autotune Convergence"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_tune_events(path: Path) -> list[dict]:
    events = []
    for line in path.open():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _get(d: dict, *keys, default=None):
    """Look up the first matching key, falling back across old/new naming."""
    for k in keys:
        if k in d:
            return d[k]
    return default


def _plot_hillclimb(events: list[dict], title: str, out_dir: Path) -> None:
    steps = [e["step"] for e in events]
    samples = [e.get("total_samples", i) for i, e in enumerate(events)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    # Key-driven so this works for both the 3-weight model
    # (prefill_weight/load_weight/rtt_weight) and the 2D model
    # (rtt_weight/queue_weight). Only plot params actually present.
    colors = {
        "prefill_weight": "tab:blue",
        "load_weight": "tab:orange",
        "queue_weight": "tab:red",
        "rtt_weight": "tab:green",
    }
    param_keys = [k for k in colors if any(k in (e.get("best_params") or {}) for e in events)]
    for k in param_keys:
        best = [_get(e.get("best_params") or {}, k, default=np.nan) for e in events]
        ax.plot(samples, best, label=f"{k} (best)", color=colors[k])
        props = [
            (e.get("total_samples", i), (e.get("proposal") or {})[k])
            for i, e in enumerate(events)
            if e.get("proposal") and k in (e.get("proposal") or {})
        ]
        if props:
            ax.scatter(
                [p[0] for p in props],
                [p[1] for p in props],
                s=8,
                alpha=0.3,
                color=colors[k],
                zorder=1,
            )
    ax.set_yscale("log")
    ax.set_xlabel("Total samples")
    ax.set_ylabel("Hyperparameter value")
    ax.set_title("Hyperparameter Trajectories")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    sigma = [e["sigma"] for e in events]
    ax.plot(samples, sigma, color="tab:green")
    ax.set_xlabel("Total samples")
    ax.set_ylabel("Sigma")
    ax.set_title("Mutation Step Size (Sigma)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    scores = [e["score"] for e in events]
    best_scores = [e["best_score"] for e in events]
    ax.plot(samples, scores, label="Window score", alpha=0.6, color="tab:red")
    ax.plot(samples, best_scores, label="Best score", color="tab:purple", linewidth=2)
    ax.set_xlabel("Total samples")
    ax.set_ylabel(f"Score ({events[0].get('objective_metric', 'neg_p95_ttft')})")
    ax.set_title("Objective Score")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    rates = [e.get("success_rate") for e in events]
    valid = [(s, r) for s, r in zip(samples, rates) if r is not None]
    if valid:
        ax.plot([v[0] for v in valid], [v[1] for v in valid], color="tab:brown")
        ax.axhline(y=0.2, color="gray", linestyle="--", alpha=0.5, label="1/5 target")
        ax.legend(fontsize=9)
    ax.set_xlabel("Total samples")
    ax.set_ylabel("Success rate")
    ax.set_title("Rechenberg 1/5 Success Rate")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "tune_convergence_hillclimb.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Hillclimb convergence: {out_path}")


def _plot_fit(events: list[dict], title: str, out_dir: Path) -> None:
    samples = [e.get("total_samples", i) for i, e in enumerate(events)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    ax = axes[0]
    tp_defaults = [_get(e["defaults"], "prefill_rate", "prefill_weight", default=0) for e in events]
    qw_defaults = [_get(e["defaults"], "load_rate", "load_weight", default=0) for e in events]
    ax.plot(samples, tp_defaults, label="prefill_rate (defaults)", color="tab:blue")
    ax.plot(samples, qw_defaults, label="load_rate (defaults)", color="tab:orange")
    ax.set_yscale("log")
    ax.set_xlabel("Total samples")
    ax.set_ylabel("Rate (ms/tok)")
    ax.set_title("Pooled Defaults")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    targets = set()
    for e in events:
        targets.update(e.get("per_target", {}).keys())
    for i, target in enumerate(sorted(targets)):
        short = target.split("-")[-1][:12] if "-" in target else target[:20]
        pt = e.get("per_target", {}).get(target, {})
        tp = [
            _get(
                e.get("per_target", {}).get(target, {}),
                "prefill_rate",
                "prefill_weight",
                default=np.nan,
            )
            for e in events
        ]
        ax.plot(samples, tp, label=f"prefill_rate ({short})", alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("Total samples")
    ax.set_ylabel("prefill_rate (ms/tok)")
    ax.set_title("Per-Replica prefill_rate")
    if targets:
        ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "tune_convergence_fit.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Fit convergence: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tune-jsonl", required=True, help="Path to tune.jsonl")
    parser.add_argument(
        "--tune-jsonl2", default="", help="Optional second tune.jsonl for comparison"
    )
    parser.add_argument("--out-dir", default="results/analysis", help="Output directory")
    parser.add_argument("--title", default="Tune Convergence", help="Plot title")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    events = _load_tune_events(Path(args.tune_jsonl))
    if not events:
        print(f"No tune events in {args.tune_jsonl}")
        return

    mode = events[0].get("mode", "unknown")
    print(f"Loaded {len(events)} tune events (mode={mode}) from {args.tune_jsonl}")

    if mode == "online-es":
        _plot_hillclimb(events, args.title, out_dir)
    elif mode == "fit":
        _plot_fit(events, args.title, out_dir)
    else:
        print(f"Unknown tune mode: {mode}")

    if args.tune_jsonl2:
        events2 = _load_tune_events(Path(args.tune_jsonl2))
        if events2:
            mode2 = events2[0].get("mode", "unknown")
            print(f"Loaded {len(events2)} tune events (mode={mode2}) from {args.tune_jsonl2}")
            if mode2 == "online-es":
                _plot_hillclimb(events2, f"{args.title} (2)", out_dir)
            elif mode2 == "fit":
                _plot_fit(events2, f"{args.title} (2)", out_dir)


if __name__ == "__main__":
    main()
