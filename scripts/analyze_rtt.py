"""Analyze per-replica network RTT from proxy metrics traces.

Reads a metrics.jsonl trace and produces:
  1. Per-replica RTT statistics (mean, std, min, max, CV)
  2. RTT over time CSV (for plotting)
  3. RTT vs scrape latency comparison
  4. Cross-replica spread summary

Usage:
    python scripts/analyze_rtt.py \
        --metrics results/proxy_traces/<run>_random/metrics.jsonl \
        --out-dir results/analysis

    # Compare W1 vs W2:
    python scripts/analyze_rtt.py \
        --metrics results/proxy_traces/<w1_run>_random/metrics.jsonl \
        --metrics2 results/proxy_traces/<w2_run>_random/metrics.jsonl \
        --label "W1" --label2 "W2" \
        --out-dir results/analysis
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _load_metrics(path: Path) -> dict[str, list[dict]]:
    per_replica: dict[str, list[dict]] = defaultdict(list)
    min_mono: float | None = None
    for line in path.open():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("kind") != "metrics" or not r.get("ok"):
            continue
        mono = r.get("monotonic_s")
        if mono is None:
            continue
        if min_mono is None:
            min_mono = mono
        url = r.get("replica_url", "?")
        short = url.split("//", 1)[-1].split(".", 1)[0][:30] if "//" in url else url[:30]
        per_replica[short].append(
            {
                "elapsed_s": mono - min_mono,
                "elapsed_min": (mono - min_mono) / 60.0,
                "network_rtt_ms": r["network_rtt_seconds"] * 1000
                if r.get("network_rtt_seconds")
                else None,
                "scrape_ms": r["scrape_latency_seconds"] * 1000
                if r.get("scrape_latency_seconds")
                else None,
                "num_running_reqs": r.get("num_running_reqs"),
                "num_queue_reqs": r.get("num_queue_reqs"),
                "region": r.get("region", "?"),
                "url": url,
            }
        )
    return dict(per_replica)


def _stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": 0, "std": 0, "min": 0, "max": 0, "cv_pct": 0}
    a = np.array(values)
    return {
        "n": len(a),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "min": float(a.min()),
        "max": float(a.max()),
        "cv_pct": float(a.std() / a.mean() * 100) if a.mean() > 0 else 0,
    }


def _analyze(per_replica: dict[str, list[dict]], label: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 80}")
    print(f"  {label} — Per-replica network RTT")
    print(f"{'=' * 80}")

    summary_rows = []
    all_rtt_rows = []

    for replica in sorted(
        per_replica,
        key=lambda r: np.mean(
            [s["network_rtt_ms"] for s in per_replica[r] if s["network_rtt_ms"] is not None] or [0]
        ),
    ):
        samples = per_replica[replica]
        rtts = [s["network_rtt_ms"] for s in samples if s["network_rtt_ms"] is not None]
        scrapes = [s["scrape_ms"] for s in samples if s["scrape_ms"] is not None]
        if not rtts:
            continue
        region = samples[0].get("region", "?")
        url = samples[0].get("url", "?")
        s = _stats(rtts)
        ss = _stats(scrapes)

        print(f"\n  {replica}")
        print(f"    region: {region}  |  {s['n']} samples")
        print(
            f"    RTT:     mean={s['mean']:.0f}ms  std={s['std']:.0f}ms  "
            f"min={s['min']:.0f}ms  max={s['max']:.0f}ms  CV={s['cv_pct']:.1f}%"
        )
        print(
            f"    Scrape:  mean={ss['mean']:.0f}ms  std={ss['std']:.0f}ms  "
            f"min={ss['min']:.0f}ms  max={ss['max']:.0f}ms"
        )
        print(
            f"    Scrape overhead vs RTT: +{ss['mean'] - s['mean']:.0f}ms avg "
            f"({ss['mean'] / s['mean']:.1f}× RTT)"
            if s["mean"] > 0
            else ""
        )

        summary_rows.append(
            {
                "replica": replica,
                "region": region,
                "rtt_mean_ms": round(s["mean"], 1),
                "rtt_std_ms": round(s["std"], 1),
                "rtt_min_ms": round(s["min"], 1),
                "rtt_max_ms": round(s["max"], 1),
                "rtt_cv_pct": round(s["cv_pct"], 1),
                "scrape_mean_ms": round(ss["mean"], 1),
                "scrape_max_ms": round(ss["max"], 1),
                "n_samples": s["n"],
            }
        )

        for sample in samples:
            if sample["network_rtt_ms"] is not None:
                all_rtt_rows.append(
                    {
                        "replica": replica,
                        "region": region,
                        "elapsed_min": round(sample["elapsed_min"], 3),
                        "network_rtt_ms": round(sample["network_rtt_ms"], 1),
                        "scrape_ms": round(sample["scrape_ms"], 1) if sample["scrape_ms"] else None,
                        "num_running_reqs": sample.get("num_running_reqs"),
                    }
                )

    # Cross-replica summary
    means = [(r["replica"], r["rtt_mean_ms"]) for r in summary_rows]
    if len(means) >= 2:
        closest = min(means, key=lambda x: x[1])
        farthest = max(means, key=lambda x: x[1])
        spread = farthest[1] - closest[1]
        ratio = farthest[1] / closest[1] if closest[1] > 0 else 0
        print(f"\n  Cross-replica spread: {spread:.0f}ms ({ratio:.1f}× ratio)")
        print(f"    Closest:  {closest[0]}  {closest[1]:.0f}ms")
        print(f"    Farthest: {farthest[0]}  {farthest[1]:.0f}ms")

    # Write CSVs
    slug = label.lower().replace(" ", "_").replace("(", "").replace(")", "")
    summary_path = out_dir / f"rtt_summary_{slug}.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\n  Summary CSV: {summary_path}")

    timeseries_path = out_dir / f"rtt_timeseries_{slug}.csv"
    with open(timeseries_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rtt_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rtt_rows)
    print(f"  Timeseries CSV: {timeseries_path}")

    # Plot
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        for replica in sorted(
            per_replica,
            key=lambda r: np.mean(
                [s["network_rtt_ms"] for s in per_replica[r] if s["network_rtt_ms"] is not None]
                or [0]
            ),
        ):
            samples = per_replica[replica]
            times = [s["elapsed_min"] for s in samples if s["network_rtt_ms"] is not None]
            rtts = [s["network_rtt_ms"] for s in samples if s["network_rtt_ms"] is not None]
            scrapes = [
                s["scrape_ms"]
                for s in samples
                if s["network_rtt_ms"] is not None and s["scrape_ms"] is not None
            ]
            scrape_times = [
                s["elapsed_min"]
                for s in samples
                if s["network_rtt_ms"] is not None and s["scrape_ms"] is not None
            ]

            region = samples[0].get("region", "?")
            short_label = f"{replica[:15]}… ({region})"

            ax1.plot(times, rtts, linewidth=1.5, alpha=0.9, label=short_label)
            ax2.plot(times, rtts, linewidth=1.5, alpha=0.9, label=f"RTT {short_label}")
            if scrapes:
                ax2.plot(
                    scrape_times,
                    scrapes,
                    linewidth=1,
                    alpha=0.5,
                    linestyle="--",
                    label=f"Scrape {short_label}",
                )

        ax1.set_ylabel("Network RTT (ms)")
        ax1.set_title(f"{label} — Per-replica network RTT over time")
        ax1.legend(fontsize=8, loc="upper right")
        ax1.grid(alpha=0.3)

        ax2.set_xlabel("Elapsed time (minutes)")
        ax2.set_ylabel("Latency (ms)")
        ax2.set_title("RTT probe (solid) vs /metrics scrape (dashed)")
        ax2.legend(fontsize=7, loc="upper right", ncols=2)
        ax2.grid(alpha=0.3)

        fig.tight_layout()
        plot_path = out_dir / f"rtt_over_time_{slug}.png"
        fig.savefig(plot_path, dpi=180)
        plt.close(fig)
        print(f"  Plot: {plot_path}")
    except ImportError:
        print("  (matplotlib not available, skipping plot)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--metrics2", type=Path, default=None)
    parser.add_argument("--label", default="Run")
    parser.add_argument("--label2", default="Run 2")
    parser.add_argument("--out-dir", default="results/analysis", type=Path)
    args = parser.parse_args()

    data1 = _load_metrics(args.metrics)
    _analyze(data1, args.label, args.out_dir)

    if args.metrics2:
        data2 = _load_metrics(args.metrics2)
        _analyze(data2, args.label2, args.out_dir)


if __name__ == "__main__":
    main()
