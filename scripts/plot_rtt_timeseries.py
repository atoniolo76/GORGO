"""Generate RTT timeseries figure from proxy metrics.jsonl.

Usage:
    python scripts/plot_rtt_timeseries.py \
        --metrics-jsonl /tmp/gorgo_dl/proxy_traces/tune_gorgo-hillclimb-p95/metrics.jsonl \
        --out figures/rtt_timeseries.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paper_style import REGION_COLORS, classify_region, apply_paper_style


def _canonical_region(region: str | None) -> str | None:
    if not region:
        return None
    r = region.strip().lower()
    if "seoul" in r:
        return "Seoul"
    if "frankfurt" in r:
        return "Frankfurt"
    if "ashburn" in r:
        return "Ashburn"
    return None


def _load_explicit_region_map(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    doc = json.loads(Path(path).read_text())
    if not isinstance(doc, dict):
        return {}
    out: dict[str, str] = {}
    for url, region in doc.items():
        if isinstance(url, str) and isinstance(region, str):
            out[url.strip().rstrip("/")] = region
    return out


def _infer_region_map_from_requests(
    requests_jsonl: str | None,
    fleet_regions: list[str],
) -> dict[str, str]:
    if not requests_jsonl or not fleet_regions:
        return {}
    try:
        with open(requests_jsonl) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                snapshot = row.get("candidate_snapshot")
                if not isinstance(snapshot, dict) or not snapshot:
                    continue
                urls = [u.strip().rstrip("/") for u in snapshot.keys()]
                if len(urls) != len(fleet_regions):
                    continue
                return {u: r for u, r in zip(urls, fleet_regions)}
    except Exception:
        return {}
    return {}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--metrics-jsonl", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ewma-alpha", type=float, default=0.3)
    parser.add_argument(
        "--replica-region-map-json",
        default="",
        help="Optional JSON file mapping replica_url -> region",
    )
    parser.add_argument(
        "--requests-jsonl",
        default="",
        help="Optional request trace JSONL for region inference by replica order",
    )
    parser.add_argument(
        "--fleet-regions",
        default="",
        help="Comma-separated region list used with --requests-jsonl",
    )
    parser.add_argument(
        "--title",
        default="Proxy \u2192 Replica Round-Trip Time (W1 Tuning Window)",
        help="Figure title text",
    )
    args = parser.parse_args()

    metrics = []
    with open(args.metrics_jsonl) as f:
        for line in f:
            if line.strip():
                try:
                    metrics.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    metrics = [m for m in metrics if m.get("ok") and m.get("network_rtt_seconds") is not None]
    if not metrics:
        raise SystemExit("No valid metrics with network_rtt_seconds")

    explicit_region_map = _load_explicit_region_map(args.replica_region_map_json or None)
    fleet_regions = [r.strip() for r in (args.fleet_regions or "").split(",") if r.strip()]
    inferred_region_map = _infer_region_map_from_requests(
        args.requests_jsonl or None, fleet_regions
    )

    t0 = min(m["monotonic_s"] for m in metrics)

    replicas: dict[str, list[dict]] = {}
    for m in metrics:
        replicas.setdefault(m["replica_url"].strip().rstrip("/"), []).append(m)

    region_map = {}
    for url, ms in replicas.items():
        med = np.median([m["network_rtt_seconds"] * 1000 for m in ms])
        trace_region = next((m.get("replica_region") for m in ms if m.get("replica_region")), None)
        raw_region = (
            trace_region or explicit_region_map.get(url) or inferred_region_map.get(url) or None
        )
        canonical = _canonical_region(raw_region) or classify_region(med)
        replica_key = next((m.get("replica_key") for m in ms if m.get("replica_key")), None)
        region_map[url] = {
            "canonical_region": canonical,
            "raw_region": raw_region,
            "median_ms": med,
            "replica_key": replica_key,
        }

    fig, ax = plt.subplots(figsize=(10, 4))

    for url in sorted(replicas, key=lambda u: region_map[u]["median_ms"], reverse=True):
        meta = region_map[url]
        region = meta["canonical_region"]
        med = meta["median_ms"]
        color = REGION_COLORS.get(region, "#6f8094")
        ms = sorted(replicas[url], key=lambda m: m["monotonic_s"])
        times = [(m["monotonic_s"] - t0) / 60.0 for m in ms]
        rtts = [m["network_rtt_seconds"] * 1000 for m in ms]
        std = np.std(rtts)

        ax.plot(times, rtts, "-", linewidth=0.5, alpha=0.3, color=color)
        ewma = [rtts[0]]
        for v in rtts[1:]:
            ewma.append(args.ewma_alpha * v + (1 - args.ewma_alpha) * ewma[-1])
        ax.plot(
            times,
            ewma,
            linewidth=2.5,
            color=color,
            label=(
                f"{region}"
                + (f" ({meta['raw_region']})" if meta["raw_region"] else "")
                + (f" [{meta['replica_key']}]" if meta["replica_key"] else "")
                + f"  (mean {med:.0f} \u00b1 {std:.0f} ms)"
            ),
        )
        ax.axhline(med, color=color, linestyle="--", linewidth=0.8, alpha=0.4)

    ax.set_xlabel("Elapsed time (minutes)", fontsize=10)
    ax.set_ylabel("RTT (ms)", fontsize=10)
    ax.set_title(args.title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, None)
    apply_paper_style(ax)

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
