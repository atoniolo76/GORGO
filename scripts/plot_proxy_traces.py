"""Plot proxy metrics/request traces saved by ``POST /trace/save``.

Usage:
    python scripts/plot_proxy_traces.py \
        --metrics /path/to/metrics.jsonl \
        --requests /path/to/requests.jsonl \
        --out /tmp/proxy-trace.png

The output is a compact three-panel timeline:
  1. per-replica metrics scrape RTT
  2. per-request selected target over time
  3. TTFT over time, colored by target
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from datetime import datetime
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _parse_ts(row: dict) -> datetime:
    return datetime.fromisoformat(row["wall_ts"].replace("Z", "+00:00"))


def _replica_index(urls: list[str]) -> dict[str, int]:
    return {u: i for i, u in enumerate(urls)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", required=True, type=Path)
    p.add_argument("--requests", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    import matplotlib.pyplot as plt

    metrics = _read_jsonl(args.metrics)
    requests = _read_jsonl(args.requests)

    replicas = list(
        OrderedDict.fromkeys(
            [r["replica_url"] for r in metrics if r.get("replica_url")]
            + [r["target"] for r in requests if r.get("target")]
        )
    )
    ridx = _replica_index(replicas)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    for replica in replicas:
        rows = [r for r in metrics if r.get("replica_url") == replica and r.get("ok")]
        if not rows:
            continue
        xs = [_parse_ts(r) for r in rows]
        ys = [float(r.get("scrape_latency_seconds") or 0.0) * 1000.0 for r in rows]
        axes[0].plot(xs, ys, label=f"replica {ridx[replica]}")
    axes[0].set_ylabel("RTT via /metrics (ms)")
    axes[0].legend(loc="upper right", ncols=max(1, min(4, len(replicas))))

    req_rows = [r for r in requests if r.get("target")]
    if req_rows:
        xs = [_parse_ts(r) for r in req_rows]
        ys = [ridx.get(r["target"], -1) for r in req_rows]
        axes[1].scatter(xs, ys, s=10, alpha=0.7)
    axes[1].set_ylabel("Selected replica")
    axes[1].set_yticks(list(range(len(replicas))))
    axes[1].set_yticklabels([f"replica {i}" for i in range(len(replicas))])

    for replica in replicas:
        rows = [r for r in requests if r.get("target") == replica and r.get("ttft_ns") is not None]
        if not rows:
            continue
        xs = [_parse_ts(r) for r in rows]
        ys = [float(r["ttft_ns"]) / 1e9 for r in rows]
        axes[2].scatter(xs, ys, s=12, alpha=0.75, label=f"replica {ridx[replica]}")
    axes[2].set_ylabel("TTFT (s)")
    axes[2].set_xlabel("Wall time")
    axes[2].legend(loc="upper right", ncols=max(1, min(4, len(replicas))))

    fig.autofmt_xdate()
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)


if __name__ == "__main__":
    main()
