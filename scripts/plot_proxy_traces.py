"""Plot proxy metrics/request traces saved by ``POST /trace/save``.

Single-trace usage:
    python scripts/plot_proxy_traces.py \
        --metrics /path/to/metrics.jsonl \
        --requests /path/to/requests.jsonl \
        --out /tmp/proxy-trace.png

Multi-policy usage:
    python scripts/plot_proxy_traces.py \
        --matrix-manifest results/policy_matrix_sweep/...json \
        --local-trace-root /tmp/downloaded_results \
        --out /tmp/policy-comparison.png

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


def _maybe_local(path: str, root: Path | None) -> Path:
    p = Path(path)
    if p.exists():
        return p
    if root is None or not p.is_absolute():
        return p
    parts = p.parts[1:]
    for i in range(len(parts)):
        candidate = root.joinpath(*parts[i:])
        if candidate.exists():
            return candidate
    return p


def _plot_single(metrics_path: Path, requests_path: Path, out: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

    metrics = _read_jsonl(metrics_path)
    requests = _read_jsonl(requests_path)

    replicas = list(
        OrderedDict.fromkeys(
            [r["replica_url"] for r in metrics if r.get("replica_url")]
            + [r["target"] for r in requests if r.get("target")]
        )
    )
    ridx = _replica_index(replicas)
    n_replicas = len(replicas)
    palette = sns.color_palette("Blues_d", n_colors=max(n_replicas, 3))

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    for ri, replica in enumerate(replicas):
        rows = [r for r in metrics if r.get("replica_url") == replica and r.get("ok")]
        if not rows:
            continue
        xs = [_parse_ts(r) for r in rows]
        ys = [float(r.get("scrape_latency_seconds") or 0.0) * 1000.0 for r in rows]
        axes[0].plot(xs, ys, label=f"replica {ridx[replica]}", color=palette[ri], linewidth=1.5)
    axes[0].set_ylabel("RTT via /metrics (ms)")
    axes[0].legend(loc="upper right", ncols=max(1, min(4, n_replicas)))

    req_rows = [r for r in requests if r.get("target")]
    if req_rows:
        xs = [_parse_ts(r) for r in req_rows]
        ys = [ridx.get(r["target"], -1) for r in req_rows]
        scatter_colors = [palette[ridx.get(r["target"], 0)] for r in req_rows]
        axes[1].scatter(xs, ys, s=10, alpha=0.7, c=scatter_colors)
    axes[1].set_ylabel("Selected replica")
    axes[1].set_yticks(list(range(n_replicas)))
    axes[1].set_yticklabels([f"replica {i}" for i in range(n_replicas)])

    for ri, replica in enumerate(replicas):
        rows = [r for r in requests if r.get("target") == replica and r.get("ttft_ns") is not None]
        if not rows:
            continue
        xs = [_parse_ts(r) for r in rows]
        ys = [float(r["ttft_ns"]) / 1e9 for r in rows]
        axes[2].scatter(
            xs, ys, s=12, alpha=0.75, label=f"replica {ridx[replica]}", color=palette[ri]
        )
    axes[2].set_ylabel("TTFT (s)")
    axes[2].set_xlabel("Wall time")
    axes[2].legend(loc="upper right", ncols=max(1, min(4, n_replicas)))

    fig.autofmt_xdate()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")


def _iter_policy_results(manifest: dict):
    """Yield per-policy result dicts from either an aggregate sweep manifest
    (``results[].manifest.results[]``) or a per-trace matrix manifest
    (``results[]`` directly)."""
    for result in manifest.get("results", []):
        nested = result.get("manifest")
        if isinstance(nested, dict) and "results" in nested:
            yield from _iter_policy_results(nested)
            continue
        yield result


def _load_policy_traces(matrix_manifest_path: Path, local_trace_root: Path | None) -> list[dict]:
    manifest = json.loads(matrix_manifest_path.read_text())
    out = []
    for result in _iter_policy_results(manifest):
        if result.get("error"):
            continue
        label = result.get("label") or result.get("policy") or result.get("run_id")
        trace_paths = (result.get("trace") or {}).get("paths") or {}
        requests_path = trace_paths.get("requests_path")
        metrics_path = trace_paths.get("metrics_path")
        if not requests_path or not metrics_path:
            continue
        req_path = _maybe_local(requests_path, local_trace_root)
        met_path = _maybe_local(metrics_path, local_trace_root)
        if not req_path.exists() or not met_path.exists():
            continue
        out.append(
            {
                "label": label,
                "requests": _read_jsonl(req_path),
                "metrics": _read_jsonl(met_path),
                "requests_path": str(req_path),
                "metrics_path": str(met_path),
            }
        )
    return out


def _request_index(rows: list[dict]) -> dict[str, int]:
    ids = sorted({r["request_id"] for r in rows if r.get("request_id")})
    return {rid: i for i, rid in enumerate(ids)}


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    return ys[min(len(ys) - 1, int(len(ys) * p))]


def _plot_multi_policy(
    matrix_manifest_path: Path, local_trace_root: Path | None, out: Path
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

    policy_traces = _load_policy_traces(matrix_manifest_path, local_trace_root)
    if not policy_traces:
        raise SystemExit(
            "No policy traces found. Download /results/proxy_traces locally or set --local-trace-root."
        )

    all_requests = [r for t in policy_traces for r in t["requests"]]
    ridx = _request_index(all_requests)
    labels = [t["label"] for t in policy_traces]
    lidx = {label: i for i, label in enumerate(labels)}
    n_policies = len(policy_traces)
    palette = sns.color_palette("Blues_d", n_colors=max(n_policies, 3))

    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)

    # Panel 1: policy-level TTFT points over common request index
    for idx, t in enumerate(policy_traces):
        rows = [
            r for r in t["requests"] if r.get("request_id") in ridx and r.get("ttft_ns") is not None
        ]
        xs = [ridx[r["request_id"]] for r in rows]
        ys = [float(r["ttft_ns"]) / 1e9 for r in rows]
        axes[0].scatter(xs, ys, s=8, alpha=0.35, label=t["label"], color=palette[idx])
    axes[0].set_ylabel("TTFT (s)")
    axes[0].legend(loc="upper right", ncols=max(1, min(4, n_policies)))

    # Panel 2: rolling per-policy p95 over 100 request buckets
    bucket = 100
    for idx, t in enumerate(policy_traces):
        rows = [
            r for r in t["requests"] if r.get("request_id") in ridx and r.get("ttft_ns") is not None
        ]
        by_bucket: dict[int, list[float]] = {}
        for r in rows:
            b = ridx[r["request_id"]] // bucket
            by_bucket.setdefault(b, []).append(float(r["ttft_ns"]) / 1e9)
        xs = [b * bucket for b in sorted(by_bucket)]
        ys = [_percentile(by_bucket[b], 0.95) for b in sorted(by_bucket)]
        axes[1].plot(
            xs, ys, marker="o", markersize=4, label=t["label"], color=palette[idx], linewidth=1.5
        )
    axes[1].set_ylabel("Bucket p95 TTFT (s)")
    axes[1].legend(loc="upper right", ncols=max(1, min(4, n_policies)))

    # Panel 3: selected replica index per policy over request index
    for idx, t in enumerate(policy_traces):
        rows = [r for r in t["requests"] if r.get("request_id") in ridx and r.get("target")]
        xs = [ridx[r["request_id"]] for r in rows]
        ys = [lidx[t["label"]]] * len(rows)
        axes[2].scatter(xs, ys, s=8, alpha=0.4, label=t["label"], color=palette[idx])
    axes[2].set_yticks(list(range(len(labels))))
    axes[2].set_yticklabels(labels)
    axes[2].set_ylabel("Policy")
    axes[2].set_xlabel("Request index (shared Mooncake request_id order)")

    fig.suptitle(matrix_manifest_path.name)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", type=Path)
    p.add_argument("--requests", type=Path)
    p.add_argument("--matrix-manifest", type=Path)
    p.add_argument("--local-trace-root", type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    if args.matrix_manifest:
        _plot_multi_policy(args.matrix_manifest, args.local_trace_root, args.out)
        return
    if not args.metrics or not args.requests:
        raise SystemExit("pass either --matrix-manifest or both --metrics and --requests")
    _plot_single(args.metrics, args.requests, args.out)


if __name__ == "__main__":
    main()
