"""Sweep runner + result writer.

A run = (policy, policy_config, cluster_topology, cost_params,
workload_params, seed). A sweep expands Cartesian combinations. Results
are written per-run as:

    results/<run_id>/config.json     full snapshot of inputs
    results/<run_id>/metrics.json    summary metrics
    results/<run_id>/records.csv     per-request rows (plot-ready)

Plus an index file `results/index.json` listing every run for later
aggregation in GRD reports.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from ..cluster import ClusterState
from ..core import PodSpec
from ..cost_model import (
    AnalyticCostModel,
    ComputeParams,
    InstrumentedCostModel,
    NetworkParams,
    SchedulerParams,
)
from ..kv_cache import KVCacheState
from ..policy import get_policy
from .engine import EngineConfig, SimulationEngine
from .metrics import MetricsCollector


def _to_jsonable(obj: Any) -> Any:
    # Exclude private (leading-underscore) fields so runtime state like
    # policy `_bindings` maps cannot leak into config snapshots and
    # silently change the content-addressed run_id.
    if is_dataclass(obj):
        return {
            k: _to_jsonable(v)
            for k, v in asdict(obj).items()
            if not k.startswith("_")
        }
    if isinstance(obj, dict):
        return {
            k: _to_jsonable(v)
            for k, v in obj.items()
            if not (isinstance(k, str) and k.startswith("_"))
        }
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _run_id(config_snapshot: dict) -> str:
    payload = json.dumps(_to_jsonable(config_snapshot), sort_keys=True).encode()
    return hashlib.blake2b(payload, digest_size=8).hexdigest()


def build_cluster_and_cache(topology: list[PodSpec]) -> tuple[ClusterState, KVCacheState]:
    cluster = ClusterState.from_specs(topology)
    cache = KVCacheState.from_specs({s.pod_id: s.kv_cache_bytes for s in topology})
    return cluster, cache


def run_single(
    policy_id: str,
    policy_kwargs: dict,
    topology: list[PodSpec],
    trace,
    compute: ComputeParams,
    network: NetworkParams,
    scheduler: SchedulerParams,
    engine_cfg: EngineConfig,
    output_root: Path | None = None,
    run_meta: dict | None = None,
    observations: dict[str, float] | None = None,
) -> dict:
    cluster, cache = build_cluster_and_cache(topology)
    policy = get_policy(policy_id, **policy_kwargs)
    analytic = AnalyticCostModel(compute=compute, network=network, scheduler=scheduler)
    cost_model = (
        InstrumentedCostModel.from_observations(analytic, observations)
        if observations
        else analytic
    )
    engine = SimulationEngine(
        cluster=cluster,
        kv_cache=cache,
        policy=policy,
        cost_model=cost_model,
        network=network,
        config=engine_cfg,
        metrics=MetricsCollector(),
    )
    metrics = engine.run(trace)
    summary = metrics.summary()

    snapshot = {
        "policy_id": policy_id,
        "policy_kwargs": policy_kwargs,
        "topology": [asdict(s) for s in topology],
        "compute": asdict(compute),
        "network": asdict(network),
        "scheduler": asdict(scheduler),
        "engine": asdict(engine_cfg),
        "trace": trace.describe() if hasattr(trace, "describe") else {},
        "meta": run_meta or {},
        "observations": dict(sorted(observations.items())) if observations else {},
    }
    rid = _run_id(snapshot)
    result = {"run_id": rid, "config": snapshot, "metrics": summary}

    if output_root is not None:
        out = Path(output_root) / rid
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps(_to_jsonable(snapshot), indent=2))
        (out / "metrics.json").write_text(json.dumps(_to_jsonable(summary), indent=2))
        _write_records_csv(out / "records.csv", metrics)
        _append_index(Path(output_root) / "index.json", rid, snapshot, summary)
    return result


def _write_records_csv(path: Path, metrics: MetricsCollector) -> None:
    fields = [
        "request_id",
        "session_id",
        "prefill_pod",
        "decode_pod",
        "total_ms",
        "routing_ms",
        "queueing_ms",
        "compute_prefill_ms",
        "compute_decode_ms",
        "network_ms",
        "kv_transport_ms",
        "cached_prefix_tokens",
        "kv_transport_bytes",
        "reuse_available_blocks",
        "reuse_captured_blocks",
        "migrated",
    ]
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(fields)
        for r in metrics.records:
            c = r.cost
            w.writerow([
                r.request.request_id,
                r.request.session_id,
                r.decision.prefill_pod_id,
                r.decision.decode_pod_id,
                f"{c.total_ms:.3f}",
                f"{c.routing_ms:.3f}",
                f"{c.queueing_ms:.3f}",
                f"{c.compute_prefill_ms:.3f}",
                f"{c.compute_decode_ms:.3f}",
                f"{c.network_ms:.3f}",
                f"{c.kv_transport_ms:.3f}",
                r.cached_prefix_tokens,
                r.kv_transport_bytes,
                r.reuse_available_blocks,
                r.reuse_captured_blocks,
                int(r.migrated),
            ])


def _append_index(path: Path, run_id: str, snapshot: dict, summary: dict) -> None:
    existing: list[dict] = []
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except json.JSONDecodeError:
            existing = []
    existing.append({
        "run_id": run_id,
        "policy_id": snapshot["policy_id"],
        "p50_ms": summary.get("latency_ms", {}).get("p50"),
        "p95_ms": summary.get("latency_ms", {}).get("p95"),
        "p99_ms": summary.get("latency_ms", {}).get("p99"),
        "ttft_p50_ms": summary.get("ttft_ms", {}).get("p50"),
        "ttft_p95_ms": summary.get("ttft_ms", {}).get("p95"),
        "ttft_p99_ms": summary.get("ttft_ms", {}).get("p99"),
        "hit_rate": summary.get("kv", {}).get("hit_rate"),
        "capture_rate_micro": summary.get("kv", {}).get("capture_rate_micro"),
        "capture_rate_macro": summary.get("kv", {}).get("capture_rate_macro"),
        "skew": summary.get("load", {}).get("skew"),
    })
    path.write_text(json.dumps(existing, indent=2))
