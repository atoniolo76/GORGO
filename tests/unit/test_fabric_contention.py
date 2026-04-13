"""Fabric contention: the engine must feed the cost model a
concurrent-bytes total that sums overlapping in-flight KV transfers.

A lone transfer (nothing else in flight) must get its own-bytes total;
a transfer that arrives before a prior transfer has finished on the
fabric must get (own + prior) and therefore a larger kv_transport_ms.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from routing_harness import policies  # noqa: F401 — register built-ins
from routing_harness.cluster import ClusterState
from routing_harness.core import CostBreakdown, Phase, PodSpec, Request
from routing_harness.kv_cache import KVCacheState, PrefixEntry, enumerate_prefix_hashes
from routing_harness.policy import get_policy
from routing_harness.simulator.engine import EngineConfig, SimulationEngine
from routing_harness.simulator.metrics import MetricsCollector


@dataclass
class RecordingCostModel:
    """Deterministic cost model that records each call for inspection.

    Returns a fixed `kv_transport_ms` regardless of bytes, so the fabric
    heap keeps a transfer "active" long enough for a later request's
    arrival to overlap. Other cost components are zero so the engine
    doesn't depend on compute/queueing math.
    """

    fixed_kv_transport_ms: float = 1000.0
    calls: list[dict] = field(default_factory=list)

    def estimate(
        self,
        request,
        decision,
        cluster,
        kv_cache,
        cached_prefix_tokens,
        kv_transport_bytes,
        concurrent_kv_transport_bytes=None,
    ):
        self.calls.append({
            "request_id": request.request_id,
            "kv_transport_bytes": kv_transport_bytes,
            "concurrent_kv_transport_bytes": concurrent_kv_transport_bytes,
        })
        kv_ms = self.fixed_kv_transport_ms if kv_transport_bytes > 0 else 0.0
        return CostBreakdown(
            routing_ms=0.0,
            queueing_ms=0.0,
            compute_prefill_ms=1.0,
            compute_decode_ms=1.0,
            network_ms=0.0,
            kv_transport_ms=kv_ms,
        )


def _pd_cluster_and_cache() -> tuple[ClusterState, KVCacheState]:
    specs = [
        PodSpec("pf0", Phase.PREFILL, 1, 32 * 1024 * 1024, 4, 0, peer_ids=("dc0",)),
        PodSpec("dc0", Phase.DECODE, 1, 32 * 1024 * 1024, 0, 8, peer_ids=("pf0",)),
    ]
    return (
        ClusterState.from_specs(specs),
        KVCacheState.from_specs({s.pod_id: s.kv_cache_bytes for s in specs}),
    )


def _engine(cluster, cache, cost_model, network):
    return SimulationEngine(
        cluster=cluster,
        kv_cache=cache,
        policy=get_policy("pd"),
        cost_model=cost_model,
        network=network,
        config=EngineConfig(kv_ewma_alpha=0.2, block_size=16, initial_warm_latency_ms=1.0),
        metrics=MetricsCollector(),
    )


def test_concurrent_pd_handoff_sees_larger_concurrent_bytes(network_params):
    cluster, cache = _pd_cluster_and_cache()
    cost = RecordingCostModel(fixed_kv_transport_ms=1000.0)
    engine = _engine(cluster, cache, cost, network_params)

    trace = [
        Request("r0", "s0", 0.0, tuple(range(64)), 4),
        Request("r1", "s1", 0.05, tuple(range(100, 164)), 4),  # 50ms later
    ]
    engine.run(trace)

    assert len(cost.calls) == 2
    r0, r1 = cost.calls
    assert r0["kv_transport_bytes"] > 0 and r1["kv_transport_bytes"] > 0
    # First transfer: fabric empty, concurrent == own bytes.
    assert r0["concurrent_kv_transport_bytes"] == r0["kv_transport_bytes"]
    # Second transfer arrives 50ms after the first — well before the
    # first's 1000ms fabric completion. Concurrent == r0 + r1 bytes.
    assert r1["concurrent_kv_transport_bytes"] == (
        r0["kv_transport_bytes"] + r1["kv_transport_bytes"]
    )


def test_serial_pd_handoff_sees_no_contention(network_params):
    cluster, cache = _pd_cluster_and_cache()
    cost = RecordingCostModel(fixed_kv_transport_ms=10.0)
    engine = _engine(cluster, cache, cost, network_params)

    # Second arrival is 5s later — long after the 10ms fabric transfer
    # of the first has drained.
    trace = [
        Request("r0", "s0", 0.0, tuple(range(64)), 4),
        Request("r1", "s1", 5.0, tuple(range(100, 164)), 4),
    ]
    engine.run(trace)
    r0, r1 = cost.calls
    assert r0["concurrent_kv_transport_bytes"] == r0["kv_transport_bytes"]
    assert r1["concurrent_kv_transport_bytes"] == r1["kv_transport_bytes"]


def test_no_transfer_means_no_fabric_contention_passed(network_params):
    # Colocated pod so no PD handoff, no peer pull — kv_transport_bytes
    # is zero and we must pass `concurrent_kv_transport_bytes=None`.
    specs = [PodSpec("p0", Phase.BOTH, 1, 8 * 1024 * 1024, 4, 8)]
    cluster = ClusterState.from_specs(specs)
    cache = KVCacheState.from_specs({"p0": specs[0].kv_cache_bytes})
    cost = RecordingCostModel()
    engine = SimulationEngine(
        cluster=cluster,
        kv_cache=cache,
        policy=get_policy("random", seed=0),
        cost_model=cost,
        network=network_params,
        config=EngineConfig(kv_ewma_alpha=0.2, block_size=16, initial_warm_latency_ms=1.0),
        metrics=MetricsCollector(),
    )
    engine.run([Request("r0", "s0", 0.0, tuple(range(32)), 4)])
    assert cost.calls[0]["kv_transport_bytes"] == 0
    assert cost.calls[0]["concurrent_kv_transport_bytes"] is None
