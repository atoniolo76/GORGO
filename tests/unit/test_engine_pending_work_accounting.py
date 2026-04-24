"""Engine pending_work_ms accounting must use service_ms, not total_ms.

cost.total_ms includes queueing_ms (an M/M/1 wait derived from
pod.active_prefill). Folding queueing_ms back into pending_work_ms
would nest the wait estimate on top of the service-time sum the
Preble L_i signal is supposed to represent, inflating the magnitude
the th_bal parameter is calibrated against.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from routing_harness import policies  # noqa: F401 — register built-ins
from routing_harness.cluster import ClusterState
from routing_harness.core import CostBreakdown, Phase, PodSpec, Request
from routing_harness.kv_cache import KVCacheState
from routing_harness.policy import get_policy
from routing_harness.simulator.engine import EngineConfig, SimulationEngine
from routing_harness.simulator.metrics import MetricsCollector


@dataclass
class FixedCostModel:
    """Returns a fixed CostBreakdown with both queueing and service cost."""

    queueing_ms: float
    compute_prefill_ms: float
    compute_decode_ms: float
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
        pod = cluster.get(decision.prefill_pod_id)
        self.calls.append({
            "request_id": request.request_id,
            "active_prefill_at_decide": pod.active_prefill,
            "pending_work_ms_at_decide": pod.pending_work_ms,
        })
        return CostBreakdown(
            routing_ms=0.0,
            queueing_ms=self.queueing_ms,
            compute_prefill_ms=self.compute_prefill_ms,
            compute_decode_ms=self.compute_decode_ms,
            network_ms=0.0,
            kv_transport_ms=0.0,
        )


def _single_pod_engine(cost_model, network):
    specs = [PodSpec("p0", Phase.BOTH, 1, 8 * 1024 * 1024, 4, 8)]
    cluster = ClusterState.from_specs(specs)
    cache = KVCacheState.from_specs({"p0": specs[0].kv_cache_bytes})
    return cluster, SimulationEngine(
        cluster=cluster,
        kv_cache=cache,
        policy=get_policy("random", seed=0),
        cost_model=cost_model,
        network=network,
        config=EngineConfig(kv_ewma_alpha=0.2, block_size=16, initial_warm_latency_ms=1.0),
        metrics=MetricsCollector(),
    )


def test_pending_work_ms_excludes_queueing(network_params):
    """Each arrival bumps pending_work_ms by service_ms (total - queueing),
    not total_ms — otherwise the M/M/1 wait compounds into the load signal.
    Check the value observed at the *next* decide() call."""
    cost = FixedCostModel(queueing_ms=50.0, compute_prefill_ms=10.0, compute_decode_ms=5.0)
    cluster, engine = _single_pod_engine(cost, network_params)

    # Two arrivals close enough in time that the first has not retired.
    trace = [
        Request("r0", "s0", 0.0, tuple(range(32)), 4),
        Request("r1", "s1", 0.001, tuple(range(32)), 4),
    ]
    engine.run(trace)

    # total_ms = 0 + 50 + max(10,0) + 5 + 0 = 65; service_ms = 15.
    # At r1's decide, r0 is still in flight and pending_work_ms must
    # reflect r0's service_ms (15), not total_ms (65).
    assert cost.calls[0]["pending_work_ms_at_decide"] == 0.0
    assert cost.calls[1]["pending_work_ms_at_decide"] == 15.0


def test_pending_work_ms_no_feedback_loop(network_params):
    """Successive arrivals on a busy pod must see pending_work_ms grow
    by service_ms per arrival, not total_ms. Before the fix, each
    arrival added total_ms (service + queueing), so later arrivals
    observed an inflated signal."""
    cost = FixedCostModel(queueing_ms=100.0, compute_prefill_ms=20.0, compute_decode_ms=5.0)
    cluster, engine = _single_pod_engine(cost, network_params)
    # service_ms per request = 25; total_ms per request = 125.
    trace = [
        Request(f"r{i}", f"s{i}", 0.0001 * i, tuple(range(32)), 4)
        for i in range(3)
    ]
    engine.run(trace)

    # Each decide sees Σ service_ms of prior still-in-flight requests.
    observed = [c["pending_work_ms_at_decide"] for c in cost.calls]
    assert observed == [0.0, 25.0, 50.0]


def test_pending_work_ms_decrements_match_increments(network_params):
    """Retirement must subtract the same service_ms that was added on
    arrival — asymmetry would drift pending_work_ms positive or negative."""
    cost = FixedCostModel(queueing_ms=40.0, compute_prefill_ms=10.0, compute_decode_ms=5.0)
    cluster, engine = _single_pod_engine(cost, network_params)

    # Two arrivals followed by a late arrival that triggers retirement
    # of the first two.
    # total_ms = 55, so first two retire by t ≈ 0.055 s.
    trace = [
        Request("r0", "s0", 0.0, tuple(range(32)), 4),
        Request("r1", "s1", 0.001, tuple(range(32)), 4),
        Request("r2", "s2", 1.0, tuple(range(32)), 4),  # well past retirement
    ]
    engine.run(trace)

    pod = cluster.pods["p0"]
    # By end-of-trace, _retire_up_to(inf) drains all pending; pending_work_ms → 0.
    assert pod.pending_work_ms == 0.0
