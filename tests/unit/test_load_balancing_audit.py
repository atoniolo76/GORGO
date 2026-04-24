"""Tests backing the load-balancing policy audit (go-651).

Extends `test_policies_individual.py` with per-policy edge-case and
end-to-end-engine tests for the six load-balancing policies. Tests that
assert a *known-bad* behavior are tagged `# NEGATIVE: documents F<n>`
so the reader knows the assertion pins a bug; when the bug is fixed,
the test must be updated and the corresponding bead closed.
"""

from __future__ import annotations

import statistics

import pytest

from routing_harness import policies  # noqa: F401 — register
from routing_harness.cluster import ClusterState
from routing_harness.core import Phase, PodRuntime, PodSpec, Request
from routing_harness.cost_model import (
    AnalyticCostModel,
    ComputeParams,
    NetworkParams,
    SchedulerParams,
)
from routing_harness.kv_cache import KVCacheState
from routing_harness.policy import get_policy
from routing_harness.simulator.engine import EngineConfig, SimulationEngine
from routing_harness.simulator.metrics import MetricsCollector


LOAD_BALANCING_POLICIES = (
    "random",
    "least-request",
    "least-busy-time",
    "least-latency",
    "least-kv-cache",
    "throughput",
)


def _specs(n: int = 3, cap_bytes: int = 8 * 1024 * 1024) -> list[PodSpec]:
    return [
        PodSpec(
            pod_id=f"p{i}",
            role=Phase.BOTH,
            gpu_count=1,
            kv_cache_bytes=cap_bytes,
            max_concurrent_prefill=2,
            max_concurrent_decode=8,
        )
        for i in range(n)
    ]


def _fresh_cluster(specs: list[PodSpec]) -> tuple[ClusterState, KVCacheState]:
    cluster = ClusterState.from_specs(specs)
    kv = KVCacheState.from_specs({s.pod_id: s.kv_cache_bytes for s in specs})
    return cluster, kv


def _cost_model() -> AnalyticCostModel:
    return AnalyticCostModel(
        ComputeParams(
            prefill_ms_per_token=0.1,
            decode_ms_per_token=5.0,
            prefill_overhead_ms=4.0,
            decode_overhead_ms=1.0,
        ),
        NetworkParams(
            client_rtt_ms=5.0,
            inter_pod_rtt_ms=0.2,
            inter_pod_bandwidth_gbps=100.0,
            kv_bytes_per_token=1024,
            serialization_overhead_ms=0.5,
        ),
        SchedulerParams(base_routing_ms=0.2, per_pod_consideration_us=5.0),
    )


def _run_trace(
    policy_id: str,
    trace: list[Request],
    specs: list[PodSpec] | None = None,
    *,
    policy_kwargs: dict | None = None,
) -> dict[str, int]:
    """Run `trace` through a fresh engine + `policy_id`; return dispatch counts."""
    specs = specs or _specs()
    cluster, kv = _fresh_cluster(specs)
    cm = _cost_model()
    policy = get_policy(policy_id, **(policy_kwargs or {}))
    metrics = MetricsCollector()
    engine = SimulationEngine(
        cluster=cluster,
        kv_cache=kv,
        policy=policy,
        cost_model=cm,
        network=cm.network,
        config=EngineConfig(),
        metrics=metrics,
    )
    engine.run(trace)
    counts: dict[str, int] = {}
    for rec in metrics.records:
        counts[rec.decision.prefill_pod_id] = (
            counts.get(rec.decision.prefill_pod_id, 0) + 1
        )
    return counts


# -------------------------------------------------------------------------
# Group-level smoke tests
# -------------------------------------------------------------------------


@pytest.mark.parametrize("policy_id", LOAD_BALANCING_POLICIES)
def test_empty_cluster_returns_none_sentinel(policy_id, kv_cache):
    empty = ClusterState.from_specs([])
    kwargs = {"seed": 0} if policy_id == "random" else {}
    p = get_policy(policy_id, **kwargs)
    d = p.decide(Request("r", "s", 0.0, (1, 2, 3), 4), empty, kv_cache)
    assert d.prefill_pod_id == "__none__"
    assert d.decode_pod_id == "__none__"


@pytest.mark.parametrize("policy_id", LOAD_BALANCING_POLICIES)
def test_single_pod_cluster_always_picks_that_pod(policy_id):
    specs = _specs(n=1)
    cluster, kv = _fresh_cluster(specs)
    kwargs = {"seed": 0} if policy_id == "random" else {}
    p = get_policy(policy_id, **kwargs)
    for i in range(5):
        d = p.decide(Request(f"r{i}", "s", float(i), (1, 2, 3), 4), cluster, kv)
        assert d.prefill_pod_id == "p0"


# -------------------------------------------------------------------------
# random
# -------------------------------------------------------------------------


def test_random_uniform_distribution():
    """300 seeded picks over 3 pods should be within a wide band of uniform."""
    cluster, kv = _fresh_cluster(_specs())
    p = get_policy("random", seed=0)
    counts = {"p0": 0, "p1": 0, "p2": 0}
    for i in range(300):
        d = p.decide(Request(f"r{i}", "s", float(i), (1, 2), 4), cluster, kv)
        counts[d.prefill_pod_id] += 1
    mean = 100.0
    for n in counts.values():
        assert 70 <= n <= 130, f"uniform-random out of band: {counts}"
    _ = mean  # documentation


# -------------------------------------------------------------------------
# least-request
# -------------------------------------------------------------------------


def test_least_request_rotates_under_load():
    """High-QPS trace forces queue buildup → perfect rotation expected."""
    # 60 requests at 50 QPS, prefill service ~ 15 ms + decode; overlaps exist.
    tokens = tuple(range(64))
    trace = [Request(f"r{i}", "s", i * 0.02, tokens, 32) for i in range(60)]
    counts = _run_trace("least-request", trace)
    assert set(counts) == {"p0", "p1", "p2"}
    assert max(counts.values()) - min(counts.values()) <= 2, counts


def test_least_request_uncontended_ties_to_lowest_id():
    """Arrivals fully staggered → all pods idle at decide → tie → p0 every time.

    Documents the §2.2.2 audit note: under uncontended load this policy
    concentrates on p0. Not a bug; documented for regression awareness.
    """
    tokens = tuple(range(32))
    trace = [Request(f"r{i}", "s", i * 10.0, tokens, 4) for i in range(8)]
    counts = _run_trace("least-request", trace)
    assert counts.get("p0", 0) == 8, counts


# -------------------------------------------------------------------------
# least-busy-time
# -------------------------------------------------------------------------


def test_least_busy_time_zero_load_ties_to_lowest_id():
    """Cold start: all busy = ewma * 0 = 0 → tie → p0."""
    cluster, kv = _fresh_cluster(_specs())
    # Warm-init is applied by engine, not policy. Simulate by hand:
    for pod in cluster.pods.values():
        pod.ewma_latency_ms = 5.0
    p = get_policy("least-busy-time")
    d = p.decide(Request("r", "s", 0.0, (1, 2), 4), cluster, kv)
    assert d.prefill_pod_id == "p0"


def test_least_busy_time_rotates_under_load():
    tokens = tuple(range(64))
    trace = [Request(f"r{i}", "s", i * 0.02, tokens, 32) for i in range(60)]
    counts = _run_trace("least-busy-time", trace)
    assert set(counts) == {"p0", "p1", "p2"}
    assert max(counts.values()) - min(counts.values()) <= 2, counts


# -------------------------------------------------------------------------
# least-latency
# -------------------------------------------------------------------------


def test_least_latency_relies_on_warm_init_to_avoid_cold_monopoly():
    """Without the engine's warm-init, all ewma_latency_ms would be 0.

    This test asserts the *current* contract: pods constructed fresh
    (without the engine) have ewma=0, and least-latency picks the first
    in iteration order. If the engine ever stops warm-initializing
    ewma_latency_ms, the dependent starvation would resurface.
    """
    cluster, kv = _fresh_cluster(_specs())
    # Deliberately do NOT run through SimulationEngine.__post_init__.
    for pod in cluster.pods.values():
        assert pod.ewma_latency_ms == 0.0
    p = get_policy("least-latency")
    d = p.decide(Request("r", "s", 0.0, (1, 2), 4), cluster, kv)
    # All ties at 0.0 → smallest pod_id wins.
    assert d.prefill_pod_id == "p0"


def test_least_latency_rotates_under_real_engine():
    """With the engine's warm-init, least-latency rotates under load."""
    tokens = tuple(range(64))
    trace = [Request(f"r{i}", "s", i * 0.02, tokens, 32) for i in range(60)]
    counts = _run_trace("least-latency", trace)
    assert set(counts) == {"p0", "p1", "p2"}
    # Least-latency cycles on EWMA decay; allow a wider band than rotation-strict.
    assert max(counts.values()) - min(counts.values()) <= 4, counts


# -------------------------------------------------------------------------
# least-kv-cache
# -------------------------------------------------------------------------


def test_least_kv_cache_rotates_on_unique_prefixes():
    """Unique-prefix workload → every dispatch installs bytes → rotation."""
    trace = [
        Request(
            f"r{i}",
            f"s{i}",
            i * 0.02,
            tuple(range(i * 100, i * 100 + 64)),
            32,
        )
        for i in range(60)
    ]
    counts = _run_trace("least-kv-cache", trace)
    assert set(counts) == {"p0", "p1", "p2"}
    assert max(counts.values()) - min(counts.values()) <= 2, counts


def test_least_kv_cache_tie_break_picks_largest_pod_id():
    """NEGATIVE: documents F9.

    All pods empty → free = cap for all → `max` with `(free, pod_id)`
    picks the largest pod_id (p2). This is inconsistent with the rest
    of the load-balancing group (which prefers smallest pod_id). When
    F9 is fixed, this test must flip and the discovered bead closed.
    """
    cluster, kv = _fresh_cluster(_specs())
    p = get_policy("least-kv-cache")
    d = p.decide(Request("r", "s", 0.0, (1, 2), 4), cluster, kv)
    assert d.prefill_pod_id == "p2", (
        "F9 expected: max(pod_id) wins ties. If this assertion flips to "
        "'p0', F9 has been fixed — update this test and close the bead."
    )


def test_least_kv_cache_starves_on_shared_prefix():
    """NEGATIVE: documents F10.

    When every request shares the same prefix, the first install on the
    tie-break winner (p2, per F9) warms its cache; subsequent installs
    on p2 are byte-level no-ops, so p2.free never shrinks. Result: p2
    captures the overwhelming majority of dispatches.

    When F10 is fixed (e.g., by adding a load-aware secondary key),
    this assertion will flip and the discovered bead must close.
    """
    tokens = tuple(range(64))
    trace = [Request(f"r{i}", "s", i * 0.02, tokens, 32) for i in range(60)]
    counts = _run_trace("least-kv-cache", trace)
    top = max(counts.values())
    assert top >= 50, (
        f"F10 expected: one pod captures ≥ 50/60 under shared prefix. "
        f"Got {counts}. If this is no longer true, F10 has been mitigated; "
        f"update this test and close the bead."
    )


# -------------------------------------------------------------------------
# throughput
# -------------------------------------------------------------------------


def test_throughput_rotates_under_real_engine():
    """POSITIVE: regression test for F7 (fix).

    With the engine warm-initializing `ewma_throughput_tps` in parallel
    to `ewma_latency_ms`, all pods start equal and the `/(1+active)`
    normalization carries the rotation. Cold-start starvation is gone
    in both low- and high-load regimes. If this test flips back to
    starvation, F7 has regressed.
    """
    tokens = tuple(range(64))
    # High-load: 60 req @ 50 QPS — the regime that originally pinned p0.
    trace = [Request(f"r{i}", "s", i * 0.02, tokens, 32) for i in range(60)]
    counts = _run_trace("throughput", trace)
    assert set(counts) == {"p0", "p1", "p2"}, counts
    assert max(counts.values()) - min(counts.values()) <= 4, counts
    # Low-load: 30 req @ 5 QPS — same pathology under the bug.
    trace = [Request(f"r{i}", "s", i * 0.2, tokens, 32) for i in range(30)]
    counts = _run_trace("throughput", trace)
    assert set(counts) == {"p0", "p1", "p2"}, counts
    assert max(counts.values()) - min(counts.values()) <= 4, counts


def test_throughput_tiebreak_uses_only_first_char_of_pod_id():
    """NEGATIVE: documents F8.

    Tie-break key is `-ord(pod_id[0])`. For pods whose first characters
    DIFFER, the tie-break resolves. For pods whose first characters MATCH
    (e.g., canonical p0/p1/p2), the tie-break is a no-op.

    We construct two pods with distinct first chars to show the tie-break
    does *something*; and we construct two pods with identical first
    chars to show it does *nothing*.
    """
    kv = KVCacheState.from_specs({"a0": 1, "b0": 1})

    # a0 and b0 have distinct first chars. First request, cold start,
    # all scores 0 → tie broken on -ord('a')=-97 vs -ord('b')=-98. max
    # picks the LARGER value → -97 → 'a0'.
    cluster_ab = ClusterState.from_specs(
        [
            PodSpec("a0", Phase.BOTH, 1, 1 << 20, 2, 8),
            PodSpec("b0", Phase.BOTH, 1, 1 << 20, 2, 8),
        ]
    )
    p = get_policy("throughput")
    d = p.decide(Request("r", "s", 0.0, (1, 2), 4), cluster_ab, kv)
    assert d.prefill_pod_id == "a0", (
        "F8: tie-break on -ord(first_char) picks smallest first char."
    )

    # a0 and a1 share first char 'a'. Scores all zero → tie-break also
    # all -97. Python's max on fully-tied sequence returns the first
    # element in iteration order.
    kv2 = KVCacheState.from_specs({"a0": 1, "a1": 1})
    cluster_aa = ClusterState.from_specs(
        [
            PodSpec("a0", Phase.BOTH, 1, 1 << 20, 2, 8),
            PodSpec("a1", Phase.BOTH, 1, 1 << 20, 2, 8),
        ]
    )
    d2 = p.decide(Request("r", "s", 0.0, (1, 2), 4), cluster_aa, kv2)
    # Iteration order follows dict insertion order: a0 inserted first.
    assert d2.prefill_pod_id == "a0", (
        "F8: when first chars are identical, tie-break collapses to "
        "iteration order. If this flips, the tie-break was strengthened "
        "to use the full pod_id — update this test and close the F8 bead."
    )
