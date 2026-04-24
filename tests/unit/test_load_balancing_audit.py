"""Tests backing the load-balancing policy audit (go-651).

Extends `test_policies_individual.py` with per-policy edge-case and
end-to-end-engine tests for the six load-balancing policies. Tests that
assert a *known-bad* behavior are tagged `# NEGATIVE: documents F<n>`
so the reader knows the assertion pins a bug; when the bug is fixed,
the test must be updated and the corresponding bead closed.
"""

from __future__ import annotations

import pytest

from routing_harness import policies  # noqa: F401 — register
from routing_harness.cluster import ClusterState
from routing_harness.core import Phase, PodSpec, Request
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
        counts[rec.decision.prefill_pod_id] = counts.get(rec.decision.prefill_pod_id, 0) + 1
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


def test_least_kv_cache_tie_break_picks_smallest_pod_id():
    """POSITIVE: regression test for F9 (fix).

    All pods empty → free = cap for all → tie on `(free, active_prefill)`
    falls through to pod_id. The fix (go-edm) restructures the selection
    as `min` over negated free bytes so the pod_id tie-break prefers the
    smallest id, matching the rest of the load-balancing group. If this
    flips back to 'p2', F9 has regressed.
    """
    cluster, kv = _fresh_cluster(_specs())
    p = get_policy("least-kv-cache")
    d = p.decide(Request("r", "s", 0.0, (1, 2), 4), cluster, kv)
    assert d.prefill_pod_id == "p0", (
        "F9 regression: tie-break must prefer smallest pod_id, matching "
        "the rest of the load-balancing group."
    )


def test_least_kv_cache_rotates_on_shared_prefix():
    """POSITIVE: regression test for F10 (fix).

    Under shared-prefix workloads, install-after-cache-hit is a byte-level
    no-op, so the first-warmed pod's `free` never shrinks. Without a
    load-aware secondary key, that pod captures the overwhelming majority
    of dispatches (reproduced at 58/1/1 on the baseline). The fix
    (go-fw8) adds `-active_prefill` as a secondary sort key so free-byte
    ties fall back to load balancing. Post-fix: 20/20/20 at 50 QPS.
    If this flips back to starvation, F10 has regressed.
    """
    tokens = tuple(range(64))
    trace = [Request(f"r{i}", "s", i * 0.02, tokens, 32) for i in range(60)]
    counts = _run_trace("least-kv-cache", trace)
    assert set(counts) == {"p0", "p1", "p2"}, counts
    assert max(counts.values()) - min(counts.values()) <= 4, counts


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


def test_throughput_tiebreak_uses_full_pod_id():
    """POSITIVE: F8 fixed.

    Tie-break uses the full pod_id (not just the first character),
    matching the 'smallest pod_id wins ties' convention of sibling
    policies (F11). If this test fails, the tie-break has regressed to
    -ord(pod_id[0]) and F8 is back.
    """
    p = get_policy("throughput")

    # Distinct first chars: smallest pod_id ('a0') wins over 'b0'.
    kv_ab = KVCacheState.from_specs({"a0": 1, "b0": 1})
    cluster_ab = ClusterState.from_specs(
        [
            PodSpec("a0", Phase.BOTH, 1, 1 << 20, 2, 8),
            PodSpec("b0", Phase.BOTH, 1, 1 << 20, 2, 8),
        ]
    )
    d = p.decide(Request("r", "s", 0.0, (1, 2), 4), cluster_ab, kv_ab)
    assert d.prefill_pod_id == "a0", (
        "F8: tie-break on full pod_id picks lexicographically smallest."
    )

    # Identical first chars, reverse insertion order: a1 inserted before
    # a0. Old bug would collapse to iteration order and pick 'a1'. Fix
    # picks smallest full pod_id → 'a0' regardless of insertion order.
    kv_aa = KVCacheState.from_specs({"a1": 1, "a0": 1})
    cluster_aa = ClusterState.from_specs(
        [
            PodSpec("a1", Phase.BOTH, 1, 1 << 20, 2, 8),
            PodSpec("a0", Phase.BOTH, 1, 1 << 20, 2, 8),
        ]
    )
    d2 = p.decide(Request("r", "s", 0.0, (1, 2), 4), cluster_aa, kv_aa)
    assert d2.prefill_pod_id == "a0", (
        "F8: when first chars match, tie-break still picks smallest full "
        "pod_id — not iteration order."
    )


# -------------------------------------------------------------------------
# F11: cross-policy tie-break convention
# -------------------------------------------------------------------------

# Policies participating in the 'smallest pod_id wins ties' convention.
# `random` is deliberately excluded — it has no tie-break. The two prefix
# policies are included because their no-match/explore branches fall back
# to the same load-balancing tie rule (bead go-dmo description).
TIE_BREAK_CONVENTION_POLICIES = (
    "least-request",
    "least-busy-time",
    "least-latency",
    "least-kv-cache",
    "throughput",
    "prefix-cache",
    "prefix-cache-preble",
)


@pytest.mark.parametrize("policy_id", TIE_BREAK_CONVENTION_POLICIES)
def test_group_tie_break_picks_smallest_pod_id(policy_id):
    """POSITIVE: umbrella regression for F11.

    All pods fresh and identical → every scoring signal ties → selection
    must fall through to the smallest pod_id. Covers the load-balancing
    group plus the prefix policies whose fallback/explore branches share
    the convention. Ids are inserted in reverse order so iteration order
    does not coincide with the expected winner — if a policy regresses
    to 'iteration order' or 'largest pod_id', this test catches it.
    """
    specs = [
        PodSpec(
            pod_id=pid,
            role=Phase.BOTH,
            gpu_count=1,
            kv_cache_bytes=8 * 1024 * 1024,
            max_concurrent_prefill=2,
            max_concurrent_decode=8,
        )
        for pid in ("p2", "p1", "p0")
    ]
    cluster, kv = _fresh_cluster(specs)
    p = get_policy(policy_id)
    d = p.decide(Request("r", "s", 0.0, (1, 2, 3), 4), cluster, kv)
    assert d.prefill_pod_id == "p0", (
        f"F11 regression in {policy_id}: tie-break must pick smallest pod_id "
        f"(got {d.prefill_pod_id}). Group convention is 'smallest pod_id "
        f"wins ties' across the load-balancing + prefix fallback policies."
    )
