"""Audit-driven tests for the PD topology policy (bead go-h0b).

Covers edge cases not exercised by `test_policies_individual.py` and
pins specific current-behavior quirks (F18, F19, F20) with comments so a
future fixer finds the assertion that needs to flip.

See `research/reports/policy_audits/pd_topology.md` §4.2 for the gap list.
"""

from __future__ import annotations

from collections import Counter

import pytest

from routing_harness import policies  # noqa: F401 — registers policies
from routing_harness.cluster import ClusterState
from routing_harness.core import Phase, PodSpec, Request
from routing_harness.cost_model import (
    AnalyticCostModel,
    ComputeParams,
    NetworkParams,
    SchedulerParams,
)
from routing_harness.kv_cache import (
    KVCacheState,
    PrefixEntry,
    enumerate_prefix_hashes,
)
from routing_harness.policy import get_policy
from routing_harness.simulator.engine import EngineConfig, SimulationEngine
from routing_harness.simulator.metrics import MetricsCollector


def _pd_cluster_2x2():
    specs = [
        PodSpec("pfA", Phase.PREFILL, 1, 16 * 1024 * 1024, 4, 0, peer_ids=("dcA",)),
        PodSpec("pfB", Phase.PREFILL, 1, 16 * 1024 * 1024, 4, 0, peer_ids=("dcB",)),
        PodSpec("dcA", Phase.DECODE, 1, 4 * 1024 * 1024, 0, 8, peer_ids=("pfA",)),
        PodSpec("dcB", Phase.DECODE, 1, 4 * 1024 * 1024, 0, 8, peer_ids=("pfB",)),
    ]
    cluster = ClusterState.from_specs(specs)
    kv = KVCacheState.from_specs({s.pod_id: s.kv_cache_bytes for s in specs})
    return specs, cluster, kv


def _real_cost_model():
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


def _engine(cluster, kv, policy):
    net = NetworkParams(
        client_rtt_ms=5.0,
        inter_pod_rtt_ms=0.2,
        inter_pod_bandwidth_gbps=100.0,
        kv_bytes_per_token=1024,
        serialization_overhead_ms=0.5,
    )
    return SimulationEngine(
        cluster=cluster,
        kv_cache=kv,
        policy=policy,
        cost_model=_real_cost_model(),
        network=net,
        config=EngineConfig(),
        metrics=MetricsCollector(),
    )


# ---------------------------------------------------------------------------
# Pool shape / degradation
# ---------------------------------------------------------------------------


def test_pd_empty_decode_pool_returns_none():
    specs = [PodSpec("pf0", Phase.PREFILL, 1, 8 * 1024 * 1024, 4, 0)]
    cluster = ClusterState.from_specs(specs)
    kv = KVCacheState.from_specs({s.pod_id: s.kv_cache_bytes for s in specs})
    p = get_policy("pd", block_size=16)
    d = p.decide(Request("r", "s", 0.0, tuple(range(16)), 4), cluster, kv)
    assert d.prefill_pod_id == "__none__"
    assert d.decode_pod_id == "__none__"
    assert "pd-pools-empty" in d.rationale


def test_pd_empty_prefill_pool_returns_none():
    specs = [PodSpec("dc0", Phase.DECODE, 1, 4 * 1024 * 1024, 0, 8)]
    cluster = ClusterState.from_specs(specs)
    kv = KVCacheState.from_specs({s.pod_id: s.kv_cache_bytes for s in specs})
    p = get_policy("pd", block_size=16)
    d = p.decide(Request("r", "s", 0.0, tuple(range(16)), 4), cluster, kv)
    assert d.prefill_pod_id == "__none__"


def test_pd_single_phase_both_pod_colocates():
    specs = [PodSpec("p0", Phase.BOTH, 1, 8 * 1024 * 1024, 2, 8)]
    cluster = ClusterState.from_specs(specs)
    kv = KVCacheState.from_specs({s.pod_id: s.kv_cache_bytes for s in specs})
    p = get_policy("pd", block_size=16)
    d = p.decide(Request("r", "s", 0.0, tuple(range(16)), 4), cluster, kv)
    # Single pod means prefill_pool == decode_pool == [only] — no split possible.
    assert d.prefill_pod_id == "p0"
    assert d.decode_pod_id == "p0"


def test_pd_imbalanced_pools_1_prefill_3_decode():
    specs = [
        PodSpec("pf0", Phase.PREFILL, 1, 16 * 1024 * 1024, 4, 0),
        PodSpec("dc0", Phase.DECODE, 1, 4 * 1024 * 1024, 0, 8),
        PodSpec("dc1", Phase.DECODE, 1, 4 * 1024 * 1024, 0, 8),
        PodSpec("dc2", Phase.DECODE, 1, 4 * 1024 * 1024, 0, 8),
    ]
    cluster = ClusterState.from_specs(specs)
    kv = KVCacheState.from_specs({s.pod_id: s.kv_cache_bytes for s in specs})
    # Load one decode pod; expect the least-busy to win. Warm the EWMA
    # directly because this test exercises the policy in isolation (no
    # engine, so __post_init__ warm-up does not run).
    for pid in ("dc0", "dc1", "dc2"):
        cluster.pods[pid].ewma_latency_ms = 10.0
    cluster.pods["dc0"].active_decode = 5
    cluster.pods["dc1"].active_decode = 0
    cluster.pods["dc2"].active_decode = 2
    p = get_policy("pd", block_size=16)
    d = p.decide(Request("r", "s", 0.0, tuple(range(32)), 4), cluster, kv)
    assert d.prefill_pod_id == "pf0"
    assert d.decode_pod_id == "dc1"


# ---------------------------------------------------------------------------
# Positive behavior pins
# ---------------------------------------------------------------------------


def test_pd_prefill_cache_match_routes_to_owner():
    specs, cluster, kv = _pd_cluster_2x2()
    tokens = tuple(range(32))
    hashes = enumerate_prefix_hashes(tokens, block_size=16)
    # Only pfA has the prefix.
    for h in hashes:
        kv.install("pfA", PrefixEntry(h, 16, 1024), now=1.0)
    p = get_policy("pd", block_size=16)
    d = p.decide(Request("r", "s", 2.0, tokens, 4), cluster, kv)
    assert d.prefill_pod_id == "pfA"


def test_pd_decode_busy_picks_least_loaded():
    specs, cluster, kv = _pd_cluster_2x2()
    # Give both decode pods warm ewma so the multiplier is non-trivial; then
    # make dcA clearly busier.
    cluster.pods["dcA"].ewma_latency_ms = 10.0
    cluster.pods["dcB"].ewma_latency_ms = 10.0
    cluster.pods["dcA"].active_decode = 8
    cluster.pods["dcB"].active_decode = 1
    p = get_policy("pd", block_size=16)
    d = p.decide(Request("r", "s", 2.0, tuple(range(32)), 4), cluster, kv)
    assert d.decode_pod_id == "dcB"


def test_pd_deterministic_under_repeated_calls():
    specs, cluster, kv = _pd_cluster_2x2()
    p = get_policy("pd", block_size=16)
    req = Request("r", "s", 0.0, tuple(range(32)), 4)
    d1 = p.decide(req, cluster, kv)
    d2 = p.decide(req, cluster, kv)
    d3 = p.decide(req, cluster, kv)
    assert d1.prefill_pod_id == d2.prefill_pod_id == d3.prefill_pod_id
    assert d1.decode_pod_id == d2.decode_pod_id == d3.decode_pod_id


def test_pd_prefix_key_path():
    specs, cluster, kv = _pd_cluster_2x2()
    # Short prompt, opaque prefix_key — the _prefix path returns [prefix_key].
    kv.install("pfA", PrefixEntry("opaque-key", 16, 1024), now=1.0)
    p = get_policy("pd", block_size=16)
    req = Request(
        "r", "s", 0.0, tuple(range(4)), 4, prefix_key="opaque-key"
    )
    d = p.decide(req, cluster, kv)
    assert d.prefill_pod_id == "pfA"


# ---------------------------------------------------------------------------
# F18: non-consecutive block-match scoring (pins current quirk)
# ---------------------------------------------------------------------------


def test_pd_f18_consecutive_match_beats_scattered_blocks():
    """F18 fix: pd scores by longest consecutive prefix, not scattered hits.

    A pod with blocks {0, 2, 4} (scattered, unreusable past block 0) scores
    1 and loses to a pod with blocks {0, 1} (genuinely reusable prefix)
    that scores 2 — matching the engine's `captured` semantics and
    `prefix_cache.py`.
    """
    specs = [
        PodSpec("pfA", Phase.PREFILL, 1, 16 * 1024 * 1024, 4, 0),
        PodSpec("pfB", Phase.PREFILL, 1, 16 * 1024 * 1024, 4, 0),
        PodSpec("dc0", Phase.DECODE, 1, 4 * 1024 * 1024, 0, 8),
    ]
    cluster = ClusterState.from_specs(specs)
    kv = KVCacheState.from_specs({s.pod_id: s.kv_cache_bytes for s in specs})
    tokens = tuple(range(80))  # 5 blocks at block_size=16
    hashes = enumerate_prefix_hashes(tokens, block_size=16)

    for i in (0, 2, 4):  # scattered (unreusable past block 0)
        kv.install("pfA", PrefixEntry(hashes[i], 16, 1024), now=1.0)
    for i in (0, 1):  # genuine consecutive prefix
        kv.install("pfB", PrefixEntry(hashes[i], 16, 1024), now=1.0)

    p = get_policy("pd", block_size=16)
    d = p.decide(Request("r", "s", 2.0, tokens, 4), cluster, kv)
    assert d.prefill_pod_id == "pfB"

    # Cross-check: prefix_cache (consecutive match) agrees on pfB.
    d_ref = get_policy("prefix-cache", block_size=16).decide(
        Request("r", "s", 2.0, tokens, 4), cluster, kv
    )
    assert d_ref.prefill_pod_id == "pfB"


# ---------------------------------------------------------------------------
# F19: cross-branch tie-break asymmetry (pins current quirk)
# ---------------------------------------------------------------------------


def test_pd_f19_colocated_fallback_tie_triggers_handoff():
    """F19: under perfect ties, prefill's max picks largest pod_id and
    decode's min picks smallest. On a BOTH-mode colocated cluster this
    forces a cross-pod handoff on every dispatch.

    When F19 is fixed (both branches agree on smallest-wins), flip to
        assert d.prefill_pod_id == d.decode_pod_id
    """
    specs = [PodSpec(f"p{i}", Phase.BOTH, 1, 8 * 1024 * 1024, 2, 8) for i in range(3)]
    cluster = ClusterState.from_specs(specs)
    kv = KVCacheState.from_specs({s.pod_id: s.kv_cache_bytes for s in specs})
    p = get_policy("pd", block_size=16)
    d = p.decide(Request("r", "s", 0.0, tuple(range(16)), 4), cluster, kv)
    # F19: prefill='p2' (max), decode='p0' (min) — pods differ.
    assert d.prefill_pod_id == "p2"
    assert d.decode_pod_id == "p0"
    assert d.prefill_pod_id != d.decode_pod_id


def test_pd_f19_engine_charges_gratuitous_handoff_on_both_cluster():
    """Engine-level reproduction of F19: pd_handoff_bytes fires on every
    request in a colocated BOTH cluster even though no role-split exists.
    """
    specs = [PodSpec(f"p{i}", Phase.BOTH, 1, 8 * 1024 * 1024, 2, 8) for i in range(3)]
    cluster = ClusterState.from_specs(specs)
    kv = KVCacheState.from_specs({s.pod_id: s.kv_cache_bytes for s in specs})
    eng = _engine(cluster, kv, get_policy("pd", block_size=16))
    reqs = [
        Request(f"r{i}", "s", float(i) * 0.1, tuple(range(32)), 4)
        for i in range(5)
    ]
    eng.run(reqs)
    # Every request migrated and carried kv_transport_bytes > 0.
    for rec in eng.metrics.records:
        assert rec.migrated is True
        assert rec.kv_transport_bytes > 0


# ---------------------------------------------------------------------------
# F20: stale ewma_latency_ms on pure-DECODE pods (pins current quirk)
# ---------------------------------------------------------------------------


def test_pd_f20_decode_pool_ewma_is_stale():
    """F20: the engine only updates ewma_latency_ms on the prefill pod.
    Pure-DECODE pods stay at initial_warm_latency_ms forever.

    When F20 is fixed (engine instruments decode EWMA, or policy drops
    the multiplier), update the assertion to verify the signal advances.
    """
    specs, cluster, kv = _pd_cluster_2x2()
    eng = _engine(cluster, kv, get_policy("pd", block_size=16))
    reqs = [
        Request(f"r{i}", "s", float(i) * 0.01, tuple(range(32)), 16)
        for i in range(30)
    ]
    eng.run(reqs)
    warm = eng.config.initial_warm_latency_ms
    # F20: both pure-DECODE pods' EWMA stayed pinned at warm-start.
    assert cluster.pods["dcA"].ewma_latency_ms == pytest.approx(warm)
    assert cluster.pods["dcB"].ewma_latency_ms == pytest.approx(warm)
    # Sanity: at least one prefill pod's EWMA *did* advance past warm.
    advanced = [
        p.ewma_latency_ms > warm * 1.5
        for p in (cluster.pods["pfA"], cluster.pods["pfB"])
    ]
    assert any(advanced)


# ---------------------------------------------------------------------------
# F23: peer_ids ignored (pins current quirk)
# ---------------------------------------------------------------------------


def test_pd_f23_peer_ids_ignored():
    """F23: the policy does not consult PodSpec.peer_ids, so it can pair a
    prefill pod with a decode pod that is not in its peer set.

    Topology here: pfA is peered only with dcA; pfB is peered only with
    dcB. We pre-load pfA's cache so prefill→pfA, then make dcA busy so
    decode→dcB. Under a peer-aware policy this would either (a) refuse
    the pair or (b) prefer the peer-matched dcA. Current pd picks (pfA,
    dcB) — a mispaired transfer across un-peered pods.
    """
    specs, cluster, kv = _pd_cluster_2x2()
    # Cache prefix only on pfA.
    tokens = tuple(range(32))
    hashes = enumerate_prefix_hashes(tokens, block_size=16)
    for h in hashes:
        kv.install("pfA", PrefixEntry(h, 16, 1024), now=1.0)
    # Make dcA (pfA's peer) the busier decode.
    cluster.pods["dcA"].ewma_latency_ms = 10.0
    cluster.pods["dcB"].ewma_latency_ms = 10.0
    cluster.pods["dcA"].active_decode = 8
    cluster.pods["dcB"].active_decode = 0
    p = get_policy("pd", block_size=16)
    d = p.decide(Request("r", "s", 2.0, tokens, 4), cluster, kv)
    assert d.prefill_pod_id == "pfA"
    # F23: policy ignored peer_ids; picks dcB even though pfA's peer is dcA.
    assert d.decode_pod_id == "dcB"
    assert d.decode_pod_id not in cluster.pods["pfA"].spec.peer_ids
