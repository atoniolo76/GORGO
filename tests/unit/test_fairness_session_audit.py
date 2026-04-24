"""Audit-driven tests for fairness/session policies (bead go-sz5).

Covers edge cases not exercised by `test_policies_individual.py`:
session binding persistence and eviction, pod add/remove mid-run,
TTL boundaries, vtc cold start and tenant_debt evolution, monotonic
counters, fairness signal lag, and an end-to-end integration pin
through the SimulationEngine.

See `research/reports/policy_audits/fairness_session.md` §4.2 for the
list of gaps these tests fill.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from routing_harness import policies  # noqa: F401 — registers policies
from routing_harness.cluster import ClusterState
from routing_harness.core import Decision, Phase, PodSpec, Request
from routing_harness.kv_cache import KVCacheState
from routing_harness.policy import get_policy


# ---------- helpers ----------


def _fresh_3_pod_cluster(pod_specs) -> tuple[ClusterState, KVCacheState]:
    cluster = ClusterState.from_specs(pod_specs)
    kv = KVCacheState.from_specs({s.pod_id: s.kv_cache_bytes for s in pod_specs})
    return cluster, kv


def _make_req(rid: str, sid: str, ts: float, metadata: dict | None = None) -> Request:
    return Request(
        request_id=rid,
        session_id=sid,
        arrival_ts=ts,
        prompt_tokens=(1, 2, 3, 4),
        max_output_tokens=4,
        metadata=metadata or {},
    )


# =============================================================================
# session-affinity
# =============================================================================


def test_session_affinity_cold_start_picks_lightest(pod_specs):
    """Brand-new session with varying pod loads lands on the lightest."""
    cluster, kv = _fresh_3_pod_cluster(pod_specs)
    cluster.pods["p0"].active_prefill = 5
    cluster.pods["p1"].active_prefill = 2
    cluster.pods["p2"].active_prefill = 0
    p = get_policy("session-affinity", stickiness_ttl_s=3600.0)
    d = p.decide(_make_req("r0", "s_new", 1.0), cluster, kv)
    assert d.prefill_pod_id == "p2"
    assert d.rationale == "new-sticky-binding"


def test_session_affinity_binding_survives_many_requests(pod_specs):
    """Same session over 10 requests routes to the same pod even as loads shift."""
    cluster, kv = _fresh_3_pod_cluster(pod_specs)
    p = get_policy("session-affinity", stickiness_ttl_s=3600.0)
    d0 = p.decide(_make_req("r0", "sX", 1.0), cluster, kv)
    bound_pod = d0.prefill_pod_id
    # Now heavily load the bound pod; subsequent same-session requests should still stick.
    cluster.pods[bound_pod].active_prefill = 100
    for i in range(10):
        d = p.decide(_make_req(f"r{i+1}", "sX", 2.0 + i), cluster, kv)
        assert d.prefill_pod_id == bound_pod, f"iter {i}: expected sticky {bound_pod}, got {d.prefill_pod_id}"
        assert d.rationale.startswith("sticky")


def test_session_affinity_ttl_boundary_exactly_at_ttl_still_sticky(pod_specs):
    """Condition is `<= ttl`, so binding at exactly the TTL must still stick."""
    cluster, kv = _fresh_3_pod_cluster(pod_specs)
    p = get_policy("session-affinity", stickiness_ttl_s=60.0)
    d1 = p.decide(_make_req("r1", "sTTL", 0.0), cluster, kv)
    d2 = p.decide(_make_req("r2", "sTTL", 60.0), cluster, kv)  # exactly TTL
    assert d1.prefill_pod_id == d2.prefill_pod_id
    assert d2.rationale.startswith("sticky")


def test_session_affinity_ttl_boundary_just_over_rebinds(pod_specs):
    """At arrival = bound_ts + TTL + ε, must rebind."""
    cluster, kv = _fresh_3_pod_cluster(pod_specs)
    p = get_policy("session-affinity", stickiness_ttl_s=60.0)
    d1 = p.decide(_make_req("r1", "sTTL", 0.0), cluster, kv)
    # Make the fallback prefer a different pod by loading d1's pick.
    cluster.pods[d1.prefill_pod_id].active_prefill = 50
    d2 = p.decide(_make_req("r2", "sTTL", 60.001), cluster, kv)  # just past TTL
    assert d2.rationale == "new-sticky-binding"
    # New binding should go to a less loaded pod.
    assert d2.prefill_pod_id != d1.prefill_pod_id


def test_session_affinity_rebinds_when_bound_pod_removed(pod_specs):
    """If the bound pod disappears from cluster.pods, policy rebinds."""
    cluster, kv = _fresh_3_pod_cluster(pod_specs)
    p = get_policy("session-affinity", stickiness_ttl_s=3600.0)
    d1 = p.decide(_make_req("r1", "sRm", 1.0), cluster, kv)
    bound = d1.prefill_pod_id
    del cluster.pods[bound]
    d2 = p.decide(_make_req("r2", "sRm", 2.0), cluster, kv)
    assert d2.prefill_pod_id != bound
    assert d2.prefill_pod_id in cluster.pods
    assert d2.rationale == "new-sticky-binding"


def test_session_affinity_other_sessions_unaffected_by_pod_removal(pod_specs):
    """Removing pod X only rebinds sessions that were bound to X."""
    cluster, kv = _fresh_3_pod_cluster(pod_specs)
    p = get_policy("session-affinity", stickiness_ttl_s=3600.0)
    # Force bindings to distinct pods by loading the cluster between decisions.
    cluster.pods["p1"].active_prefill = 99
    cluster.pods["p2"].active_prefill = 99
    d_a = p.decide(_make_req("rA", "sA", 1.0), cluster, kv)  # → p0
    cluster.pods["p0"].active_prefill = 99
    cluster.pods["p2"].active_prefill = 99
    cluster.pods["p1"].active_prefill = 0
    d_b = p.decide(_make_req("rB", "sB", 2.0), cluster, kv)  # → p1
    assert d_a.prefill_pod_id == "p0"
    assert d_b.prefill_pod_id == "p1"
    del cluster.pods["p0"]
    # sA must rebind; sB must stick.
    d_a2 = p.decide(_make_req("rA2", "sA", 3.0), cluster, kv)
    d_b2 = p.decide(_make_req("rB2", "sB", 3.1), cluster, kv)
    assert d_a2.prefill_pod_id in ("p1", "p2")
    assert d_a2.rationale == "new-sticky-binding"
    assert d_b2.prefill_pod_id == "p1"
    assert d_b2.rationale.startswith("sticky")


def test_session_affinity_pod_added_midrun_does_not_rebalance(pod_specs):
    """F14 pin: new pods do not cause existing sessions to rebind."""
    cluster, kv = _fresh_3_pod_cluster(pod_specs)
    p = get_policy("session-affinity", stickiness_ttl_s=3600.0)
    d1 = p.decide(_make_req("r1", "sHold", 1.0), cluster, kv)
    bound = d1.prefill_pod_id
    # Add a brand-new, unloaded pod.
    new_spec = PodSpec(
        pod_id="p9",
        role=Phase.BOTH,
        gpu_count=1,
        kv_cache_bytes=8 * 1024 * 1024,
        max_concurrent_prefill=2,
        max_concurrent_decode=8,
    )
    from routing_harness.core import PodRuntime
    cluster.pods["p9"] = PodRuntime(spec=new_spec)
    d2 = p.decide(_make_req("r2", "sHold", 2.0), cluster, kv)
    # Still bound to the original pod, not "p9".
    assert d2.prefill_pod_id == bound
    assert d2.prefill_pod_id != "p9"


def test_session_affinity_fallback_tiebreak_on_pod_id(pod_specs):
    """All pods identically idle → lowest pod_id wins (new session)."""
    cluster, kv = _fresh_3_pod_cluster(pod_specs)
    p = get_policy("session-affinity", stickiness_ttl_s=3600.0)
    d = p.decide(_make_req("r", "sTie", 0.0), cluster, kv)
    assert d.prefill_pod_id == "p0"


def test_session_affinity_two_sessions_independent(pod_specs):
    """Different sessions get independent bindings."""
    cluster, kv = _fresh_3_pod_cluster(pod_specs)
    p = get_policy("session-affinity", stickiness_ttl_s=3600.0)
    # Stagger: load pod so second session picks a different one.
    d_a = p.decide(_make_req("rA", "sA", 0.0), cluster, kv)
    cluster.pods[d_a.prefill_pod_id].active_prefill = 50
    d_b = p.decide(_make_req("rB", "sB", 0.1), cluster, kv)
    # Re-decide both — bindings must persist independently.
    d_a2 = p.decide(_make_req("rA2", "sA", 1.0), cluster, kv)
    d_b2 = p.decide(_make_req("rB2", "sB", 1.1), cluster, kv)
    assert d_a2.prefill_pod_id == d_a.prefill_pod_id
    assert d_b2.prefill_pod_id == d_b.prefill_pod_id


def test_session_affinity_empty_cluster_returns_none_sentinel(pod_specs):
    empty = ClusterState.from_specs([])
    kv = KVCacheState.from_specs({})
    p = get_policy("session-affinity", stickiness_ttl_s=3600.0)
    d = p.decide(_make_req("r", "s", 0.0), empty, kv)
    assert d.prefill_pod_id == "__none__"
    assert d.decode_pod_id == "__none__"


def test_session_affinity_single_pod_cluster(pod_specs):
    one = [pod_specs[0]]
    cluster = ClusterState.from_specs(one)
    kv = KVCacheState.from_specs({one[0].pod_id: one[0].kv_cache_bytes})
    p = get_policy("session-affinity", stickiness_ttl_s=3600.0)
    d1 = p.decide(_make_req("r1", "sOne", 0.0), cluster, kv)
    d2 = p.decide(_make_req("r2", "sOne", 10.0), cluster, kv)
    assert d1.prefill_pod_id == one[0].pod_id
    assert d2.prefill_pod_id == one[0].pod_id


# =============================================================================
# vtc-basic
# =============================================================================


def test_vtc_cold_start_picks_lowest_pod_id(pod_specs):
    cluster, kv = _fresh_3_pod_cluster(pod_specs)
    p = get_policy("vtc-basic")
    d = p.decide(_make_req("r", "s_cold", 0.0), cluster, kv)
    assert d.prefill_pod_id == "p0"


def test_vtc_observe_completion_updates_both_counters(pod_specs):
    cluster, kv = _fresh_3_pod_cluster(pod_specs)
    p = get_policy("vtc-basic")
    req = _make_req("r", "sA", 0.0)
    d = p.decide(req, cluster, kv)
    p.observe_completion(req, d, tokens_consumed=500.0)
    assert p.counters["sA"] == pytest.approx(500.0)
    assert p.pod_tenant_tokens[d.prefill_pod_id]["sA"] == pytest.approx(500.0)
    # Other pods untouched.
    for pid in ("p0", "p1", "p2"):
        if pid != d.prefill_pod_id:
            assert p.pod_tenant_tokens[pid]["sA"] == 0.0


def test_vtc_heavy_tenant_burst_without_completions_pins_F17(pod_specs):
    """F17: during a burst, tenant_debt stays zero; only `busy()` differentiates.

    Pod-level load (active_prefill) is simulated in-test: we load pod p0 so
    `busy()` clearly favors p1/p2, and verify the burst does NOT stick to p0.
    """
    cluster, kv = _fresh_3_pod_cluster(pod_specs)
    for pid in ("p0", "p1", "p2"):
        cluster.pods[pid].ewma_latency_ms = 10.0  # nonzero so busy() is meaningful
    cluster.pods["p0"].active_prefill = 5  # p0 is the heavy pod
    p = get_policy("vtc-basic")
    decisions = [p.decide(_make_req(f"r{i}", "sHeavy", float(i)), cluster, kv) for i in range(4)]
    # All tenant_debts are 0 (no completions) → tie-break falls to (busy, pod_id).
    # busy for p0 is 10.0 * 5 = 50; p1/p2 are 0 → all decisions avoid p0.
    assert all(d.prefill_pod_id != "p0" for d in decisions)


def test_vtc_heavy_tenant_steered_away_after_completion(pod_specs):
    """After a completion records tokens against a pod, future requests from that
    tenant prefer any pod with smaller tenant_debt (i.e. any other pod)."""
    cluster, kv = _fresh_3_pod_cluster(pod_specs)
    p = get_policy("vtc-basic")
    r1 = _make_req("r1", "sHeavy", 0.0)
    d1 = p.decide(r1, cluster, kv)
    p.observe_completion(r1, d1, tokens_consumed=1_000_000.0)
    # Next request: tenant_debt on d1.prefill_pod_id = 1e6; elsewhere = 0.
    r2 = _make_req("r2", "sHeavy", 1.0)
    d2 = p.decide(r2, cluster, kv)
    assert d2.prefill_pod_id != d1.prefill_pod_id


def test_vtc_light_tenant_indifferent_when_all_zero(pod_specs):
    """A light tenant with no history picks on (busy, pod_id) only."""
    cluster, kv = _fresh_3_pod_cluster(pod_specs)
    # Poison another tenant's history; this should not affect the light tenant.
    p = get_policy("vtc-basic")
    r_heavy = _make_req("rH", "sHeavy", 0.0)
    d_heavy = p.decide(r_heavy, cluster, kv)
    p.observe_completion(r_heavy, d_heavy, tokens_consumed=1_000_000.0)
    # Light tenant: no entry in pod_tenant_tokens[*]["sLight"] → all debts 0.
    d_light = p.decide(_make_req("rL", "sLight", 1.0), cluster, kv)
    # Indifferent → picks lowest pod_id since busy() is zero for idle pods.
    assert d_light.prefill_pod_id == "p0"


def test_vtc_metadata_fairness_key_used_when_present(pod_specs):
    cluster, kv = _fresh_3_pod_cluster(pod_specs)
    p = get_policy("vtc-basic", fairness_key="tenant")
    req = _make_req("r", "sIgnored", 0.0, metadata={"tenant": "T1"})
    d = p.decide(req, cluster, kv)
    p.observe_completion(req, d, tokens_consumed=100.0)
    assert p.counters["T1"] == 100.0
    assert p.counters.get("sIgnored", 0.0) == 0.0


def test_vtc_metadata_fairness_key_falls_back_to_session_id(pod_specs):
    """Missing metadata key → _key returns str(session_id)."""
    cluster, kv = _fresh_3_pod_cluster(pod_specs)
    p = get_policy("vtc-basic", fairness_key="tenant")
    req = _make_req("r", "sFallback", 0.0, metadata={})  # no `tenant` key
    d = p.decide(req, cluster, kv)
    p.observe_completion(req, d, tokens_consumed=50.0)
    assert p.counters["sFallback"] == 50.0


def test_vtc_is_deterministic_under_same_input(pod_specs):
    """Two independent instances, same input sequence → same decisions."""
    cluster1, kv1 = _fresh_3_pod_cluster(pod_specs)
    cluster2, kv2 = _fresh_3_pod_cluster(pod_specs)
    p1 = get_policy("vtc-basic")
    p2 = get_policy("vtc-basic")
    reqs = [_make_req(f"r{i}", f"s{i % 3}", float(i)) for i in range(15)]
    decisions1 = []
    decisions2 = []
    for r in reqs:
        d1 = p1.decide(r, cluster1, kv1)
        d2 = p2.decide(r, cluster2, kv2)
        decisions1.append(d1.prefill_pod_id)
        decisions2.append(d2.prefill_pod_id)
        p1.observe_completion(r, d1, 100.0)
        p2.observe_completion(r, d2, 100.0)
    assert decisions1 == decisions2


def test_vtc_counters_monotonic_when_window_unset_F16(pod_specs):
    """F16: default (window_s=None) keeps monotonic behavior for backward-compat."""
    cluster, kv = _fresh_3_pod_cluster(pod_specs)
    p = get_policy("vtc-basic")
    assert p.window_s is None
    req = _make_req("r", "sMono", 0.0)
    d = p.decide(req, cluster, kv)
    for _ in range(5):
        p.observe_completion(req, d, tokens_consumed=200.0)
    assert p.counters["sMono"] == pytest.approx(1000.0)
    # Even re-deciding far in the future does not age out consumption.
    p.decide(_make_req("r2", "sMono", 1e9), cluster, kv)
    assert p.counters["sMono"] == pytest.approx(1000.0)


def test_vtc_sliding_window_ages_out_consumption_F16(pod_specs):
    """F16: with window_s set, consumption beyond the window stops counting."""
    cluster, kv = _fresh_3_pod_cluster(pod_specs)
    p = get_policy("vtc-basic", window_s=60.0)
    # Heavy burst at t=0..2
    for i in range(5):
        r = _make_req(f"r{i}", "sHeavy", float(i))
        d = p.decide(r, cluster, kv)
        p.observe_completion(r, d, tokens_consumed=200.0)
    # Within window: still 1000
    p.decide(_make_req("probe_a", "sHeavy", 50.0), cluster, kv)
    assert p.counters["sHeavy"] == pytest.approx(1000.0)
    # Past the window: all events aged out
    p.decide(_make_req("probe_b", "sHeavy", 200.0), cluster, kv)
    assert p.counters["sHeavy"] == pytest.approx(0.0)
    for pid in ("p0", "p1", "p2"):
        assert p.pod_tenant_tokens[pid].get("sHeavy", 0.0) == pytest.approx(0.0)


def test_vtc_sliding_window_unpins_heavy_tenant_F16(pod_specs):
    """F16: after the window elapses, a previously-heavy tenant is no longer
    steered away from its earlier warmed pod."""
    cluster, kv = _fresh_3_pod_cluster(pod_specs)
    p = get_policy("vtc-basic", window_s=60.0)
    r1 = _make_req("r1", "sH", 0.0)
    d1 = p.decide(r1, cluster, kv)
    p.observe_completion(r1, d1, tokens_consumed=1_000_000.0)
    # Mid-window: should still avoid the heavy pod.
    d_mid = p.decide(_make_req("r_mid", "sH", 30.0), cluster, kv)
    assert d_mid.prefill_pod_id != d1.prefill_pod_id
    # Past the window: debt aged out; tie-break falls back to lowest pod_id.
    d_late = p.decide(_make_req("r_late", "sH", 120.0), cluster, kv)
    assert d_late.prefill_pod_id == "p0"


def test_vtc_reset_clears_all_state_F16(pod_specs):
    """F16: reset() exposed for experiment isolation."""
    cluster, kv = _fresh_3_pod_cluster(pod_specs)
    p = get_policy("vtc-basic")
    r = _make_req("r", "sX", 0.0)
    d = p.decide(r, cluster, kv)
    p.observe_completion(r, d, tokens_consumed=500.0)
    assert p.counters["sX"] == 500.0
    p.reset()
    assert p.counters == {}
    assert p.pod_tenant_tokens == {}


def test_vtc_empty_cluster_returns_none_sentinel(pod_specs):
    empty = ClusterState.from_specs([])
    kv = KVCacheState.from_specs({})
    p = get_policy("vtc-basic")
    d = p.decide(_make_req("r", "s", 0.0), empty, kv)
    assert d.prefill_pod_id == "__none__"


def test_vtc_single_pod_cluster(pod_specs):
    one = [pod_specs[0]]
    cluster = ClusterState.from_specs(one)
    kv = KVCacheState.from_specs({one[0].pod_id: one[0].kv_cache_bytes})
    p = get_policy("vtc-basic")
    # Heavy tenant with history; still must pick the only pod.
    r1 = _make_req("r1", "sA", 0.0)
    d1 = p.decide(r1, cluster, kv)
    p.observe_completion(r1, d1, tokens_consumed=1e9)
    d2 = p.decide(_make_req("r2", "sA", 1.0), cluster, kv)
    assert d2.prefill_pod_id == one[0].pod_id


# =============================================================================
# Integration: vtc-basic end-to-end through the engine spreads a heavy tenant
# =============================================================================


def test_vtc_spreads_heavy_tenant_across_pods_under_engine(
    pod_specs, compute_params, network_params, scheduler_params
):
    """On a heavy-user+light-user trace, vtc-basic must spread the heavy
    tenant's requests across pods. Exact counts are engine-dependent, but
    the heavy tenant must not be pinned to a single pod."""
    from routing_harness.cost_model import AnalyticCostModel
    from routing_harness.simulator.engine import EngineConfig, SimulationEngine
    from routing_harness.simulator.metrics import MetricsCollector

    cluster = ClusterState.from_specs(pod_specs)
    kv = KVCacheState.from_specs({s.pod_id: s.kv_cache_bytes for s in pod_specs})
    policy = get_policy("vtc-basic")
    cost_model = AnalyticCostModel(
        compute=compute_params, network=network_params, scheduler=scheduler_params
    )
    engine = SimulationEngine(
        cluster=cluster,
        kv_cache=kv,
        policy=policy,
        cost_model=cost_model,
        network=network_params,
        config=EngineConfig(block_size=16),
        metrics=MetricsCollector(),
    )

    # 20 heavy-tenant requests interleaved with 20 light-tenant requests.
    trace = []
    for i in range(20):
        trace.append(
            Request(
                request_id=f"H{i}",
                session_id="heavy",
                arrival_ts=i * 0.05,  # 20 QPS
                prompt_tokens=tuple(range(64)),
                max_output_tokens=32,
            )
        )
        trace.append(
            Request(
                request_id=f"L{i}",
                session_id=f"light_{i % 5}",
                arrival_ts=i * 0.05 + 0.025,
                prompt_tokens=tuple(range(32)),
                max_output_tokens=16,
            )
        )

    metrics = engine.run(trace)
    # Count heavy-tenant dispatches per pod.
    heavy_pod_counts: dict[str, int] = {pid: 0 for pid in cluster.pods}
    for rec in metrics.records:
        if rec.request.session_id == "heavy":
            heavy_pod_counts[rec.decision.prefill_pod_id] += 1
    # Require the heavy tenant to touch at least 2 of the 3 pods.
    touched = sum(1 for c in heavy_pod_counts.values() if c > 0)
    assert touched >= 2, (
        f"vtc-basic failed to spread heavy tenant: {heavy_pod_counts}"
    )
    # And the max-single-pod share must be under 90% (spread, not pinned).
    total_heavy = sum(heavy_pod_counts.values())
    assert total_heavy == 20
    assert max(heavy_pod_counts.values()) / total_heavy < 0.9, (
        f"heavy tenant too concentrated: {heavy_pod_counts}"
    )
