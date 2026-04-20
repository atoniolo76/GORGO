"""Per-policy targeted behavior tests.

Contract tests cover shared invariants; these tests pin down the
behavior that distinguishes each policy.
"""

from __future__ import annotations

import pytest

from routing_harness import policies  # noqa: F401
from routing_harness.core import Decision, PodSpec, Phase, Request
from routing_harness.cluster import ClusterState
from routing_harness.kv_cache import KVCacheState, PrefixEntry, enumerate_prefix_hashes
from routing_harness.policy import get_policy


def test_random_is_deterministic_under_seed(cluster, kv_cache):
    p1 = get_policy("random", seed=42)
    p2 = get_policy("random", seed=42)
    req = Request("r", "s", 0.0, (1, 2, 3), 4)
    assert p1.decide(req, cluster, kv_cache).prefill_pod_id == p2.decide(req, cluster, kv_cache).prefill_pod_id


def test_least_request_picks_min_load(pod_specs, kv_cache):
    cluster = ClusterState.from_specs(pod_specs)
    cluster.pods["p0"].active_prefill = 5
    cluster.pods["p1"].active_prefill = 0
    cluster.pods["p2"].active_prefill = 2
    p = get_policy("least-request")
    d = p.decide(Request("r", "s", 0.0, (1, 2), 1), cluster, kv_cache)
    assert d.prefill_pod_id == "p1"


def test_prefix_cache_routes_to_owner(cluster, kv_cache):
    tokens = tuple(range(32))
    hashes = enumerate_prefix_hashes(tokens, block_size=16)
    for h in hashes:
        kv_cache.install("p2", PrefixEntry(h, 16, 1024), now=1.0)
    p = get_policy("prefix-cache", block_size=16)
    d = p.decide(Request("r", "s", 2.0, tokens, 4), cluster, kv_cache)
    assert d.prefill_pod_id == "p2"


def test_prefix_cache_preble_avoids_hotspot(pod_specs, kv_cache):
    cluster = ClusterState.from_specs(pod_specs)
    tokens = tuple(range(32))
    hashes = enumerate_prefix_hashes(tokens, block_size=16)
    for h in hashes:
        # Prefix lives ONLY on p0 (mono-homing).
        kv_cache.install("p0", PrefixEntry(h, 16, 1024), now=1.0)
    # p0 is a hotspot: high pending_work_ms from in-flight requests.
    cluster.pods["p0"].active_prefill = 20
    cluster.pods["p0"].pending_work_ms = 1000.0  # 1s of pending work
    # p1 and p2 are idle (pending_work_ms = 0).
    p = get_policy("prefix-cache-preble", block_size=16, th_bal=1.5)
    d = p.decide(Request("r", "s", 3.0, tokens, 4), cluster, kv_cache)
    # Exploit branch fires (cached > missed) but hotspot redirect
    # steers away from p0 because p0.load_ms >> th_bal * min_load.
    # No match>0 gate — redirect works even under mono-homing.
    assert d.prefill_pod_id != "p0"
    assert "hotspot-redirect" in d.rationale


def test_prefix_cache_preble_exploit_binds_to_owner(pod_specs, kv_cache):
    """When no hotspot, exploit binds to the prefix owner."""
    cluster = ClusterState.from_specs(pod_specs)
    tokens = tuple(range(32))
    hashes = enumerate_prefix_hashes(tokens, block_size=16)
    for h in hashes:
        kv_cache.install("p2", PrefixEntry(h, 16, 1024), now=1.0)
    p = get_policy("prefix-cache-preble", block_size=16)
    d = p.decide(Request("r", "s", 2.0, tokens, 4), cluster, kv_cache)
    assert d.prefill_pod_id == "p2"
    assert "exploit" in d.rationale


def test_prefix_cache_preble_explore_picks_lightest(pod_specs, kv_cache):
    """When cache reuse is insufficient, explore picks lightest pod."""
    cluster = ClusterState.from_specs(pod_specs)
    # Short prompt with no cached prefix → missed >= cached → explore.
    tokens = tuple(range(48))
    cluster.pods["p0"].active_prefill = 5
    cluster.pods["p0"].pending_work_ms = 500.0
    cluster.pods["p1"].active_prefill = 1
    cluster.pods["p1"].pending_work_ms = 100.0
    # p2 is idle → lightest (pending_work_ms = 0).
    p = get_policy("prefix-cache-preble", block_size=16)
    d = p.decide(Request("r", "s", 2.0, tokens, 4), cluster, kv_cache)
    assert d.prefill_pod_id == "p2"
    assert "explore" in d.rationale


def test_session_affinity_sticks(cluster, kv_cache):
    p = get_policy("session-affinity", stickiness_ttl_s=3600.0)
    req1 = Request("r1", "sX", 1.0, (1, 2, 3), 4)
    req2 = Request("r2", "sX", 2.0, (9, 8, 7), 4)
    d1 = p.decide(req1, cluster, kv_cache)
    d2 = p.decide(req2, cluster, kv_cache)
    assert d1.prefill_pod_id == d2.prefill_pod_id


def test_pd_separates_roles(pd_specs, kv_cache):
    cluster = ClusterState.from_specs(pd_specs)
    kv_cache = KVCacheState.from_specs({s.pod_id: s.kv_cache_bytes for s in pd_specs})
    p = get_policy("pd", block_size=16)
    d = p.decide(Request("r", "s", 0.0, tuple(range(32)), 8), cluster, kv_cache)
    assert cluster.pods[d.prefill_pod_id].spec.role == Phase.PREFILL
    assert cluster.pods[d.decode_pod_id].spec.role == Phase.DECODE


def test_vtc_fairness_annotation(cluster, kv_cache):
    p = get_policy("vtc-basic")
    # Route a heavy tenant's request first so the engine records their
    # tokens against a specific pod, then observe completion.
    r_heavy = Request("r1", "sA", 0.0, (1, 2), 1)
    d_first = p.decide(r_heavy, cluster, kv_cache)
    p.observe_completion(r_heavy, d_first, tokens_consumed=1_000_000)
    r_light = Request("r2", "sB", 0.1, (1, 2), 1)
    d_heavy = p.decide(r_heavy, cluster, kv_cache)
    d_light = p.decide(r_light, cluster, kv_cache)
    # Score reflects the heavy tenant's global counter (-tokens); the
    # light tenant has zero.
    assert (d_heavy.score or 0) < (d_light.score or 0)
    # The heavy tenant should now be steered *away* from the pod where
    # they just consumed tokens, while the light tenant has no debt to
    # avoid — so the two decisions should differ on at least the
    # heavy-tenant's former pod.
    assert d_heavy.prefill_pod_id != d_first.prefill_pod_id or len(cluster.pods) == 1
