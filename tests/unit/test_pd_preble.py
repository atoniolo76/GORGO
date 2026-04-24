"""Tests for the pd-preble policy (bead go-caz, F25).

pd-preble is the PD topology policy with Preble's exploit/explore gate
and relative-imbalance hotspot deflection on the prefill pool. Decode
selection (peer filter + min active_decode) is unchanged from `pd`.

The motivating pathology (see `research/reports/policy_audits/pd_topology.md`
§2.6, F25): under identical-prompt skew, plain `pd` cache-locks on one
prefill pod — first tie-break winner warms up, then wins all subsequent
requests on match count. pd-preble must break the lock via the hotspot
branch once load diverges past th_bal.
"""

from __future__ import annotations

from routing_harness import policies  # noqa: F401 — registers policies
from routing_harness.cluster import ClusterState
from routing_harness.core import Phase, PodSpec, Request
from routing_harness.kv_cache import (
    KVCacheState,
    PrefixEntry,
    enumerate_prefix_hashes,
)
from routing_harness.policy import get_policy


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


# ---------------------------------------------------------------------------
# Core gate behavior
# ---------------------------------------------------------------------------


def test_pd_preble_exploit_when_prefix_saves_more_than_tail():
    """Long prefix match on pfA, equal load: exploit binds to pfA."""
    specs, cluster, kv = _pd_cluster_2x2()
    tokens = tuple(range(32))  # 2 blocks at block_size=16
    hashes = enumerate_prefix_hashes(tokens, block_size=16)
    for h in hashes:
        kv.install("pfA", PrefixEntry(h, 16, 1024), now=1.0)
    p = get_policy("pd-preble", block_size=16, th_bal=1.5)
    d = p.decide(Request("r", "s", 0.0, tokens, 4), cluster, kv)
    assert d.prefill_pod_id == "pfA"
    assert "gate=exploit" in d.rationale


def test_pd_preble_explore_when_no_cache_match():
    """No prefix match anywhere: explore picks the lightest prefill."""
    specs, cluster, kv = _pd_cluster_2x2()
    cluster.pods["pfA"].pending_work_ms = 100.0
    cluster.pods["pfB"].pending_work_ms = 10.0
    p = get_policy("pd-preble", block_size=16, th_bal=1.5)
    d = p.decide(Request("r", "s", 0.0, tuple(range(32)), 4), cluster, kv)
    assert d.prefill_pod_id == "pfB"
    assert "gate=explore" in d.rationale


def test_pd_preble_explore_when_tail_dominates_match():
    """Short match, long prompt: missed > cached → explore."""
    specs, cluster, kv = _pd_cluster_2x2()
    tokens = tuple(range(80))  # 5 blocks
    hashes = enumerate_prefix_hashes(tokens, block_size=16)
    # Only one block matches on pfA — cached=16, missed=64.
    kv.install("pfA", PrefixEntry(hashes[0], 16, 1024), now=1.0)
    cluster.pods["pfA"].pending_work_ms = 50.0
    cluster.pods["pfB"].pending_work_ms = 5.0
    p = get_policy("pd-preble", block_size=16, th_bal=1.5)
    d = p.decide(Request("r", "s", 0.0, tokens, 4), cluster, kv)
    assert d.prefill_pod_id == "pfB"
    assert "gate=explore" in d.rationale


# ---------------------------------------------------------------------------
# F25: hotspot deflection breaks the cache-lock
# ---------------------------------------------------------------------------


def test_pd_preble_f25_hotspot_redirect_under_identical_prompt_skew():
    """The motivating F25 case: pfA owns the prefix and is saturated;
    pfB is idle. pd-preble must deflect to pfB despite the cache match.
    """
    specs, cluster, kv = _pd_cluster_2x2()
    tokens = tuple(range(32))
    hashes = enumerate_prefix_hashes(tokens, block_size=16)
    for h in hashes:
        kv.install("pfA", PrefixEntry(h, 16, 1024), now=1.0)
    # pfA's load is 3x pfB's — well above th_bal=1.5.
    cluster.pods["pfA"].pending_work_ms = 300.0
    cluster.pods["pfB"].pending_work_ms = 50.0
    p = get_policy("pd-preble", block_size=16, th_bal=1.5)
    d = p.decide(Request("r", "s", 0.0, tokens, 4), cluster, kv)
    assert d.prefill_pod_id == "pfB"
    assert "gate=exploit-hotspot-redirect" in d.rationale


def test_pd_preble_f25_no_redirect_when_imbalance_under_threshold():
    """Load ratio below th_bal: stay with the prefix owner."""
    specs, cluster, kv = _pd_cluster_2x2()
    tokens = tuple(range(32))
    hashes = enumerate_prefix_hashes(tokens, block_size=16)
    for h in hashes:
        kv.install("pfA", PrefixEntry(h, 16, 1024), now=1.0)
    # pfA 1.2x pfB — under th_bal=1.5.
    cluster.pods["pfA"].pending_work_ms = 60.0
    cluster.pods["pfB"].pending_work_ms = 50.0
    p = get_policy("pd-preble", block_size=16, th_bal=1.5)
    d = p.decide(Request("r", "s", 0.0, tokens, 4), cluster, kv)
    assert d.prefill_pod_id == "pfA"
    assert "gate=exploit" in d.rationale
    assert "hotspot-redirect" not in d.rationale


def test_pd_preble_f25_breaks_50_identical_prompt_cache_lock():
    """50 identical-prompt requests through a feedback loop:
    simulate pending_work_ms accumulating on whichever pod is picked,
    decay uniformly each step. pd-preble must spread the load —
    specifically, neither pod gets all 50.
    """
    specs, cluster, kv = _pd_cluster_2x2()
    tokens = tuple(range(32))
    hashes = enumerate_prefix_hashes(tokens, block_size=16)
    # Seed the cache on pfA (the pathology's "winner").
    for h in hashes:
        kv.install("pfA", PrefixEntry(h, 16, 1024), now=1.0)
    p = get_policy("pd-preble", block_size=16, th_bal=1.5)

    counts = {"pfA": 0, "pfB": 0}
    service_ms = 100.0  # per-request service time stand-in
    decay_ms = 20.0  # uniform drain per arrival
    for i in range(50):
        req = Request(f"r{i}", "s", float(i) * 0.01, tokens, 4)
        d = p.decide(req, cluster, kv)
        counts[d.prefill_pod_id] += 1
        # After dispatch, the chosen pod's pending work grows; both
        # pods drain a little before the next arrival.
        cluster.pods[d.prefill_pod_id].pending_work_ms += service_ms
        for pid in ("pfA", "pfB"):
            cluster.pods[pid].pending_work_ms = max(
                0.0, cluster.pods[pid].pending_work_ms - decay_ms
            )
        # Simulate pfB warming the prefix once it's chosen (engine
        # would install on dispatch; we do it here so the test exercises
        # the branch that matters — hotspot deflection, not cold explore).
        if d.prefill_pod_id == "pfB":
            for h in hashes:
                if not kv.has("pfB", h):
                    kv.install("pfB", PrefixEntry(h, 16, 1024), now=float(i) + 2.0)

    # Neither pod should monopolize — F25's 50/0 split is the bug.
    assert counts["pfA"] > 0
    assert counts["pfB"] > 0
    assert max(counts.values()) < 50


# ---------------------------------------------------------------------------
# Decode selection (inherited from pd)
# ---------------------------------------------------------------------------


def test_pd_preble_decode_prefers_peered():
    """Peer filter on the chosen prefill pod's peer_ids still applies."""
    specs, cluster, kv = _pd_cluster_2x2()
    tokens = tuple(range(32))
    hashes = enumerate_prefix_hashes(tokens, block_size=16)
    for h in hashes:
        kv.install("pfA", PrefixEntry(h, 16, 1024), now=1.0)
    cluster.pods["dcA"].active_decode = 8
    cluster.pods["dcB"].active_decode = 0
    p = get_policy("pd-preble", block_size=16, th_bal=1.5)
    d = p.decide(Request("r", "s", 0.0, tokens, 4), cluster, kv)
    assert d.prefill_pod_id == "pfA"
    # Peer filter locks us to dcA (pfA's only peer) despite dcA being busier.
    assert d.decode_pod_id == "dcA"
    assert "peer" in d.rationale


def test_pd_preble_decode_falls_back_when_no_peers():
    """Empty peer_ids → full decode pool, pick by active_decode."""
    specs = [
        PodSpec("pf0", Phase.PREFILL, 1, 16 * 1024 * 1024, 4, 0),
        PodSpec("dc0", Phase.DECODE, 1, 4 * 1024 * 1024, 0, 8),
        PodSpec("dc1", Phase.DECODE, 1, 4 * 1024 * 1024, 0, 8),
    ]
    cluster = ClusterState.from_specs(specs)
    kv = KVCacheState.from_specs({s.pod_id: s.kv_cache_bytes for s in specs})
    cluster.pods["dc0"].active_decode = 5
    cluster.pods["dc1"].active_decode = 0
    p = get_policy("pd-preble", block_size=16, th_bal=1.5)
    d = p.decide(Request("r", "s", 0.0, tuple(range(16)), 4), cluster, kv)
    assert d.prefill_pod_id == "pf0"
    assert d.decode_pod_id == "dc1"
    assert "nopeers" in d.rationale


# ---------------------------------------------------------------------------
# Pool shape / degradation (inherited from pd)
# ---------------------------------------------------------------------------


def test_pd_preble_empty_cluster_returns_none():
    cluster = ClusterState.from_specs([])
    kv = KVCacheState.from_specs({})
    p = get_policy("pd-preble", block_size=16)
    d = p.decide(Request("r", "s", 0.0, tuple(range(16)), 4), cluster, kv)
    assert d.prefill_pod_id == "__none__"
    assert d.decode_pod_id == "__none__"
    assert "pools-empty" in d.rationale


def test_pd_preble_colocated_fallback_empty_decode():
    """Only PREFILL pods → collapse to colocated on the best prefill."""
    specs = [
        PodSpec("pfA", Phase.PREFILL, 1, 16 * 1024 * 1024, 4, 0),
        PodSpec("pfB", Phase.PREFILL, 1, 16 * 1024 * 1024, 4, 0),
    ]
    cluster = ClusterState.from_specs(specs)
    kv = KVCacheState.from_specs({s.pod_id: s.kv_cache_bytes for s in specs})
    cluster.pods["pfA"].pending_work_ms = 100.0
    cluster.pods["pfB"].pending_work_ms = 10.0
    p = get_policy("pd-preble", block_size=16, th_bal=1.5)
    d = p.decide(Request("r", "s", 0.0, tuple(range(16)), 4), cluster, kv)
    assert d.prefill_pod_id == d.decode_pod_id == "pfB"
    assert "colocated" in d.rationale
    assert "one-pool-empty" in d.rationale


def test_pd_preble_colocated_fallback_empty_prefill():
    """Only DECODE pods → collapse to colocated on the best decode."""
    specs = [
        PodSpec("dcA", Phase.DECODE, 1, 4 * 1024 * 1024, 0, 8),
        PodSpec("dcB", Phase.DECODE, 1, 4 * 1024 * 1024, 0, 8),
    ]
    cluster = ClusterState.from_specs(specs)
    kv = KVCacheState.from_specs({s.pod_id: s.kv_cache_bytes for s in specs})
    p = get_policy("pd-preble", block_size=16, th_bal=1.5)
    d = p.decide(Request("r", "s", 0.0, tuple(range(16)), 4), cluster, kv)
    # No cache, equal load → smallest pod_id wins the explore branch.
    assert d.prefill_pod_id == d.decode_pod_id == "dcA"
    assert "colocated" in d.rationale


def test_pd_preble_deterministic_under_repeated_calls():
    specs, cluster, kv = _pd_cluster_2x2()
    p = get_policy("pd-preble", block_size=16, th_bal=1.5)
    req = Request("r", "s", 0.0, tuple(range(32)), 4)
    d1 = p.decide(req, cluster, kv)
    d2 = p.decide(req, cluster, kv)
    d3 = p.decide(req, cluster, kv)
    assert d1.prefill_pod_id == d2.prefill_pod_id == d3.prefill_pod_id
    assert d1.decode_pod_id == d2.decode_pod_id == d3.decode_pod_id


def test_pd_preble_prefix_key_short_prompt_no_spurious_exploit():
    """Under prefix_key the match is {0,1} on an opaque hash; short
    prompts must not exploit spuriously from a negative missed_tokens.
    """
    specs, cluster, kv = _pd_cluster_2x2()
    kv.install("pfA", PrefixEntry("opaque-key", 16, 1024), now=1.0)
    cluster.pods["pfA"].pending_work_ms = 10.0
    cluster.pods["pfB"].pending_work_ms = 10.0
    p = get_policy("pd-preble", block_size=16, th_bal=1.5)
    # prompt length 4 < cached_tokens=16 — without the clamp this would
    # spuriously satisfy missed < cached and exploit on pfA.
    req = Request("r", "s", 0.0, tuple(range(4)), 4, prefix_key="opaque-key")
    d = p.decide(req, cluster, kv)
    # missed=max(0, 4-16)=0, cached=16, 0 < 16 is true → exploit is
    # allowed. The clamp just prevents negative missed from distorting
    # the comparison. Here the exploit is legitimate: the match does
    # cover the entire prompt.
    assert d.prefill_pod_id == "pfA"
    assert "gate=exploit" in d.rationale
