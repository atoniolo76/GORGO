"""Unit tests for the gorgo policy (port of main's ``route_gorgo``).

Covers the scoring rule's structural properties:

* ``decide`` returns a ``__none__`` Decision when the cluster has no
  prefill-capable pods.
* The score is monotonic in each of its three terms (latency, uncached
  prefill tokens, and queue+used-kv tokens).
* Tie-break is deterministic on pod_id (lowest wins) when scores are
  equal.
* Setting ``t_prefill = 0`` collapses the rule to ``latency + queue``
  (no prefix preference).
* Setting ``queued_tokens_weight = 0`` collapses the rule to a
  prefix-cache + latency policy (no load awareness).
* Stale signals (large ``ewma_latency_ms``) propagate as expected
  through the score (a known-good pod with stale latency loses to a
  fresh one).
* Empty prompt + cached match yields the no-prefill term clamped at 0
  (no negative scoring).
"""

from __future__ import annotations

from routing_harness import policies  # noqa: F401 — register built-ins
from routing_harness.cluster import ClusterState
from routing_harness.core import Phase, PodSpec, Request
from routing_harness.kv_cache import KVCacheState, PrefixEntry, enumerate_prefix_hashes
from routing_harness.policies.gorgo import GorgoPolicy
from routing_harness.policy import get_policy


def _three_pods(kv_cache_bytes: int = 8 * 1024 * 1024) -> list[PodSpec]:
    return [
        PodSpec(
            pod_id=f"p{i}",
            role=Phase.BOTH,
            gpu_count=1,
            kv_cache_bytes=kv_cache_bytes,
            max_concurrent_prefill=4,
            max_concurrent_decode=8,
        )
        for i in range(3)
    ]


def _fresh_state(specs: list[PodSpec]) -> tuple[ClusterState, KVCacheState]:
    cluster = ClusterState.from_specs(specs)
    kv = KVCacheState.from_specs({s.pod_id: s.kv_cache_bytes for s in specs})
    return cluster, kv


def test_no_prefill_capable_returns_sentinel():
    # All decode-only pods → no prefill_capable() candidates.
    specs = [
        PodSpec("d0", Phase.DECODE, 1, 8 * 1024 * 1024, 0, 8),
        PodSpec("d1", Phase.DECODE, 1, 8 * 1024 * 1024, 0, 8),
    ]
    cluster, kv = _fresh_state(specs)
    p = GorgoPolicy()
    d = p.decide(Request("r", "s", 0.0, (1, 2, 3), 4), cluster, kv)
    assert d.prefill_pod_id == "__none__"
    assert d.decode_pod_id == "__none__"


def test_registered_via_get_policy():
    p = get_policy("gorgo", block_size=16, t_prefill=0.05, queued_tokens_weight=0.001)
    assert p.policy_id == "gorgo"
    assert isinstance(p, GorgoPolicy)


def test_picks_lowest_latency_when_only_signal_differs():
    specs = _three_pods()
    cluster, kv = _fresh_state(specs)
    cluster.pods["p0"].ewma_latency_ms = 50.0
    cluster.pods["p1"].ewma_latency_ms = 10.0
    cluster.pods["p2"].ewma_latency_ms = 30.0
    p = GorgoPolicy()
    d = p.decide(Request("r", "s", 0.0, (1, 2, 3), 4), cluster, kv)
    assert d.prefill_pod_id == "p1"


def test_prefix_match_pulls_traffic_to_owner():
    specs = _three_pods()
    cluster, kv = _fresh_state(specs)
    # Equal latency everywhere, no queue → prefix term is the only
    # discriminator. Install full prefix on p2.
    for pod in cluster.pods.values():
        pod.ewma_latency_ms = 10.0
    tokens = tuple(range(64))
    hashes = enumerate_prefix_hashes(tokens, block_size=16)
    for h in hashes:
        kv.install("p2", PrefixEntry(h, 16, 1024), now=1.0)
    p = GorgoPolicy(block_size=16, t_prefill=0.5, queued_tokens_weight=0.0)
    d = p.decide(Request("r", "s", 2.0, tokens, 4), cluster, kv)
    assert d.prefill_pod_id == "p2"


def test_queue_signal_deflects_from_busy_pod():
    specs = _three_pods()
    cluster, kv = _fresh_state(specs)
    for pod in cluster.pods.values():
        pod.ewma_latency_ms = 10.0
    # p0 is the natural pick on latency tie (lowest pod_id) — pile on
    # queued_prompt_tokens to flip the decision.
    cluster.pods["p0"].queued_prompt_tokens = 100_000
    p = GorgoPolicy(t_prefill=0.0, queued_tokens_weight=1.0)
    d = p.decide(Request("r", "s", 0.0, (1, 2, 3), 4), cluster, kv)
    assert d.prefill_pod_id != "p0"


def test_used_kv_tokens_contribute_to_queue_term():
    specs = _three_pods()
    cluster, kv = _fresh_state(specs)
    for pod in cluster.pods.values():
        pod.ewma_latency_ms = 10.0
    # Install many prefix entries on p0 so its used_kv_tokens dominates.
    # Use distinct hashes to avoid LRU update of the same key.
    for i in range(50):
        kv.install("p0", PrefixEntry(f"h{i}", 16, 1024), now=1.0)
    p = GorgoPolicy(t_prefill=0.0, queued_tokens_weight=1.0)
    d = p.decide(Request("r", "s", 0.0, (1, 2, 3), 4), cluster, kv)
    assert d.prefill_pod_id != "p0"


def test_tie_break_is_lowest_pod_id():
    specs = _three_pods()
    cluster, kv = _fresh_state(specs)
    # All equal: latency tied, queue tied, no prefix anywhere.
    for pod in cluster.pods.values():
        pod.ewma_latency_ms = 10.0
    p = GorgoPolicy()
    d = p.decide(Request("r", "s", 0.0, (1, 2, 3), 4), cluster, kv)
    # Stable min over pod_id → smallest wins.
    assert d.prefill_pod_id == "p0"
    # Determinism: two calls give the same answer.
    d2 = p.decide(Request("r2", "s", 1.0, (4, 5, 6), 4), cluster, kv)
    assert d2.prefill_pod_id == "p0"


def test_t_prefill_zero_reduces_to_latency_plus_queue():
    """t_prefill=0 must remove the prefix-cache bias entirely."""
    specs = _three_pods()
    cluster, kv = _fresh_state(specs)
    for pod in cluster.pods.values():
        pod.ewma_latency_ms = 10.0
    tokens = tuple(range(64))
    hashes = enumerate_prefix_hashes(tokens, block_size=16)
    # Owner has the *whole* prefix but is also slightly slower.
    for h in hashes:
        kv.install("p2", PrefixEntry(h, 16, 1024), now=1.0)
    cluster.pods["p2"].ewma_latency_ms = 11.0
    p = GorgoPolicy(block_size=16, t_prefill=0.0, queued_tokens_weight=0.0)
    d = p.decide(Request("r", "s", 2.0, tokens, 4), cluster, kv)
    # Prefix should not pull traffic to p2 once t_prefill=0; tied
    # latency on p0/p1, lower pod_id wins.
    assert d.prefill_pod_id == "p0"


def test_queued_tokens_weight_zero_reduces_to_prefix_plus_latency():
    """queued_tokens_weight=0 must remove load awareness."""
    specs = _three_pods()
    cluster, kv = _fresh_state(specs)
    for pod in cluster.pods.values():
        pod.ewma_latency_ms = 10.0
    # p1 is the prefix owner, but is buried in queue + KV.
    tokens = tuple(range(64))
    hashes = enumerate_prefix_hashes(tokens, block_size=16)
    for h in hashes:
        kv.install("p1", PrefixEntry(h, 16, 1024), now=1.0)
    cluster.pods["p1"].queued_prompt_tokens = 1_000_000
    # Weight=0 → queue/KV term has no influence; prefix wins.
    p = GorgoPolicy(block_size=16, t_prefill=0.5, queued_tokens_weight=0.0)
    d = p.decide(Request("r", "s", 2.0, tokens, 4), cluster, kv)
    assert d.prefill_pod_id == "p1"


def test_score_monotonic_in_uncached_tail():
    """Holding everything else equal, longer uncached tails increase
    score linearly with t_prefill. Verifies the prefill term's slope."""
    specs = _three_pods()
    cluster, kv = _fresh_state(specs)
    for pod in cluster.pods.values():
        pod.ewma_latency_ms = 5.0
    p = GorgoPolicy(block_size=16, t_prefill=0.1, queued_tokens_weight=0.0)
    short_req = Request("r1", "s", 0.0, tuple(range(16)), 4)
    long_req = Request("r2", "s", 0.0, tuple(range(160)), 4)
    d_short = p.decide(short_req, cluster, kv)
    d_long = p.decide(long_req, cluster, kv)
    # Same pod won (tie-break lowest pod_id), but the long request's
    # score is strictly higher than the short one's.
    assert d_short.score is not None and d_long.score is not None
    assert d_long.score > d_short.score
    # Score differs by exactly (160 - 16) * 0.1 = 14.4.
    assert abs((d_long.score - d_short.score) - 14.4) < 1e-9


def test_stale_latency_propagates_to_score():
    """Pods with stale-but-large ewma_latency_ms should be avoided —
    confirms gorgo doesn't ignore the latency signal even when the
    other terms favor that pod."""
    specs = _three_pods()
    cluster, kv = _fresh_state(specs)
    cluster.pods["p0"].ewma_latency_ms = 1000.0  # stale, very slow
    cluster.pods["p1"].ewma_latency_ms = 5.0
    cluster.pods["p2"].ewma_latency_ms = 5.0
    # Install a long prefix on p0 so the prefix term *would* prefer it
    # if latency were ignored.
    tokens = tuple(range(160))
    hashes = enumerate_prefix_hashes(tokens, block_size=16)
    for h in hashes:
        kv.install("p0", PrefixEntry(h, 16, 1024), now=1.0)
    # With t_prefill=0.05 default, max prefix savings = 160 * 0.05 = 8.
    # That can't outweigh the ~995ms latency gap.
    p = GorgoPolicy(block_size=16, t_prefill=0.05, queued_tokens_weight=0.001)
    d = p.decide(Request("r", "s", 2.0, tokens, 4), cluster, kv)
    assert d.prefill_pod_id != "p0"


def test_uncached_tail_clamped_at_zero():
    """When the cached prefix exceeds the actual prompt length (can
    happen under prefix_key opaque hashing), the prefill term must not
    go negative."""
    specs = _three_pods()
    cluster, kv = _fresh_state(specs)
    for pod in cluster.pods.values():
        pod.ewma_latency_ms = 10.0
    # Prefix-key path: install one opaque block on p2 — the full
    # cached_tokens (16) exceeds the request's 8-token prompt.
    kv.install("p2", PrefixEntry("opaque", 16, 1024), now=1.0)
    # Same opaque key on p0 too, so the prefix term ties between p0/p2.
    kv.install("p0", PrefixEntry("opaque", 16, 1024), now=1.0)
    p = GorgoPolicy(block_size=16, t_prefill=10.0, queued_tokens_weight=0.0)
    short = Request(
        "r", "s", 0.0, tuple(range(8)), 4, prefix_key="opaque"
    )
    d = p.decide(short, cluster, kv)
    # Score must be exactly latency (10.0), not 10 + (8 - 16) * 10 = -70.
    assert d.score is not None
    assert d.score == 10.0
