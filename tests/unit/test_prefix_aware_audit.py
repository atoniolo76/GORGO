"""Audit-driven tests for prefix-aware policies (bead go-5m8).

Covers edge cases not exercised by `test_policies_individual.py`:
cold start, single-pod cluster, tie-breaks, prefix_key path, exploit
gate boundary, hotspot threshold boundary, cache eviction, no-match
fallback. See `research/reports/policy_audits/prefix_aware.md`.
"""

from __future__ import annotations

from routing_harness import policies  # noqa: F401 — registers policies
from routing_harness.cluster import ClusterState
from routing_harness.core import Phase, PodSpec, Request
from routing_harness.kv_cache import KVCacheState, PrefixEntry, enumerate_prefix_hashes
from routing_harness.policy import get_policy

# ---------- helpers ----------


def _fresh(pod_specs):
    cluster = ClusterState.from_specs(pod_specs)
    kv = KVCacheState.from_specs({s.pod_id: s.kv_cache_bytes for s in pod_specs})
    return cluster, kv


def _install_prefix(kv, pod_id, hashes, *, now=1.0, byte_size=1024, tok_count=16):
    for h in hashes:
        kv.install(pod_id, PrefixEntry(h, tok_count, byte_size), now=now)


# ---------- prefix_cache: shared-behavior ----------


def test_prefix_cache_cold_start_falls_back_to_lrq(pod_specs):
    """Empty cache, zero load → no-match fallback → lowest pod_id."""
    cluster, kv = _fresh(pod_specs)
    p = get_policy("prefix-cache", block_size=16)
    d = p.decide(Request("r", "s", 0.0, tuple(range(32)), 4), cluster, kv)
    assert d.prefill_pod_id == "p0"
    assert "fallback" in d.rationale


def test_prefix_cache_no_match_uses_least_request(pod_specs):
    """No cache hit anywhere; fallback picks the least-loaded pod."""
    cluster, kv = _fresh(pod_specs)
    cluster.pods["p0"].active_prefill = 5
    cluster.pods["p1"].active_prefill = 1
    cluster.pods["p2"].active_prefill = 3
    p = get_policy("prefix-cache", block_size=16)
    d = p.decide(Request("r", "s", 0.0, tuple(range(32)), 4), cluster, kv)
    assert d.prefill_pod_id == "p1"


def test_prefix_cache_tie_on_match_picks_lowest_pod_id(pod_specs):
    """Two pods have identical prefix match → lowest pod_id wins."""
    cluster, kv = _fresh(pod_specs)
    tokens = tuple(range(32))
    hashes = enumerate_prefix_hashes(tokens, block_size=16)
    _install_prefix(kv, "p0", hashes)
    _install_prefix(kv, "p2", hashes)
    p = get_policy("prefix-cache", block_size=16)
    d = p.decide(Request("r", "s", 2.0, tokens, 4), cluster, kv)
    assert d.prefill_pod_id == "p0"  # lowest pod_id


def test_prefix_cache_longer_match_wins_over_shorter(pod_specs):
    """p0 has 1 block, p2 has 2 blocks → p2 wins."""
    cluster, kv = _fresh(pod_specs)
    tokens = tuple(range(32))
    hashes = enumerate_prefix_hashes(tokens, block_size=16)
    _install_prefix(kv, "p0", hashes[:1])
    _install_prefix(kv, "p2", hashes)
    p = get_policy("prefix-cache", block_size=16)
    d = p.decide(Request("r", "s", 2.0, tokens, 4), cluster, kv)
    assert d.prefill_pod_id == "p2"
    assert d.score == 2.0


def test_prefix_cache_prefix_key_single_block_path(pod_specs):
    """When request.prefix_key is set, policy treats it as a single-block match."""
    cluster, kv = _fresh(pod_specs)
    # Opaque precomputed key — no real tokens.
    _install_prefix(kv, "p1", ["opaque-prefix-key"])
    p = get_policy("prefix-cache", block_size=16)
    d = p.decide(
        Request("r", "s", 0.0, tuple(range(8)), 4, prefix_key="opaque-prefix-key"),
        cluster,
        kv,
    )
    assert d.prefill_pod_id == "p1"


def test_prefix_cache_single_pod_cluster():
    """Single pod → always routes there regardless of match."""
    specs = [PodSpec("only", Phase.BOTH, 1, 8 * 1024 * 1024, 2, 8)]
    cluster, kv = _fresh(specs)
    p = get_policy("prefix-cache", block_size=16)
    # No cache, no match → fallback.
    d = p.decide(Request("r", "s", 0.0, tuple(range(32)), 4), cluster, kv)
    assert d.prefill_pod_id == "only"


# ---------- prefix_cache_preble: gates and boundaries ----------


def test_preble_cold_start_explore_picks_lowest_pod_id(pod_specs):
    """All pending_work_ms=0, no cache → explore → lowest pod_id."""
    cluster, kv = _fresh(pod_specs)
    p = get_policy("prefix-cache-preble", block_size=16)
    d = p.decide(Request("r", "s", 0.0, tuple(range(48)), 4), cluster, kv)
    assert d.prefill_pod_id == "p0"
    assert "explore" in d.rationale


def test_preble_explore_tie_break_lowest_pod_id(pod_specs):
    """Tied zero load in explore branch → lowest pod_id."""
    cluster, kv = _fresh(pod_specs)
    # No cache, equal load everywhere.
    p = get_policy("prefix-cache-preble", block_size=16)
    d = p.decide(Request("r", "s", 0.0, tuple(range(48)), 4), cluster, kv)
    assert d.prefill_pod_id == "p0"
    assert "explore" in d.rationale


def test_preble_exploit_tie_break_lowest_pod_id(pod_specs):
    """F3 fix: exploit branch tie-breaks on LOWEST pod_id, matching explore.

    Identical match, identical load → smallest pod_id wins. Symmetric with
    the explore and hotspot-redirect branches so that borderline requests
    flipping between branches don't also bounce between pods.
    """
    cluster, kv = _fresh(pod_specs)
    tokens = tuple(range(32))
    hashes = enumerate_prefix_hashes(tokens, block_size=16)
    _install_prefix(kv, "p0", hashes)
    _install_prefix(kv, "p2", hashes)
    # Identical match, identical load (zero).
    p = get_policy("prefix-cache-preble", block_size=16)
    d = p.decide(Request("r", "s", 2.0, tokens, 4), cluster, kv)
    assert d.prefill_pod_id == "p0"
    assert "exploit" in d.rationale


def test_preble_single_pod_cluster_never_redirects():
    """Single-pod cluster: exploit has no alternative to redirect to."""
    specs = [PodSpec("only", Phase.BOTH, 1, 8 * 1024 * 1024, 2, 8)]
    cluster, kv = _fresh(specs)
    tokens = tuple(range(32))
    hashes = enumerate_prefix_hashes(tokens, block_size=16)
    _install_prefix(kv, "only", hashes)
    cluster.pods["only"].pending_work_ms = 10000.0  # massive load
    p = get_policy("prefix-cache-preble", block_size=16, th_bal=1.5)
    d = p.decide(Request("r", "s", 2.0, tokens, 4), cluster, kv)
    assert d.prefill_pod_id == "only"


def test_preble_exploit_gate_exact_half_is_explore(pod_specs):
    """`missed_tokens == cached_tokens` → gate fails → explore.

    32-token prompt (2 blocks), one block cached → cached=16, missed=16.
    Paper's condition is strict `<`, so we fall through to explore.
    """
    cluster, kv = _fresh(pod_specs)
    tokens = tuple(range(32))
    hashes = enumerate_prefix_hashes(tokens, block_size=16)
    _install_prefix(kv, "p2", hashes[:1])  # 1 block only
    p = get_policy("prefix-cache-preble", block_size=16)
    d = p.decide(Request("r", "s", 2.0, tokens, 4), cluster, kv)
    assert "explore" in d.rationale


def test_preble_exploit_gate_just_over_half_is_exploit(pod_specs):
    """missed < cached → exploit binds to owner.

    31-token prompt, 1 full block hash exists → cached=16, missed=15.
    """
    cluster, kv = _fresh(pod_specs)
    tokens = tuple(range(31))  # 1 full block + 15-tok tail
    hashes = enumerate_prefix_hashes(tokens, block_size=16)
    assert len(hashes) == 1
    _install_prefix(kv, "p2", hashes)
    p = get_policy("prefix-cache-preble", block_size=16)
    d = p.decide(Request("r", "s", 2.0, tokens, 4), cluster, kv)
    assert d.prefill_pod_id == "p2"
    assert "exploit" in d.rationale


def test_preble_hotspot_at_exact_threshold_no_redirect(pod_specs):
    """load == th_bal * min_load → strict `>` fails → no redirect."""
    cluster, kv = _fresh(pod_specs)
    tokens = tuple(range(32))
    hashes = enumerate_prefix_hashes(tokens, block_size=16)
    _install_prefix(kv, "p0", hashes)
    # th_bal=1.5; p0=150, min=100 → 150 > 1.5*100 is False (equal).
    cluster.pods["p0"].pending_work_ms = 150.0
    cluster.pods["p1"].pending_work_ms = 100.0
    cluster.pods["p2"].pending_work_ms = 100.0
    p = get_policy("prefix-cache-preble", block_size=16, th_bal=1.5)
    d = p.decide(Request("r", "s", 2.0, tokens, 4), cluster, kv)
    assert d.prefill_pod_id == "p0"
    assert "exploit" in d.rationale
    assert "hotspot" not in d.rationale


def test_preble_hotspot_just_over_threshold_redirects(pod_specs):
    """load > th_bal * min_load → redirect to lightest."""
    cluster, kv = _fresh(pod_specs)
    tokens = tuple(range(32))
    hashes = enumerate_prefix_hashes(tokens, block_size=16)
    _install_prefix(kv, "p0", hashes)
    cluster.pods["p0"].pending_work_ms = 150.1
    cluster.pods["p1"].pending_work_ms = 100.0
    cluster.pods["p2"].pending_work_ms = 100.0
    p = get_policy("prefix-cache-preble", block_size=16, th_bal=1.5)
    d = p.decide(Request("r", "s", 2.0, tokens, 4), cluster, kv)
    assert d.prefill_pod_id != "p0"
    assert "hotspot-redirect" in d.rationale


def test_preble_hotspot_redirect_target_is_lightest_regardless_of_match(pod_specs):
    """F: redirect goes to lightest pod even if it has no prefix match."""
    cluster, kv = _fresh(pod_specs)
    tokens = tuple(range(32))
    hashes = enumerate_prefix_hashes(tokens, block_size=16)
    # Prefix on p0 (hot). p1 has some match but is medium-loaded. p2 is cold.
    _install_prefix(kv, "p0", hashes)
    _install_prefix(kv, "p1", hashes[:1])
    cluster.pods["p0"].pending_work_ms = 1000.0
    cluster.pods["p1"].pending_work_ms = 500.0
    cluster.pods["p2"].pending_work_ms = 10.0  # lightest
    p = get_policy("prefix-cache-preble", block_size=16, th_bal=1.5)
    d = p.decide(Request("r", "s", 2.0, tokens, 4), cluster, kv)
    assert d.prefill_pod_id == "p2"


def test_preble_prefix_key_path_binds_to_owner(pod_specs):
    """prefix_key path: single-block match, no load → exploit binds."""
    cluster, kv = _fresh(pod_specs)
    _install_prefix(kv, "p1", ["opaque-key"])
    p = get_policy("prefix-cache-preble", block_size=16)
    # Long prompt so cached_tokens=16 > missed=(N-16) fails — adjust.
    # For exploit under prefix_key, need len(prompt_tokens) < 2*block_size.
    d = p.decide(
        Request("r", "s", 0.0, tuple(range(20)), 4, prefix_key="opaque-key"),
        cluster,
        kv,
    )
    assert d.prefill_pod_id == "p1"
    assert "exploit" in d.rationale


def test_preble_prefix_key_short_prompt_missed_tokens_nonnegative(pod_specs):
    """F4 regression: prefix_key with len(prompt) < block_size.

    Under prefix_key, best_match∈{0,1} on a single opaque hash. If the
    prompt is shorter than block_size, cached_tokens (= block_size) > len,
    so missed_tokens would go negative without clamping. The exploit gate
    still fires (full-prompt hit is defensible), but the missed_tokens
    value fed to the gate must be a non-negative token count.
    """
    cluster, kv = _fresh(pod_specs)
    _install_prefix(kv, "p1", ["opaque-key"])
    p = get_policy("prefix-cache-preble", block_size=16)
    # 8-token prompt, block_size=16 → pre-fix: cached=16, missed=-8.
    d = p.decide(
        Request("r", "s", 0.0, tuple(range(8)), 4, prefix_key="opaque-key"),
        cluster,
        kv,
    )
    assert d.prefill_pod_id == "p1"
    assert "exploit" in d.rationale


def test_preble_stable_under_repeated_identical_requests(pod_specs):
    """Without load changes, exploit stays sticky on the owner."""
    cluster, kv = _fresh(pod_specs)
    tokens = tuple(range(32))
    hashes = enumerate_prefix_hashes(tokens, block_size=16)
    _install_prefix(kv, "p1", hashes)
    p = get_policy("prefix-cache-preble", block_size=16)
    # No side effects: policy doesn't mutate. Repeat 10 calls.
    seen = {
        p.decide(Request(f"r{i}", "s", float(i), tokens, 4), cluster, kv).prefill_pod_id
        for i in range(10)
    }
    assert seen == {"p1"}


# ---------- cache-eviction interaction ----------


def test_preble_policy_sees_post_eviction_state():
    """After LRU eviction, policy no longer sees evicted prefix on that pod."""
    # Tiny pods: 1 KiB KV budget → must evict.
    specs = [PodSpec(f"p{i}", Phase.BOTH, 1, 1024, 2, 8) for i in range(2)]
    cluster, kv = _fresh(specs)
    tokens_a = tuple(range(32))
    tokens_b = tuple(range(100, 132))
    hashes_a = enumerate_prefix_hashes(tokens_a, block_size=16)
    hashes_b = enumerate_prefix_hashes(tokens_b, block_size=16)
    # Install A on p0. Byte size equals capacity so B evicts A.
    kv.install("p0", PrefixEntry(hashes_a[0], 16, 512), now=1.0)
    kv.install("p0", PrefixEntry(hashes_a[1], 16, 512), now=1.1)
    # Now install B — evicts A (LRU).
    kv.install("p0", PrefixEntry(hashes_b[0], 16, 512), now=2.0)
    kv.install("p0", PrefixEntry(hashes_b[1], 16, 512), now=2.1)
    assert not kv.has("p0", hashes_a[0])
    assert kv.has("p0", hashes_b[0])
    # Policy must see post-eviction state: request A now has no cache hit.
    p = get_policy("prefix-cache-preble", block_size=16)
    d = p.decide(Request("r", "s", 3.0, tokens_a, 4), cluster, kv)
    assert "explore" in d.rationale  # no match anywhere


# ---------- hash utility regressions ----------


def test_enumerate_prefix_hashes_stable_across_block_boundaries():
    """Incremental blake2b (fc039e4) matches legacy join-then-hash."""
    import hashlib

    tokens = list(range(96))
    block_size = 16
    # Legacy form: each block's hash is blake2b over b",".join(str(t).encode() for t in seq[:end]).
    legacy = []
    for end in range(block_size, len(tokens) + 1, block_size):
        h = hashlib.blake2b(digest_size=16)
        h.update(b",".join(str(t).encode() for t in tokens[:end]))
        legacy.append(h.hexdigest())
    incremental = enumerate_prefix_hashes(tokens, block_size=block_size)
    assert incremental == legacy


def test_enumerate_prefix_hashes_partial_block_not_emitted():
    """Tokens that don't fill the final block are ignored (matches legacy)."""
    tokens = list(range(31))  # 1 full block + 15 stragglers
    hashes = enumerate_prefix_hashes(tokens, block_size=16)
    assert len(hashes) == 1


def test_enumerate_prefix_hashes_empty_input():
    """Empty input → no hashes."""
    assert enumerate_prefix_hashes([], block_size=16) == []


# ---------- contract reinforcement ----------


def test_preble_no_cache_and_equal_loads_is_deterministic(pod_specs):
    """Same inputs, two calls, two policies → identical decisions."""
    cluster, kv = _fresh(pod_specs)
    req = Request("r", "s", 0.0, tuple(range(48)), 4)
    p1 = get_policy("prefix-cache-preble", block_size=16)
    p2 = get_policy("prefix-cache-preble", block_size=16)
    assert p1.decide(req, cluster, kv).prefill_pod_id == p2.decide(req, cluster, kv).prefill_pod_id
