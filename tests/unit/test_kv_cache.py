"""KVCacheState: LRU eviction, owner lookup, prefix hashing determinism."""

from __future__ import annotations

from routing_harness.kv_cache import KVCacheState, PrefixEntry, enumerate_prefix_hashes


def test_enumerate_prefix_hashes_is_deterministic():
    tokens = list(range(48))
    a = enumerate_prefix_hashes(tokens, block_size=16)
    b = enumerate_prefix_hashes(tokens, block_size=16)
    assert a == b
    assert len(a) == 3


def test_enumerate_prefix_hashes_prefix_structure():
    short = list(range(32))
    long = list(range(48))
    # First two hashes must match since they cover the same blocks.
    assert enumerate_prefix_hashes(short)[:2] == enumerate_prefix_hashes(long)[:2]


def test_lru_eviction_by_byte_budget():
    kv = KVCacheState.from_specs({"p": 100})
    for i in range(10):
        kv.install("p", PrefixEntry(f"h{i}", token_count=4, byte_size=30), now=float(i))
    # Capacity 100 / 30 per entry => ~3 entries fit.
    pod = kv.pods["p"]
    assert pod.bytes_used <= 100
    assert len(pod.entries) <= 4
    # Most-recent ids should remain.
    assert "h9" in pod.entries


def test_owners_of_and_best_owner():
    kv = KVCacheState.from_specs({"a": 1000, "b": 1000})
    kv.install("a", PrefixEntry("x", 4, 32), now=1.0)
    kv.install("b", PrefixEntry("x", 4, 32), now=5.0)
    assert set(kv.owners_of("x")) == {"a", "b"}
    assert kv.best_owner("x") == "b"


def test_owners_of_requires_consecutive_prefix_residency():
    """Scattered residency must not qualify a pod as an owner.

    Pod `consec` has blocks h0..h2 end-to-end; pod `scatter` holds h0 and
    h2 but is missing h1. Only `consec` is a legitimate reuse source
    for h2 because block 2's KV depends on block 1.
    """
    kv = KVCacheState.from_specs({"consec": 1000, "scatter": 1000})
    for h in ("h0", "h1", "h2"):
        kv.install("consec", PrefixEntry(h, 4, 32), now=1.0)
    kv.install("scatter", PrefixEntry("h0", 4, 32), now=2.0)
    kv.install("scatter", PrefixEntry("h2", 4, 32), now=3.0)

    hashes = ["h0", "h1", "h2"]
    # Strict: only the consecutively-resident pod qualifies for h2.
    assert kv.owners_of("h2", hashes) == ["consec"]
    # Both pods still qualify as owners of h0 (first block, no predecessors).
    assert set(kv.owners_of("h0", hashes)) == {"consec", "scatter"}
    # Legacy per-block semantics still available and would over-estimate.
    assert set(kv.owners_of("h2")) == {"consec", "scatter"}
    # best_owner honors the context.
    assert kv.best_owner("h2", hashes) == "consec"


def test_owners_of_unknown_hash_returns_empty():
    kv = KVCacheState.from_specs({"a": 1000})
    kv.install("a", PrefixEntry("h0", 4, 32), now=0.0)
    assert kv.owners_of("missing", ["h0", "h1"]) == []
