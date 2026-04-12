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
