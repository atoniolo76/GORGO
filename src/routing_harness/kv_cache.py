"""KVCacheState: per-pod prefix cache with eviction and reuse accounting.

Design:
- Each pod owns an LRU set of prefix entries keyed by (token-tuple) hash.
- We track length in tokens and an estimated byte size so the cost model
  can charge transport cost when a prefix must be pulled to a different
  pod.
- We distinguish "available reuse" (any pod in the cluster has the
  prefix) from "captured reuse" (the routed pod has the prefix).

This is a *simulation* of KV state. It does not hold real tensors.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class PrefixEntry:
    prefix_hash: str
    token_count: int
    byte_size: int
    last_access_ts: float = 0.0
    access_count: int = 0


@dataclass
class PodCache:
    """LRU-by-access cache for one pod. Eviction is by byte budget."""

    capacity_bytes: int
    bytes_used: int = 0
    entries: "OrderedDict[str, PrefixEntry]" = field(default_factory=OrderedDict)

    def get(self, prefix_hash: str, now: float) -> PrefixEntry | None:
        entry = self.entries.get(prefix_hash)
        if entry is None:
            return None
        entry.last_access_ts = now
        entry.access_count += 1
        self.entries.move_to_end(prefix_hash)
        return entry

    def put(self, entry: PrefixEntry, now: float) -> list[PrefixEntry]:
        """Insert or update; evict LRU to fit. Returns evicted entries."""
        evicted: list[PrefixEntry] = []
        existing = self.entries.get(entry.prefix_hash)
        if existing is not None:
            self.bytes_used -= existing.byte_size
        while (
            self.bytes_used + entry.byte_size > self.capacity_bytes
            and self.entries
        ):
            _, victim = self.entries.popitem(last=False)
            self.bytes_used -= victim.byte_size
            evicted.append(victim)
        self.entries[entry.prefix_hash] = entry
        self.bytes_used += entry.byte_size
        entry.last_access_ts = now
        self.entries.move_to_end(entry.prefix_hash)
        return evicted


@dataclass
class KVCacheState:
    """Cluster-wide prefix cache state, partitioned by pod."""

    pods: dict[str, PodCache] = field(default_factory=dict)

    @classmethod
    def from_specs(cls, pod_capacities: dict[str, int]) -> "KVCacheState":
        return cls(pods={pid: PodCache(capacity_bytes=cap) for pid, cap in pod_capacities.items()})

    def has(self, pod_id: str, prefix_hash: str) -> bool:
        pod = self.pods.get(pod_id)
        return pod is not None and prefix_hash in pod.entries

    def owners_of(
        self,
        prefix_hash: str,
        prefix_hashes: Iterable[str] | None = None,
    ) -> list[str]:
        """Pods whose cache contains a consecutive prefix ending at ``prefix_hash``.

        A paged KV block is only reusable if its predecessors in the
        request's prefix are also resident on the same pod: block K's
        attention state depends on blocks 0..K-1. A pod that holds a
        scattered subset (say blocks 2 and 5 but not 0, 1, 3, 4) cannot
        serve block 5 as a reuse source.

        ``prefix_hashes`` is the ordered block-hash sequence for the
        request (as produced by ``enumerate_prefix_hashes``). If
        omitted, ``prefix_hash`` is treated as a single-block prefix
        (the legacy per-block membership check, correct only for the
        first block of any request).
        """
        if prefix_hashes is None:
            required = (prefix_hash,)
        else:
            required = []
            for h in prefix_hashes:
                required.append(h)
                if h == prefix_hash:
                    break
            else:
                return []
            required = tuple(required)
        return [
            pid
            for pid, pc in self.pods.items()
            if all(h in pc.entries for h in required)
        ]

    def best_owner(
        self,
        prefix_hash: str,
        prefix_hashes: Iterable[str] | None = None,
    ) -> str | None:
        owners = self.owners_of(prefix_hash, prefix_hashes)
        if not owners:
            return None
        # Prefer the most-recently-used owner as a tiebreaker.
        return max(
            owners,
            key=lambda pid: self.pods[pid].entries[prefix_hash].last_access_ts,
        )

    def reuse_available(self, prefix_hash: str) -> bool:
        return any(prefix_hash in pc.entries for pc in self.pods.values())

    def install(
        self, pod_id: str, entry: PrefixEntry, now: float
    ) -> list[PrefixEntry]:
        return self.pods[pod_id].put(entry, now)

    def touch(self, pod_id: str, prefix_hash: str, now: float) -> PrefixEntry | None:
        return self.pods[pod_id].get(prefix_hash, now)

    def size_bytes(self, pod_id: str) -> int:
        return self.pods[pod_id].bytes_used


def enumerate_prefix_hashes(tokens: Iterable[int], block_size: int = 16) -> list[str]:
    """Deterministic block-level prefix hashes for a token sequence.

    Mirrors the common "paged" KV layout where prefix reuse is measured
    in blocks of `block_size` tokens. The hash is an opaque string
    (content-addressed) so it is stable across processes.

    Streams tokens into a single incremental blake2b rather than
    re-hashing every prefix from scratch — hashing is O(n) total, not
    O(n²). The on-wire hash output is unchanged from the prior
    implementation: each block's hash is still blake2b over
    ``b",".join(str(t).encode() for t in seq[:end])``.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    import hashlib

    hasher = hashlib.blake2b(digest_size=16)
    out: list[str] = []
    first = True
    i = 0
    for tok in tokens:
        if not first:
            hasher.update(b",")
        hasher.update(str(tok).encode())
        first = False
        i += 1
        if i % block_size == 0:
            out.append(hasher.copy().hexdigest())
    return out
