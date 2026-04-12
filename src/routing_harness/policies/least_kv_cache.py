"""Least-kv-cache: route to the pod with the most free KV-cache bytes.

A capacity-aware policy that avoids routing into a pod whose cache is
near-full (which would force heavy eviction). Ignores prefix match; pair
with prefix_cache_preble if you want both signals.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..cluster import ClusterState
from ..core import Decision, Request
from ..kv_cache import KVCacheState
from ..policy import register_policy


@register_policy("least-kv-cache")
@dataclass
class LeastKVCachePolicy:
    def decide(
        self,
        request: Request,
        cluster: ClusterState,
        kv_cache: KVCacheState,
    ) -> Decision:
        cands = cluster.prefill_capable()
        if not cands:
            return Decision("__none__", "__none__", "no-prefill-capable-pod")

        def free(p):
            cap = p.spec.kv_cache_bytes
            used = kv_cache.size_bytes(p.spec.pod_id) if p.spec.pod_id in kv_cache.pods else 0
            return cap - used

        pick = max(cands, key=lambda p: (free(p), p.spec.pod_id))
        return Decision(
            pick.spec.pod_id,
            pick.spec.pod_id,
            rationale="max-free-kv",
            score=float(free(pick)),
        )
