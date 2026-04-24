"""Least-kv-cache: route to the pod with the most free KV-cache bytes.

A capacity-aware policy that avoids routing into a pod whose cache is
near-full (which would force heavy eviction). Ignores prefix match; pair
with prefix_cache_preble if you want both signals.

Taxonomy (see `research/reports/routing-comparison.md` §3):
    selection=capacity, state=stateless, fairness=best-effort,
    topology=any, migration=none.
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

        # Secondary key -active_prefill falls back to load-balancing when free
        # bytes tie. Without it, shared-prefix workloads starve 2/3 of pods:
        # the first pod warmed absorbs all subsequent requests because install
        # on an already-cached prefix is a byte-level no-op, so its `free`
        # never shrinks (F10, go-fw8).
        pick = max(
            cands,
            key=lambda p: (free(p), -p.active_prefill, p.spec.pod_id),
        )
        return Decision(
            pick.spec.pod_id,
            pick.spec.pod_id,
            rationale="max-free-kv",
            score=float(free(pick)),
        )
