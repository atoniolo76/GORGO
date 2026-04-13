"""Least-request: route to the pod with fewest in-flight requests.

Classic load balancer. Ties broken by pod_id lexicographically for
determinism.

Taxonomy (see `research/reports/routing-comparison.md` §3):
    selection=load, state=stateless, fairness=best-effort,
    topology=any, migration=none.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..cluster import ClusterState
from ..core import Decision, Request
from ..kv_cache import KVCacheState
from ..policy import register_policy


@register_policy("least-request")
@dataclass
class LeastRequestPolicy:
    def decide(
        self,
        request: Request,
        cluster: ClusterState,
        kv_cache: KVCacheState,
    ) -> Decision:
        cands = cluster.prefill_capable()
        if not cands:
            return Decision("__none__", "__none__", "no-prefill-capable-pod")
        pick = min(
            cands,
            key=lambda p: (p.active_prefill + p.active_decode + p.queued, p.spec.pod_id),
        )
        return Decision(
            pick.spec.pod_id,
            pick.spec.pod_id,
            rationale="min-active+queued",
            score=float(pick.active_prefill + pick.active_decode + pick.queued),
        )
