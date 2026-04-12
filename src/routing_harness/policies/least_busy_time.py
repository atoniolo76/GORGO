"""Least-busy-time: route to pod with lowest projected busy time.

Busy time = EWMA(latency) * (active + queued). Balances utilization and
latency; generally dominates least-request under heterogeneous request
sizes.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..cluster import ClusterState
from ..core import Decision, Request
from ..kv_cache import KVCacheState
from ..policy import register_policy


@register_policy("least-busy-time")
@dataclass
class LeastBusyTimePolicy:
    def decide(
        self,
        request: Request,
        cluster: ClusterState,
        kv_cache: KVCacheState,
    ) -> Decision:
        cands = cluster.prefill_capable()
        if not cands:
            return Decision("__none__", "__none__", "no-prefill-capable-pod")

        def busy(p):
            return p.ewma_latency_ms * (p.active_prefill + p.active_decode + p.queued)

        pick = min(cands, key=lambda p: (busy(p), p.spec.pod_id))
        return Decision(
            pick.spec.pod_id,
            pick.spec.pod_id,
            rationale="min-busy-time",
            score=busy(pick),
        )
