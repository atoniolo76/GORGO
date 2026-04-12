"""Least-latency: route to the pod with lowest EWMA latency.

Simple and brittle: a cold pod with no requests has latency 0 and will
attract the first wave. The simulator's EWMA is initialized to a
non-zero "warm" value so this policy behaves reasonably at t=0.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..cluster import ClusterState
from ..core import Decision, Request
from ..kv_cache import KVCacheState
from ..policy import register_policy


@register_policy("least-latency")
@dataclass
class LeastLatencyPolicy:
    def decide(
        self,
        request: Request,
        cluster: ClusterState,
        kv_cache: KVCacheState,
    ) -> Decision:
        cands = cluster.prefill_capable()
        if not cands:
            return Decision("__none__", "__none__", "no-prefill-capable-pod")
        pick = min(cands, key=lambda p: (p.ewma_latency_ms, p.spec.pod_id))
        return Decision(
            pick.spec.pod_id,
            pick.spec.pod_id,
            rationale="min-ewma-latency",
            score=pick.ewma_latency_ms,
        )
