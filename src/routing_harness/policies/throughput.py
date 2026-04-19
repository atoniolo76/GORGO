"""Throughput-maximizing: route to the pod with highest available capacity.

Scores each pod by ``ewma_throughput_tps / (1 + active_requests)`` so
a pod with high throughput but high concurrency is ranked below a pod
with moderate throughput and headroom. This prevents the pathological
single-pod starvation where the first-warm pod monopolizes all traffic
via a reinforcing EWMA feedback loop.

Taxonomy (see ``research/reports/routing-comparison.md`` §3):
    selection=load, state=stateless, fairness=best-effort,
    topology=any, migration=none.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..cluster import ClusterState
from ..core import Decision, Request
from ..kv_cache import KVCacheState
from ..policy import register_policy


@register_policy("throughput")
@dataclass
class ThroughputPolicy:
    def decide(
        self,
        request: Request,
        cluster: ClusterState,
        kv_cache: KVCacheState,
    ) -> Decision:
        cands = cluster.prefill_capable()
        if not cands:
            return Decision("__none__", "__none__", "no-prefill-capable-pod")
        pick = max(
            cands,
            key=lambda p: (
                p.ewma_throughput_tps / (1 + p.active_prefill + p.active_decode),
                -ord(p.spec.pod_id[0]),
            ),
        )
        return Decision(
            pick.spec.pod_id,
            pick.spec.pod_id,
            rationale="max-available-throughput",
            score=pick.ewma_throughput_tps,
        )
