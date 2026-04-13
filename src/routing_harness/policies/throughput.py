"""Throughput-maximizing: route to the pod with highest recent tokens/s.

Uses EWMA throughput maintained by the simulator. This policy optimizes
for aggregate cluster throughput but can starve cold pods; see
least-busy-time for a fairness-aware counterpart.

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
        pick = max(cands, key=lambda p: (p.ewma_throughput_tps, -ord(p.spec.pod_id[0])))
        return Decision(
            pick.spec.pod_id,
            pick.spec.pod_id,
            rationale="max-ewma-throughput",
            score=pick.ewma_throughput_tps,
        )
