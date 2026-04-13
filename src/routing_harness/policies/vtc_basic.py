"""VTC-basic: Virtual Token Counter fairness policy.

Tracks per-tenant (or per-session) token consumption and penalizes pods
that are currently serving heavy tenants. Each pod has a per-tenant
"debt" implicit in its in-flight mix; VTC routes the request to the pod
whose aggregate debt for the request's tenant is lowest, breaking ties
by least-busy-time. The engine's `observe_completion` hook updates
counters when requests finish.

Taxonomy (see `research/reports/routing-comparison.md` §3):
    selection=fairness-debt (per-pod × per-tenant token counter),
    state=per-tenant (`counters` and `pod_tenant_tokens`),
    fairness=tenant-weighted, topology=any, migration=none.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from ..cluster import ClusterState
from ..core import Decision, Request
from ..kv_cache import KVCacheState
from ..policy import register_policy


@register_policy("vtc-basic")
@dataclass
class VTCBasicPolicy:
    fairness_key: str = "session_id"  # or "tenant" if set in metadata
    counters: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    pod_tenant_tokens: dict[str, dict[str, float]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(float))
    )

    def _key(self, r: Request) -> str:
        if self.fairness_key == "session_id":
            return r.session_id
        return str(r.metadata.get(self.fairness_key, r.session_id))

    def observe_completion(
        self,
        request: Request,
        decision: Decision,
        tokens_consumed: float,
    ) -> None:
        """Engine calls this when a request finishes.

        Updates the global per-tenant counter (for observability/score
        reporting) and the per-pod×tenant counter (used by `decide` as
        the primary selection priority).
        """
        k = self._key(request)
        self.counters[k] += tokens_consumed
        self.pod_tenant_tokens[decision.prefill_pod_id][k] += tokens_consumed

    def decide(
        self,
        request: Request,
        cluster: ClusterState,
        kv_cache: KVCacheState,
    ) -> Decision:
        cands = cluster.prefill_capable()
        if not cands:
            return Decision("__none__", "__none__", "no-prefill-capable-pod")
        k = self._key(request)
        vtc_score = self.counters.get(k, 0.0)

        def busy(p):
            return p.ewma_latency_ms * (p.active_prefill + p.active_decode + p.queued)

        def tenant_debt(p):
            return self.pod_tenant_tokens.get(p.spec.pod_id, {}).get(k, 0.0)

        pick = min(cands, key=lambda p: (tenant_debt(p), busy(p), p.spec.pod_id))
        return Decision(
            pick.spec.pod_id,
            pick.spec.pod_id,
            rationale=f"vtc k={k} tokens={vtc_score:.0f} debt={tenant_debt(pick):.0f}",
            score=-vtc_score,
        )
