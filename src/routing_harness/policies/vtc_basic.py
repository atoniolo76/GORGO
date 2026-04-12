"""VTC-basic: Virtual Token Counter fairness policy.

Tracks per-session (or per-tenant) token consumption and prefers sessions
with lower consumption. Pod selection is delegated to least-busy-time;
VTC's contribution is admission ordering and tenant fairness. In this
harness we expose VTC as a routing policy that (a) picks a pod via
least-busy-time and (b) annotates the request with a VTC priority for
the simulator's queue to honor.
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

    def _key(self, r: Request) -> str:
        if self.fairness_key == "session_id":
            return r.session_id
        return str(r.metadata.get(self.fairness_key, r.session_id))

    def observe_completion(self, request: Request, tokens_consumed: float) -> None:
        """Simulator calls this when a request completes to update counters."""
        self.counters[self._key(request)] += tokens_consumed

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

        pick = min(cands, key=lambda p: (busy(p), p.spec.pod_id))
        return Decision(
            pick.spec.pod_id,
            pick.spec.pod_id,
            rationale=f"vtc k={k} tokens={vtc_score:.0f}",
            score=-vtc_score,
        )
