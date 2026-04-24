"""Session affinity: sticky sessions by session_id hashed to a pod.

New sessions land on the pod chosen by a fallback policy (default:
least-request); subsequent requests from the same session stick. Evicts
stickiness if the sticky pod is no longer in cluster.pods (pod removed
/ scaled down) or if stickiness_ttl_s seconds have elapsed since the
binding was recorded.

Memory bound (F13): TTL is checked on read, so never-returning sessions
would otherwise accumulate in `_bindings` forever. Every
`purge_interval` decisions we sweep entries whose bound_ts is older than
stickiness_ttl_s relative to the current arrival_ts. The sweep is
amortized O(N / purge_interval) per decide().

Taxonomy (see `research/reports/routing-comparison.md` §3):
    selection=identity (session_id), state=per-session (bindings map),
    fairness=session-sticky, topology=any, migration=rebind-on-fail.
    Note that session-sticky is not a fairness-balancing model: it
    isolates sessions onto pods for cache warm-up, without attempting
    to equalize throughput across sessions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..cluster import ClusterState
from ..core import Decision, Request
from ..kv_cache import KVCacheState
from ..policy import register_policy


@register_policy("session-affinity")
@dataclass
class SessionAffinityPolicy:
    stickiness_ttl_s: float = 3600.0
    purge_interval: int = 1024
    _bindings: dict[str, tuple[str, float]] = field(default_factory=dict)
    _decide_count: int = 0

    def decide(
        self,
        request: Request,
        cluster: ClusterState,
        kv_cache: KVCacheState,
    ) -> Decision:
        cands = cluster.prefill_capable()
        if not cands:
            return Decision("__none__", "__none__", "no-prefill-capable-pod")
        self._decide_count += 1
        if self.purge_interval > 0 and self._decide_count % self.purge_interval == 0:
            self._purge_stale(request.arrival_ts)
        bound = self._bindings.get(request.session_id)
        if bound is not None:
            pod_id, bound_ts = bound
            if pod_id in cluster.pods and request.arrival_ts - bound_ts <= self.stickiness_ttl_s:
                return Decision(pod_id, pod_id, rationale=f"sticky session={request.session_id}")
        pick = min(
            cands,
            key=lambda p: (p.active_prefill + p.active_decode, p.spec.pod_id),
        )
        self._bindings[request.session_id] = (pick.spec.pod_id, request.arrival_ts)
        return Decision(pick.spec.pod_id, pick.spec.pod_id, rationale="new-sticky-binding")

    def _purge_stale(self, now: float) -> None:
        ttl = self.stickiness_ttl_s
        self._bindings = {
            sid: (pid, ts) for sid, (pid, ts) in self._bindings.items() if now - ts <= ttl
        }
