"""Per-tenant load balance: route by lowest per-pod × per-tenant token debt.

Tracks per-tenant (or per-session) token consumption and steers a tenant's
requests toward the pod that has served the least of *that tenant's* tokens
so far. Each pod has a per-tenant "debt" implicit in its in-flight mix; the
policy routes the request to the pod whose aggregate debt for the request's
tenant is lowest, breaking ties by least-busy-time. The engine's
`observe_completion` hook updates counters when requests finish.

This is **not** the VTC policy from Sheng et al. (OSDI'24). That paper
schedules *admission order among queued requests* to bound max-min fairness
|U_i - U_j| <= U; this policy has no admission queue, no fairness bound, and
no starvation protection. It is a per-pod tenant-aware load balancer, useful
for spreading a heavy tenant across pods, but it does not provide the
guarantees of paper VTC. (See go-362.)

Taxonomy (see `research/reports/routing-comparison.md` §3):
    selection=fairness-debt (per-pod × per-tenant token counter),
    state=per-tenant (`counters` and `pod_tenant_tokens`),
    fairness=tenant-weighted, topology=any, migration=none.

Windowing (F16): when `window_s` is set, consumption aged out beyond
that window stops counting. When `window_s` is None, counters are
monotonic (kept for backward compatibility with short-trace sweeps where
no visible difference occurs).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from ..cluster import ClusterState
from ..core import Decision, Request
from ..kv_cache import KVCacheState
from ..policy import register_policy


@register_policy("per-tenant-load-balance")
@dataclass
class PerTenantLoadBalancePolicy:
    fairness_key: str = "session_id"  # or "tenant" if set in metadata
    window_s: float | None = None  # None → monotonic; float → sliding window in seconds
    counters: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    pod_tenant_tokens: dict[str, dict[str, float]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(float))
    )
    _counter_events: dict[str, list[tuple[float, float]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    _pod_tenant_events: dict[str, dict[str, list[tuple[float, float]]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(list))
    )

    def _key(self, r: Request) -> str:
        if self.fairness_key == "session_id":
            return r.session_id
        return str(r.metadata.get(self.fairness_key, r.session_id))

    def _evict_expired(self, now_s: float) -> None:
        """Drop events older than `window_s` and rebalance running totals.

        No-op when `window_s is None` (monotonic mode).
        """
        if self.window_s is None:
            return
        cutoff = now_s - self.window_s
        for k, events in self._counter_events.items():
            expired = 0.0
            kept: list[tuple[float, float]] = []
            for ts, tokens in events:
                if ts < cutoff:
                    expired += tokens
                else:
                    kept.append((ts, tokens))
            if expired:
                self.counters[k] = max(0.0, self.counters[k] - expired)
            self._counter_events[k] = kept
        for pod_id, per_tenant in self._pod_tenant_events.items():
            for k, events in per_tenant.items():
                expired = 0.0
                kept = []
                for ts, tokens in events:
                    if ts < cutoff:
                        expired += tokens
                    else:
                        kept.append((ts, tokens))
                if expired:
                    cur = self.pod_tenant_tokens[pod_id][k]
                    self.pod_tenant_tokens[pod_id][k] = max(0.0, cur - expired)
                per_tenant[k] = kept

    def reset(self) -> None:
        """Clear all accumulated state — for experiment isolation."""
        self.counters.clear()
        self.pod_tenant_tokens.clear()
        self._counter_events.clear()
        self._pod_tenant_events.clear()

    def observe_completion(
        self,
        request: Request,
        decision: Decision,
        tokens_consumed: float,
    ) -> None:
        """Engine calls this when a request finishes.

        Updates the global per-tenant counter (for observability/score
        reporting) and the per-pod×tenant counter (used by `decide` as
        the primary selection priority). When `window_s` is set, records
        the event timestamp so old consumption can age out.
        """
        k = self._key(request)
        self.counters[k] += tokens_consumed
        self.pod_tenant_tokens[decision.prefill_pod_id][k] += tokens_consumed
        if self.window_s is not None:
            ts = request.arrival_ts
            self._counter_events[k].append((ts, tokens_consumed))
            self._pod_tenant_events[decision.prefill_pod_id][k].append((ts, tokens_consumed))
            self._evict_expired(ts)

    def decide(
        self,
        request: Request,
        cluster: ClusterState,
        kv_cache: KVCacheState,
    ) -> Decision:
        cands = cluster.prefill_capable()
        if not cands:
            return Decision("__none__", "__none__", "no-prefill-capable-pod")
        self._evict_expired(request.arrival_ts)
        k = self._key(request)
        tenant_score = self.counters.get(k, 0.0)

        def busy(p):
            return p.ewma_latency_ms * (p.active_prefill + p.active_decode)

        def tenant_debt(p):
            return self.pod_tenant_tokens.get(p.spec.pod_id, {}).get(k, 0.0)

        pick = min(cands, key=lambda p: (tenant_debt(p), busy(p), p.spec.pod_id))
        return Decision(
            pick.spec.pod_id,
            pick.spec.pod_id,
            rationale=f"ptlb k={k} tokens={tenant_score:.0f} debt={tenant_debt(pick):.0f}",
            score=-tenant_score,
        )
