"""Preble-inspired prefix-cache routing with exploit/explore gate and
relative-imbalance hotspot mitigation.

Implements a minimum-viable approximation of Preble (Zhong et al.,
"Efficient Distributed Prompt Scheduling for LLM Serving", ICLR 2025).
Three mechanisms match the paper:

1. **Exploit/explore gate (E2).** If the best prefix match saves more
   tokens than the uncached tail would cost (``missed < cached``), bind
   to the prefix owner (exploit). Otherwise fall back to the pod with
   the lowest estimated wait (explore). This replaces the prior linear
   ``alpha * match - beta * load`` combination with the conditional
   structure the paper actually uses.

2. **Time-domain load signal.** Per-pod load is ``pending_work_ms`` —
   the sum of predicted service times (ms) for all in-flight requests
   on that pod, maintained by the engine. This is Preble's ``L_i``
   term: a direct measure of how long until the pod drains its
   current work, in units of milliseconds.

3. **Relative hotspot threshold.** Hotspot fires when the exploit
   target's load exceeds ``th_bal`` times the lightest pod's load — a
   ratio, not an absolute threshold. The deflection target is the
   lightest pod regardless of prefix match (the prior ``match > 0``
   gate was the binding constraint under mono-homing).

Deferred: eviction cost M_i, prefix auto-scaling, radix tree, local
priority-group scheduling. See ``docs/preble_paper_vs_impl.md`` for
the full divergence analysis (go-ggf).

The registered id ``prefix-cache-preble`` is intentionally kept stable
(config compatibility). Prior parameters ``alpha``, ``beta``, and
``hotspot_threshold`` are silently dropped by the registry for
backwards-compatible sweep configs (they are no longer accepted).

Taxonomy (see ``research/reports/routing-comparison.md`` §3):
    selection=composite (cache-affinity + load), state=stateless,
    fairness=best-effort, topology=any, migration=none.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..cluster import ClusterState
from ..core import Decision, PodRuntime, Request
from ..kv_cache import KVCacheState, enumerate_prefix_hashes
from ..policy import register_policy


@register_policy("prefix-cache-preble")
@dataclass
class PreblePrefixCachePolicy:
    block_size: int = 16
    th_bal: float = 1.5

    def _prefix_hashes(self, r: Request) -> list[str]:
        if r.prefix_key:
            return [r.prefix_key]
        return enumerate_prefix_hashes(r.prompt_tokens, block_size=self.block_size)

    @staticmethod
    def _load_ms(pod: PodRuntime) -> float:
        """Preble L_i: sum of predicted service times for in-flight requests."""
        return pod.pending_work_ms

    def decide(
        self,
        request: Request,
        cluster: ClusterState,
        kv_cache: KVCacheState,
    ) -> Decision:
        cands = cluster.prefill_capable()
        if not cands:
            return Decision("__none__", "__none__", "no-prefill-capable-pod")
        hashes = self._prefix_hashes(request)

        # Per-pod: prefix match length + time-domain load.
        pod_data: list[tuple[int, float, PodRuntime]] = []
        for p in cands:
            match = 0
            for h in hashes:
                if kv_cache.has(p.spec.pod_id, h):
                    match += 1
                else:
                    break
            pod_data.append((match, self._load_ms(p), p))

        # Exploit candidate: most prefix match, then least load, then lowest pod_id.
        # Tie-break on pod_id must match the explore/hotspot branches (smallest
        # wins) so identical requests don't bounce between branches on ties.
        best_match, best_load, best_pod = min(
            pod_data,
            key=lambda t: (-t[0], t[1], t[2].spec.pod_id),
        )

        # Exploit/explore gate (Preble E2).
        cached_tokens = best_match * self.block_size
        missed_tokens = len(request.prompt_tokens) - cached_tokens

        if best_match > 0 and missed_tokens < cached_tokens:
            # EXPLOIT: prefix reuse dominates. Bind to owner unless
            # the owner is a hotspot relative to the lightest pod.
            min_load = min(t[1] for t in pod_data)
            if best_load > self.th_bal * min_load:
                lightest = min(pod_data, key=lambda t: (t[1], t[2].spec.pod_id))
                return Decision(
                    lightest[2].spec.pod_id,
                    lightest[2].spec.pod_id,
                    rationale=(f"exploit-hotspot-redirect from={best_pod.spec.pod_id}"),
                    score=float(lightest[0]),
                )
            return Decision(
                best_pod.spec.pod_id,
                best_pod.spec.pod_id,
                rationale=f"exploit match={best_match} load_ms={best_load:.1f}",
                score=float(best_match),
            )

        # EXPLORE: cache reuse insufficient. Minimize estimated wait.
        lightest = min(pod_data, key=lambda t: (t[1], t[2].spec.pod_id))
        return Decision(
            lightest[2].spec.pod_id,
            lightest[2].spec.pod_id,
            rationale=f"explore load_ms={lightest[1]:.1f}",
            score=float(-lightest[1]),
        )
