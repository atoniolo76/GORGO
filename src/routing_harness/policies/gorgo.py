"""GORGO multi-objective scoring policy (port of main's ``route_gorgo``).

Ports the production scoring rule deployed in
``utils/lb_aibrix.py:route_gorgo`` on the ``main`` branch into rome's
simulator so it can be benchmarked under identical conditions as the
rest of the policy library.

The score is a weighted sum that wins on the *minimum*::

    score(u) = ewma_latency
             + max(0, request_tokens - cached_tokens) * t_prefill
             + (queued_tokens + used_kv_tokens) * queued_tokens_weight

Component meaning:

* ``ewma_latency`` — observed end-to-end latency on this pod (the
  baseline preference: avoid pods that have been slow lately).
* ``max(0, request_tokens - cached_tokens) * t_prefill`` — estimated
  marginal prefill cost for the *uncached* tail of this request. Pods
  with longer matching prefixes pay less here, so the term implements
  a soft prefix-affinity bias.
* ``(queued_tokens + used_kv_tokens) * queued_tokens_weight`` — load
  signal combining queue depth (sum of prompt-token lengths in
  flight) and resident KV-cache usage. Pods that are already saturated
  in either dimension are penalized linearly.

Hyperparameters ``t_prefill`` and ``queued_tokens_weight`` are scalar
weights that trade off the three components. On main they are tuned
online; here we expose them as policy params so a sweep can
characterize sensitivity. Defaults (``t_prefill=0.05``,
``queued_tokens_weight=0.001``) mirror the values in production at the
time of porting.

Signal mapping main → rome simulator:

* ``m.latency`` → ``pod.ewma_latency_ms``
* ``cached_tokens`` → block-level prefix match (count blocks via
  ``kv_cache.has``, multiply by ``block_size``)
* ``request_tokens`` → ``len(request.prompt_tokens)``
* ``queued_tokens`` → ``pod.queued_prompt_tokens`` (engine-tracked Σ
  of in-flight prompt-token lengths; chosen over the alternative of
  approximating from ``active_prefill`` × mean prompt length for
  fidelity to the live signal main reads from SGLang ``/metrics``)
* ``m.num_used_tokens`` → Σ ``token_count`` over the pod's
  ``KVCacheState`` entries (resident KV tokens)

Taxonomy (see ``research/reports/routing-comparison.md`` §3):
    selection=composite (latency + cache-affinity + load),
    state=stateless, fairness=best-effort, topology=any,
    migration=none.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..cluster import ClusterState
from ..core import Decision, PodRuntime, Request
from ..kv_cache import KVCacheState, enumerate_prefix_hashes
from ..policy import register_policy


@register_policy("gorgo")
@dataclass
class GorgoPolicy:
    block_size: int = 16
    t_prefill: float = 0.05
    queued_tokens_weight: float = 0.001

    def _prefix_hashes(self, r: Request) -> list[str]:
        if r.prefix_key:
            return [r.prefix_key]
        return enumerate_prefix_hashes(r.prompt_tokens, block_size=self.block_size)

    def _cached_tokens(
        self, pod: PodRuntime, hashes: list[str], kv_cache: KVCacheState
    ) -> int:
        match = 0
        for h in hashes:
            if kv_cache.has(pod.spec.pod_id, h):
                match += 1
            else:
                break
        return match * self.block_size

    @staticmethod
    def _used_kv_tokens(pod_id: str, kv_cache: KVCacheState) -> int:
        cache = kv_cache.pods.get(pod_id)
        if cache is None:
            return 0
        return sum(entry.token_count for entry in cache.entries.values())

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
        request_tokens = len(request.prompt_tokens)

        scored: list[tuple[float, str, PodRuntime]] = []
        for p in cands:
            cached = self._cached_tokens(p, hashes, kv_cache)
            effective_prefill = max(0, request_tokens - cached)
            prefill_cost = effective_prefill * self.t_prefill
            used_kv = self._used_kv_tokens(p.spec.pod_id, kv_cache)
            queue_cost = (p.queued_prompt_tokens + used_kv) * self.queued_tokens_weight
            score = p.ewma_latency_ms + prefill_cost + queue_cost
            scored.append((score, p.spec.pod_id, p))

        # Tie-break on pod_id (lowest wins) for determinism — matches
        # the rest of the load-balancing group's convention. The score
        # tuple sorts naturally by (score, pod_id), so min() is stable.
        best_score, best_pod_id, best_pod = min(scored, key=lambda t: (t[0], t[1]))

        return Decision(
            best_pod_id,
            best_pod_id,
            rationale=f"gorgo score={best_score:.3f}",
            score=float(best_score),
        )
