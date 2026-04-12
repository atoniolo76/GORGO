"""Preble-*inspired* prefix-cache routing with hotspot-aware balancing.

Deliberately simplified formulation, not a faithful reimplementation of
Preble. The published Preble system optimizes routing over a reuse
graph with load prediction; this policy reduces that to a linear
combination:

    score = alpha * prefix_match_blocks - beta * load_factor

where load_factor is normalized (active + queued)/capacity. Hotspot
mitigation: if the top-scoring pod's load_factor exceeds
`hotspot_threshold`, defer to a less-loaded pod that already has *some*
prefix overlap, even if shorter. The coefficients (alpha, beta,
hotspot_threshold) are magic numbers chosen to expose the trade-off in
this harness; empirical tuning is expected before drawing any
conclusion about Preble-the-system from this policy.

The registered id `prefix-cache-preble` is intentionally kept stable
(config compatibility). Refer to this class name or docstring when
citing the algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..cluster import ClusterState
from ..core import Decision, Request
from ..kv_cache import KVCacheState, enumerate_prefix_hashes
from ..policy import register_policy


@register_policy("prefix-cache-preble")
@dataclass
class PreblePrefixCachePolicy:
    block_size: int = 16
    alpha: float = 1.0
    beta: float = 0.5
    hotspot_threshold: float = 0.9

    def _prefix_hashes(self, r: Request) -> list[str]:
        if r.prefix_key:
            return [r.prefix_key]
        return enumerate_prefix_hashes(r.prompt_tokens, block_size=self.block_size)

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
        scored = []
        for p in cands:
            match = 0
            for h in hashes:
                if kv_cache.has(p.spec.pod_id, h):
                    match += 1
                else:
                    break
            cap = max(1, p.spec.max_concurrent_prefill + p.spec.max_concurrent_decode)
            load = (p.active_prefill + p.active_decode + p.queued) / cap
            score = self.alpha * match - self.beta * load
            scored.append((score, match, load, p))
        scored.sort(key=lambda t: (-t[0], t[3].spec.pod_id))
        top_score, top_match, top_load, top_pod = scored[0]
        if top_load > self.hotspot_threshold:
            for _, match, load, p in scored[1:]:
                if match > 0 and load < self.hotspot_threshold:
                    return Decision(
                        p.spec.pod_id,
                        p.spec.pod_id,
                        rationale=f"hotspot-avoid top={top_pod.spec.pod_id}",
                        score=match - load,
                    )
        return Decision(
            top_pod.spec.pod_id,
            top_pod.spec.pod_id,
            rationale=f"preble match={top_match} load={top_load:.2f}",
            score=top_score,
        )
