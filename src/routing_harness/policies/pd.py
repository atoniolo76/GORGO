"""PD: prefill-decode disaggregation-aware routing.

For clusters where some pods are Phase.PREFILL and others Phase.DECODE,
picks:
  - a prefill pod using prefix-cache (reuse wins)
  - a decode pod using least-busy-time (decode is memory-bound, fairness
    matters more than cache locality)

If no PD roles are present (all pods are Phase.BOTH), degrades to the
prefix-cache policy to keep the harness well-defined.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..cluster import ClusterState
from ..core import Decision, Phase, Request
from ..kv_cache import KVCacheState, enumerate_prefix_hashes
from ..policy import register_policy


@register_policy("pd")
@dataclass
class PDPolicy:
    block_size: int = 16

    def _prefix(self, r: Request) -> list[str]:
        if r.prefix_key:
            return [r.prefix_key]
        return enumerate_prefix_hashes(r.prompt_tokens, block_size=self.block_size)

    def decide(
        self,
        request: Request,
        cluster: ClusterState,
        kv_cache: KVCacheState,
    ) -> Decision:
        prefill = [p for p in cluster.pods.values() if p.spec.role in (Phase.PREFILL, Phase.BOTH)]
        decode = [p for p in cluster.pods.values() if p.spec.role in (Phase.DECODE, Phase.BOTH)]
        if not prefill or not decode:
            return Decision("__none__", "__none__", "pd-pools-empty")

        hashes = self._prefix(request)
        best_prefill = max(
            prefill,
            key=lambda p: (
                sum(1 for h in hashes if kv_cache.has(p.spec.pod_id, h)),
                -p.active_prefill,
                p.spec.pod_id,
            ),
        )

        def busy(p):
            return p.ewma_latency_ms * (p.active_decode + p.queued)

        best_decode = min(decode, key=lambda p: (busy(p), p.spec.pod_id))
        rationale = (
            f"pd prefill={best_prefill.spec.pod_id} decode={best_decode.spec.pod_id}"
        )
        return Decision(best_prefill.spec.pod_id, best_decode.spec.pod_id, rationale=rationale)
