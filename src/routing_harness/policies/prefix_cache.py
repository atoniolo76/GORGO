"""Prefix-cache: route to the pod that already owns the longest matching prefix.

Baseline KV-cache aware routing (inspired by SGLang router and similar
designs). No cross-pod transport; if nobody has the prefix, fall back to
least-request to avoid hotspotting on the first pod that happens to
cache a popular prefix.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..cluster import ClusterState
from ..core import Decision, Request
from ..kv_cache import KVCacheState, enumerate_prefix_hashes
from ..policy import register_policy


@register_policy("prefix-cache")
@dataclass
class PrefixCachePolicy:
    block_size: int = 16

    def _hashes(self, request: Request) -> list[str]:
        if request.prefix_key:
            return [request.prefix_key]
        return enumerate_prefix_hashes(request.prompt_tokens, block_size=self.block_size)

    def decide(
        self,
        request: Request,
        cluster: ClusterState,
        kv_cache: KVCacheState,
    ) -> Decision:
        cands = cluster.prefill_capable()
        if not cands:
            return Decision("__none__", "__none__", "no-prefill-capable-pod")
        hashes = self._hashes(request)
        best_pod: str | None = None
        best_len = -1
        for p in cands:
            match_len = 0
            for h in hashes:
                if kv_cache.has(p.spec.pod_id, h):
                    match_len += 1
                else:
                    break
            if match_len > best_len or (
                match_len == best_len
                and best_pod is not None
                and p.spec.pod_id < best_pod
            ):
                best_len = match_len
                best_pod = p.spec.pod_id
        if best_pod is None or best_len <= 0:
            pick = min(
                cands,
                key=lambda p: (p.active_prefill + p.queued, p.spec.pod_id),
            )
            return Decision(
                pick.spec.pod_id, pick.spec.pod_id, rationale="no-prefix-match/fallback-LRQ"
            )
        return Decision(
            best_pod, best_pod, rationale=f"prefix-match-blocks={best_len}", score=float(best_len)
        )
