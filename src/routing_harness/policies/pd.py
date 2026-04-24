"""PD: prefill-decode disaggregation-aware routing.

For clusters where some pods are Phase.PREFILL and others Phase.DECODE,
picks:
  - a prefill pod using prefix-cache (reuse wins)
  - a decode pod using least-active-decode (decode is memory-bound,
    fairness matters more than cache locality)

Decode selection is topology-aware: if the chosen prefill pod declares
`peer_ids` (NVLink islands, RDMA groups, etc.), the decode candidate
set is filtered to those peers. If the filter yields no candidates
(peer_ids unset, or no declared peers are in the decode pool) the
policy falls back to the full decode pool — degraded but available —
rather than refusing the request. See go-6i2 (F23) and
`research/reports/policy_audits/pd_topology.md` §2.4.

Partial-availability fallback: if only one role-pool is populated
(e.g. every DECODE pod is unreachable but PREFILL/BOTH pods remain),
the policy collapses to colocated execution on a single pod chosen by
prefill criteria rather than dropping every request. The chosen pod
serves both phases; the engine treats `decode_pod_id == prefill_pod_id`
as a no-handoff path, so degraded mode still produces tokens. See
go-997 (F24) and `research/reports/policy_audits/pd_topology.md` §2.x.

If no PD roles are present (all pods are Phase.BOTH), the two pools
collapse to the same set, but prefill and decode are still chosen
independently — prefix-match for prefill, active-decode for decode —
so they may land on different BOTH pods when those two signals
disagree. When they do, the engine treats it as a same-cluster KV
handoff and charges `pd_handoff_bytes` accordingly
(engine.py:168–175). Under perfect ties both branches settle on the
smallest `pod_id` and colocate (no gratuitous handoff); see go-5dt
(F19) and `research/reports/policy_audits/pd_topology.md` §2.5.

The decode signal is pod.active_decode directly, not
ewma_latency_ms * active_decode. The engine only updates
ewma_latency_ms on the prefill pod (engine.py:255), so a multiplier on
pure-DECODE pods would reduce to warm_constant * active_decode —
latency-unaware. Under the analytic cost model decode latency is
roughly constant per token, so active_decode already approximates a
time-domain load signal under continuous batching. See go-vix (F20)
and `research/reports/policy_audits/pd_topology.md` §2.4.

Taxonomy (see `research/reports/routing-comparison.md` §3):
    selection=composite (phase-split: cache-affinity for prefill + load
    for decode), state=stateless, fairness=best-effort,
    topology=pd-aware (exploits role split when present, tolerates
    colocated), migration=none.
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
        if not prefill and not decode:
            return Decision("__none__", "__none__", "pd-pools-empty")

        colocate_fallback = not prefill or not decode
        if not prefill:
            prefill = list(decode)
        if not decode:
            decode = list(prefill)

        hashes = self._prefix(request)

        def _match_len(pod_id: str) -> int:
            n = 0
            for h in hashes:
                if not kv_cache.has(pod_id, h):
                    break
                n += 1
            return n

        best_prefill = min(
            prefill,
            key=lambda p: (
                -_match_len(p.spec.pod_id),
                p.active_prefill,
                p.spec.pod_id,
            ),
        )

        if colocate_fallback:
            rationale = f"pd colocated={best_prefill.spec.pod_id} one-pool-empty"
            return Decision(best_prefill.spec.pod_id, best_prefill.spec.pod_id, rationale=rationale)

        peers = set(best_prefill.spec.peer_ids)
        peered_decode = [p for p in decode if p.spec.pod_id in peers]
        decode_pool = peered_decode if peered_decode else decode
        peer_tag = "peer" if peered_decode else ("unpeered" if peers else "nopeers")

        best_decode = min(decode_pool, key=lambda p: (p.active_decode, p.spec.pod_id))
        rationale = (
            f"pd prefill={best_prefill.spec.pod_id} decode={best_decode.spec.pod_id} {peer_tag}"
        )
        return Decision(best_prefill.spec.pod_id, best_decode.spec.pod_id, rationale=rationale)
