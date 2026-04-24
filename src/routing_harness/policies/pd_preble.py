"""PD-Preble: disaggregation-aware routing with Preble hotspot deflection
on the prefill pool.

Fixes F25 (see ``research/reports/policy_audits/pd_topology.md`` §2.6).

Plain ``pd`` selects the prefill pod by longest prefix match, tie-broken
by ``active_prefill`` and then ``pod_id``. Under a high-skew workload
(many identical-prompt requests) this cache-locks onto whichever pod
won the first tie-break: the winner warms the prefix, then wins every
subsequent request on match count, regardless of load. The 2×2 PD
repro in §2.4 routed all 50 identical-prompt requests to ``pfB``.

``pd-preble`` applies the ``prefix-cache-preble`` exploit/explore gate
to the prefill pool:

1. **Exploit/explore gate (E2).** If the best prefix match saves more
   tokens than the uncached tail would cost (``missed < cached``), bind
   to the prefix owner; otherwise pick the pod with the lowest
   ``pending_work_ms`` (explore).
2. **Relative hotspot deflection.** Even when exploit wins, if the
   owner's ``pending_work_ms`` exceeds ``th_bal`` times the lightest
   prefill pod's load, deflect to the lightest pod. This prevents the
   cache-lock pathology on repeated identical prompts.

Decode selection is unchanged from ``pd``:
  - filter decode candidates by the chosen prefill pod's ``peer_ids``
    (NVLink islands, RDMA groups), falling back to the full pool if
    the peer set is empty or yields no candidates (F23, go-6i2);
  - pick by ``active_decode`` ascending, tie-broken by ``pod_id`` (F20,
    go-vix).

Colocated fallback (F24, go-997) is preserved: when one role-pool is
empty, collapse to single-pod execution using the Preble gate on the
surviving pool. Both ``prefill_pod_id`` and ``decode_pod_id`` are set
to the chosen pod; the engine treats this as a no-handoff path.

If no PD roles are present (all pods are Phase.BOTH) the two pools are
the same set; the gate runs on the prefill pool and the decode pick
uses the same set under its own tie-break, preserving the F19 fix
(smallest pod_id wins both branches under ties, so colocated clusters
avoid gratuitous handoff).

Taxonomy (see ``research/reports/routing-comparison.md`` §3):
    selection=composite (phase-split: exploit/explore on prefill +
    load on decode), state=stateless, fairness=best-effort,
    topology=pd-aware (exploits role split, tolerates colocated),
    migration=none.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..cluster import ClusterState
from ..core import Decision, Phase, PodRuntime, Request
from ..kv_cache import KVCacheState, enumerate_prefix_hashes
from ..policy import register_policy


@register_policy("pd-preble")
@dataclass
class PDPreblePolicy:
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

    def _pick_prefill(
        self,
        prefill: list[PodRuntime],
        request: Request,
        kv_cache: KVCacheState,
    ) -> tuple[PodRuntime, str]:
        """Preble exploit/explore gate with relative-imbalance deflection.

        Returns (chosen_pod, gate_tag) where gate_tag is one of
        ``exploit``, ``exploit-hotspot-redirect``, or ``explore``.
        """
        hashes = self._prefix_hashes(request)

        pod_data: list[tuple[int, float, PodRuntime]] = []
        for p in prefill:
            match = 0
            for h in hashes:
                if kv_cache.has(p.spec.pod_id, h):
                    match += 1
                else:
                    break
            pod_data.append((match, self._load_ms(p), p))

        # Exploit candidate: most prefix match, then least load, then
        # lowest pod_id. Tie-break on pod_id must match the other
        # branches so identical requests don't oscillate on ties.
        best_match, best_load, best_pod = min(
            pod_data,
            key=lambda t: (-t[0], t[1], t[2].spec.pod_id),
        )

        # Clamp missed_tokens at 0: under prefix_key, best_match is
        # {0,1} on a single opaque hash regardless of prompt length, so
        # cached_tokens can exceed the actual prompt length. Without
        # the clamp, short prompts spuriously satisfy missed < cached.
        cached_tokens = best_match * self.block_size
        missed_tokens = max(0, len(request.prompt_tokens) - cached_tokens)

        if best_match > 0 and missed_tokens < cached_tokens:
            min_load = min(t[1] for t in pod_data)
            if best_load > self.th_bal * min_load:
                lightest = min(pod_data, key=lambda t: (t[1], t[2].spec.pod_id))
                return lightest[2], "exploit-hotspot-redirect"
            return best_pod, "exploit"

        lightest = min(pod_data, key=lambda t: (t[1], t[2].spec.pod_id))
        return lightest[2], "explore"

    def decide(
        self,
        request: Request,
        cluster: ClusterState,
        kv_cache: KVCacheState,
    ) -> Decision:
        prefill = [p for p in cluster.pods.values() if p.spec.role in (Phase.PREFILL, Phase.BOTH)]
        decode = [p for p in cluster.pods.values() if p.spec.role in (Phase.DECODE, Phase.BOTH)]
        if not prefill and not decode:
            return Decision("__none__", "__none__", "pd-preble-pools-empty")

        colocate_fallback = not prefill or not decode
        if not prefill:
            prefill = list(decode)
        if not decode:
            decode = list(prefill)

        best_prefill, gate = self._pick_prefill(prefill, request, kv_cache)

        if colocate_fallback:
            rationale = f"pd-preble colocated={best_prefill.spec.pod_id} gate={gate} one-pool-empty"
            return Decision(
                best_prefill.spec.pod_id,
                best_prefill.spec.pod_id,
                rationale=rationale,
            )

        peers = set(best_prefill.spec.peer_ids)
        peered_decode = [p for p in decode if p.spec.pod_id in peers]
        decode_pool = peered_decode if peered_decode else decode
        peer_tag = "peer" if peered_decode else ("unpeered" if peers else "nopeers")

        best_decode = min(decode_pool, key=lambda p: (p.active_decode, p.spec.pod_id))
        rationale = (
            f"pd-preble prefill={best_prefill.spec.pod_id} "
            f"decode={best_decode.spec.pod_id} gate={gate} {peer_tag}"
        )
        return Decision(
            best_prefill.spec.pod_id,
            best_decode.spec.pod_id,
            rationale=rationale,
        )
