"""Discrete-event simulator.

The engine iterates requests in arrival order, asks the policy to
decide, charges cost via the cost model, updates cluster/kv state, and
records metrics. It is deliberately simple (no priority queue, no
preemption) — the point is a deterministic comparison harness, not a
cycle-accurate simulator. More sophisticated scheduling can be added
behind the same interface without touching policies.

Load-counter approximation: because the engine is one-pass without a
true time advance, `active_prefill` / `active_decode` are incremented
on `decide` and decremented once the next request's arrival_ts is
past the projected completion. This gives load-aware policies a
non-degenerate load signal while remaining deterministic. Documented as
an approximation in docs/peer_review_v1.md §7.

Fabric contention: concurrent KV transfers share the inter-pod
fabric's bandwidth. The engine tracks in-flight transfers in a heap
keyed by projected fabric completion and sums overlapping bytes; the
cost model uses that sum under a fluid fair-share model so an
individual transfer slows as the fabric saturates. A lone transfer
reduces to the uncontended `rtt + bytes/B` formula.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from heapq import heappop, heappush

from ..cluster import ClusterState
from ..core import Phase, PodRuntime, Request
from ..cost_model import CostModel, NetworkParams
from ..kv_cache import KVCacheState, PrefixEntry, enumerate_prefix_hashes
from ..policy import RoutingPolicy
from .metrics import MetricsCollector, RequestRecord


@dataclass
class EngineConfig:
    kv_ewma_alpha: float = 0.2
    block_size: int = 16
    initial_warm_latency_ms: float = 5.0


@dataclass
class SimulationEngine:
    cluster: ClusterState
    kv_cache: KVCacheState
    policy: RoutingPolicy
    cost_model: CostModel
    network: NetworkParams
    config: EngineConfig
    metrics: MetricsCollector
    _pending: list[tuple[float, int, str, str, Request, "object", float]] = field(
        default_factory=list
    )
    _fabric_inflight: list[tuple[float, int, int]] = field(
        default_factory=list
    )
    _seq: int = 0

    def __post_init__(self) -> None:
        for p in self.cluster.pods.values():
            if p.ewma_latency_ms == 0.0:
                p.ewma_latency_ms = self.config.initial_warm_latency_ms

    def _prefix_hashes(self, req: Request) -> list[str]:
        if req.prefix_key:
            return [req.prefix_key]
        return enumerate_prefix_hashes(
            req.prompt_tokens, block_size=self.config.block_size
        )

    def _drain_fabric(self, now_s: float) -> int:
        """Drop transfers whose projected fabric completion is past.

        Returns the sum of bytes still in flight on the fabric at
        `now_s`. The sum is used to charge contention on the next
        transfer via the cost model's fair-share formula.
        """
        while self._fabric_inflight and self._fabric_inflight[0][0] <= now_s:
            heappop(self._fabric_inflight)
        return sum(b for (_, _, b) in self._fabric_inflight)

    def _retire_up_to(self, now_s: float) -> None:
        """Retire in-flight requests whose projected completion is past.

        Drains the pending heap, decrements pod active counters, and
        invokes the policy's optional `observe_completion` hook.
        """
        hook = getattr(self.policy, "observe_completion", None)
        while self._pending and self._pending[0][0] <= now_s:
            _, _, prefill_id, decode_id, req, decision, service_ms = heappop(self._pending)
            pod_p = self.cluster.pods.get(prefill_id)
            if pod_p is not None and pod_p.active_prefill > 0:
                pod_p.active_prefill -= 1
            if pod_p is not None and pod_p.queued > 0:
                pod_p.queued -= 1
            if pod_p is not None:
                pod_p.pending_work_ms = max(0.0, pod_p.pending_work_ms - service_ms)
            pod_d = self.cluster.pods.get(decode_id)
            if pod_d is not None and pod_d.active_decode > 0:
                pod_d.active_decode -= 1
            if hook is not None:
                tokens = len(req.prompt_tokens) + req.max_output_tokens
                hook(req, decision, float(tokens))

    def run(self, trace) -> MetricsCollector:
        now = 0.0
        for req in trace:
            now = max(now, req.arrival_ts)
            self._retire_up_to(now)
            decision = self.policy.decide(req, self.cluster, self.kv_cache)
            if decision.prefill_pod_id == "__none__":
                continue
            pod = self.cluster.get(decision.prefill_pod_id)
            decode_pod_id = decision.decode_pod_id
            decode_pod = self.cluster.pods.get(decode_pod_id, pod)
            hashes = self._prefix_hashes(req)

            reuse_avail_blocks = sum(
                1 for h in hashes if self.kv_cache.reuse_available(h)
            )
            captured = 0
            for h in hashes:
                if self.kv_cache.has(pod.spec.pod_id, h):
                    captured += 1
                else:
                    break
            cached_prefix_tokens = captured * self.config.block_size

            # Decide whether to pull a missing prefix from a peer (simple rule:
            # pull if there is a strictly longer cluster-wide prefix available).
            owner_pull_bytes = 0
            owner = None
            pull_blocks = 0
            if captured < reuse_avail_blocks:
                for h in hashes[captured:]:
                    owners = self.kv_cache.owners_of(h, hashes)
                    if owners:
                        owner = max(
                            owners,
                            key=lambda pid: self.kv_cache.pods[pid].entries[h].last_access_ts,
                        )
                        break
                if owner and owner != pod.spec.pod_id:
                    for h in hashes[captured:]:
                        if self.kv_cache.has(owner, h):
                            pull_blocks += 1
                        else:
                            break
                    owner_pull_bytes = (
                        pull_blocks
                        * self.config.block_size
                        * self.network.kv_bytes_per_token
                    )
                    cached_prefix_tokens += pull_blocks * self.config.block_size
                    captured += pull_blocks

            # PD-disaggregation handoff: if prefill and decode pods differ,
            # the computed KV must cross the fabric to the decode pod.
            pd_handoff_bytes = 0
            if (
                decode_pod_id != pod.spec.pod_id
                and decode_pod_id in self.cluster.pods
            ):
                pd_handoff_bytes = (
                    len(req.prompt_tokens) * self.network.kv_bytes_per_token
                )

            kv_transport_bytes = owner_pull_bytes + pd_handoff_bytes
            # Fabric contention: retire transfers whose fabric-side
            # completion has passed, then charge this transfer against
            # the sum of (other in-flight bytes) + (own bytes). The
            # cost model implements fluid fair-share on that sum.
            if kv_transport_bytes > 0:
                other_inflight_bytes = self._drain_fabric(now)
                concurrent_bytes = other_inflight_bytes + kv_transport_bytes
            else:
                self._drain_fabric(now)
                concurrent_bytes = 0

            cost = self.cost_model.estimate(
                request=req,
                decision=decision,
                cluster=self.cluster,
                kv_cache=self.kv_cache,
                cached_prefix_tokens=cached_prefix_tokens,
                kv_transport_bytes=kv_transport_bytes,
                concurrent_kv_transport_bytes=concurrent_bytes or None,
            )

            if kv_transport_bytes > 0:
                self._seq += 1
                heappush(
                    self._fabric_inflight,
                    (
                        now + cost.kv_transport_ms / 1000.0,
                        self._seq,
                        kv_transport_bytes,
                    ),
                )

            self._apply_side_effects(
                pod=pod,
                decode_pod=decode_pod,
                req=req,
                hashes=hashes,
                now=now,
                captured_blocks=captured,
                pull_blocks_from_peer=pull_blocks,
                pd_handoff=pd_handoff_bytes > 0,
                observed_latency_ms=cost.total_ms,
                decision=decision,
            )

            self.metrics.observe(
                RequestRecord(
                    request=req,
                    decision=decision,
                    cost=cost,
                    cached_prefix_tokens=cached_prefix_tokens,
                    reuse_available_blocks=reuse_avail_blocks,
                    reuse_captured_blocks=captured,
                    kv_transport_bytes=kv_transport_bytes,
                    migrated=decision.prefill_pod_id != decision.decode_pod_id,
                )
            )

        # Drain any still-pending completions at end-of-trace so that
        # observe_completion fires for every request.
        self._retire_up_to(float("inf"))
        return self.metrics

    def _apply_side_effects(
        self,
        pod: PodRuntime,
        decode_pod: PodRuntime,
        req: Request,
        hashes: list[str],
        now: float,
        captured_blocks: int,
        pull_blocks_from_peer: int,
        pd_handoff: bool,
        observed_latency_ms: float,
        decision,
    ) -> None:
        alpha = self.config.kv_ewma_alpha
        pod.ewma_latency_ms = (1 - alpha) * pod.ewma_latency_ms + alpha * observed_latency_ms
        throughput = (len(req.prompt_tokens) + req.max_output_tokens) / max(
            1e-9, observed_latency_ms / 1000.0
        )
        pod.ewma_throughput_tps = (1 - alpha) * pod.ewma_throughput_tps + alpha * throughput
        pod.last_update_ts = now

        # Record the in-flight arrival on both pods for load-aware policies.
        pod.active_prefill += 1
        pod.queued += 1
        pod.pending_work_ms += observed_latency_ms
        decode_pod.active_decode += 1
        # Schedule retirement at now + observed latency (ms → s).
        self._seq += 1
        heappush(
            self._pending,
            (
                now + observed_latency_ms / 1000.0,
                self._seq,
                pod.spec.pod_id,
                decode_pod.spec.pod_id,
                req,
                decision,
                observed_latency_ms,
            ),
        )

        # Install block-level prefix entries up to and including what was
        # used. For PD handoff and for peer-pulled blocks, also install on
        # the *destination* cache so subsequent reuse claims are honest.
        byte_size = self.config.block_size * self.network.kv_bytes_per_token
        for i, h in enumerate(hashes, start=1):
            tokens_so_far = i * self.config.block_size
            entry = PrefixEntry(
                prefix_hash=h,
                token_count=self.config.block_size,
                byte_size=byte_size,
            )
            self.kv_cache.install(pod.spec.pod_id, entry, now)
            if pd_handoff and decode_pod.spec.pod_id != pod.spec.pod_id:
                # Decode pod materializes the same prefix blocks it just
                # received so it can be a reuse source for future requests.
                self.kv_cache.install(
                    decode_pod.spec.pod_id,
                    PrefixEntry(
                        prefix_hash=h,
                        token_count=self.config.block_size,
                        byte_size=byte_size,
                    ),
                    now,
                )
            if tokens_so_far >= len(req.prompt_tokens):
                break

        # Peer-pulled blocks must be installed on the destination pod's
        # cache. Without this the capture-rate metric claims credit for
        # blocks that were never actually materialized locally, and
        # subsequent requests route as if the prefix were cached when it
        # is not. (Defect found in peer review v1 §3.)
        if pull_blocks_from_peer > 0:
            start = captured_blocks - pull_blocks_from_peer
            for h in hashes[start : start + pull_blocks_from_peer]:
                self.kv_cache.install(
                    pod.spec.pod_id,
                    PrefixEntry(
                        prefix_hash=h,
                        token_count=self.config.block_size,
                        byte_size=byte_size,
                    ),
                    now,
                )
