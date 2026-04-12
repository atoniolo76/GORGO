"""Discrete-event simulator.

The engine iterates requests in arrival order, asks the policy to
decide, charges cost via the cost model, updates cluster/kv state, and
records metrics. It is deliberately simple (no priority queue, no
preemption) — the point is a deterministic comparison harness, not a
cycle-accurate simulator. More sophisticated scheduling can be added
behind the same interface without touching policies.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..cluster import ClusterState
from ..core import PodRuntime, Request
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

    def run(self, trace) -> MetricsCollector:
        now = 0.0
        for req in trace:
            now = max(now, req.arrival_ts)
            decision = self.policy.decide(req, self.cluster, self.kv_cache)
            if decision.prefill_pod_id == "__none__":
                continue
            pod = self.cluster.get(decision.prefill_pod_id)
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
            if captured < reuse_avail_blocks:
                for h in hashes[captured:]:
                    owners = self.kv_cache.owners_of(h)
                    if owners:
                        owner = max(owners, key=lambda pid: self.kv_cache.pods[pid].entries[h].last_access_ts)
                        break
                if owner and owner != pod.spec.pod_id:
                    pull_blocks = 0
                    for h in hashes[captured:]:
                        if self.kv_cache.has(owner, h):
                            pull_blocks += 1
                        else:
                            break
                    owner_pull_bytes = pull_blocks * self.config.block_size * self.network.kv_bytes_per_token
                    cached_prefix_tokens += pull_blocks * self.config.block_size
                    captured += pull_blocks

            cost = self.cost_model.estimate(
                request=req,
                decision=decision,
                cluster=self.cluster,
                kv_cache=self.kv_cache,
                cached_prefix_tokens=cached_prefix_tokens,
                kv_transport_bytes=owner_pull_bytes,
            )

            self._apply_side_effects(pod, req, hashes, now, captured, cost.total_ms)

            self.metrics.observe(
                RequestRecord(
                    request=req,
                    decision=decision,
                    cost=cost,
                    cached_prefix_tokens=cached_prefix_tokens,
                    reuse_available_blocks=reuse_avail_blocks,
                    reuse_captured_blocks=captured,
                    kv_transport_bytes=owner_pull_bytes,
                    migrated=decision.prefill_pod_id != decision.decode_pod_id,
                )
            )
        return self.metrics

    def _apply_side_effects(
        self,
        pod: PodRuntime,
        req: Request,
        hashes: list[str],
        now: float,
        captured_blocks: int,
        observed_latency_ms: float,
    ) -> None:
        alpha = self.config.kv_ewma_alpha
        pod.ewma_latency_ms = (1 - alpha) * pod.ewma_latency_ms + alpha * observed_latency_ms
        throughput = (len(req.prompt_tokens) + req.max_output_tokens) / max(
            1e-9, observed_latency_ms / 1000.0
        )
        pod.ewma_throughput_tps = (1 - alpha) * pod.ewma_throughput_tps + alpha * throughput
        pod.last_update_ts = now

        # Install block-level prefix entries up to and including what was used.
        for i, h in enumerate(hashes, start=1):
            tokens_so_far = i * self.config.block_size
            byte_size = self.config.block_size * self.network.kv_bytes_per_token
            self.kv_cache.install(
                pod.spec.pod_id,
                PrefixEntry(
                    prefix_hash=h,
                    token_count=self.config.block_size,
                    byte_size=byte_size,
                ),
                now,
            )
            if tokens_so_far >= len(req.prompt_tokens):
                break
