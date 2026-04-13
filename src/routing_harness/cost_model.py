"""CostModel: estimate end-to-end latency components.

The default `AnalyticCostModel` is closed-form and dependency-free. It
charges:
  - compute_prefill_ms proportional to (prompt_tokens - cached_prefix),
  - compute_decode_ms proportional to max_output_tokens, optionally
    amortized by continuous-batching when `decode_batch_k > 0`,
  - network_ms from a simple (latency + bytes/bandwidth) model,
  - kv_transport_ms when a prefix must be pulled from a peer pod,
  - routing_ms = constant policy-specific budget,
  - queueing_ms from an M/M/1 single-server approximation of the pod's
    prefill slots: W_q = rho/(1-rho) * S, where rho is slot occupancy
    (active_prefill / max_concurrent_prefill) clamped below 1.0, and S
    is the representative prefill service time for the workload.

The queueing term superlinearly diverges near saturation — which is
the point: prior linear scaling under-reported absolute p99 by ~8x at
high load (peer review v1, critic C). Relative ordering across
policies is preserved because every policy sees the same formula; the
change is that saturation is now penalized non-linearly.

KV transport uses a fluid fair-share fabric model: when multiple
transfers overlap on the inter-pod fabric, each sees effective
bandwidth `B / (Σb_in_flight / b_self)` so that the individual
transfer time becomes `rtt + Σb_in_flight / B`. The caller passes
`concurrent_kv_transport_bytes` — the sum of this transfer's bytes
plus the bytes of transfers still in flight when this one starts —
and the single-transfer case (no overlap) recovers the uncontended
formula `rtt + bytes/B`.

All coefficients are explicit in NetworkParams / ComputeParams. No
silent defaults. An `InstrumentedCostModel` subclass is scaffolded for
the future, where measured values replace analytic estimates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from .cluster import ClusterState
from .core import CostBreakdown, Decision, Request
from .kv_cache import KVCacheState


@dataclass(frozen=True)
class ComputeParams:
    """Compute-cost coefficients.

    `decode_batch_k` controls the continuous-batching amortization of
    per-request decode latency. At k=0 (default), decode is charged at a
    constant `decode_ms_per_token` regardless of concurrency — the
    original, deliberately-pessimistic behavior preserved for
    backwards-compatible run_ids. At k>0, the effective per-token decode
    cost is

        decode_ms_per_token / (1 + k * log(1 + max(0, batch - 1)))

    where `batch` is the number of concurrent decodes on the decode pod
    (inclusive of the request being scheduled). The formula is pinned so
    batch=1 reproduces the single-request baseline, and growth is
    logarithmic (sublinear) in batch — matching observed continuous-
    batching throughput curves on vLLM / TRT-LLM until calibration.
    Reasonable values land in roughly [0.3, 1.5]; the exact number needs
    to be fit to real hardware.
    """

    prefill_ms_per_token: float
    decode_ms_per_token: float
    prefill_overhead_ms: float
    decode_overhead_ms: float
    decode_batch_k: float = 0.0


@dataclass(frozen=True)
class NetworkParams:
    client_rtt_ms: float
    inter_pod_rtt_ms: float
    inter_pod_bandwidth_gbps: float
    kv_bytes_per_token: int
    serialization_overhead_ms: float


@dataclass(frozen=True)
class SchedulerParams:
    base_routing_ms: float
    per_pod_consideration_us: float  # scales with cluster size


class CostModel(Protocol):
    def estimate(
        self,
        request: Request,
        decision: Decision,
        cluster: ClusterState,
        kv_cache: KVCacheState,
        cached_prefix_tokens: int,
        kv_transport_bytes: int,
        concurrent_kv_transport_bytes: int | None = None,
    ) -> CostBreakdown:
        ...


# Clamp utilization below 1.0 before applying the M/M/1 waiting-time
# formula. Without this the expression diverges at saturation; real
# systems don't stop responding at rho=1, they just back up quickly.
# 0.99 is a pragmatic cap — high enough that tails still explode
# superlinearly, low enough to stay finite when a pod momentarily
# exceeds its concurrency budget (which active_prefill can, since it
# is a soft count, not a hard admission limit).
_RHO_MAX = 0.99


def _mm1_wait_ms(active: int, capacity: int, service_ms: float) -> float:
    """Mean queue-wait under an M/M/1 approximation.

    Treats a pod's prefill slots as one aggregate server with utilization
    rho = active / capacity. Returns 0 when idle or service_ms <= 0.
    """
    if service_ms <= 0.0 or active <= 0:
        return 0.0
    c = max(1, capacity)
    rho = active / c
    if rho >= _RHO_MAX:
        rho = _RHO_MAX
    return (rho / (1.0 - rho)) * service_ms


@dataclass
class AnalyticCostModel:
    compute: ComputeParams
    network: NetworkParams
    scheduler: SchedulerParams

    def estimate(
        self,
        request: Request,
        decision: Decision,
        cluster: ClusterState,
        kv_cache: KVCacheState,
        cached_prefix_tokens: int,
        kv_transport_bytes: int,
        concurrent_kv_transport_bytes: int | None = None,
    ) -> CostBreakdown:
        prompt_len = len(request.prompt_tokens)
        uncached = max(0, prompt_len - cached_prefix_tokens)
        pod = cluster.get(decision.prefill_pod_id)
        # Representative prefill service time for M/M/1 wait: use the
        # current request's *uncached* prefill cost as a proxy. Cache
        # hits ahead in the queue shorten S, so it's correct that cached
        # requests contribute less to downstream wait. Guard with a
        # tiny floor so a pure cache hit with a non-empty queue still
        # records the overhead-driven wait.
        avg_service_ms = (
            self.compute.prefill_overhead_ms
            + max(1, uncached) * self.compute.prefill_ms_per_token
        )
        queueing_ms = _mm1_wait_ms(
            pod.active_prefill,
            pod.spec.max_concurrent_prefill,
            avg_service_ms,
        )
        routing_ms = (
            self.scheduler.base_routing_ms
            + self.scheduler.per_pod_consideration_us * len(cluster) / 1000.0
        )
        compute_prefill = (
            self.compute.prefill_overhead_ms
            + uncached * self.compute.prefill_ms_per_token
        )
        decode_pod = cluster.pods.get(decision.decode_pod_id, pod)
        # Concurrent decode batch at dispatch time, inclusive of this
        # request. See ComputeParams.decode_batch_k for the rationale.
        decode_batch = decode_pod.active_decode + 1
        k = self.compute.decode_batch_k
        if k > 0.0:
            amortization = 1.0 + k * math.log(1 + max(0, decode_batch - 1))
            effective_decode_ms_per_token = (
                self.compute.decode_ms_per_token / amortization
            )
        else:
            effective_decode_ms_per_token = self.compute.decode_ms_per_token
        compute_decode = (
            self.compute.decode_overhead_ms
            + request.max_output_tokens * effective_decode_ms_per_token
        )
        network_ms = (
            2 * self.network.client_rtt_ms
            + self.network.serialization_overhead_ms
        )
        if kv_transport_bytes > 0:
            bytes_per_ms = (
                self.network.inter_pod_bandwidth_gbps * 1e9 / 8.0 / 1000.0
            )
            # Fluid fair-share contention: if other transfers are in
            # flight on the fabric, this transfer waits for the shared
            # bandwidth backlog to drain. The caller computes the sum
            # of overlapping transfer bytes (including our own). When
            # nothing else is in flight, contended == our own bytes
            # and the formula reduces to the uncontended case.
            contended = (
                concurrent_kv_transport_bytes
                if concurrent_kv_transport_bytes is not None
                else kv_transport_bytes
            )
            if contended < kv_transport_bytes:
                contended = kv_transport_bytes
            kv_transport_ms = (
                self.network.inter_pod_rtt_ms + contended / bytes_per_ms
            )
        else:
            kv_transport_ms = 0.0
        return CostBreakdown(
            routing_ms=routing_ms,
            queueing_ms=queueing_ms,
            compute_prefill_ms=compute_prefill,
            compute_decode_ms=compute_decode,
            network_ms=network_ms,
            kv_transport_ms=kv_transport_ms,
        )


class InstrumentedCostModel:
    """Placeholder for a future cost model backed by real measurements.

    Design intent: a decorator around AnalyticCostModel that, when a
    run is executed against a real server, overrides any component it
    has observed (e.g. measured prefill latency per token) while falling
    back to analytic estimates for unobserved components.

    Not implemented — instrumentation hooks live here so the interface
    is stable.
    """

    def __init__(self, inner: AnalyticCostModel) -> None:
        self.inner = inner
        self._observed: dict[str, float] = {}

    def record(self, key: str, value_ms: float) -> None:
        self._observed[key] = value_ms

    def estimate(self, *args, **kwargs) -> CostBreakdown:  # pragma: no cover
        # Future: blend observed with analytic. For now, delegate.
        return self.inner.estimate(*args, **kwargs)
