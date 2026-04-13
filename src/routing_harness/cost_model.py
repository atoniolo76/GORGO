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

Prefill/transport overlap: `kv_transport_ms` in the returned
`CostBreakdown` is the raw wire time of the pull. `total_ms`
composes it with prefill under an async-initiation assumption: the
pull starts at dispatch and runs in parallel with prefill compute,
so the prefill phase completes in `max(compute_prefill_ms,
kv_transport_ms)` — the slower of the two. This narrows a prior
synchronous-pull bias (go-npl) that over-penalized small cross-pod
pulls by charging transport strictly additively. The fabric is
still occupied for the full `kv_transport_ms` (compute overlap does
not free the wire), so concurrency tracking in the engine is
unchanged.

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


def decode_batch_bucket(batch: int) -> int:
    """Power-of-two floor bucket for decode batch size.

    Decode throughput curves saturate roughly logarithmically in batch,
    so we key observations by doubling bands (1, 2, 4, 8, 16, ...) rather
    than exact counts — this keeps the observation table small and lets
    a single sample cover a plausible throughput plateau. `batch` is the
    inclusive count (the request being scheduled counts as 1); callers
    that already subtract the request should add it back before calling.
    """
    if batch <= 1:
        return 1
    return 1 << (int(batch).bit_length() - 1)


def load_observations_csv(path) -> dict[str, float]:
    """Load (key, value_ms) pairs from a CSV file.

    Accepts an optional header row (`key,value_ms`). Blank lines and
    lines starting with `#` are ignored. Later values override earlier
    ones for the same key.
    """
    import csv
    from pathlib import Path as _Path

    observations: dict[str, float] = {}
    with _Path(path).open(newline="") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row or not row[0].strip() or row[0].lstrip().startswith("#"):
                continue
            if len(row) < 2:
                raise ValueError(f"instrumentation row needs 2 columns: {row!r}")
            key = row[0].strip()
            if key == "key":  # header
                continue
            observations[key] = float(row[1])
    return observations


class InstrumentedCostModel:
    """Cost model that blends measured values with analytic estimates.

    Decorator around `AnalyticCostModel`. For each request it consults
    the observation table and, when a matching key has been recorded,
    substitutes the measured coefficient into the corresponding
    component; components without a matching observation fall through to
    the analytic estimate unchanged.

    Observation-key schema (all values in milliseconds):
      - `prefill_ms_per_token:<pod_id>` — per-token prefill rate on the
        prefill pod. Replaces `ComputeParams.prefill_ms_per_token` only
        for the matching pod; `prefill_overhead_ms` and the uncached-
        token count are still taken from the analytic path.
      - `decode_ms_per_token:<pod_id>:<batch_bucket>` — per-token decode
        rate for the decode pod at a given batch bucket (power-of-two
        floor of the inclusive concurrent decode count; see
        `decode_batch_bucket`). When present, replaces the batched
        analytic decode cost for that (pod, bucket) combination.
      - `queueing_ms:<pod_id>` — measured mean wait for the prefill pod.
        Replaces the M/M/1 estimate as a flat value (the analytic model
        is itself a steady-state approximation, so a directly measured
        mean is the apples-to-apples replacement).

    Components without an override — routing, network, kv_transport —
    pass through from the analytic model. Use `record(key, value_ms)` to
    add observations at runtime, or construct via `from_observations`.
    """

    def __init__(self, inner: AnalyticCostModel) -> None:
        self.inner = inner
        self._observed: dict[str, float] = {}

    @classmethod
    def from_observations(
        cls, inner: AnalyticCostModel, observations: dict[str, float]
    ) -> "InstrumentedCostModel":
        m = cls(inner)
        for k, v in observations.items():
            m.record(k, v)
        return m

    def record(self, key: str, value_ms: float) -> None:
        self._observed[key] = value_ms

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
        base = self.inner.estimate(
            request,
            decision,
            cluster,
            kv_cache,
            cached_prefix_tokens,
            kv_transport_bytes,
            concurrent_kv_transport_bytes,
        )
        compute_prefill = base.compute_prefill_ms
        compute_decode = base.compute_decode_ms
        queueing = base.queueing_ms

        prefill_key = f"prefill_ms_per_token:{decision.prefill_pod_id}"
        if prefill_key in self._observed:
            uncached = max(0, len(request.prompt_tokens) - cached_prefix_tokens)
            compute_prefill = (
                self.inner.compute.prefill_overhead_ms
                + uncached * self._observed[prefill_key]
            )

        decode_pod = cluster.pods.get(
            decision.decode_pod_id, cluster.get(decision.prefill_pod_id)
        )
        bucket = decode_batch_bucket(decode_pod.active_decode + 1)
        decode_key = f"decode_ms_per_token:{decision.decode_pod_id}:{bucket}"
        if decode_key in self._observed:
            compute_decode = (
                self.inner.compute.decode_overhead_ms
                + request.max_output_tokens * self._observed[decode_key]
            )

        queue_key = f"queueing_ms:{decision.prefill_pod_id}"
        if queue_key in self._observed:
            queueing = self._observed[queue_key]

        return CostBreakdown(
            routing_ms=base.routing_ms,
            queueing_ms=queueing,
            compute_prefill_ms=compute_prefill,
            compute_decode_ms=compute_decode,
            network_ms=base.network_ms,
            kv_transport_ms=base.kv_transport_ms,
        )
