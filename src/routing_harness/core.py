"""Core data types shared across policies, simulator, and cost model.

All types are immutable dataclasses (frozen=True) where practical so they
are safe to pass across threads / processes and so test fixtures are
trivially comparable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Phase(str, Enum):
    """LLM inference phase.

    PREFILL is compute-bound, amortizable across shared prefixes, and the
    primary driver of KV-cache population. DECODE is memory-bound and the
    primary driver of sustained GPU utilization and KV-cache residency.
    """

    PREFILL = "prefill"
    DECODE = "decode"
    BOTH = "both"


@dataclass(frozen=True)
class Request:
    """A single inference request.

    Attributes:
        request_id: Monotonic unique id (string for safe serialization).
        session_id: Conversation/session id. Requests sharing a
            session_id are assumed to share prefix structure and may be
            sticky under session-affinity policies.
        arrival_ts: Seconds since simulation start (float).
        prompt_tokens: Tuple of token ids (or stable hashes). The *prefix*
            of this tuple is what the KV cache indexes. Kept as a tuple
            so it is hashable and slice-cheap.
        max_output_tokens: Upper bound on decode length used for compute
            and memory accounting.
        prefix_key: Optional precomputed prefix hash (for datasets where
            we don't have real tokens). If None, policies may hash
            prompt_tokens themselves.
        priority: Lower is more urgent. Used by VTC and fairness policies.
        metadata: Free-form; policies may read, must not mutate.
    """

    request_id: str
    session_id: str
    arrival_ts: float
    prompt_tokens: tuple[int, ...]
    max_output_tokens: int
    prefix_key: str | None = None
    priority: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PodSpec:
    """Static description of a serving pod.

    A pod is one addressable replica. For PD-disaggregated clusters, a
    "prefill pod" and a "decode pod" are modeled as separate pods with
    `role` set accordingly and linked via `peer_ids`.
    """

    pod_id: str
    role: Phase  # PREFILL, DECODE, or BOTH
    gpu_count: int
    kv_cache_bytes: int
    max_concurrent_prefill: int
    max_concurrent_decode: int
    peer_ids: tuple[str, ...] = ()


@dataclass
class PodRuntime:
    """Mutable runtime state of a pod, owned by ClusterState.

    Policies read this; only the simulator mutates it.
    """

    spec: PodSpec
    active_prefill: int = 0
    active_decode: int = 0
    queued: int = 0
    ewma_latency_ms: float = 0.0
    ewma_throughput_tps: float = 0.0
    pending_work_ms: float = 0.0  # Preble L_i: Σ predicted service_ms for in-flight requests
    last_update_ts: float = 0.0
    # KV accounting is owned by KVCacheState, keyed by pod_id.


@dataclass(frozen=True)
class Decision:
    """Output of a RoutingPolicy.

    `prefill_pod_id` and `decode_pod_id` may be the same (colocated) or
    different (PD-disaggregated). For non-PD policies, set both equal.
    `rationale` is a short string for logging/debugging. `score` is an
    optional policy-internal score exposed for diagnostics.
    """

    prefill_pod_id: str
    decode_pod_id: str
    rationale: str
    score: float | None = None


@dataclass(frozen=True)
class CostBreakdown:
    """Per-request latency decomposition, in milliseconds.

    Fields hold raw component measurements for transparency — each is
    the uncombined cost of its phase. `total_ms` composes them into
    an end-to-end latency under the assumption that KV transport and
    prefill compute overlap: an async/RDMA pull initiated at dispatch
    runs in parallel with prefill work, so the bottleneck is
    `max(compute_prefill_ms, kv_transport_ms)` rather than their sum.

    All components are non-negative; `kv_transport_ms` is nonzero only
    when a request is routed to a pod that must pull KV state from a
    peer. When `kv_transport_ms == 0`, `total_ms` reduces to the plain
    sum of the remaining components.
    """

    routing_ms: float
    queueing_ms: float
    compute_prefill_ms: float
    compute_decode_ms: float
    network_ms: float
    kv_transport_ms: float

    @property
    def prefill_block_ms(self) -> float:
        """Effective prefill-phase cost after KV-transport overlap.

        Returns `max(compute_prefill_ms, kv_transport_ms)`. Prefill on
        uncached tail tokens and the KV pull of the cached prefix are
        assumed to proceed in parallel; the phase completes when the
        slower of the two finishes.
        """
        return max(self.compute_prefill_ms, self.kv_transport_ms)

    @property
    def total_ms(self) -> float:
        return (
            self.routing_ms
            + self.queueing_ms
            + self.prefill_block_ms
            + self.compute_decode_ms
            + self.network_ms
        )
