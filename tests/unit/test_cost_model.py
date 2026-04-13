"""AnalyticCostModel: decomposition sums, KV transport charging."""

from __future__ import annotations

import math

from routing_harness.cluster import ClusterState
from routing_harness.core import Decision, Phase, PodSpec, Request
from routing_harness.cost_model import (
    AnalyticCostModel,
    ComputeParams,
    NetworkParams,
    SchedulerParams,
)
from routing_harness.kv_cache import KVCacheState


def _req(prompt_len=128, max_out=32) -> Request:
    return Request("r0", "sA", 0.0, tuple(range(prompt_len)), max_out)


def test_decomposition_sums(cost_model, cluster, kv_cache):
    r = _req()
    d = Decision("p0", "p0", "test")
    c = cost_model.estimate(r, d, cluster, kv_cache, cached_prefix_tokens=0, kv_transport_bytes=0)
    assert c.total_ms > 0
    assert c.kv_transport_ms == 0.0
    assert abs(
        c.total_ms
        - (c.routing_ms + c.queueing_ms + c.compute_prefill_ms + c.compute_decode_ms + c.network_ms)
    ) < 1e-9


def test_cached_prefix_reduces_prefill_cost(cost_model, cluster, kv_cache):
    r = _req(prompt_len=256)
    d = Decision("p0", "p0", "test")
    cold = cost_model.estimate(r, d, cluster, kv_cache, cached_prefix_tokens=0, kv_transport_bytes=0)
    warm = cost_model.estimate(r, d, cluster, kv_cache, cached_prefix_tokens=200, kv_transport_bytes=0)
    assert warm.compute_prefill_ms < cold.compute_prefill_ms


def test_kv_transport_charged_only_when_nonzero(cost_model, cluster, kv_cache):
    r = _req()
    d = Decision("p0", "p0", "test")
    no_transport = cost_model.estimate(r, d, cluster, kv_cache, 0, 0)
    with_transport = cost_model.estimate(r, d, cluster, kv_cache, 0, 1_000_000)
    assert no_transport.kv_transport_ms == 0.0
    assert with_transport.kv_transport_ms > 0.0


def _batched_cost_model(decode_batch_k: float) -> AnalyticCostModel:
    return AnalyticCostModel(
        compute=ComputeParams(
            prefill_ms_per_token=0.1,
            decode_ms_per_token=5.0,
            prefill_overhead_ms=4.0,
            decode_overhead_ms=1.0,
            decode_batch_k=decode_batch_k,
        ),
        network=NetworkParams(
            client_rtt_ms=5.0,
            inter_pod_rtt_ms=0.2,
            inter_pod_bandwidth_gbps=100.0,
            kv_bytes_per_token=1024,
            serialization_overhead_ms=0.5,
        ),
        scheduler=SchedulerParams(
            base_routing_ms=0.2, per_pod_consideration_us=5.0
        ),
    )


def test_decode_batch_k_default_is_constant(cost_model, cluster, kv_cache):
    """Default decode_batch_k=0 preserves the legacy constant-decode
    behavior independent of the decode pod's active_decode count."""
    r = _req(prompt_len=64, max_out=64)
    d = Decision("p0", "p0", "test")
    base = cost_model.estimate(r, d, cluster, kv_cache, 0, 0)
    cluster.pods["p0"].active_decode = 16
    loaded = cost_model.estimate(r, d, cluster, kv_cache, 0, 0)
    assert base.compute_decode_ms == loaded.compute_decode_ms


def test_decode_batch_k_batch_one_reproduces_baseline(cluster, kv_cache):
    """With k>0, a single-concurrent decode (batch=1) charges exactly
    the legacy per-token baseline — the formula is pinned so existing
    calibration holds at low concurrency."""
    baseline = _batched_cost_model(decode_batch_k=0.0)
    batched = _batched_cost_model(decode_batch_k=1.0)
    r = _req(prompt_len=64, max_out=64)
    d = Decision("p0", "p0", "test")
    assert cluster.pods["p0"].active_decode == 0
    cb = baseline.estimate(r, d, cluster, kv_cache, 0, 0)
    cc = batched.estimate(r, d, cluster, kv_cache, 0, 0)
    assert cb.compute_decode_ms == cc.compute_decode_ms


def test_decode_batch_k_amortizes_at_high_concurrency(cluster, kv_cache):
    """k>0: per-request decode cost drops monotonically as the decode
    pod's concurrent batch grows, but sublinearly (logarithmic)."""
    cm = _batched_cost_model(decode_batch_k=1.0)
    r = _req(prompt_len=64, max_out=128)
    d = Decision("p0", "p0", "test")

    costs: list[float] = []
    for active in (0, 1, 3, 7, 15, 31):
        cluster.pods["p0"].active_decode = active
        c = cm.estimate(r, d, cluster, kv_cache, 0, 0)
        costs.append(c.compute_decode_ms)

    # Strictly decreasing in batch size.
    for prev, nxt in zip(costs, costs[1:]):
        assert nxt < prev

    # Sublinear: doubling batch does NOT halve the decode cost (unlike
    # a linear-throughput model would). Compare batch=1 vs batch=32.
    overhead = 1.0  # decode_overhead_ms from _batched_cost_model
    net_b1 = costs[0] - overhead
    net_b32 = costs[-1] - overhead
    assert net_b32 > net_b1 / 32.0 * 4  # well above a naive 1/N curve

    # Exact formula check at batch=8: base / (1 + log 8).
    cluster.pods["p0"].active_decode = 7
    c8 = cm.estimate(r, d, cluster, kv_cache, 0, 0)
    expected = overhead + 128 * (5.0 / (1.0 + math.log(8)))
    assert abs(c8.compute_decode_ms - expected) < 1e-9


def test_decode_batch_k_uses_decode_pod_not_prefill_pod(cluster, kv_cache):
    """Under PD-disaggregation the decode pod, not the prefill pod,
    determines the batch. This is the bias the feature exists to fix."""
    cm = _batched_cost_model(decode_batch_k=1.0)
    r = _req(prompt_len=64, max_out=128)
    d = Decision(prefill_pod_id="p0", decode_pod_id="p1", rationale="pd")

    cluster.pods["p0"].active_decode = 0
    cluster.pods["p1"].active_decode = 15
    c_busy_decode = cm.estimate(r, d, cluster, kv_cache, 0, 0)

    cluster.pods["p1"].active_decode = 0
    cluster.pods["p0"].active_decode = 15
    c_busy_prefill = cm.estimate(r, d, cluster, kv_cache, 0, 0)

    assert c_busy_decode.compute_decode_ms < c_busy_prefill.compute_decode_ms


def test_fabric_contention_slows_transfer(cost_model, cluster, kv_cache):
    r = _req()
    d = Decision("p0", "p0", "test")
    solo = cost_model.estimate(
        r, d, cluster, kv_cache, 0, 1_000_000,
        concurrent_kv_transport_bytes=1_000_000,
    )
    contended = cost_model.estimate(
        r, d, cluster, kv_cache, 0, 1_000_000,
        concurrent_kv_transport_bytes=4_000_000,
    )
    assert contended.kv_transport_ms > solo.kv_transport_ms
    # Fluid fair-share: slowdown scales with the ratio of concurrent
    # bytes to this transfer's own bytes (post-rtt).
    bytes_per_ms = cost_model.network.inter_pod_bandwidth_gbps * 1e9 / 8.0 / 1000.0
    expected = cost_model.network.inter_pod_rtt_ms + 4_000_000 / bytes_per_ms
    assert abs(contended.kv_transport_ms - expected) < 1e-6


def test_fabric_contention_default_is_uncontended(cost_model, cluster, kv_cache):
    r = _req()
    d = Decision("p0", "p0", "test")
    default = cost_model.estimate(r, d, cluster, kv_cache, 0, 500_000)
    explicit_solo = cost_model.estimate(
        r, d, cluster, kv_cache, 0, 500_000,
        concurrent_kv_transport_bytes=500_000,
    )
    assert abs(default.kv_transport_ms - explicit_solo.kv_transport_ms) < 1e-9


def test_fabric_contention_floor_on_self_bytes(cost_model, cluster, kv_cache):
    # Passing a "concurrent" value smaller than this transfer must not
    # cheat the transfer into going faster than its own-bytes time.
    r = _req()
    d = Decision("p0", "p0", "test")
    c = cost_model.estimate(
        r, d, cluster, kv_cache, 0, 1_000_000,
        concurrent_kv_transport_bytes=100_000,
    )
    bytes_per_ms = cost_model.network.inter_pod_bandwidth_gbps * 1e9 / 8.0 / 1000.0
    expected_own = cost_model.network.inter_pod_rtt_ms + 1_000_000 / bytes_per_ms
    assert abs(c.kv_transport_ms - expected_own) < 1e-6


def test_kv_transport_overlaps_prefill_when_shorter(cost_model, cluster, kv_cache):
    """go-npl: a small cross-pod pull that completes before prefill does
    must not add to end-to-end latency. The transport runs in parallel
    with prefill compute, so the phase bottleneck is the longer of the
    two. Here transport is tiny (~0.28 ms) versus prefill on 128 tokens
    (~16.8 ms) — total_ms must equal the no-transport total."""
    r = _req(prompt_len=128)
    d = Decision("p0", "p0", "test")
    no_pull = cost_model.estimate(r, d, cluster, kv_cache, 0, 0)
    small_pull = cost_model.estimate(r, d, cluster, kv_cache, 0, 1_000_000)
    # Transport is strictly shorter than prefill in this regime.
    assert small_pull.kv_transport_ms > 0.0
    assert small_pull.kv_transport_ms < small_pull.compute_prefill_ms
    # Overlap absorbs the transport entirely.
    assert abs(small_pull.total_ms - no_pull.total_ms) < 1e-9


def test_kv_transport_dominates_when_longer(cost_model, cluster, kv_cache):
    """When the pull is slower than prefill (large bytes, low bandwidth-
    relative cost), total_ms is driven by transport, not prefill.
    Prefill on a short prompt is absorbed by the pull latency."""
    r = _req(prompt_len=32)  # prefill ~ 4 + 32*0.1 = 7.2 ms
    d = Decision("p0", "p0", "test")
    # 100 MB pull at 100 Gbps => 0.2 + 100_000_000/12_500_000 = 8.2 ms > 7.2
    huge_pull = cost_model.estimate(r, d, cluster, kv_cache, 0, 100_000_000)
    assert huge_pull.kv_transport_ms > huge_pull.compute_prefill_ms
    # Prefill-block cost is pinned to transport, not the sum.
    assert huge_pull.prefill_block_ms == huge_pull.kv_transport_ms
    # total_ms excludes the overlapped prefill portion.
    expected = (
        huge_pull.routing_ms
        + huge_pull.queueing_ms
        + huge_pull.kv_transport_ms
        + huge_pull.compute_decode_ms
        + huge_pull.network_ms
    )
    assert abs(huge_pull.total_ms - expected) < 1e-9


def test_total_ms_monotone_in_transport_above_prefill(cost_model, cluster, kv_cache):
    """Once transport exceeds prefill, further growth in transport bytes
    translates into proportional latency growth (no more free overlap)."""
    r = _req(prompt_len=32)
    d = Decision("p0", "p0", "test")
    a = cost_model.estimate(r, d, cluster, kv_cache, 0, 100_000_000)
    b = cost_model.estimate(r, d, cluster, kv_cache, 0, 200_000_000)
    # Both are transport-dominated in this regime.
    assert a.kv_transport_ms > a.compute_prefill_ms
    assert b.kv_transport_ms > b.compute_prefill_ms
    # Latency grows 1-for-1 with added transport bytes.
    bytes_per_ms = cost_model.network.inter_pod_bandwidth_gbps * 1e9 / 8.0 / 1000.0
    extra_transport = 100_000_000 / bytes_per_ms
    assert abs((b.total_ms - a.total_ms) - extra_transport) < 1e-6


def test_prefill_block_ms_zero_transport_is_prefill(cost_model, cluster, kv_cache):
    """`prefill_block_ms` reduces to compute_prefill_ms when no pull
    occurs — the overlap model is a no-op for colocated requests."""
    r = _req(prompt_len=64)
    d = Decision("p0", "p0", "test")
    c = cost_model.estimate(r, d, cluster, kv_cache, 0, 0)
    assert c.kv_transport_ms == 0.0
    assert c.prefill_block_ms == c.compute_prefill_ms
