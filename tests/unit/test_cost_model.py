"""AnalyticCostModel: decomposition sums, KV transport charging."""

from __future__ import annotations

from routing_harness.core import Decision, Phase, PodSpec, Request
from routing_harness.cluster import ClusterState
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
