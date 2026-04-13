"""InstrumentedCostModel: measured-value override, analytic fallthrough."""

from __future__ import annotations

import textwrap

from routing_harness.core import Decision, Request
from routing_harness.cost_model import (
    AnalyticCostModel,
    InstrumentedCostModel,
    decode_batch_bucket,
    load_observations_csv,
)


def _req(prompt_len=128, max_out=32) -> Request:
    return Request("r0", "sA", 0.0, tuple(range(prompt_len)), max_out)


def test_empty_observations_match_analytic(cost_model, cluster, kv_cache):
    """No recorded observations → every component equals analytic."""
    inst = InstrumentedCostModel(cost_model)
    r = _req()
    d = Decision("p0", "p0", "test")
    a = cost_model.estimate(r, d, cluster, kv_cache, 0, 0)
    i = inst.estimate(r, d, cluster, kv_cache, 0, 0)
    assert i == a


def test_prefill_override_replaces_per_token_rate(cost_model, cluster, kv_cache):
    """Measured prefill_ms_per_token overrides the analytic rate but
    retains analytic overhead and uncached-token count."""
    inst = InstrumentedCostModel(cost_model)
    # Analytic: 0.1 ms/token. Observed: 0.25 ms/token.
    inst.record("prefill_ms_per_token:p0", 0.25)
    r = _req(prompt_len=128)
    d = Decision("p0", "p0", "test")
    c = inst.estimate(r, d, cluster, kv_cache, cached_prefix_tokens=0, kv_transport_bytes=0)
    expected = cost_model.compute.prefill_overhead_ms + 128 * 0.25
    assert abs(c.compute_prefill_ms - expected) < 1e-9


def test_prefill_override_is_pod_scoped(cost_model, cluster, kv_cache):
    """Recording against p0 does not affect a request routed to p1."""
    inst = InstrumentedCostModel(cost_model)
    inst.record("prefill_ms_per_token:p0", 99.0)
    r = _req(prompt_len=128)
    d = Decision("p1", "p1", "test")
    analytic = cost_model.estimate(r, d, cluster, kv_cache, 0, 0)
    instrumented = inst.estimate(r, d, cluster, kv_cache, 0, 0)
    assert analytic.compute_prefill_ms == instrumented.compute_prefill_ms


def test_decode_override_uses_batch_bucket(cost_model, cluster, kv_cache):
    """Decode override keyed by (pod, batch_bucket). An observation for
    bucket=1 applies when the decode pod has no other active requests."""
    inst = InstrumentedCostModel(cost_model)
    inst.record("decode_ms_per_token:p0:1", 2.0)
    r = _req(prompt_len=64, max_out=64)
    d = Decision("p0", "p0", "test")
    c = inst.estimate(r, d, cluster, kv_cache, 0, 0)
    expected = cost_model.compute.decode_overhead_ms + 64 * 2.0
    assert abs(c.compute_decode_ms - expected) < 1e-9


def test_decode_override_bucket_match_is_exact(cost_model, cluster, kv_cache):
    """A recording for bucket=1 must NOT apply when decode_batch falls
    in bucket=4 — otherwise one sample would poison every batch size."""
    inst = InstrumentedCostModel(cost_model)
    inst.record("decode_ms_per_token:p0:1", 99.0)
    cluster.pods["p0"].active_decode = 5  # batch=6 → bucket=4
    r = _req(prompt_len=64, max_out=64)
    d = Decision("p0", "p0", "test")
    c = inst.estimate(r, d, cluster, kv_cache, 0, 0)
    a = cost_model.estimate(r, d, cluster, kv_cache, 0, 0)
    assert c.compute_decode_ms == a.compute_decode_ms


def test_decode_override_matches_its_own_bucket(cost_model, cluster, kv_cache):
    """When active_decode puts the request in bucket=4, a matching
    bucket-keyed observation wins."""
    inst = InstrumentedCostModel(cost_model)
    inst.record("decode_ms_per_token:p0:4", 3.0)
    cluster.pods["p0"].active_decode = 5  # batch=6, bucket=4
    r = _req(prompt_len=64, max_out=64)
    d = Decision("p0", "p0", "test")
    c = inst.estimate(r, d, cluster, kv_cache, 0, 0)
    expected = cost_model.compute.decode_overhead_ms + 64 * 3.0
    assert abs(c.compute_decode_ms - expected) < 1e-9


def test_queueing_override_is_flat(cost_model, cluster, kv_cache):
    """A queueing_ms observation is treated as the measured mean wait;
    it replaces the M/M/1 estimate regardless of pod occupancy."""
    inst = InstrumentedCostModel(cost_model)
    inst.record("queueing_ms:p0", 12.5)
    cluster.pods["p0"].active_prefill = 2  # would produce large M/M/1 wait
    r = _req()
    d = Decision("p0", "p0", "test")
    c = inst.estimate(r, d, cluster, kv_cache, 0, 0)
    assert c.queueing_ms == 12.5


def test_fallthrough_preserves_unobserved_components(cost_model, cluster, kv_cache):
    """Setting one override must not disturb the others — they still
    come straight from the analytic model."""
    inst = InstrumentedCostModel(cost_model)
    inst.record("prefill_ms_per_token:p0", 0.25)
    r = _req()
    d = Decision("p0", "p0", "test")
    a = cost_model.estimate(r, d, cluster, kv_cache, 0, 1_000_000)
    c = inst.estimate(r, d, cluster, kv_cache, 0, 1_000_000)
    assert c.routing_ms == a.routing_ms
    assert c.queueing_ms == a.queueing_ms
    assert c.compute_decode_ms == a.compute_decode_ms
    assert c.network_ms == a.network_ms
    assert c.kv_transport_ms == a.kv_transport_ms
    # And the one we DID override is different.
    assert c.compute_prefill_ms != a.compute_prefill_ms


def test_from_observations_constructor(cost_model, cluster, kv_cache):
    inst = InstrumentedCostModel.from_observations(
        cost_model, {"prefill_ms_per_token:p0": 0.5}
    )
    r = _req(prompt_len=100)
    d = Decision("p0", "p0", "test")
    c = inst.estimate(r, d, cluster, kv_cache, 0, 0)
    expected = cost_model.compute.prefill_overhead_ms + 100 * 0.5
    assert abs(c.compute_prefill_ms - expected) < 1e-9


def test_decode_batch_bucket_boundaries():
    assert decode_batch_bucket(1) == 1
    assert decode_batch_bucket(2) == 2
    assert decode_batch_bucket(3) == 2
    assert decode_batch_bucket(4) == 4
    assert decode_batch_bucket(7) == 4
    assert decode_batch_bucket(8) == 8
    assert decode_batch_bucket(31) == 16
    assert decode_batch_bucket(32) == 32
    # Degenerate inputs clamp to the smallest bucket.
    assert decode_batch_bucket(0) == 1


def test_load_observations_csv_parses_pairs(tmp_path):
    path = tmp_path / "obs.csv"
    path.write_text(
        textwrap.dedent(
            """\
            key,value_ms
            prefill_ms_per_token:p0,0.12
            # a comment line
            decode_ms_per_token:p0:4,3.5

            queueing_ms:p1,8.0
            """
        )
    )
    obs = load_observations_csv(path)
    assert obs == {
        "prefill_ms_per_token:p0": 0.12,
        "decode_ms_per_token:p0:4": 3.5,
        "queueing_ms:p1": 8.0,
    }


def test_load_observations_csv_last_write_wins(tmp_path):
    path = tmp_path / "obs.csv"
    path.write_text("prefill_ms_per_token:p0,0.1\nprefill_ms_per_token:p0,0.2\n")
    obs = load_observations_csv(path)
    assert obs == {"prefill_ms_per_token:p0": 0.2}
