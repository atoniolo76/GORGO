"""Workload adapters: synthetic determinism + lmsys stub loader."""

from __future__ import annotations

from routing_harness.workload.lmsys import LmsysConfig, StubLoader, TraceParams, build_trace
from routing_harness.workload.synthetic import SyntheticParams, generate


def _params(seed: int = 0) -> SyntheticParams:
    return SyntheticParams(
        n_requests=30,
        arrival_rate_qps=10.0,
        n_prefix_families=4,
        zipf_s=1.1,
        prompt_len_min=8,
        prompt_len_max=16,
        max_output_tokens=8,
        n_sessions=5,
        seed=seed,
    )


def test_synthetic_is_deterministic():
    a = generate(_params(7))
    b = generate(_params(7))
    assert [r.request_id for r in a.requests] == [r.request_id for r in b.requests]
    assert [r.prompt_tokens for r in a.requests] == [r.prompt_tokens for r in b.requests]
    assert [r.prefix_key for r in a.requests] == [r.prefix_key for r in b.requests]


def test_synthetic_arrivals_monotonic():
    t = generate(_params(1))
    ts = [r.arrival_ts for r in t.requests]
    assert ts == sorted(ts)


def test_synthetic_describe():
    t = generate(_params(1))
    d = t.describe()
    assert d["n"] == 30
    assert d["t_start"] >= 0.0


def test_lmsys_stub_loader_builds_trace():
    stub = StubLoader(n_convs=3, turns_per_conv=2, seed=0)
    cfg = LmsysConfig(local_path="/nonexistent/but/unused", seed=0)
    params = TraceParams(arrival_rate_qps=5.0, seed=1)
    trace = build_trace(cfg, params, loader=stub.iter_turns)
    assert len(trace.requests) > 0
    # Common system prompt => consecutive requests share token prefixes.
    r0, r1 = trace.requests[0], trace.requests[1]
    assert r0.prompt_tokens[:16] == r1.prompt_tokens[:16]


def test_lmsys_missing_file_raises():
    import pytest

    cfg = LmsysConfig(local_path="/no/such/path.jsonl", seed=0)
    params = TraceParams(arrival_rate_qps=5.0, seed=0)
    with pytest.raises(FileNotFoundError):
        build_trace(cfg, params)  # uses real loader -> raises
