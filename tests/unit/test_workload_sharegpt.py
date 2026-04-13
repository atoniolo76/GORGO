"""ShareGPT adapter: fixture-JSONL replay + stub/tokenizer coverage."""

from __future__ import annotations

from routing_harness.workload.sharegpt import (
    ShareGPTConfig,
    StubLoader,
    TraceParams,
    build_trace,
    load_sessions,
)


def test_sharegpt_stub_loader_builds_trace():
    stub = StubLoader(n_convs=3, turns_per_conv=4, seed=0)
    cfg = ShareGPTConfig(local_path="/unused", seed=0)
    params = TraceParams(arrival_rate_qps=5.0, seed=1)
    trace = build_trace(cfg, params, loader=stub.iter_turns)

    # 3 convs * 4 turns = 12 turns, half are "user" => 6 requests.
    assert len(trace.requests) == 6
    # Shared system preamble => consecutive user requests share a token prefix.
    r0, r1 = trace.requests[0], trace.requests[1]
    assert r0.prompt_tokens[:16] == r1.prompt_tokens[:16]
    # Arrivals are monotonic (invariant for InMemoryTrace).
    ts = [r.arrival_ts for r in trace.requests]
    assert ts == sorted(ts)


def test_sharegpt_stub_loader_is_deterministic():
    cfg = ShareGPTConfig(local_path="/unused", seed=0)
    params = TraceParams(arrival_rate_qps=5.0, seed=42)
    a = build_trace(cfg, params, loader=StubLoader(n_convs=2, seed=7).iter_turns)
    b = build_trace(cfg, params, loader=StubLoader(n_convs=2, seed=7).iter_turns)
    assert [r.request_id for r in a.requests] == [r.request_id for r in b.requests]
    assert [r.prompt_tokens for r in a.requests] == [r.prompt_tokens for r in b.requests]


def test_sharegpt_missing_file_raises():
    import pytest

    cfg = ShareGPTConfig(local_path="/no/such/sharegpt.jsonl", seed=0)
    params = TraceParams(arrival_rate_qps=5.0, seed=0)
    with pytest.raises(FileNotFoundError):
        build_trace(cfg, params)  # uses real loader -> raises


def test_sharegpt_fixture_jsonl_replay(fixtures_root):
    """Deterministic replay: two build_trace calls against the same JSONL
    fixture produce identical request_ids, tokens, and session ids."""
    cfg = ShareGPTConfig(
        local_path=str(fixtures_root / "sharegpt_tiny.jsonl"), seed=0
    )
    params = TraceParams(arrival_rate_qps=5.0, seed=3)
    a = build_trace(cfg, params)
    b = build_trace(cfg, params)
    assert len(a.requests) > 0
    assert [r.request_id for r in a.requests] == [r.request_id for r in b.requests]
    assert [r.prompt_tokens for r in a.requests] == [r.prompt_tokens for r in b.requests]
    assert [r.session_id for r in a.requests] == [r.session_id for r in b.requests]
    # Session ids propagate from "id" field in the fixture.
    assert {r.session_id for r in a.requests} <= {
        "sg_fixture_0", "sg_fixture_1", "sg_fixture_2"
    }


def test_sharegpt_fixture_yields_one_request_per_human_turn(fixtures_root):
    cfg = ShareGPTConfig(
        local_path=str(fixtures_root / "sharegpt_tiny.jsonl"), seed=0
    )
    # Fixture has 2 human turns in conv 0, 1 in conv 1, 1 in conv 2 => 4 total.
    all_turns = list(load_sessions(cfg))
    user_turns = [t for t in all_turns if t.role == "user"]
    assert len(user_turns) == 4

    params = TraceParams(arrival_rate_qps=5.0, seed=0)
    trace = build_trace(cfg, params)
    assert len(trace.requests) == 4


def test_sharegpt_max_conversations_caps_convs(fixtures_root):
    cfg = ShareGPTConfig(
        local_path=str(fixtures_root / "sharegpt_tiny.jsonl"),
        max_conversations=1,
        seed=0,
    )
    params = TraceParams(arrival_rate_qps=5.0, seed=0)
    trace = build_trace(cfg, params)
    # Only the first conversation (2 human turns) is emitted.
    assert len({r.session_id for r in trace.requests}) == 1
    assert len(trace.requests) == 2


def test_sharegpt_unknown_tokenizer_raises():
    import pytest

    stub = StubLoader(n_convs=1, turns_per_conv=2, seed=0)
    cfg = ShareGPTConfig(local_path="/unused", seed=0)
    params = TraceParams(arrival_rate_qps=5.0, seed=0, tokenizer="bogus")
    with pytest.raises(ValueError, match="Unknown tokenizer"):
        build_trace(cfg, params, loader=stub.iter_turns)


def test_sharegpt_tiktoken_override_produces_real_tokens():
    import pytest

    tiktoken = pytest.importorskip("tiktoken")

    from routing_harness.workload import lmsys as _lm

    _lm._load_tiktoken_encoding.cache_clear()

    stub = StubLoader(n_convs=2, turns_per_conv=2, seed=0)
    cfg = ShareGPTConfig(local_path="/unused", seed=0)
    params = TraceParams(
        arrival_rate_qps=5.0, seed=0, tokenizer="tiktoken:cl100k_base"
    )
    trace = build_trace(cfg, params, loader=stub.iter_turns)
    assert len(trace.requests) > 0

    # Shared "You are ShareGPT..." preamble => shared real-token prefix.
    r0, r1 = trace.requests[0], trace.requests[1]
    assert r0.prompt_tokens[:4] == r1.prompt_tokens[:4]
