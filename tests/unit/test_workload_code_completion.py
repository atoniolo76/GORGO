"""Code-completion adapter: fixture-JSONL replay + stub/tokenizer coverage."""

from __future__ import annotations

from routing_harness.workload.code_completion import (
    CodeCompletionConfig,
    StubLoader,
    TraceParams,
    build_trace,
    load_tasks,
)


def test_code_completion_stub_loader_builds_trace():
    stub = StubLoader(n_tasks=4, seed=0)
    cfg = CodeCompletionConfig(
        local_path="/unused",
        instruction_prefix="# Complete the following Python function.\n" * 4,
        seed=0,
    )
    params = TraceParams(arrival_rate_qps=5.0, seed=1)
    trace = build_trace(cfg, params, loader=stub.iter_tasks)

    assert len(trace.requests) == 4
    # Every task is prepended with the same instruction_prefix, so tokens
    # at position 0 should match across all requests.
    token_heads = [r.prompt_tokens[:8] for r in trace.requests]
    assert all(h == token_heads[0] for h in token_heads)
    # Each task is its own session.
    assert len({r.session_id for r in trace.requests}) == 4


def test_code_completion_stub_is_deterministic():
    cfg = CodeCompletionConfig(local_path="/unused", seed=0)
    params = TraceParams(arrival_rate_qps=5.0, seed=9)
    a = build_trace(cfg, params, loader=StubLoader(n_tasks=3, seed=2).iter_tasks)
    b = build_trace(cfg, params, loader=StubLoader(n_tasks=3, seed=2).iter_tasks)
    assert [r.request_id for r in a.requests] == [r.request_id for r in b.requests]
    assert [r.prompt_tokens for r in a.requests] == [r.prompt_tokens for r in b.requests]


def test_code_completion_missing_file_raises():
    import pytest

    cfg = CodeCompletionConfig(local_path="/no/such/code.jsonl", seed=0)
    params = TraceParams(arrival_rate_qps=5.0, seed=0)
    with pytest.raises(FileNotFoundError):
        build_trace(cfg, params)  # uses real loader -> raises


def test_code_completion_fixture_jsonl_replay(fixtures_root):
    cfg = CodeCompletionConfig(
        local_path=str(fixtures_root / "code_completion_tiny.jsonl"), seed=0
    )
    params = TraceParams(arrival_rate_qps=5.0, seed=4)
    a = build_trace(cfg, params)
    b = build_trace(cfg, params)
    assert len(a.requests) == 4
    assert [r.request_id for r in a.requests] == [r.request_id for r in b.requests]
    assert [r.prompt_tokens for r in a.requests] == [r.prompt_tokens for r in b.requests]
    # Session ids match the task_id field; one fixture row uses MBPP-style
    # "text" instead of "prompt" — it must still be ingested.
    sids = [r.session_id for r in a.requests]
    assert "MBPP/11" in sids


def test_code_completion_fixture_accepts_mbpp_text_fallback(fixtures_root):
    cfg = CodeCompletionConfig(
        local_path=str(fixtures_root / "code_completion_tiny.jsonl"), seed=0
    )
    tasks = list(load_tasks(cfg))
    mbpp = next(t for t in tasks if t.task_id == "MBPP/11")
    assert mbpp.prompt.startswith("Write a Python function")


def test_code_completion_language_filter(fixtures_root):
    cfg = CodeCompletionConfig(
        local_path=str(fixtures_root / "code_completion_tiny.jsonl"),
        language_filter=("python",),
        seed=0,
    )
    params = TraceParams(arrival_rate_qps=5.0, seed=0)
    trace = build_trace(cfg, params)
    # The JS row is dropped.
    assert len(trace.requests) == 3
    assert all(r.metadata["language"] == "python" for r in trace.requests)


def test_code_completion_max_tasks_caps_output(fixtures_root):
    cfg = CodeCompletionConfig(
        local_path=str(fixtures_root / "code_completion_tiny.jsonl"),
        max_tasks=2,
        seed=0,
    )
    params = TraceParams(arrival_rate_qps=5.0, seed=0)
    trace = build_trace(cfg, params)
    assert len(trace.requests) == 2


def test_code_completion_instruction_prefix_drives_reuse(fixtures_root):
    cfg_no_prefix = CodeCompletionConfig(
        local_path=str(fixtures_root / "code_completion_tiny.jsonl"),
        instruction_prefix="",
        seed=0,
    )
    cfg_with_prefix = CodeCompletionConfig(
        local_path=str(fixtures_root / "code_completion_tiny.jsonl"),
        instruction_prefix="# Standard instruction. " * 8,
        seed=0,
    )
    params = TraceParams(arrival_rate_qps=5.0, seed=0)

    # Without an instruction prefix the 4 tasks are unrelated; the
    # prompts don't share leading content, so their token heads differ.
    no_pref = build_trace(cfg_no_prefix, params)
    heads_no = {r.prompt_tokens[:4] for r in no_pref.requests}
    assert len(heads_no) > 1

    # With a non-trivial shared prefix, every request now starts with
    # the same token sequence.
    with_pref = build_trace(cfg_with_prefix, params)
    heads_with = {r.prompt_tokens[:4] for r in with_pref.requests}
    assert len(heads_with) == 1


def test_code_completion_unknown_tokenizer_raises():
    import pytest

    stub = StubLoader(n_tasks=1, seed=0)
    cfg = CodeCompletionConfig(local_path="/unused", seed=0)
    params = TraceParams(arrival_rate_qps=5.0, seed=0, tokenizer="bogus")
    with pytest.raises(ValueError, match="Unknown tokenizer"):
        build_trace(cfg, params, loader=stub.iter_tasks)


def test_code_completion_tiktoken_override_produces_real_tokens():
    import pytest

    tiktoken = pytest.importorskip("tiktoken")

    from routing_harness.workload import lmsys as _lm

    _lm._load_tiktoken_encoding.cache_clear()

    stub = StubLoader(n_tasks=3, seed=0)
    cfg = CodeCompletionConfig(
        local_path="/unused",
        instruction_prefix="# Code assistant. " * 4,
        seed=0,
    )
    params = TraceParams(
        arrival_rate_qps=5.0, seed=0, tokenizer="tiktoken:cl100k_base"
    )
    trace = build_trace(cfg, params, loader=stub.iter_tasks)
    assert len(trace.requests) == 3

    # All three requests share the `instruction_prefix`; under a real
    # BPE tokenizer their leading tokens must agree.
    r0, r1, r2 = trace.requests
    assert r0.prompt_tokens[:6] == r1.prompt_tokens[:6] == r2.prompt_tokens[:6]
