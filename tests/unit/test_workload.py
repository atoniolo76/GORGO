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


def test_synthetic_describe_includes_generator_params():
    """describe() must surface generator inputs so the runner's
    content-addressed run_id discriminates on workload-axis sweeps
    (zipf_s, n_prefix_families, prompt_len_*, shared_prefix_tokens).
    Without this, sweeps over those axes collide on run_id and overwrite
    each other's results/<id>/ dir.
    """
    from dataclasses import replace

    base = _params(1)
    a = generate(base).describe()
    b = generate(replace(base, zipf_s=base.zipf_s + 0.5)).describe()
    c = generate(replace(base, n_prefix_families=base.n_prefix_families + 1)).describe()
    d = generate(replace(base, shared_prefix_tokens=8)).describe()
    # All four describe() outputs must differ — that is what feeds run_id.
    assert len({repr(x) for x in (a, b, c, d)}) == 4
    assert a["params"]["zipf_s"] == base.zipf_s


def test_synthetic_shared_head_produces_matching_token_prefixes():
    """Shared-head mode: requests from the same family share a real
    token-level prefix, and prefix_key is cleared so the engine will
    enumerate block-level hashes over prompt_tokens instead of
    collapsing the prefix to a single opaque block.
    """
    params = SyntheticParams(
        n_requests=200,
        arrival_rate_qps=10.0,
        n_prefix_families=4,
        zipf_s=1.1,
        prompt_len_min=64,
        prompt_len_max=128,
        max_output_tokens=8,
        n_sessions=5,
        seed=0,
        shared_prefix_tokens=32,
    )
    trace = generate(params)
    assert len(trace.requests) == 200
    # All prefix_keys are None in shared-head mode.
    assert all(r.prefix_key is None for r in trace.requests)
    # Group by family, verify same-family requests share the first 32 tokens.
    by_fam: dict[int, list[tuple[int, ...]]] = {}
    for r in trace.requests:
        by_fam.setdefault(r.metadata["family"], []).append(r.prompt_tokens)
    for fam, toks in by_fam.items():
        heads = {t[:32] for t in toks}
        assert len(heads) == 1, f"family {fam} has divergent heads"
    # Different families produce different heads.
    distinct_heads = {toks[0][:32] for toks in by_fam.values()}
    assert len(distinct_heads) == len(by_fam)


def test_synthetic_shared_head_is_deterministic():
    params = SyntheticParams(
        n_requests=50, arrival_rate_qps=10.0, n_prefix_families=4,
        zipf_s=1.1, prompt_len_min=64, prompt_len_max=128,
        max_output_tokens=8, n_sessions=5, seed=3,
        shared_prefix_tokens=32,
    )
    a = generate(params)
    b = generate(params)
    assert [r.prompt_tokens for r in a.requests] == [r.prompt_tokens for r in b.requests]


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


def test_lmsys_tokenizer_default_is_mock():
    # Default must stay mock so the base install works with no extras.
    params = TraceParams(arrival_rate_qps=5.0)
    assert params.tokenizer == "mock"


def test_lmsys_unknown_tokenizer_raises():
    import pytest

    stub = StubLoader(n_convs=1, turns_per_conv=1, seed=0)
    cfg = LmsysConfig(local_path="/unused", seed=0)
    params = TraceParams(arrival_rate_qps=5.0, seed=0, tokenizer="bogus")
    with pytest.raises(ValueError, match="Unknown tokenizer"):
        build_trace(cfg, params, loader=stub.iter_turns)


def test_lmsys_tiktoken_missing_raises_helpfully():
    import importlib.util
    import pytest

    if importlib.util.find_spec("tiktoken") is not None:
        pytest.skip("tiktoken is installed; this test covers the missing-dep path")

    stub = StubLoader(n_convs=1, turns_per_conv=1, seed=0)
    cfg = LmsysConfig(local_path="/unused", seed=0)
    params = TraceParams(
        arrival_rate_qps=5.0, seed=0, tokenizer="tiktoken:cl100k_base"
    )
    with pytest.raises(RuntimeError, match="tokenizers"):
        build_trace(cfg, params, loader=stub.iter_turns)


def test_lmsys_tiktoken_cl100k_produces_real_tokens():
    import pytest

    tiktoken = pytest.importorskip("tiktoken")

    # Clear the LRU cache so this test doesn't depend on prior-test state.
    from routing_harness.workload import lmsys as _lm
    _lm._load_tiktoken_encoding.cache_clear()

    stub = StubLoader(n_convs=2, turns_per_conv=2, seed=0)
    cfg = LmsysConfig(local_path="/unused", seed=0)
    params = TraceParams(
        arrival_rate_qps=5.0, seed=0, tokenizer="tiktoken:cl100k_base"
    )
    trace = build_trace(cfg, params, loader=stub.iter_turns)
    assert len(trace.requests) > 0

    # Shared "You are a helpful assistant." prefix => shared real-token prefix.
    r0, r1 = trace.requests[0], trace.requests[1]
    assert r0.prompt_tokens[:4] == r1.prompt_tokens[:4]

    # Real cl100k tokens for English land near ~0.25 tokens/char (word-ish);
    # this is WAY above the mock's 0.25-char heuristic on the same text
    # for short strings. Sanity: real tokenization of "You are a helpful
    # assistant." is 7 tokens, not 32 * 0.25 = 8, but crucially the
    # tokens are real cl100k ids, not blake2b hashes.
    enc = tiktoken.get_encoding("cl100k_base")
    expected = tuple(enc.encode("You are a helpful assistant. " * 8 + " turn 0 payload "))
    # r0.prompt_tokens starts with the system prompt's real encoding.
    assert r0.prompt_tokens[: len(expected) - 4] == expected[: len(expected) - 4]
