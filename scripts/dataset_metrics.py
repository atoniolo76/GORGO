"""Dataset-metrics extraction CLI, shared across workload datasets.

Usage:
    python scripts/dataset_metrics.py --dataset synthetic [--n-requests 10000]
    python scripts/dataset_metrics.py --dataset code_completion ...
    python scripts/dataset_metrics.py --dataset lmsys ...
    python scripts/dataset_metrics.py --dataset sharegpt ...

Writes:
    research/data/dataset_metrics/<dataset>.json   # raw numbers
    research/reports/dataset_metrics_<dataset>.md  # narrative + plots
    research/reports/figures/<dataset>_*.png       # plots linked from md

Only `synthetic` is implemented at the moment (see go-igz). Other
datasets dispatch through ``DATASET_LOADERS`` and share the rest of the
pipeline — add a loader entry and everything downstream (metrics,
report, plots) works unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from routing_harness.core import Request  # noqa: E402
from routing_harness.kv_cache import enumerate_prefix_hashes  # noqa: E402
from routing_harness.workload.code_completion import (  # noqa: E402
    CodeCompletionConfig,
)
from routing_harness.workload.code_completion import (  # noqa: E402
    TraceParams as CCTraceParams,
)
from routing_harness.workload.code_completion import (  # noqa: E402
    build_trace as build_code_completion_trace,
)
from routing_harness.workload.synthetic import SyntheticParams  # noqa: E402
from routing_harness.workload.synthetic import generate as generate_synthetic  # noqa: E402
from routing_harness.workload.trace import InMemoryTrace  # noqa: E402

# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------


def load_synthetic(n_requests: int, seed: int) -> tuple[InMemoryTrace, dict]:
    """Generate a synthetic trace sized at ~n_requests.

    Parameters mirror ``configs/example_run.yaml`` so the metrics reflect
    the default configuration the harness is evaluated against. The
    shared-head mode is on so prefix hashes are block-level (the default
    opaque-key mode collapses every family to one block and makes
    prefix-concentration metrics meaningless).
    """
    params = SyntheticParams(
        n_requests=n_requests,
        arrival_rate_qps=8.0,
        n_prefix_families=64,
        zipf_s=1.1,
        prompt_len_min=256,
        prompt_len_max=2048,
        max_output_tokens=128,
        n_sessions=200,
        seed=seed,
        shared_prefix_tokens=512,
    )
    trace = generate_synthetic(params)
    return trace, {"kind": "synthetic", "params": asdict(params)}


# A realistic "system template" for a code-completion endpoint. Every
# deployed copilot-style service wraps each user task in a preamble like
# this — the specifics vary, but the structural role is identical:
# a constant, moderately long token prefix that dominates KV reuse
# across independent tasks. The prefix below is ~200 cl100k tokens,
# which is in the same ballpark as real production templates.
CODE_COMPLETION_INSTRUCTION_PREFIX = (
    "You are an AI programming assistant. Follow the user's "
    "requirements carefully and to the letter. First, think "
    "step-by-step: describe your plan in pseudocode, written out "
    "in detail. Then, output the final code in a single code "
    "block. Minimize any other prose. Use the language the user "
    "specifies; if unspecified, prefer Python 3.11. Do not invent "
    "APIs or libraries; if a standard library suffices, use it. "
    "Handle edge cases (empty input, None, off-by-one, overflow) "
    "unless the task explicitly excludes them. Match the exact "
    "function signature the user gives, including type hints and "
    "docstring format. Never add input() calls, print statements, "
    "or example invocations unless asked. If the task is "
    "ambiguous, state the ambiguity once and proceed with the "
    "most reasonable interpretation. Task follows.\n\n"
)


def load_code_completion(
    n_requests: int,
    seed: int,
    local_path: str | None = None,
) -> tuple[InMemoryTrace, dict]:
    """Load a real code-completion dataset (HumanEval + MBPP).

    Tasks are independent — each task is its own session. A constant
    ``instruction_prefix`` is prepended to every prompt before
    tokenization, matching how production code-assistants wrap each
    request in a static system template. This is the dominant source
    of cross-request KV prefix reuse for this workload, so we include
    it explicitly rather than profiling a denuded dataset.

    Tokenizer is ``cl100k_base`` (GPT-4/Claude-ish) rather than the mock
    tokenizer: code tokenizes at a noticeably different tokens/char ratio
    than prose, and the mock bias would misstate prompt-length
    distributions here.
    """
    abs_path = local_path or str(REPO_ROOT / "data" / "code_completion" / "combined.jsonl")
    cfg = CodeCompletionConfig(
        local_path=abs_path,
        max_tasks=n_requests,  # naturally caps to dataset size if smaller
        instruction_prefix=CODE_COMPLETION_INSTRUCTION_PREFIX,
        seed=seed,
    )
    params = CCTraceParams(
        arrival_rate_qps=4.0,
        max_output_tokens=256,
        tokenizer="tiktoken:cl100k_base",
        seed=seed,
    )
    trace = build_code_completion_trace(cfg, params)
    # Normalize source to a repo-relative form so the rendered report
    # doesn't leak the machine's absolute home directory.
    try:
        rel = Path(abs_path).resolve().relative_to(REPO_ROOT)
        trace.source = f"code_completion:{rel.as_posix()}"
        path_for_info = rel.as_posix()
    except ValueError:
        path_for_info = abs_path
    # Record the prefix's token length so the report can explain where
    # the cache-hit curve saturates (once the cache fits the prefix,
    # additional capacity buys near-zero hits for independent tasks).
    try:
        import tiktoken

        prefix_token_len = int(
            len(tiktoken.get_encoding("cl100k_base").encode(CODE_COMPLETION_INSTRUCTION_PREFIX))
        )
    except Exception:  # pragma: no cover — tiktoken is required by the loader
        prefix_token_len = None
    prefix_block_len = None if prefix_token_len is None else int(prefix_token_len // 16)
    return trace, {
        "kind": "code_completion",
        "local_path": path_for_info,
        "config": {
            "max_tasks": cfg.max_tasks,
            "language_filter": list(cfg.language_filter),
            "instruction_prefix_tokens": prefix_token_len,
            "instruction_prefix_blocks_16": prefix_block_len,
            "seed": cfg.seed,
        },
        "params": {
            "arrival_rate_qps": params.arrival_rate_qps,
            "max_output_tokens": params.max_output_tokens,
            "tokenizer": params.tokenizer,
            "seed": params.seed,
        },
        "data_acquisition": [
            "# Fetch HumanEval + MBPP into data/code_completion/ (gitignored).",
            "python scripts/fetch_code_completion_data.py",
        ],
    }


def _not_implemented(name: str) -> Callable[..., tuple[InMemoryTrace, dict]]:
    def _raise(*_args, **_kwargs):
        raise NotImplementedError(
            f"Loader for dataset '{name}' not implemented yet. "
            f"Add an entry to scripts/dataset_metrics.py::DATASET_LOADERS."
        )

    return _raise


DATASET_LOADERS: dict[str, Callable[..., tuple[InMemoryTrace, dict]]] = {
    "synthetic": load_synthetic,
    "code_completion": load_code_completion,
    "lmsys": _not_implemented("lmsys"),
    "sharegpt": _not_implemented("sharegpt"),
}


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------


def percentiles(x: np.ndarray, ps: Iterable[float]) -> dict[str, float]:
    if x.size == 0:
        return {f"p{int(p * 100)}": math.nan for p in ps}
    return {f"p{int(p * 100)}": float(np.quantile(x, p)) for p in ps}


def describe_numeric(x: np.ndarray) -> dict:
    if x.size == 0:
        return {"n": 0}
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=0)),
        "min": float(x.min()),
        "max": float(x.max()),
        **percentiles(x, [0.5, 0.9, 0.95, 0.99]),
    }


def gini(x: np.ndarray) -> float:
    """Gini coefficient on non-negative values. 0=uniform, 1=pure skew."""
    if x.size == 0:
        return math.nan
    x = np.sort(np.asarray(x, dtype=float))
    if x[0] < 0:
        x = x - x[0]
    n = x.size
    cumx = np.cumsum(x)
    total = cumx[-1]
    if total == 0:
        return 0.0
    # Standard Gini = (2 * sum(i * x_i) / (n * total)) - (n + 1) / n
    i = np.arange(1, n + 1)
    return float((2.0 * np.sum(i * x) / (n * total)) - (n + 1) / n)


def zipf_fit(counts: np.ndarray) -> dict:
    """Fit count_k = C * (k+1)^(-s) by OLS on log-log.

    `counts` is the sorted-descending count vector per rank.
    Returns exponent s, intercept log C, and R^2.
    """
    c = np.asarray(counts, dtype=float)
    c = c[c > 0]
    if c.size < 3:
        return {"s": math.nan, "log_C": math.nan, "r2": math.nan, "n_ranks": int(c.size)}
    ranks = np.arange(1, c.size + 1)
    log_r = np.log(ranks)
    log_c = np.log(c)
    A = np.vstack([log_r, np.ones_like(log_r)]).T
    slope, intercept = np.linalg.lstsq(A, log_c, rcond=None)[0]
    pred = slope * log_r + intercept
    ss_res = float(np.sum((log_c - pred) ** 2))
    ss_tot = float(np.sum((log_c - log_c.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan
    return {"s": float(-slope), "log_C": float(intercept), "r2": float(r2), "n_ranks": int(c.size)}


def fano_factor(event_times: np.ndarray, window: float) -> float:
    """Fano factor F = var(N)/mean(N) over fixed windows.

    F=1 for homogeneous Poisson; >1 indicates bursty; <1 regular.
    """
    if event_times.size < 2 or window <= 0:
        return math.nan
    t0, t1 = float(event_times[0]), float(event_times[-1])
    if t1 <= t0:
        return math.nan
    edges = np.arange(t0, t1 + window, window)
    if edges.size < 2:
        return math.nan
    counts, _ = np.histogram(event_times, bins=edges)
    mu = counts.mean()
    if mu == 0:
        return math.nan
    return float(counts.var(ddof=0) / mu)


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------


def compute_metrics(
    trace: InMemoryTrace,
    *,
    block_size: int = 16,
    cache_capacities: tuple[int, ...] = (64, 256, 1024, 4096, 16384),
    is_natural_language: bool = False,
) -> dict:
    """Extract the full descriptive-metrics bundle for a trace.

    Policy-agnostic. Runs one oracle LRU simulation per capacity for the
    cache-hit curve; everything else is a single pass over the requests.
    """
    reqs: list[Request] = list(trace.requests)
    if not reqs:
        raise ValueError("empty trace")

    # -- volume -----------------------------------------------------------
    prompt_lens = np.array([len(r.prompt_tokens) for r in reqs], dtype=int)
    output_lens = np.array([r.max_output_tokens for r in reqs], dtype=int)
    session_ids = [r.session_id for r in reqs]
    family_ids = [r.metadata.get("family") for r in reqs]
    has_families = all(f is not None for f in family_ids)
    volume = {
        "n_requests": int(len(reqs)),
        "n_sessions": int(len(set(session_ids))),
        "n_families": int(len(set(family_ids))) if has_families else None,
        "trace_duration_s": float(reqs[-1].arrival_ts - reqs[0].arrival_ts),
        "source": trace.source,
    }

    # -- length distributions --------------------------------------------
    lengths = {
        "prompt_tokens": describe_numeric(prompt_lens),
        "output_tokens_budget": describe_numeric(output_lens),
        "prompt_histogram": np.histogram(prompt_lens, bins=32)[0].tolist(),
        "prompt_histogram_edges": np.histogram(prompt_lens, bins=32)[1].tolist(),
    }

    # -- interarrival / burstiness ---------------------------------------
    arrivals = np.array([r.arrival_ts for r in reqs], dtype=float)
    iats = np.diff(arrivals)
    iats = iats[iats >= 0]
    iat_mean = float(iats.mean()) if iats.size else math.nan
    iat_var = float(iats.var(ddof=0)) if iats.size else math.nan
    cv2 = iat_var / (iat_mean**2) if iat_mean and iat_mean > 0 else math.nan
    duration = float(arrivals[-1] - arrivals[0]) if arrivals.size > 1 else math.nan
    window_1s = fano_factor(arrivals, 1.0) if np.isfinite(duration) else math.nan
    window_10s = fano_factor(arrivals, 10.0) if np.isfinite(duration) else math.nan
    timing = {
        "interarrival": describe_numeric(iats),
        "cv2_of_interarrival": cv2,
        "empirical_qps": float(len(reqs) / duration) if duration and duration > 0 else math.nan,
        "fano_factor_1s": window_1s,
        "fano_factor_10s": window_10s,
        "arrivals_gini": gini(iats),
    }

    # -- prefix structure --------------------------------------------------
    # Compute block-level prefix hashes per request. The engine uses
    # blake2b over the prompt token prefix, so content-addressed blocks
    # line up across requests that share a leading token subsequence.
    per_req_blocks: list[list[str]] = [
        enumerate_prefix_hashes(r.prompt_tokens, block_size=block_size) for r in reqs
    ]
    block_lens = np.array([len(b) for b in per_req_blocks], dtype=int)
    # Also track "any prefix overlap" by first-block identity — a cheap
    # cross-request reuse signal.
    first_blocks = [b[0] if b else None for b in per_req_blocks]
    first_block_counts = Counter(b for b in first_blocks if b is not None)
    # Unique-block stats across the corpus.
    all_blocks = Counter()
    for blks in per_req_blocks:
        for b in blks:
            all_blocks[b] += 1
    n_unique_blocks = len(all_blocks)
    n_total_block_occurrences = int(sum(all_blocks.values()))
    # Power-law fit on first-block popularity (that's what families share).
    sorted_first_block = np.array(sorted(first_block_counts.values(), reverse=True), dtype=float)
    first_block_zipf = zipf_fit(sorted_first_block)
    first_block_top10_share = (
        float(sorted_first_block[:10].sum() / sorted_first_block.sum())
        if sorted_first_block.size
        else math.nan
    )
    # Also a power-law fit on all-block popularity.
    sorted_all_block = np.array(sorted(all_blocks.values(), reverse=True), dtype=float)
    all_block_zipf = zipf_fit(sorted_all_block)
    # Family-frequency Zipf (if families exist).
    if has_families:
        fam_counts = Counter(family_ids)
        sorted_fam_counts = np.array(sorted(fam_counts.values(), reverse=True), dtype=float)
        family_zipf = zipf_fit(sorted_fam_counts)
        family_gini = gini(sorted_fam_counts)
        family_top10_share = float(sorted_fam_counts[:10].sum() / sorted_fam_counts.sum())
    else:
        family_zipf = {"s": math.nan, "log_C": math.nan, "r2": math.nan, "n_ranks": 0}
        family_gini = math.nan
        family_top10_share = math.nan

    prefix = {
        "block_size": block_size,
        "blocks_per_request": describe_numeric(block_lens),
        "n_unique_blocks": int(n_unique_blocks),
        "n_total_block_occurrences": n_total_block_occurrences,
        "block_reuse_ratio": (
            float(1 - n_unique_blocks / n_total_block_occurrences)
            if n_total_block_occurrences
            else math.nan
        ),
        "n_unique_first_blocks": int(len(first_block_counts)),
        "first_block_top10_share": first_block_top10_share,
        "first_block_zipf": first_block_zipf,
        "all_block_zipf": all_block_zipf,
        "family_zipf": family_zipf,
        "family_gini": family_gini,
        "family_top10_share": family_top10_share,
    }

    # -- oracle cache-hit curve -------------------------------------------
    # Single global LRU over blocks, block-granular. Hit-rate is
    # (reused blocks) / (total block lookups). This is an *upper bound*
    # on what a prefix-aware policy can achieve with a unified cache of
    # that capacity — real policies partition across pods and will do
    # strictly worse. OrderedDict + move_to_end keeps each op O(1).
    from collections import OrderedDict

    hit_curve = []
    for cap in cache_capacities:
        hits = 0
        total = 0
        cache: OrderedDict[str, None] = OrderedDict()
        cap_i = int(cap)
        for blks in per_req_blocks:
            for b in blks:
                total += 1
                if b in cache:
                    hits += 1
                    cache.move_to_end(b)
                else:
                    if len(cache) >= cap_i:
                        cache.popitem(last=False)
                    cache[b] = None
        hit_curve.append(
            {
                "capacity_blocks": int(cap),
                "capacity_tokens": int(cap * block_size),
                "hit_rate": float(hits / total) if total else math.nan,
                "hits": int(hits),
                "lookups": int(total),
            }
        )

    # -- session/turn structure -------------------------------------------
    sessions: dict[str, list[int]] = defaultdict(list)
    for idx, r in enumerate(reqs):
        sessions[r.session_id].append(idx)
    turns = np.array([len(v) for v in sessions.values()], dtype=int)
    # Intra-session prefix reuse: within a session, how often does
    # request N+1's first block match any of request 0..N's first
    # blocks? (shared-head mode makes this a family-recurrence proxy.)
    # Reuses per_req_blocks rather than re-hashing — halves runtime.
    intra_hits = 0
    intra_opps = 0
    prompt_growth_slopes: list[float] = []
    for sid, idxs in sessions.items():
        idxs_sorted = sorted(idxs, key=lambda i: reqs[i].arrival_ts)
        seen_first_blocks: set[str] = set()
        for i in idxs_sorted:
            blks = per_req_blocks[i]
            if not blks:
                continue
            if seen_first_blocks:
                intra_opps += 1
                if blks[0] in seen_first_blocks:
                    intra_hits += 1
            seen_first_blocks.add(blks[0])
        # prompt-length growth across turns: OLS slope on turn index
        if len(idxs_sorted) >= 3:
            xs = np.arange(len(idxs_sorted), dtype=float)
            ys = np.array([len(reqs[i].prompt_tokens) for i in idxs_sorted], dtype=float)
            A = np.vstack([xs, np.ones_like(xs)]).T
            slope, _ = np.linalg.lstsq(A, ys, rcond=None)[0]
            prompt_growth_slopes.append(float(slope))

    sessions_block = {
        "turns_per_session": describe_numeric(turns),
        "turns_per_session_gini": gini(turns),
        "intra_session_first_block_reuse_rate": (
            float(intra_hits / intra_opps) if intra_opps else math.nan
        ),
        "n_sessions_with_gte_3_turns": int(len(prompt_growth_slopes)),
        "prompt_growth_slope_tokens_per_turn": (
            describe_numeric(np.array(prompt_growth_slopes, dtype=float))
            if prompt_growth_slopes
            else {"n": 0}
        ),
    }

    # -- text statistics ---------------------------------------------------
    # Tokens in synthetic are random ints, not real text. We report the
    # raw shape of the token-id distribution as a sanity check and flag
    # that language/code classification does not apply.
    sample_tokens = [t for r in reqs[: min(500, len(reqs))] for t in r.prompt_tokens]
    if sample_tokens:
        sample_arr = np.array(sample_tokens, dtype=int)
        token_id_summary = {
            "sampled_from_first_n_requests": min(500, len(reqs)),
            "unique_token_ids_in_sample": int(np.unique(sample_arr).size),
            "id_min": int(sample_arr.min()),
            "id_max": int(sample_arr.max()),
            "id_mean": float(sample_arr.mean()),
        }
    else:
        token_id_summary = {}
    text = {
        "is_natural_language": bool(is_natural_language),
        "token_id_summary": token_id_summary,
        "n_empty_prompts": int(np.sum(prompt_lens == 0)),
        "n_degenerate_prompts_lt_16": int(np.sum(prompt_lens < block_size)),
    }

    return {
        "volume": volume,
        "lengths": lengths,
        "timing": timing,
        "prefix": prefix,
        "cache_hit_curve": hit_curve,
        "sessions": sessions_block,
        "text": text,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _save_plot(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def make_plots(
    trace: InMemoryTrace,
    metrics: dict,
    *,
    dataset: str,
    figures_dir: Path,
    block_size: int = 16,
) -> dict[str, str]:
    """Write per-metric PNGs; return dict of logical-name -> relative path."""
    reqs = trace.requests
    prompt_lens = np.array([len(r.prompt_tokens) for r in reqs], dtype=int)
    arrivals = np.array([r.arrival_ts for r in reqs], dtype=float)
    iats = np.diff(arrivals)
    family_ids = [r.metadata.get("family") for r in reqs]
    has_families = all(f is not None for f in family_ids)

    paths: dict[str, str] = {}

    # Prompt-length histogram.
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(prompt_lens, bins=48, color="#3a7ca5")
    ax.set_xlabel("prompt length (tokens)")
    ax.set_ylabel("requests")
    ax.set_title(f"{dataset}: prompt-length distribution")
    p = figures_dir / f"{dataset}_prompt_length_hist.png"
    _save_plot(fig, p)
    paths["prompt_length_hist"] = p.name

    # Interarrival CCDF (log-log).
    if iats.size:
        s = np.sort(iats)
        ccdf = 1.0 - np.arange(1, s.size + 1) / s.size
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.loglog(s[s > 0], ccdf[s > 0], color="#e63946")
        ax.set_xlabel("interarrival time (s)")
        ax.set_ylabel("P(IAT > x)")
        ax.set_title(f"{dataset}: interarrival CCDF (log-log)")
        p = figures_dir / f"{dataset}_interarrival_ccdf.png"
        _save_plot(fig, p)
        paths["interarrival_ccdf"] = p.name

    # Family Zipf rank-frequency plot.
    if has_families:
        from collections import Counter as _Counter

        fam_counts = _Counter(family_ids)
        ys = np.array(sorted(fam_counts.values(), reverse=True), dtype=float)
        xs = np.arange(1, ys.size + 1)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.loglog(xs, ys, "o-", color="#6a4c93", ms=3)
        fit = metrics["prefix"]["family_zipf"]
        if math.isfinite(fit["s"]):
            pred = np.exp(fit["log_C"]) * xs ** (-fit["s"])
            ax.loglog(
                xs, pred, "--", color="black", label=f"fit s={fit['s']:.2f} R²={fit['r2']:.3f}"
            )
            ax.legend()
        ax.set_xlabel("family rank")
        ax.set_ylabel("request count")
        ax.set_title(f"{dataset}: prefix-family frequency (Zipf fit)")
        p = figures_dir / f"{dataset}_family_zipf.png"
        _save_plot(fig, p)
        paths["family_zipf"] = p.name

    # Cache-hit curve.
    curve = metrics["cache_hit_curve"]
    caps = [c["capacity_blocks"] for c in curve]
    hrs = [c["hit_rate"] for c in curve]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.semilogx(caps, hrs, "o-", color="#2a9d8f")
    ax.set_xlabel("unified cache capacity (blocks)")
    ax.set_ylabel("oracle hit rate")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"{dataset}: oracle LRU hit rate vs capacity (block_size={block_size})")
    ax.grid(True, which="both", alpha=0.3)
    p = figures_dir / f"{dataset}_cache_hit_curve.png"
    _save_plot(fig, p)
    paths["cache_hit_curve"] = p.name

    # Arrivals over time (QPS in 10s windows).
    if arrivals.size > 1:
        duration = float(arrivals[-1] - arrivals[0])
        if duration > 0:
            bins = np.linspace(arrivals[0], arrivals[-1], 80)
            counts, edges = np.histogram(arrivals, bins=bins)
            dt = edges[1] - edges[0]
            centers = 0.5 * (edges[:-1] + edges[1:])
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.plot(centers, counts / dt, color="#f4a261")
            ax.axhline(
                metrics["timing"]["empirical_qps"],
                linestyle="--",
                color="black",
                label=f"mean {metrics['timing']['empirical_qps']:.2f} qps",
            )
            ax.set_xlabel("time (s)")
            ax.set_ylabel("qps (binned)")
            ax.set_title(f"{dataset}: arrival rate over time")
            ax.legend()
            p = figures_dir / f"{dataset}_qps_timeseries.png"
            _save_plot(fig, p)
            paths["qps_timeseries"] = p.name

    return paths


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def render_report(
    dataset: str,
    metrics: dict,
    loader_info: dict,
    figure_paths: dict[str, str],
    figures_reldir: str,
    distinctive_summary: str,
) -> str:
    lines: list[str] = []
    v = metrics["volume"]
    p = metrics["prefix"]
    t = metrics["timing"]
    s = metrics["sessions"]
    L = metrics["lengths"]

    lines.append(f"# Dataset metrics: {dataset}")
    lines.append("")
    lines.append(f"> {distinctive_summary}")
    lines.append("")
    lines.append("## Source")
    lines.append("")
    lines.append(f"- **kind**: `{loader_info.get('kind', dataset)}`")
    lines.append(f"- **trace source**: `{v['source']}`")
    if "config" in loader_info:
        lines.append("- **loader config**:")
        lines.append("")
        lines.append("  ```json")
        for line in json.dumps(loader_info["config"], indent=2).splitlines():
            lines.append(f"  {line}")
        lines.append("  ```")
    if "params" in loader_info:
        lines.append("- **loader params**:")
        lines.append("")
        lines.append("  ```json")
        for line in json.dumps(loader_info["params"], indent=2).splitlines():
            lines.append(f"  {line}")
        lines.append("  ```")
    lines.append("")

    lines.append("## Volume")
    lines.append("")
    lines.append(f"- Requests: **{v['n_requests']:,}**")
    lines.append(f"- Sessions: **{v['n_sessions']:,}**")
    if v["n_families"] is not None:
        lines.append(f"- Prefix families: **{v['n_families']:,}**")
    lines.append(f"- Trace duration: **{v['trace_duration_s']:.1f} s**")
    lines.append(f"- Empirical QPS: **{t['empirical_qps']:.2f}**")
    lines.append("")

    lines.append("## Prompt / output length")
    lines.append("")
    lines.append("| metric | prompt_tokens | output_tokens_budget |")
    lines.append("|---|---:|---:|")
    pt = L["prompt_tokens"]
    ot = L["output_tokens_budget"]
    for key in ("n", "mean", "std", "min", "p50", "p90", "p95", "p99", "max"):
        a = pt.get(key, math.nan)
        b = ot.get(key, math.nan)

        def _fmt(x):
            if isinstance(x, int):
                return f"{x:,}"
            return f"{x:,.1f}" if math.isfinite(x) else "—"

        lines.append(f"| {key} | {_fmt(a)} | {_fmt(b)} |")
    lines.append("")
    if "prompt_length_hist" in figure_paths:
        lines.append(
            f"![prompt length histogram]({figures_reldir}/{figure_paths['prompt_length_hist']})"
        )
        lines.append("")

    lines.append("## Interarrival / burstiness")
    lines.append("")
    iat = t["interarrival"]
    lines.append(f"- Mean IAT: **{iat['mean']:.4f} s** (std {iat['std']:.4f})")
    lines.append(f"- CV² of IAT: **{t['cv2_of_interarrival']:.3f}** (≈1.0 → Poisson-like)")
    lines.append(f"- Fano factor (1s windows): **{t['fano_factor_1s']:.3f}**")
    lines.append(f"- Fano factor (10s windows): **{t['fano_factor_10s']:.3f}**")
    lines.append(f"- Gini on interarrival gaps: **{t['arrivals_gini']:.3f}**")
    lines.append("")
    if "interarrival_ccdf" in figure_paths:
        lines.append(f"![interarrival ccdf]({figures_reldir}/{figure_paths['interarrival_ccdf']})")
        lines.append("")
    if "qps_timeseries" in figure_paths:
        lines.append(f"![qps over time]({figures_reldir}/{figure_paths['qps_timeseries']})")
        lines.append("")

    lines.append("## Prefix structure")
    lines.append("")
    lines.append(f"- Block size: **{p['block_size']} tokens**")
    bpr = p["blocks_per_request"]
    lines.append(
        f"- Blocks per request: mean **{bpr['mean']:.1f}**, "
        f"p50 {bpr['p50']:.0f}, p95 {bpr['p95']:.0f}"
    )
    lines.append(
        f"- Unique blocks: **{p['n_unique_blocks']:,}** "
        f"(of {p['n_total_block_occurrences']:,} lookups)"
    )
    lines.append(f"- Block-reuse ratio: **{p['block_reuse_ratio']:.3f}** (1 − unique/lookups)")
    lines.append(f"- Unique first-blocks: **{p['n_unique_first_blocks']:,}**")
    lines.append(f"- Top-10 first-blocks share: **{p['first_block_top10_share']:.3f}**")
    fb = p["first_block_zipf"]
    lines.append(f"- First-block Zipf fit: s=**{fb['s']:.2f}**, R²={fb['r2']:.3f}")
    ab = p["all_block_zipf"]
    lines.append(f"- All-block Zipf fit: s=**{ab['s']:.2f}**, R²={ab['r2']:.3f}")
    if v["n_families"] is not None:
        fz = p["family_zipf"]
        lines.append(f"- Family Zipf fit: s=**{fz['s']:.2f}**, R²={fz['r2']:.3f}")
        lines.append(
            f"- Family Gini: **{p['family_gini']:.3f}**, "
            f"top-10 share **{p['family_top10_share']:.3f}**"
        )
    lines.append("")
    if "family_zipf" in figure_paths:
        lines.append(f"![family zipf]({figures_reldir}/{figure_paths['family_zipf']})")
        lines.append("")

    lines.append("## Oracle cache hit-rate curve")
    lines.append("")
    lines.append("Single unified LRU over blocks. Upper bound on what a prefix-aware")
    lines.append("policy can achieve at that capacity; real multi-pod policies pay")
    lines.append("partition overhead and will do strictly worse.")
    lines.append("")
    lines.append("| capacity (blocks) | capacity (tokens) | hit rate |")
    lines.append("|---:|---:|---:|")
    for row in metrics["cache_hit_curve"]:
        lines.append(
            f"| {row['capacity_blocks']:,} | {row['capacity_tokens']:,} | {row['hit_rate']:.3f} |"
        )
    lines.append("")
    if "cache_hit_curve" in figure_paths:
        lines.append(f"![cache hit curve]({figures_reldir}/{figure_paths['cache_hit_curve']})")
        lines.append("")

    lines.append("## Session / turn structure")
    lines.append("")
    tps = s["turns_per_session"]
    lines.append(
        f"- Turns per session: mean **{tps['mean']:.1f}**, "
        f"p50 {tps['p50']:.0f}, p95 {tps['p95']:.0f}, max {tps['max']:.0f}"
    )
    lines.append(f"- Turns-per-session Gini: **{s['turns_per_session_gini']:.3f}**")
    lines.append(
        f"- Intra-session first-block reuse rate: "
        f"**{s['intra_session_first_block_reuse_rate']:.3f}**"
    )
    slope = s["prompt_growth_slope_tokens_per_turn"]
    if slope.get("n", 0):
        lines.append(
            f"- Prompt-length growth across turns (OLS slope, tokens/turn): "
            f"mean **{slope['mean']:.2f}** over {slope['n']:,} sessions with ≥3 turns"
        )
    lines.append("")

    lines.append("## Text statistics")
    lines.append("")
    text = metrics["text"]
    lines.append(f"- Natural language: **{text['is_natural_language']}**")
    lines.append(f"- Empty prompts: **{text['n_empty_prompts']:,}**")
    lines.append(f"- Degenerate prompts (<1 block): **{text['n_degenerate_prompts_lt_16']:,}**")
    if text["token_id_summary"]:
        ts = text["token_id_summary"]
        lines.append(
            f"- Token-id sample ({ts['sampled_from_first_n_requests']} reqs): "
            f"id range [{ts['id_min']},{ts['id_max']}], "
            f"unique ids {ts['unique_token_ids_in_sample']:,}, mean {ts['id_mean']:.0f}"
        )
    lines.append("")

    lines.append("## Reproduction")
    lines.append("")
    if loader_info.get("data_acquisition"):
        lines.append("```bash")
        for cmd in loader_info["data_acquisition"]:
            lines.append(cmd)
        lines.append(f"python scripts/dataset_metrics.py --dataset {dataset}")
        lines.append("```")
    else:
        lines.append("```bash")
        lines.append(f"python scripts/dataset_metrics.py --dataset {dataset}")
        lines.append("```")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Distinctive-summary tag (per-dataset, one-liner for the bead)
# ---------------------------------------------------------------------------


def distinctive_summary(dataset: str, metrics: dict, loader_info: dict | None = None) -> str:
    if dataset == "code_completion":
        v = metrics["volume"]
        p = metrics["prefix"]
        curve = metrics["cache_hit_curve"]
        # Pick the capacity just above the prefix's block count to show
        # where the hit rate saturates.
        prefix_blocks = None
        if loader_info is not None:
            prefix_blocks = loader_info.get("config", {}).get("instruction_prefix_blocks_16")
        sat = curve[-1]
        for row in curve:
            if prefix_blocks is not None and row["capacity_blocks"] >= prefix_blocks:
                sat = row
                break
        return (
            "code_completion is the single-turn dataset: every request is its own "
            f"session (n_sessions={v['n_sessions']:,} ≈ n_requests={v['n_requests']:,}, "
            "turns≈1), so prefix reuse is NOT driven by conversation history — it "
            "comes entirely from the constant instruction template we prepend to "
            f"every task. That template is {prefix_blocks} blocks; the oracle hit "
            f"rate at capacity={sat['capacity_blocks']} blocks is {sat['hit_rate']:.2f}, "
            f"and first-block top-10 share is {p['first_block_top10_share']:.3f} "
            "(one block wins essentially all first-block slots). Routing gain above "
            "that saturation point comes from somewhere other than cross-task reuse."
        )
    if dataset == "synthetic":
        fz = metrics["prefix"]["family_zipf"]
        cv2 = metrics["timing"]["cv2_of_interarrival"]
        p = metrics["cache_hit_curve"]
        # Hit rate at the middle capacity point — representative.
        mid = p[len(p) // 2]
        return (
            "Synthetic is the only dataset where arrivals are a clean homogeneous "
            f"Poisson process (CV²≈{cv2:.2f}), prefix popularity follows a tunable "
            f"Zipf law (fit s≈{fz['s']:.2f}), and cache-hit rate at capacity="
            f"{mid['capacity_blocks']} blocks is a knob-controlled "
            f"{mid['hit_rate']:.2f} — the other three datasets inherit real-world burstiness "
            "and prompt structure that the synthetic trace suppresses on purpose."
        )
    return f"{dataset}: distinctive-summary not yet written."


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(DATASET_LOADERS.keys()),
        help="Which dataset to profile.",
    )
    parser.add_argument(
        "--n-requests",
        type=int,
        default=10000,
        help="Target request budget (synthetic only; real datasets use their natural size).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT,
        help="Root for research/ outputs (defaults to repo root).",
    )
    args = parser.parse_args(argv)

    t0 = time.time()
    loader = DATASET_LOADERS[args.dataset]
    trace, loader_info = loader(n_requests=args.n_requests, seed=args.seed)
    t_load = time.time() - t0

    # Real-text datasets use tiktoken; synthetic emits random token ids.
    is_nl = loader_info.get("kind") in {"code_completion", "lmsys", "sharegpt"}
    t0 = time.time()
    metrics = compute_metrics(trace, block_size=args.block_size, is_natural_language=is_nl)
    metrics["_runtime"] = {"load_s": t_load, "compute_s": time.time() - t0}

    # Write JSON.
    data_dir = args.output_root / "research" / "data" / "dataset_metrics"
    data_dir.mkdir(parents=True, exist_ok=True)
    json_path = data_dir / f"{args.dataset}.json"
    payload = {
        "dataset": args.dataset,
        "loader": loader_info,
        "metrics": metrics,
    }
    json_path.write_text(json.dumps(payload, indent=2, default=_json_default))
    print(f"wrote {json_path}")

    # Plots + report.
    reports_dir = args.output_root / "research" / "reports"
    figures_dir = reports_dir / "figures"
    figure_paths = make_plots(
        trace, metrics, dataset=args.dataset, figures_dir=figures_dir, block_size=args.block_size
    )
    summary = distinctive_summary(args.dataset, metrics, loader_info)
    md = render_report(
        args.dataset,
        metrics,
        loader_info,
        figure_paths,
        figures_reldir="figures",
        distinctive_summary=summary,
    )
    md_path = reports_dir / f"dataset_metrics_{args.dataset}.md"
    md_path.write_text(md)
    print(f"wrote {md_path}")
    print(f"distinctive-summary: {summary}")
    return 0


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        f = float(o)
        if math.isnan(f):
            return None
        return f
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not json-serializable: {type(o).__name__}")


if __name__ == "__main__":
    raise SystemExit(main())
