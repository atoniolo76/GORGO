"""Online tuner for the GORGO policy hyperparameters.

Sits one layer above ``proxy/workload.py``: this module is the
*orchestrator*, the ``Tuner`` classes are the algorithms. The
orchestrator runs the same workload that ``proxy/workload.py`` runs,
hands the resulting stats dict to the tuner, asks the tuner for the
next candidate hyperparameters, POSTs them to the proxy, and repeats.

Each tuning iteration::

    POST /flush                  -> reset radix trie + replica caches
    POST /hyperparameters        -> stage candidate {t_prefill, queued_tokens_weight}
    workload.replay.remote(...)  -> run the workload, get stats dict
    score = score_fn(stats)      -> reduce to a scalar to maximize
    tuner.report(candidate, score)
    candidate = tuner.propose()  -> next candidate (or None when done)

Because ``proxy/workload.py`` replays the same chronological slice of
GLM 5.1 traffic on every call (deterministic sampling), repeated runs
with the same hyperparameters land at the same score modulo small
SGLang non-determinism. That's the property both tuners rely on; it
means a 0.5% relative-tolerance threshold is enough to filter noise
from real improvement.

Algorithms (selected via ``--algorithm``):

* ``hill-climb`` (``HillClimbTuner``): coordinate hill-climb with
  shrinking multiplicative step. For each axis, try ``v*(1+step)`` and
  ``v*(1-step)``; accept the candidate that improves the incumbent by
  more than ``tol``. When a full sweep finds nothing better, halve
  ``step``. Deterministic. ``2*d`` evals per sweep.

* ``gaussian-es`` (``GaussianESTuner``): (1+1)-Evolution Strategy in
  log-space with Rechenberg's 1/5 success rule. Samples one candidate
  per ``propose()`` from an isotropic log-Gaussian centered on the
  incumbent; adapts ``sigma`` so the long-run accept rate stays near
  1/5. Eval-efficient (1 eval/step), handles diagonal interactions,
  reproducible via ``--seed``.

Usage (flag-driven CLI)::

    modal run proxy/tuning.py::tune_cli --proxy-url https://... \\
        --start-time 2026-04-01T12:00:00 \\
        --num-requests 200 \\
        --concurrency 32 \\
        --metric output_throughput \\
        --algorithm gaussian-es \\
        --max-steps 16 \\
        --seed 0

Usage (interactive TUI -- prompts for every knob, including the
hyperparameter search ranges, and confirms before launching)::

    modal run proxy/tuning.py::tune_interactive --proxy-url https://...

Both entrypoints accept the same set of knobs. The workload-shaped
flags (``--start-time``/``--end-time``/``--offset``/``--num-requests``/
``--concurrency``/``--model``/``--stream``/``--max-tokens``) mirror
``proxy/workload.py`` exactly: pick a small, fast slice (a few hundred
requests is typical) so each tuning trial takes seconds, not minutes.
The hyperparameter search ranges (``--t-prefill-min`` /
``--t-prefill-max`` / ``--queued-tokens-weight-min`` /
``--queued-tokens-weight-max``) default to :data:`HYPERPARAM_RANGES`
but can be tightened or widened per-run without editing the source.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from datetime import datetime, timezone
from typing import Callable

import modal

from app import app, bench_results_volume, completions_volume
from proxy.workload import DEFAULT_MODEL, replay

REGION = os.getenv("REGION", "us-east")

image = (
    modal.Image.debian_slim()
    .pip_install("httpx[http2]", "pyarrow")
    .add_local_python_source("app", "proxy", "policy", "utils")
)


# Default search ranges per hyperparameter. Keys mirror
# ``proxy/modal_proxy.py``::ALLOWED_HYPERPARAM_KEYS, so a typo here
# would surface as a 400 from the proxy on first POST. The tuners use
# multiplicative steps, so the range only needs to be order-of-magnitude
# correct.
#
# Bounds are sized so calibrate.py's typical recommended values
# (per-token prefill / decode rates, on the order of 1e-4 to 1e-2 s/tok
# on current GPUs) sit comfortably inside the interior, with ~2 OOM of
# headroom above for slow paths and ~1-2 OOM below for fast paths.
# Upper bound of 1.0 also coincides with the proxy default
# (``DEFAULT_GORGO_HYPERPARAMETERS``), so a bootstrap from that default
# starts at the upper edge and the search can only move inward.
#
# These act as defaults for :func:`tune` -- the per-key bounds can be
# overridden at runtime via ``--t-prefill-{min,max}`` /
# ``--queued-tokens-weight-{min,max}`` (or the matching prompts in
# :func:`tune_interactive`) without editing this file.
HYPERPARAM_RANGES: dict[str, tuple[float, float]] = {
    "t_prefill": (1e-5, 1.0),
    "queued_tokens_weight": (1e-5, 1.0),
}


def _validated_ranges(
    overrides: dict[str, tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    """Merge ``overrides`` on top of :data:`HYPERPARAM_RANGES` and check
    each pair satisfies ``0 < lo < hi``. The lower bound must be
    strictly positive because both tuners sample in log-space; a zero
    or negative ``lo`` would crash with a domain error inside
    :class:`GaussianESTuner`.
    """
    merged = {k: tuple(v) for k, v in HYPERPARAM_RANGES.items()}
    merged.update({k: tuple(v) for k, v in overrides.items()})
    for k, (lo, hi) in merged.items():
        if lo <= 0:
            raise SystemExit(f"{k} lower bound must be > 0 (log-space sampling), got {lo}")
        if lo >= hi:
            raise SystemExit(f"{k} range invalid: lo ({lo}) must be < hi ({hi})")
    return merged


# Each entry maps a CLI ``--metric`` value to a function that pulls a
# scalar to *maximize* out of a workload stats dict. Negate latencies
# so the optimization direction stays uniform across metrics.
SCORE_FUNCTIONS: dict[str, Callable[[dict], float]] = {
    "output_throughput": lambda s: float(s["output_token_throughput"]),
    "total_throughput": lambda s: float(s["total_token_throughput"]),
    "request_throughput": lambda s: float(s["request_throughput_rps"]),
    "neg_p95_ttft": lambda s: -float(s["ttft_seconds"]["p95"]),
    "neg_p99_ttft": lambda s: -float(s["ttft_seconds"]["p99"]),
    "neg_p95_e2e": lambda s: -float(s["request_e2e_seconds"]["p95"]),
    "neg_avg_itl": lambda s: -float(s["itl_ms"]["avg"]),
}


def _now_iso_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# Tuner protocol (duck-typed; both tuners below implement it):
#   propose()        -> dict | None    # next candidate, or None when done
#   report(c, s)     -> bool           # was c the new incumbent? updates state
#   state            -> dict           # internal scalars worth logging per trial
#   best_params      -> dict
#   best_score       -> float | None
#
# First propose() always returns the bootstrap params unchanged so the
# orchestrator gets a baseline measurement before any search step.


class HillClimbTuner:
    """Coordinate hill-climb with shrinking multiplicative step.

    State machine driven by ``propose() -> report() -> propose() ...``:
    the orchestrator never has to know about sweeps or steps. ``propose``
    returns ``None`` exactly once the tuner is done.
    """

    name = "coordinate-hill-climb-shrink"

    def __init__(
        self,
        initial_params: dict[str, float],
        ranges: dict[str, tuple[float, float]],
        *,
        initial_step: float = 0.5,
        min_step: float = 0.05,
        tol: float = 0.005,
        max_steps: int = 16,
    ) -> None:
        self.ranges = ranges
        self.best_params: dict[str, float] = {
            k: self._clamp(k, float(initial_params.get(k, sum(ranges[k]) / 2))) for k in ranges
        }
        self.best_score: float | None = None
        self.step = float(initial_step)
        self.min_step = float(min_step)
        self.tol = float(tol)
        self.max_steps = int(max_steps)
        self.evaluated_after_baseline = 0
        self._sweep: list[tuple[str, int]] = []
        self._sweep_improved = False
        self._build_sweep()

    def _clamp(self, key: str, v: float) -> float:
        lo, hi = self.ranges[key]
        return max(lo, min(hi, v))

    def _build_sweep(self) -> None:
        # One full sweep = one direction along each axis. Fixed order keeps
        # the saved history easy to skim by eye.
        self._sweep = [(key, sign) for key in self.ranges for sign in (+1, -1)]
        self._sweep_improved = False

    @property
    def state(self) -> dict:
        return {"step": self.step, "sweep_improved": self._sweep_improved}

    def propose(self) -> dict[str, float] | None:
        """Next candidate to evaluate, or ``None`` if tuning is done."""
        if self.best_score is None:
            return dict(self.best_params)
        if self.evaluated_after_baseline >= self.max_steps:
            return None
        while True:
            while self._sweep:
                key, sign = self._sweep.pop(0)
                factor = 1.0 + sign * self.step
                candidate_val = self._clamp(key, self.best_params[key] * factor)
                if abs(candidate_val - self.best_params[key]) < 1e-12:
                    continue
                cand = dict(self.best_params)
                cand[key] = candidate_val
                return cand
            if not self._sweep_improved:
                self.step *= 0.5
                if self.step < self.min_step:
                    return None
            self._build_sweep()

    def report(self, candidate: dict[str, float], score: float) -> bool:
        """Tell the tuner the score for the last proposed candidate.

        Returns ``True`` iff the candidate is the new incumbent.
        """
        if self.best_score is None:
            self.best_score = score
            self.best_params = dict(candidate)
            return True
        self.evaluated_after_baseline += 1
        if score > self.best_score * (1.0 + self.tol):
            self.best_score = score
            self.best_params = dict(candidate)
            self._sweep_improved = True
            return True
        return False


class GaussianESTuner:
    """(1+1)-Evolution Strategy with Rechenberg's 1/5 success rule.

    Samples an isotropic Gaussian perturbation in *log-space* on top of
    the current incumbent::

        log(theta') = log(theta) + sigma * z,   z ~ N(0, I)
        theta'      = exp(log(theta'))          (then clipped to range)

    Log-space sampling means a single isotropic ``sigma`` works across
    hyperparameters whose natural ranges span orders of magnitude
    (which is exactly the case for ``t_prefill`` and
    ``queued_tokens_weight``).

    Rechenberg's 1/5 rule (Rechenberg 1973): track the acceptance rate
    over a sliding window; if it's above ``target_rate``, grow ``sigma``;
    if it's below, shrink. The optimal acceptance rate on the sphere
    function is ~1/5, and this empirically generalizes.

    1 evaluation per ``propose()``, vs ``2 * d`` per sweep for
    ``HillClimbTuner``. Stops when ``max_steps`` trials have been used
    or when ``sigma < sigma_min``.
    """

    name = "gaussian-es-1plus1-1over5"

    def __init__(
        self,
        initial_params: dict[str, float],
        ranges: dict[str, tuple[float, float]],
        *,
        sigma: float = 0.5,
        sigma_min: float = 0.02,
        sigma_decay: float = 0.817,  # Schwefel ~= 1/1.224
        success_window: int = 8,
        target_rate: float = 0.2,
        tol: float = 0.005,
        max_steps: int = 16,
        seed: int | None = None,
    ) -> None:
        self.ranges = ranges
        self.keys = list(ranges.keys())
        self.best_params: dict[str, float] = {
            k: self._clamp(k, float(initial_params.get(k, sum(ranges[k]) / 2))) for k in ranges
        }
        self.best_score: float | None = None
        self.sigma = float(sigma)
        self.sigma_min = float(sigma_min)
        self.sigma_decay = float(sigma_decay)
        self.success_window = int(success_window)
        self.target_rate = float(target_rate)
        self.tol = float(tol)
        self.max_steps = int(max_steps)
        self.evaluated_after_baseline = 0
        self._recent: list[bool] = []
        self._rng = random.Random(seed)
        self.seed = seed

    def _clamp(self, key: str, v: float) -> float:
        lo, hi = self.ranges[key]
        return max(lo, min(hi, v))

    def propose(self) -> dict[str, float] | None:
        """Next candidate to evaluate, or ``None`` when done."""
        if self.best_score is None:
            return dict(self.best_params)
        if self.evaluated_after_baseline >= self.max_steps:
            return None
        if self.sigma < self.sigma_min:
            return None
        cand: dict[str, float] = {}
        for key in self.keys:
            v = self.best_params[key]
            # Range floors are positive (``lo > 0``) so log() is safe; the
            # _clamp() on the way out keeps proposals strictly inside.
            log_new = math.log(max(v, 1e-300)) + self.sigma * self._rng.gauss(0.0, 1.0)
            cand[key] = self._clamp(key, math.exp(log_new))
        return cand

    def report(self, candidate: dict[str, float], score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.best_params = dict(candidate)
            return True
        self.evaluated_after_baseline += 1
        accepted = score > self.best_score * (1.0 + self.tol)
        if accepted:
            self.best_score = score
            self.best_params = dict(candidate)
        # 1/5 rule: only adapt once we have a representative window so we
        # don't oscillate sigma based on the first 2-3 trials.
        self._recent.append(accepted)
        if len(self._recent) > self.success_window:
            self._recent.pop(0)
        if len(self._recent) >= self.success_window:
            rate = sum(self._recent) / len(self._recent)
            if rate > self.target_rate:
                # Too many accepts -> we're being timid, take bigger steps.
                self.sigma /= self.sigma_decay
            elif rate < self.target_rate:
                self.sigma *= self.sigma_decay
        return accepted

    @property
    def state(self) -> dict:
        recent_rate = sum(self._recent) / len(self._recent) if self._recent else None
        return {
            "sigma": self.sigma,
            "recent_success_rate": recent_rate,
            "recent_window_filled": len(self._recent),
        }


# Tuner instances are interchangeable under the protocol above; the
# orchestrator picks one by name.
TunerLike = HillClimbTuner | GaussianESTuner


def _build_tuner(
    algorithm: str,
    *,
    initial_params: dict[str, float],
    ranges: dict[str, tuple[float, float]],
    max_steps: int,
    relative_tolerance: float,
    initial_step: float,
    min_step: float,
    sigma: float,
    sigma_min: float,
    seed: int | None,
) -> TunerLike:
    if algorithm == "hill-climb":
        return HillClimbTuner(
            initial_params=initial_params,
            ranges=ranges,
            initial_step=initial_step,
            min_step=min_step,
            tol=relative_tolerance,
            max_steps=max_steps,
        )
    if algorithm == "gaussian-es":
        return GaussianESTuner(
            initial_params=initial_params,
            ranges=ranges,
            sigma=sigma,
            sigma_min=sigma_min,
            tol=relative_tolerance,
            max_steps=max_steps,
            seed=seed,
        )
    raise SystemExit(f"unknown --algorithm {algorithm!r}; choices: 'hill-climb', 'gaussian-es'")


@app.function(
    image=image,
    region=REGION,
    timeout=24 * 60 * 60,
    volumes={
        "/data": completions_volume,
        "/results": bench_results_volume,
    },
)
def tune(
    proxy_url: str,
    *,
    start_time: str | None = None,
    end_time: str | None = None,
    offset: int = 0,
    num_requests: int | None = None,
    concurrency: int = 16,
    model: str | None = DEFAULT_MODEL,
    stream: bool | None = None,
    max_tokens: int | None = None,
    max_input_tokens: int | None = None,
    metric: str = "output_throughput",
    algorithm: str = "hill-climb",
    max_steps: int = 16,
    initial_step: float = 0.5,
    min_step: float = 0.05,
    sigma: float = 0.5,
    sigma_min: float = 0.02,
    seed: int | None = None,
    relative_tolerance: float = 0.005,
    output_dir: str | None = None,
    t_prefill_min: float = HYPERPARAM_RANGES["t_prefill"][0],
    t_prefill_max: float = HYPERPARAM_RANGES["t_prefill"][1],
    queued_tokens_weight_min: float = HYPERPARAM_RANGES["queued_tokens_weight"][0],
    queued_tokens_weight_max: float = HYPERPARAM_RANGES["queued_tokens_weight"][1],
) -> dict:
    """Tune ``t_prefill`` / ``queued_tokens_weight`` on the proxy online.

    Args:
        proxy_url: Base URL of the proxy (the ``modal.forward`` tunnel).
        start_time/end_time/offset/num_requests/concurrency/model/
        stream/max_tokens/max_input_tokens: Forwarded verbatim to
            ``replay``. Pick a short slice (e.g. ``--num-requests
            200``) so each tuning trial completes quickly. See
            ``proxy/workload.py``::``replay`` for ``max_input_tokens``
            semantics (``None``/``0`` auto-detects from
            ``/get_server_info``); leaving it at the default is
            recommended so trials don't burn budget on context-overflow
            failures.
        metric: Key into :data:`SCORE_FUNCTIONS`; the metric to maximize.
        algorithm: ``"hill-climb"`` (coordinate hill-climb with
            shrinking step; deterministic) or ``"gaussian-es"`` ((1+1)-ES
            with Rechenberg's 1/5 rule in log-space; 1 eval/step).
        max_steps: Maximum number of trials *after* the baseline. The
            orchestrator may stop early if the algorithm signals
            convergence (step or sigma below its floor).
        initial_step / min_step: ``hill-climb`` only -- multiplicative
            step bounds (``0.5`` = first trials probe ±50%).
        sigma / sigma_min: ``gaussian-es`` only -- log-space Gaussian
            std-dev bounds. ``sigma=0.5`` means initial perturbations
            scale params by roughly ``exp(±0.5)`` (~0.6x..1.65x).
        seed: ``gaussian-es`` only -- RNG seed for reproducibility;
            ``None`` -> nondeterministic.
        relative_tolerance: Treat improvements smaller than this as
            noise. ``0.005`` = 0.5%.
        output_dir: Where to write the per-tune-run artifact folder
            (relative paths are rooted at ``/results``). ``None`` ->
            ``/results/tune_<UTC-timestamp>``.
        t_prefill_min/max, queued_tokens_weight_min/max: Per-key
            search-range bounds. Default to :data:`HYPERPARAM_RANGES`.
            Lower bounds must be strictly positive (log-space) and
            strictly less than the upper bounds; otherwise the run is
            rejected before any traffic is sent.

    Returns:
        Summary dict including ``best_params``, ``best_score``,
        ``baseline_score``, the full per-trial ``history``, and the
        resolved ``output_path`` of the saved JSON artifacts.
    """
    import httpx

    if metric not in SCORE_FUNCTIONS:
        raise SystemExit(f"unknown metric {metric!r}; choices: {sorted(SCORE_FUNCTIONS)}")
    score_fn = SCORE_FUNCTIONS[metric]

    ranges = _validated_ranges(
        {
            "t_prefill": (t_prefill_min, t_prefill_max),
            "queued_tokens_weight": (
                queued_tokens_weight_min,
                queued_tokens_weight_max,
            ),
        }
    )

    proxy_url = proxy_url.rstrip("/")
    workload_kwargs = dict(
        start_time=start_time,
        end_time=end_time,
        offset=offset,
        num_requests=num_requests,
        concurrency=concurrency,
        model=model,
        stream=stream,
        max_tokens=max_tokens,
        max_input_tokens=max_input_tokens,
        save_per_request=False,
    )

    # Each tuning run gets a dedicated folder so per-trial replay JSONs
    # don't collide and are trivially groupable for later analysis.
    run_started_at = datetime.now(timezone.utc)
    if output_dir is None:
        out_dir = f"/results/tune_{run_started_at.strftime('%Y%m%d_%H%M%S')}"
    else:
        out_dir = output_dir if os.path.isabs(output_dir) else os.path.join("/results", output_dir)
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "summary.json")

    # Flush takes up to FLUSH_UPSTREAM_TIMEOUT_SECONDS=120s on a hot
    # cache; pad to 180 so we don't trip the client read timeout.
    http = httpx.Client(base_url=proxy_url, timeout=httpx.Timeout(180.0))
    try:
        r = http.get("/policy")
        r.raise_for_status()
        policy_doc = r.json()
        current_hp = dict(policy_doc.get("hyperparameters") or {})
        active_policy = policy_doc.get("policy")

        tuner: TunerLike = _build_tuner(
            algorithm,
            initial_params=current_hp,
            ranges=ranges,
            max_steps=max_steps,
            relative_tolerance=relative_tolerance,
            initial_step=initial_step,
            min_step=min_step,
            sigma=sigma,
            sigma_min=sigma_min,
            seed=seed,
        )

        history: list[dict] = []
        baseline_score: float | None = None

        print(
            f"[tune] algorithm={tuner.name} policy={active_policy!r} "
            f"starting hyperparameters={current_hp} "
            f"metric={metric} max_steps={max_steps}"
        )

        def evaluate(params: dict[str, float], trial_idx: int) -> tuple[float, dict]:
            """Flush, set hyperparameters, run workload, return (score, stats)."""
            print(f"[tune] trial {trial_idx}: evaluating {params}")
            t0 = time.perf_counter()

            http.post("/flush").raise_for_status()
            http.post("/hyperparameters", json=params).raise_for_status()

            trial_output = os.path.join(out_dir, f"replay_trial_{trial_idx:03d}.json")
            stats = replay.remote(
                proxy_url=proxy_url,
                output_path=trial_output,
                **workload_kwargs,
            )
            score = score_fn(stats)
            elapsed = time.perf_counter() - t0
            print(f"[tune]   trial {trial_idx} score={score:.4f} ({metric}) elapsed={elapsed:.1f}s")
            return score, stats

        trial_idx = 0
        while True:
            candidate = tuner.propose()
            if candidate is None:
                break

            is_baseline = tuner.best_score is None
            score, stats = evaluate(candidate, trial_idx)
            accepted = tuner.report(candidate, score)

            tuner_state = dict(tuner.state)
            if is_baseline:
                baseline_score = score
                print(f"[tune]   baseline score={score:.4f}")
            else:
                tag = "ACCEPTED" if accepted else "rejected"
                state_str = " ".join(
                    f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in tuner_state.items()
                )
                print(f"[tune]   {tag}: incumbent score={tuner.best_score:.4f} {state_str}")

            history.append(
                {
                    "trial": trial_idx,
                    "kind": "baseline" if is_baseline else "trial",
                    "params": candidate,
                    "score": score,
                    "incumbent_score": tuner.best_score,
                    "incumbent_params": dict(tuner.best_params),
                    "tuner_state": tuner_state,
                    "accepted": accepted,
                    "metric": metric,
                    # Embed the workload's aggregate stats so the
                    # tuning artifact is self-contained even without the
                    # per-trial replay JSON.
                    "stats": stats,
                }
            )

            # Persist incrementally so a long tune run can be inspected
            # / resumed-from-history even if the function is interrupted.
            partial = _build_summary(
                run_started_at=run_started_at,
                proxy_url=proxy_url,
                workload_kwargs=workload_kwargs,
                metric=metric,
                tuner=tuner,
                baseline_score=baseline_score,
                history=history,
                active_policy=active_policy,
                ranges=ranges,
                finished_at=None,
                output_path=summary_path,
            )
            with open(summary_path, "w") as f:
                json.dump(partial, f)
            bench_results_volume.commit()

            trial_idx += 1

        # Apply the winner so the proxy stays in the best state we found
        # (the last evaluated candidate may not be the incumbent).
        print(
            f"[tune] best_params={tuner.best_params} best_score={tuner.best_score:.4f} "
            f"(baseline={baseline_score:.4f})"
        )
        try:
            http.post("/hyperparameters", json=tuner.best_params).raise_for_status()
            http.post("/flush").raise_for_status()
        except Exception as e:
            print(f"[tune] final apply failed: {e}")
    finally:
        http.close()

    summary = _build_summary(
        run_started_at=run_started_at,
        proxy_url=proxy_url,
        workload_kwargs=workload_kwargs,
        metric=metric,
        tuner=tuner,
        baseline_score=baseline_score,
        history=history,
        active_policy=active_policy,
        ranges=ranges,
        finished_at=datetime.now(timezone.utc),
        output_path=summary_path,
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f)
    bench_results_volume.commit()
    print(f"[tune] saved tuning summary to {summary_path}")

    return summary


def _build_summary(
    *,
    run_started_at: datetime,
    proxy_url: str,
    workload_kwargs: dict,
    metric: str,
    tuner: TunerLike,
    baseline_score: float | None,
    history: list[dict],
    active_policy: str | None,
    ranges: dict[str, tuple[float, float]],
    finished_at: datetime | None,
    output_path: str,
) -> dict:
    return {
        "started_at": run_started_at.isoformat().replace("+00:00", "Z"),
        "finished_at": finished_at.isoformat().replace("+00:00", "Z") if finished_at else None,
        "proxy_url": proxy_url,
        "active_policy": active_policy,
        "metric": metric,
        "workload": workload_kwargs,
        "tuning": {
            "algorithm": tuner.name,
            "ranges": {k: list(v) for k, v in ranges.items()},
            "max_steps": tuner.max_steps,
            "relative_tolerance": tuner.tol,
            "trials_run": len(history),
            "current_state": dict(tuner.state),
        },
        "best_params": dict(tuner.best_params),
        "best_score": tuner.best_score,
        "baseline_score": baseline_score,
        "improvement_over_baseline": (
            (tuner.best_score - baseline_score) / abs(baseline_score)
            if baseline_score not in (None, 0)
            else None
        ),
        "history": history,
        "output_path": output_path,
    }


def _parse_stream_flag(stream: str) -> bool | None:
    """Stream sentinel parser shared by ``tune_cli`` and ``tune_interactive``.
    Empty string -> ``None`` (use replay default); ``true/false/1/0/...``
    coerce as expected. Anything else is rejected before any traffic."""
    s = stream.strip().lower()
    if s == "":
        return None
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    raise SystemExit(f"invalid stream={stream!r}; expected true/false")


def _dispatch_tune(
    *,
    proxy_url: str,
    start_time: str,
    end_time: str,
    offset: int,
    num_requests: int,
    concurrency: int,
    model: str,
    stream: str,
    max_tokens: int,
    max_input_tokens: int,
    metric: str,
    algorithm: str,
    max_steps: int,
    initial_step: float,
    min_step: float,
    sigma: float,
    sigma_min: float,
    seed: int,
    relative_tolerance: float,
    output_dir: str,
    t_prefill_min: float,
    t_prefill_max: float,
    queued_tokens_weight_min: float,
    queued_tokens_weight_max: float,
):
    """Single point that converts CLI/TUI sentinels and forwards to
    ``tune.remote``. Sentinel mapping matches ``proxy/workload.py``::

        empty string for start_time/end_time/model/stream/output_dir
        0 for num_requests/max_tokens
        0 for max_input_tokens (auto-detect); negative disables filter
        seed=-1 -> nondeterministic (only consulted by gaussian-es)
    """
    return tune.remote(
        proxy_url=proxy_url,
        start_time=start_time or None,
        end_time=end_time or None,
        offset=offset,
        num_requests=num_requests or None,
        concurrency=concurrency,
        model=model or None,
        stream=_parse_stream_flag(stream),
        max_tokens=max_tokens or None,
        max_input_tokens=max_input_tokens,
        metric=metric,
        algorithm=algorithm,
        max_steps=max_steps,
        initial_step=initial_step,
        min_step=min_step,
        sigma=sigma,
        sigma_min=sigma_min,
        seed=None if seed < 0 else seed,
        relative_tolerance=relative_tolerance,
        output_dir=output_dir or None,
        t_prefill_min=t_prefill_min,
        t_prefill_max=t_prefill_max,
        queued_tokens_weight_min=queued_tokens_weight_min,
        queued_tokens_weight_max=queued_tokens_weight_max,
    )


# Spec for the interactive TUI: ordered groups of (name, kind, default,
# help, choices?) entries. ``name`` matches ``_dispatch_tune`` kwargs
# 1:1 so the prompted dict can be **-splat into the dispatcher with no
# remapping. ``kind`` is one of {"str","int","float","choice"}. Hoisted
# to module scope so it's discoverable / importable for testing.
_TUNE_TUI_SPEC: list[tuple[str, list[dict]]] = [
    (
        "Workload (mirrors proxy/workload.py)",
        [
            {
                "name": "start_time",
                "kind": "str",
                "default": "",
                "help": "YYYY-MM-DD start of replay window; empty = use first row",
            },
            {
                "name": "end_time",
                "kind": "str",
                "default": "",
                "help": "YYYY-MM-DD end of replay window; empty = unbounded",
            },
            {
                "name": "offset",
                "kind": "int",
                "default": 0,
                "help": "Skip the first N rows of the replay window",
            },
            {
                "name": "num_requests",
                "kind": "int",
                "default": 0,
                "help": "Cap requests per trial (0 = full window)",
            },
            {
                "name": "concurrency",
                "kind": "int",
                "default": 16,
                "help": "Max in-flight requests per trial",
            },
            {
                "name": "model",
                "kind": "str",
                "default": DEFAULT_MODEL,
                "help": "Model name override sent to upstream",
            },
            {
                "name": "stream",
                "kind": "str",
                "default": "",
                "help": "true/false/empty -- empty uses replay default",
            },
            {
                "name": "max_tokens",
                "kind": "int",
                "default": 0,
                "help": "0 = use replay default",
            },
            {
                "name": "max_input_tokens",
                "kind": "int",
                "default": 0,
                "help": "Per-request input cap; 0=auto from /get_server_info, <0=disable, >0=use",
            },
        ],
    ),
    (
        "Tuner",
        [
            {
                "name": "metric",
                "kind": "choice",
                "default": "output_throughput",
                "help": "Score function to maximize",
                "choices": sorted(SCORE_FUNCTIONS),
            },
            {
                "name": "algorithm",
                "kind": "choice",
                "default": "hill-climb",
                "help": "Search strategy",
                "choices": ["hill-climb", "gaussian-es"],
            },
            {
                "name": "max_steps",
                "kind": "int",
                "default": 16,
                "help": "Max trials after the baseline",
            },
            {
                "name": "relative_tolerance",
                "kind": "float",
                "default": 0.005,
                "help": "Improvement fraction to count as real (0.005 = 0.5%)",
            },
            {
                "name": "initial_step",
                "kind": "float",
                "default": 0.5,
                "help": "hill-climb initial multiplicative step",
            },
            {
                "name": "min_step",
                "kind": "float",
                "default": 0.05,
                "help": "hill-climb step floor",
            },
            {
                "name": "sigma",
                "kind": "float",
                "default": 0.5,
                "help": "gaussian-es initial log-space sigma",
            },
            {
                "name": "sigma_min",
                "kind": "float",
                "default": 0.02,
                "help": "gaussian-es sigma floor",
            },
            {
                "name": "seed",
                "kind": "int",
                "default": -1,
                "help": "RNG seed for gaussian-es (-1 = nondeterministic)",
            },
        ],
    ),
    (
        "Hyperparameter search ranges (log-space; lower bound > 0)",
        [
            {
                "name": "t_prefill_min",
                "kind": "float",
                "default": HYPERPARAM_RANGES["t_prefill"][0],
                "help": "Lower bound for t_prefill (s/tok)",
            },
            {
                "name": "t_prefill_max",
                "kind": "float",
                "default": HYPERPARAM_RANGES["t_prefill"][1],
                "help": "Upper bound for t_prefill (s/tok)",
            },
            {
                "name": "queued_tokens_weight_min",
                "kind": "float",
                "default": HYPERPARAM_RANGES["queued_tokens_weight"][0],
                "help": "Lower bound for queued_tokens_weight (s/tok)",
            },
            {
                "name": "queued_tokens_weight_max",
                "kind": "float",
                "default": HYPERPARAM_RANGES["queued_tokens_weight"][1],
                "help": "Upper bound for queued_tokens_weight (s/tok)",
            },
        ],
    ),
    (
        "Output",
        [
            {
                "name": "output_dir",
                "kind": "str",
                "default": "",
                "help": "Empty = /results/tune_<UTC-timestamp>",
            },
        ],
    ),
]


def _coerce_value(kind: str, raw: str, default):
    """Parse a TUI string answer into the spec's declared type. Empty
    answer falls back to ``default`` (the printed value)."""
    if raw == "":
        return default
    if kind == "int":
        return int(raw)
    if kind == "float":
        return float(raw)
    return raw


def _prompt_field_plain(field: dict) -> object:
    """Stdlib-only prompt fallback (used when ``rich`` is unavailable).

    Round-trips through :func:`_coerce_value` so the resulting type
    exactly matches the rich-prompt path."""
    name = field["name"]
    default = field["default"]
    label = f"  {name}"
    if field.get("help"):
        label += f"  ({field['help']})"
    if field.get("choices"):
        label += f"  [{'|'.join(field['choices'])}]"
    label += f" [{default!r}]: "
    while True:
        raw = input(label).strip()
        try:
            value = _coerce_value(field["kind"], raw, default)
        except ValueError as e:
            print(f"    invalid: {e}")
            continue
        if field.get("choices") and value not in field["choices"]:
            print(f"    must be one of {field['choices']}")
            continue
        return value


def _print_config_table_plain(chosen: dict) -> None:
    name_width = max(len(k) for k in chosen)
    print()
    print("Final configuration:")
    print("-" * (name_width + 30))
    for k, v in chosen.items():
        print(f"  {k.ljust(name_width)}  {v!r}")
    print("-" * (name_width + 30))


def _run_tui(initial_proxy_url: str = "") -> dict:
    """Drive the interactive flow. Returns the dict of chosen kwargs."""
    try:
        from rich.console import Console
        from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt
        from rich.table import Table

        rich_available = True
    except ImportError:
        rich_available = False

    chosen: dict = {}

    if rich_available:
        console = Console()
        console.rule("[bold cyan]GORGO online tuner[/]")
        proxy_url = initial_proxy_url or Prompt.ask(
            "[bold]proxy_url[/] [dim](base URL of running proxy)[/]"
        )
        chosen["proxy_url"] = proxy_url.strip()

        for section, fields in _TUNE_TUI_SPEC:
            console.print(f"\n[bold magenta]{section}[/]")
            for f in fields:
                label = f"  [cyan]{f['name']}[/]"
                if f.get("help"):
                    label += f" [dim]{f['help']}[/]"
                if f["kind"] == "int":
                    chosen[f["name"]] = IntPrompt.ask(
                        label, default=int(f["default"]), console=console
                    )
                elif f["kind"] == "float":
                    chosen[f["name"]] = FloatPrompt.ask(
                        label, default=float(f["default"]), console=console
                    )
                elif f["kind"] == "choice":
                    chosen[f["name"]] = Prompt.ask(
                        label,
                        default=str(f["default"]),
                        choices=f["choices"],
                        console=console,
                    )
                else:
                    chosen[f["name"]] = Prompt.ask(
                        label, default=str(f["default"]), console=console
                    )

        # Echo final config; cross-check ranges before shipping to remote.
        try:
            _validated_ranges(
                {
                    "t_prefill": (chosen["t_prefill_min"], chosen["t_prefill_max"]),
                    "queued_tokens_weight": (
                        chosen["queued_tokens_weight_min"],
                        chosen["queued_tokens_weight_max"],
                    ),
                }
            )
        except SystemExit as e:
            console.print(f"[bold red]invalid ranges:[/] {e}")
            raise
        table = Table(title="Final configuration", show_lines=False)
        table.add_column("param", style="cyan", no_wrap=True)
        table.add_column("value", style="white")
        for k, v in chosen.items():
            table.add_row(k, repr(v))
        console.print()
        console.print(table)
        if not Confirm.ask("Launch this run?", default=True):
            console.print("[yellow]aborted[/]")
            raise SystemExit(0)
    else:
        # Plain-text fallback: same prompt names + ordering, no styling.
        print("=== GORGO online tuner ===")
        proxy_url = initial_proxy_url or input("  proxy_url (base URL of running proxy): ")
        chosen["proxy_url"] = proxy_url.strip()
        for section, fields in _TUNE_TUI_SPEC:
            print(f"\n[{section}]")
            for f in fields:
                chosen[f["name"]] = _prompt_field_plain(f)
        try:
            _validated_ranges(
                {
                    "t_prefill": (chosen["t_prefill_min"], chosen["t_prefill_max"]),
                    "queued_tokens_weight": (
                        chosen["queued_tokens_weight_min"],
                        chosen["queued_tokens_weight_max"],
                    ),
                }
            )
        except SystemExit as e:
            print(f"invalid ranges: {e}")
            raise
        _print_config_table_plain(chosen)
        confirm = input("Launch this run? [Y/n]: ").strip().lower()
        if confirm in ("n", "no"):
            print("aborted")
            raise SystemExit(0)

    return chosen


@app.local_entrypoint()
def tune_cli(
    proxy_url: str,
    start_time: str = "",
    end_time: str = "",
    offset: int = 0,
    num_requests: int = 0,
    concurrency: int = 16,
    model: str = DEFAULT_MODEL,
    stream: str = "",
    max_tokens: int = 0,
    max_input_tokens: int = 0,
    metric: str = "output_throughput",
    algorithm: str = "hill-climb",
    max_steps: int = 16,
    initial_step: float = 0.5,
    min_step: float = 0.05,
    sigma: float = 0.5,
    sigma_min: float = 0.02,
    seed: int = -1,
    relative_tolerance: float = 0.005,
    output_dir: str = "",
    t_prefill_min: float = HYPERPARAM_RANGES["t_prefill"][0],
    t_prefill_max: float = HYPERPARAM_RANGES["t_prefill"][1],
    queued_tokens_weight_min: float = HYPERPARAM_RANGES["queued_tokens_weight"][0],
    queued_tokens_weight_max: float = HYPERPARAM_RANGES["queued_tokens_weight"][1],
):
    """Flag-driven CLI wrapper around ``tune.remote``. Defaults are
    intentionally identical to :data:`HYPERPARAM_RANGES` /
    :func:`tune` so a no-flag invocation reproduces the canonical
    setup; any subset of flags can be overridden per-run."""
    _dispatch_tune(
        proxy_url=proxy_url,
        start_time=start_time,
        end_time=end_time,
        offset=offset,
        num_requests=num_requests,
        concurrency=concurrency,
        model=model,
        stream=stream,
        max_tokens=max_tokens,
        max_input_tokens=max_input_tokens,
        metric=metric,
        algorithm=algorithm,
        max_steps=max_steps,
        initial_step=initial_step,
        min_step=min_step,
        sigma=sigma,
        sigma_min=sigma_min,
        seed=seed,
        relative_tolerance=relative_tolerance,
        output_dir=output_dir,
        t_prefill_min=t_prefill_min,
        t_prefill_max=t_prefill_max,
        queued_tokens_weight_min=queued_tokens_weight_min,
        queued_tokens_weight_max=queued_tokens_weight_max,
    )


def _suppress_modal_status_spinner() -> None:
    """Stop the ``modal run`` "Running app..." live spinner so it doesn't
    repaint over our Rich prompts. Touches private Modal internals
    (``modal._output.manager``); wrapped in try/except so a Modal SDK
    refactor degrades to "user sees a spinner, runs with -q" instead of
    crashing the TUI."""
    try:
        from modal._output.manager import OutputManager

        mgr = OutputManager.get()
        stop = getattr(mgr, "stop_status_spinner", None)
        if callable(stop):
            stop()
    except Exception:
        pass


@app.local_entrypoint()
def tune_interactive(proxy_url: str = ""):
    """Interactive TUI walking through every tuning knob.

    Invoke with ``modal run proxy/tuning.py::tune_interactive`` and
    optionally pre-fill the proxy URL via ``--proxy-url``. Every other
    parameter is prompted with its current default; pressing Enter
    accepts the default. Uses ``rich`` for nicer rendering when
    available, falls back to a stdlib-only flow otherwise. After
    showing the full configuration the user confirms before any
    traffic is sent; range bounds are validated client-side so an
    invalid pair is rejected without paying for a remote launch.

    Note: ``modal run`` shows a Rich live "Running app..." spinner that
    repaints over interactive prompts; we stop it explicitly here so the
    TUI is usable without ``modal run -q``."""
    _suppress_modal_status_spinner()
    chosen = _run_tui(initial_proxy_url=proxy_url)
    _dispatch_tune(**chosen)
