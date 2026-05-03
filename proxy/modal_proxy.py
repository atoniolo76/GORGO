import asyncio
import json
import logging
import math
import os
import random
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Callable

# Force ``logging.Formatter`` to format ``%(asctime)s`` in UTC so the
# ``Z`` suffix in the uvicorn log config below is honest regardless of
# the container's tzdata. Touches the base class so it propagates to
# uvicorn's ``DefaultFormatter`` / ``AccessFormatter`` subclasses too.
logging.Formatter.converter = time.gmtime


def _log(message: str) -> None:
    """Emit a ``[proxy]`` log line prefixed with an ISO 8601 UTC
    timestamp (millisecond precision). Wraps ``print`` so it composes
    with Modal's stdout streaming; ``flush=True`` keeps lines visible
    on long-running uvicorn workers where stdout would otherwise
    line-buffer."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    print(f"{ts} [proxy] {message}", flush=True)


# uvicorn ships a sensible LOGGING_CONFIG but its formatters have no
# timestamps -- fine for short request/response loops, but we run the
# proxy for hours and need to correlate access lines with the
# ``_log()`` output above (replica syncs, hyperparameter applies, etc.)
# and with the workload-side logs. Same datefmt as ``_log`` (sans
# milliseconds, which uvicorn's logging.Formatter doesn't support
# natively without a custom Formatter class) so visual scan order
# matches across streams.
_UVICORN_LOG_CONFIG: dict = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": "%(asctime)sZ %(levelprefix)s %(message)s",
            "datefmt": "%Y-%m-%dT%H:%M:%S",
            "use_colors": None,
        },
        "access": {
            "()": "uvicorn.logging.AccessFormatter",
            "fmt": '%(asctime)sZ %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
            "datefmt": "%Y-%m-%dT%H:%M:%S",
        },
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
        "access": {
            "formatter": "access",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
        "uvicorn.error": {"level": "INFO"},
        "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
    },
}

import httpx
import modal
import tiktoken

from app import (
    app,
    bench_results_volume,
    completions_volume,
    hf_datasets_volume,
    lmsys_chat_1m_volume,
    proxies,
    replicas,
)
from proxy.measure import (
    NS_PER_S,
    consume_sse_stream,
    recommend_hyperparameters,
    recommend_hyperparameters_per_target,
    summarize_samples,
)
from policy import (
    POLICY_REGISTRY,
    ROUTING_POLICIES,
    PolicyDef,
    ReplicaSnapshot,
    RouteContext,
    normalize_policy,
)
from policy.gorgo import (
    ALLOWED_HYPERPARAM_KEYS,
    DEFAULT_GORGO_HYPERPARAMETERS,
    make_default_store,
    merge_update,
    prune_per_target,
    validate_update,
)
from utils.radix_trie import RadixTrie

# Match engine/modal_sglang.py so the proxy lands in the same datacenter as
# its replicas (and as the workload generator in proxy/workload.py).
REGION = os.getenv("REGION", "us-east")

DEFAULT_POLICY = "random"
METRICS_REFRESH_INTERVAL_SECONDS = float(os.getenv("METRICS_REFRESH_INTERVAL_SECONDS", 30.0))
METRICS_FETCH_TIMEOUT_SECONDS = 2.0
# SGLang may wait until idle; allow a generous read window for POST /flush_cache.
FLUSH_UPSTREAM_TIMEOUT_SECONDS = 120.0


HYPERPARAM_RANGES: dict[str, tuple[float, float]] = {
    "t_prefill": (1e-5, 1.0),
    "queued_tokens_weight": (1e-5, 1.0),
}


def validated_ranges(
    overrides: dict[str, tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    """Merge ``overrides`` with defaults and check each pair is ``0 < lo < hi``."""
    merged = {k: tuple(v) for k, v in HYPERPARAM_RANGES.items()}
    merged.update({k: tuple(v) for k, v in overrides.items()})
    for k, (lo, hi) in merged.items():
        if lo <= 0:
            raise ValueError(f"{k} lower bound must be > 0 (log-space sampling), got {lo}")
        if lo >= hi:
            raise ValueError(f"{k} range invalid: lo ({lo}) must be < hi ({hi})")
    return merged


SCORE_FUNCTIONS: dict[str, Callable[[dict], float]] = {
    "output_throughput": lambda s: float(s["output_token_throughput"]),
    "total_throughput": lambda s: float(s["total_token_throughput"]),
    "request_throughput": lambda s: float(s["request_throughput_rps"]),
    "neg_p95_ttft": lambda s: -float(s["ttft_seconds"]["p95"]),
    "neg_p99_ttft": lambda s: -float(s["ttft_seconds"]["p99"]),
    "neg_p95_e2e": lambda s: -float(s["request_e2e_seconds"]["p95"]),
    "neg_avg_itl": lambda s: -float(s["itl_ms"]["avg"]),
}


class HillClimbTuner:
    """Coordinate hill-climb with shrinking multiplicative step."""

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
        self._sweep = [(key, sign) for key in self.ranges for sign in (+1, -1)]
        self._sweep_improved = False

    @property
    def state(self) -> dict:
        return {"step": self.step, "sweep_improved": self._sweep_improved}

    def propose(self) -> dict[str, float] | None:
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
    """(1+1)-Evolution Strategy with Rechenberg's 1/5 success rule."""

    name = "gaussian-es-1plus1-1over5"

    def __init__(
        self,
        initial_params: dict[str, float],
        ranges: dict[str, tuple[float, float]],
        *,
        sigma: float = 0.5,
        sigma_min: float = 0.02,
        sigma_decay: float = 0.817,
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

    def _clamp(self, key: str, v: float) -> float:
        lo, hi = self.ranges[key]
        return max(lo, min(hi, v))

    def propose(self) -> dict[str, float] | None:
        if self.best_score is None:
            return dict(self.best_params)
        if self.evaluated_after_baseline >= self.max_steps:
            return None
        if self.sigma < self.sigma_min:
            return None
        cand: dict[str, float] = {}
        for key in self.keys:
            v = self.best_params[key]
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
        self._recent.append(accepted)
        if len(self._recent) > self.success_window:
            self._recent.pop(0)
        if len(self._recent) >= self.success_window:
            rate = sum(self._recent) / len(self._recent)
            if rate > self.target_rate:
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


TunerLike = HillClimbTuner | GaussianESTuner


def build_tuner(
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
    raise ValueError(f"unknown algorithm {algorithm!r}; choices: 'hill-climb', 'gaussian-es'")


def build_summary(
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


# On-the-fly tuning. The proxy keeps a bounded ring buffer of per-request
# samples (TTFT / total / token counts / per-token rates); when the
# auto-tuner is enabled (``POST /tune``) it sliding-window-recomputes
# ``gorgo`` hyperparameters from this buffer via
# ``proxy.measure.recommend_hyperparameters`` -- the same primitive
# ``proxy/calibrate.py`` uses for one-shot calibration.
#
# The auto-tuner is a stateful toggle, *not* a one-shot endpoint: once
# enabled it keeps recomputing every ``hop_size`` new samples until
# explicitly disabled (``POST /tune {"enabled": false}``). Subsequent
# POSTs while it's running atomically reconfigure the live tuner.
MAX_REQUEST_SAMPLES = 1000
DEFAULT_TUNE_WINDOW_SIZE = 100
DEFAULT_TUNE_HOP_SIZE = 50  # recompute every N new samples once warm
DEFAULT_TUNE_APPLY = True  # default to actually mutating hyperparameters

# Headers that are connection-local and must not be forwarded verbatim when
# we proxy upstream responses back to the client. Re-emitting these would
# either confuse uvicorn (e.g. it wants to manage ``transfer-encoding`` and
# ``content-length`` itself when we stream chunks) or leak upstream-specific
# keep-alive semantics to the client. We ask the upstream not to compress
# via ``accept-encoding: identity`` so ``content-encoding`` is also safe to
# drop unconditionally.
_HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "transfer-encoding",
        "upgrade",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "content-encoding",
        "content-length",
    }
)

_ENCODER = None


def _get_encoder():
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = tiktoken.encoding_for_model("gpt-4o")
    return _ENCODER


def _message_text(content) -> str:
    """Extract the text payload from an OpenAI chat-completions ``content`` field.

    ``content`` is either a string, or a list of content blocks
    (``{"type": "text", "text": "..."}``, ``{"type": "image_url", ...}``, ...).
    Non-text blocks are dropped -- they don't contribute to the input token
    count we care about for routing decisions.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text = block.get("text", "") or ""
                    if text:
                        parts.append(text)
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return ""


def tokenize_input(messages: list[dict]) -> list[int]:
    """Tokenize an OpenAI chat-completions ``messages`` payload into a flat
    list of token ids.

    Each message's text content is encoded individually via tiktoken's
    batched encoder and the per-message ids are concatenated in order. This
    isn't an exact match for the server-side count (we ignore per-message
    formatting tokens and tool-call arguments) but it's directionally
    correct and cheap enough to run on every request for both routing
    decisions (``len(ids)``) and prefix-sharing tracking (the ids
    themselves).
    """
    if not isinstance(messages, list) or not messages:
        return []
    enc = _get_encoder()
    texts: list[str] = []
    for msg in messages:
        if isinstance(msg, dict):
            text = _message_text(msg.get("content"))
            if text:
                texts.append(text)
        elif isinstance(msg, str):
            texts.append(msg)
    if not texts:
        return []
    encoded = enc.encode_batch(texts, num_threads=4, disallowed_special=())
    out: list[int] = []
    for ids in encoded:
        out.extend(ids)
    return out


def _parse_metrics_text(text: str) -> dict[str, float]:
    """Parse a Prometheus exposition snippet into ``{metric_name: value}``,
    dropping label suffixes and skipping non-numeric or commented lines."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        value = parts[1]
        if (
            not value.replace(".", "", 1)
            .replace("e+", "", 1)
            .replace("e-", "", 1)
            .lstrip("-")
            .isdigit()
        ):
            continue
        out[parts[0].split("{")[0]] = float(value)
    return out


@app.function(
    image=modal.Image.debian_slim()
    .pip_install("httpx[http2]", "uvicorn", "tiktoken", "pyarrow", "datasets>=3.0")
    .add_local_python_source("app", "proxy", "policy", "utils"),
    region=REGION,
    timeout=(24 * 60 * 60),
    volumes={
        "/data": completions_volume,
        "/lmsys": lmsys_chat_1m_volume,
        "/datasets": hf_datasets_volume,
        "/results": bench_results_volume,
    },
)
def proxy(registry_key: str = ""):
    import time

    import httpx
    import uvicorn

    replica_urls: list[str] = []

    # Routing state. Kept in a dict so the asgi_app closure can mutate it in
    # place from the /policy handler; uvicorn is single-process / single-loop
    # so we don't need a lock (all writes happen on the asyncio thread).
    #
    # ``upstream_client`` is a long-lived httpx.AsyncClient shared across all
    # handlers (metrics refresh + chat completions). Keeping it alive means
    # TCP + TLS handshakes to each replica happen once and keepalive
    # connections are reused for every subsequent request, which is a big
    # TTFT win on cold proxies. Created in lifespan.startup so it binds to
    # the right event loop; closed in lifespan.shutdown.
    state: dict = {
        "policy": DEFAULT_POLICY,
        # Structured GORGO hyperparameter store (see policy/gorgo.py).
        # ``defaults`` applies to every replica; ``per_target`` keys
        # specific replicas to override individual scalars. The auto-
        # tuner writes per-target overrides automatically once enough
        # live samples accumulate per replica; offline tools (calibrate,
        # tuning.py) typically write only ``defaults``.
        "hyperparameters": make_default_store(),
        "upstream_client": None,
        "metrics_task": None,
        # Total samples appended over the proxy lifetime. Doesn't
        # saturate at ``MAX_REQUEST_SAMPLES`` (unlike ``len(samples)``)
        # so it can be diffed across snapshots to count arrivals.
        "total_samples_appended": 0,
        # Live auto-tuner config. Reconfigured atomically by ``POST
        # /tune``; consumed by ``_record_request_sample`` after every
        # successful sample append. ``enabled=False`` means the
        # background recompute is dormant and the rest of the dict is
        # frozen at its last value.
        "auto_tune": {
            "enabled": False,
            "window_size": DEFAULT_TUNE_WINDOW_SIZE,
            "hop_size": DEFAULT_TUNE_HOP_SIZE,
            "apply": DEFAULT_TUNE_APPLY,
            # Counter zeroed on enable / on apply, incremented per
            # sample. Triggers a recompute when it crosses ``hop_size``.
            "samples_since_last_apply": 0,
            # Diagnostics surfaced by GET /tune.
            "applied_count": 0,
            "last_applied_at_monotonic": None,
            "last_recommendation": None,
            "enabled_at_monotonic": None,
        },
        "batch_tuning": {
            "running": False,
            "status": "idle",
            "task": None,
            "run_id": None,
            "started_at": None,
            "finished_at": None,
            "config": None,
            "trial": None,
            "history": [],
            "best_params": None,
            "best_score": None,
            "baseline_score": None,
            "output_path": None,
            "error": None,
            "stop_requested": False,
        },
        "workload_run": {
            "running": False,
            "status": "idle",
            "phase": "idle",
            "task": None,
            "run_id": None,
            "started_at": None,
            "finished_at": None,
            "config": None,
            "stats": None,
            "output_path": None,
            "error": None,
            "stop_requested": False,
        },
        "trace": {
            "enabled": False,
            "trace_id": None,
            "sample_metrics": True,
            "sample_requests": True,
            "max_events": 200_000,
            "started_at": None,
            "stopped_at": None,
            "dropped_metrics": 0,
            "dropped_requests": 0,
            "saved_paths": None,
        },
    }
    endpoints_queued_tokens: dict[str, int] = {url: 0 for url in replica_urls}

    # Bounded ring buffer of per-request samples produced by the
    # SSE-tee in ``_handle_chat_completions``. Each entry has the same
    # shape as ``proxy.measure.measure_chat_completion`` returns, plus
    # ``target`` (chosen replica) and ``recorded_at_monotonic`` so /tune
    # consumers can do their own time-windowing if desired.
    samples: deque[dict] = deque(maxlen=MAX_REQUEST_SAMPLES)
    metrics_trace_events: deque[dict] = deque(maxlen=state["trace"]["max_events"])
    request_trace_events: deque[dict] = deque(maxlen=state["trace"]["max_events"])

    # Live radix trie of every prompt we've forwarded. Each node along a
    # sequence's insertion path is tagged with the replica URL that received
    # that prompt, so the trie doubles as an approximate "which replica
    # currently has this prefix cached in KV" index. Updated inline in the
    # /v1/chat/completions handler after the upstream request is dispatched.
    #
    # Note: currently unbounded; a long-running proxy will accumulate state
    # until the container is recycled. That's fine for the evaluation setup;
    # a production deployment would need LRU/TTL eviction.
    radix_trie = RadixTrie()

    # Default timeout used by the shared client. ``read=None`` is critical for
    # chat completions: httpx treats ``read`` as "seconds between bytes", so a
    # generation that thinks for a while before emitting its first token would
    # otherwise trip the timeout. Metrics calls pass their own shorter
    # override via the per-request ``timeout=`` kwarg.
    default_upstream_timeout = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0)
    # Keep enough warm connections around to saturate concurrent load without
    # re-handshaking. ``keepalive_expiry=None`` means "never expire idle
    # connections"; Modal will eventually tear down the container anyway.
    upstream_limits = httpx.Limits(
        max_connections=200, max_keepalive_connections=100, keepalive_expiry=None
    )

    # Live mirror of each replica's /metrics output. A background task refreshes
    # this every ``METRICS_REFRESH_INTERVAL_SECONDS``; policy functions read
    # snapshots of it per request instead of fetching synchronously.
    live_metrics: dict[str, ReplicaSnapshot] = {}
    metrics_meta: dict = {
        "last_refresh_monotonic": 0.0,
        "last_refresh_wall_ts": None,
        "refresh_seq": 0,
        "last_refresh_errors": {},  # url -> str
    }
    # EWMA-smoothed pure-network RTT per replica, refreshed by ``_probe_rtt``
    # alongside the metrics scrape. Kept outside ``ReplicaSnapshot`` so the
    # smoothed value survives across snapshots even when a single probe fails.
    network_rtt_ewma: dict[str, float] = {}
    NETWORK_RTT_EWMA_ALPHA = 0.3
    NETWORK_RTT_PROBE_TIMEOUT_SECONDS = 2.0

    def _registry_from_items(items) -> dict[str, str]:
        registry: dict[str, str] = {}
        for key, value in items:
            if not isinstance(key, str):
                key = str(key)
            registry[key] = value if isinstance(value, str) else ""
        return registry

    def _active_urls_from_registry(registry: dict[str, str]) -> list[str]:
        seen: set[str] = set()
        normalized: list[str] = []
        for url in registry.values():
            url = url.strip().rstrip("/")
            if not url or not (url.startswith("http://") or url.startswith("https://")):
                continue
            if url in seen:
                continue
            seen.add(url)
            normalized.append(url)
        return normalized

    def _replace_replica_urls(normalized: list[str], *, source: str) -> tuple[list[str], list[str]]:
        old = set(replica_urls)
        new = set(normalized)
        added = sorted(new - old)
        removed = sorted(old - new)
        if not added and not removed and list(replica_urls) == normalized:
            return added, removed

        replica_urls.clear()
        replica_urls.extend(normalized)
        for url in added:
            endpoints_queued_tokens[url] = 0
        for url in removed:
            endpoints_queued_tokens.pop(url, None)
            live_metrics.pop(url, None)
            metrics_meta["last_refresh_errors"].pop(url, None)
        prune_per_target(state["hyperparameters"], set(replica_urls))
        _log(
            f"replicas synced from {source}: "
            f"+{len(added)} -{len(removed)} (total={len(replica_urls)})"
        )
        return added, removed

    def _sync_replicas_from_modal_dict() -> tuple[dict[str, str], list[str], list[str]]:
        registry = _registry_from_items(replicas.items())
        added, removed = _replace_replica_urls(
            _active_urls_from_registry(registry),
            source="modal dict",
        )
        return registry, added, removed

    async def _read_registry_async() -> dict[str, str]:
        """Read the global ``replicas`` modal Dict without mutating local state.

        Used by ``GET /replicas`` so callers can observe the registry
        contents for debugging without triggering a sync that would
        clobber a controller-set ``replica_urls`` (matrix mode pins its
        per-policy 3-replica list via POST /replicas; a concurrent GET
        that re-synced from the global Dict would replace the pin with
        whatever the last writer left there).
        """
        items = []
        async for item in replicas.items.aio():
            items.append(item)
        return _registry_from_items(items)

    async def _sync_replicas_from_modal_dict_async() -> tuple[dict[str, str], list[str], list[str]]:
        registry = await _read_registry_async()
        added, removed = _replace_replica_urls(
            _active_urls_from_registry(registry),
            source="modal dict",
        )
        return registry, added, removed

    def _sync_replicas_from_manual_urls(normalized: list[str]) -> tuple[list[str], list[str]]:
        return _replace_replica_urls(
            normalized,
            source="/replicas",
        )

    # Experiment proxies are configured explicitly by the controller with
    # POST /replicas. Do not ingest every URL in the global replica registry on
    # startup, otherwise a one-fleet smoke proxy can temporarily see unrelated
    # replicas before the controller narrows it back down.
    if not registry_key:
        _sync_replicas_from_modal_dict()

    # ---------- ASGI helpers ----------

    async def _read_json_body(receive) -> dict | list:
        """Drain the ASGI request body and parse it as JSON.

        Combines body assembly with JSON decode so handler call sites stay
        a single ``try / except`` instead of two. Returns ``{}`` for an
        empty body. Raises ``json.JSONDecodeError`` on malformed input
        (handlers convert that to a 400).
        """
        chunks: list[bytes] = []
        while True:
            msg = await receive()
            if msg["type"] != "http.request":
                continue
            chunks.append(msg.get("body", b"") or b"")
            if not msg.get("more_body"):
                break
        body = b"".join(chunks)
        return json.loads(body.decode()) if body else {}

    async def _send_json(send, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": body})

    def _now_wall_ts() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def _trace_append(kind: str, event: dict) -> None:
        tr = state["trace"]
        if not tr["enabled"]:
            return
        if kind == "metrics":
            if not tr["sample_metrics"]:
                return
            target = metrics_trace_events
            dropped_key = "dropped_metrics"
        elif kind == "request":
            if not tr["sample_requests"]:
                return
            target = request_trace_events
            dropped_key = "dropped_requests"
        else:
            return
        if target.maxlen is not None and len(target) >= target.maxlen:
            tr[dropped_key] += 1
        target.append(event)

    def _trace_status_payload() -> dict:
        tr = state["trace"]
        first_ts = None
        last_ts = None
        for buf in (metrics_trace_events, request_trace_events):
            if not buf:
                continue
            b_first = buf[0].get("wall_ts")
            b_last = buf[-1].get("wall_ts")
            first_ts = b_first if first_ts is None else min(first_ts, b_first)
            last_ts = b_last if last_ts is None else max(last_ts, b_last)
        return {
            **tr,
            "metrics_events": len(metrics_trace_events),
            "request_events": len(request_trace_events),
            "first_event_ts": first_ts,
            "last_event_ts": last_ts,
        }

    def _write_jsonl(path: str, rows: deque[dict]) -> None:
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row, separators=(",", ":")) + "\n")

    def _save_trace_to_volume() -> dict:
        tr = state["trace"]
        trace_id = tr["trace_id"] or f"trace-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        out_dir = os.path.join("/results", "proxy_traces", trace_id)
        os.makedirs(out_dir, exist_ok=True)
        metrics_path = os.path.join(out_dir, "metrics.jsonl")
        requests_path = os.path.join(out_dir, "requests.jsonl")
        manifest_path = os.path.join(out_dir, "manifest.json")
        _write_jsonl(metrics_path, metrics_trace_events)
        _write_jsonl(requests_path, request_trace_events)
        manifest = {
            "trace": _trace_status_payload(),
            "metrics_path": metrics_path,
            "requests_path": requests_path,
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        bench_results_volume.commit()
        paths = {
            "metrics_path": metrics_path,
            "requests_path": requests_path,
            "manifest_path": manifest_path,
        }
        tr["saved_paths"] = paths
        return paths

    # ---------- Upstream client + metrics refresh ----------

    def _new_upstream_client() -> httpx.AsyncClient:
        # ``http2=True`` negotiates HTTP/2 via ALPN on HTTPS replicas; if the
        # upstream (SGLang-on-uvicorn) only speaks HTTP/1.1, httpx transparently
        # falls back. Either way we keep the keep-alive pool benefit.
        return httpx.AsyncClient(
            http2=True,
            timeout=default_upstream_timeout,
            limits=upstream_limits,
        )

    async def _probe_rtt(client: httpx.AsyncClient, url: str) -> float | None:
        """Lightweight RTT probe to a replica's base URL.

        Issues a ``GET /`` and times the round-trip. SGLang typically
        404s the root path (no router for ``/``), which is fine here --
        we want pure HTTP RTT, not a useful response. Returns ``None``
        on failure so the caller skips the EWMA update for this tick.

        Reuses the upstream ``client``'s connection pool, so steady-state
        we measure data RTT (no TCP/TLS handshake), matching what real
        chat-completions requests pay.
        """
        t0 = time.monotonic()
        try:
            await client.get(url + "/", timeout=NETWORK_RTT_PROBE_TIMEOUT_SECONDS)
        except Exception:
            return None
        return time.monotonic() - t0

    async def _refresh_one(client: httpx.AsyncClient, url: str, seq: int) -> None:
        """Scrape one replica's /metrics + probe its RTT into ``live_metrics[url]``."""
        t0 = time.monotonic()
        wall_ts = _now_wall_ts()
        # Run the metrics scrape and the lightweight RTT probe concurrently.
        # The probe is independent of /metrics handler load (which is what
        # makes ``snap.latency`` a noisy stand-in for pure network RTT).
        scrape_task = asyncio.create_task(
            client.get(f"{url}/metrics", timeout=METRICS_FETCH_TIMEOUT_SECONDS)
        )
        probe_task = asyncio.create_task(_probe_rtt(client, url))
        try:
            resp = await scrape_task
            resp.raise_for_status()
        except Exception as e:
            # Cancel the in-flight probe defensively if the scrape failed
            # before it completed (we still record whatever the probe got).
            try:
                rtt = await probe_task
            except Exception:
                rtt = None
            if rtt is not None:
                prev = network_rtt_ewma.get(url)
                network_rtt_ewma[url] = (
                    rtt
                    if prev is None
                    else NETWORK_RTT_EWMA_ALPHA * rtt + (1 - NETWORK_RTT_EWMA_ALPHA) * prev
                )
            latency = time.monotonic() - t0
            metrics_meta["last_refresh_errors"][url] = repr(e)
            _trace_append(
                "metrics",
                {
                    "kind": "metrics",
                    "trace_id": state["trace"]["trace_id"],
                    "seq": seq,
                    "wall_ts": wall_ts,
                    "monotonic_s": t0,
                    "replica_url": url,
                    "region": REGION,
                    "scrape_latency_seconds": latency,
                    "network_rtt_seconds": network_rtt_ewma.get(url),
                    "ok": False,
                    "num_running_reqs": None,
                    "num_queue_reqs": None,
                    "num_used_tokens": None,
                    "utilization": None,
                    "gen_throughput": None,
                    "error": repr(e),
                },
            )
            return
        latency = time.monotonic() - t0
        try:
            rtt = await probe_task
        except Exception:
            rtt = None
        if rtt is not None:
            prev = network_rtt_ewma.get(url)
            network_rtt_ewma[url] = (
                rtt
                if prev is None
                else NETWORK_RTT_EWMA_ALPHA * rtt + (1 - NETWORK_RTT_EWMA_ALPHA) * prev
            )
        parsed = _parse_metrics_text(resp.text)
        snap = ReplicaSnapshot(
            num_running_reqs=int(parsed.get("sglang:num_running_reqs", 0)),
            num_queue_reqs=int(parsed.get("sglang:num_queue_reqs", 0)),
            num_used_tokens=int(parsed.get("sglang:num_used_tokens", 0)),
            latency=latency,
            network_rtt=network_rtt_ewma.get(url, 0.0),
            gen_throughput=float(parsed.get("sglang:gen_throughput", 0.0)),
            utilization=float(parsed.get("sglang:utilization", 0.0)),
        )
        live_metrics[url] = snap
        metrics_meta["last_refresh_errors"].pop(url, None)
        _trace_append(
            "metrics",
            {
                "kind": "metrics",
                "trace_id": state["trace"]["trace_id"],
                "seq": seq,
                "wall_ts": wall_ts,
                "monotonic_s": t0,
                "replica_url": url,
                "region": REGION,
                "scrape_latency_seconds": latency,
                "network_rtt_seconds": snap.network_rtt,
                "ok": True,
                "num_running_reqs": snap.num_running_reqs,
                "num_queue_reqs": snap.num_queue_reqs,
                "num_used_tokens": snap.num_used_tokens,
                "utilization": snap.utilization,
                "gen_throughput": snap.gen_throughput,
                "error": None,
            },
        )

    async def _refresh_all(client: httpx.AsyncClient | None) -> None:
        """One pass: refresh every registered replica in parallel."""
        # Auto-discover from the global Dict only when this proxy isn't
        # under explicit controller control. Matrix-mode proxies
        # (``registry_key`` set) pin their 3-replica list via POST
        # /replicas; auto-discovering would re-read every engine in the
        # global Dict and re-introduce the cross-policy clobber. Same
        # gate as the synchronous startup prime at module load time.
        if not replica_urls and not registry_key:
            await _sync_replicas_from_modal_dict_async()
        if client is None or not replica_urls:
            return
        seq = metrics_meta["refresh_seq"] + 1
        await asyncio.gather(
            *[_refresh_one(client, url, seq) for url in replica_urls],
            return_exceptions=True,
        )
        metrics_meta["refresh_seq"] = seq
        metrics_meta["last_refresh_monotonic"] = time.monotonic()
        metrics_meta["last_refresh_wall_ts"] = _now_wall_ts()

    async def _metrics_refresh_loop() -> None:
        """Background task: refresh every ``METRICS_REFRESH_INTERVAL_SECONDS``.
        Cancelled in lifespan.shutdown. Per-iteration exceptions are
        logged so a single transient failure doesn't kill the loop."""
        try:
            while True:
                try:
                    await _refresh_all(state["upstream_client"])
                except Exception as e:
                    _log(f"metrics refresh iteration failed: {e}")
                await asyncio.sleep(METRICS_REFRESH_INTERVAL_SECONDS)
        except asyncio.CancelledError:
            pass

    async def _flush_upstream_replica(client: httpx.AsyncClient, base_url: str) -> tuple[str, dict]:
        url = f"{base_url.rstrip('/')}/flush_cache"
        try:
            resp = await client.post(
                url,
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=FLUSH_UPSTREAM_TIMEOUT_SECONDS,
                    write=30.0,
                    pool=10.0,
                ),
            )
            out = {"ok": resp.is_success, "status_code": resp.status_code}
            if not resp.is_success:
                body = (resp.text or "")[:512]
                if body:
                    out["body_preview"] = body
            return base_url, out
        except Exception as e:
            return base_url, {"ok": False, "error": repr(e)}

    # ---------- Routing ----------

    def _select_endpoint(token_ids: list[int]) -> tuple[str, str, str]:
        """Pick an upstream URL using the policy registry in :mod:`policy`.

        Returns ``(target, configured, effective_policy)``.

        * ``configured`` is the policy name read from ``state["policy"]``
          atomically at the start of the call. The caller writes this
          (rather than re-reading state) into the trace event's
          ``policy`` field so a concurrent ``POST /policy`` mid-request
          can't produce a row whose ``policy`` and ``effective_policy``
          disagree just from a race.
        * ``effective_policy`` is what actually selected ``target``:
          ``configured`` when the policy function ran successfully,
          ``"single-replica"`` when there was no choice to make, or
          ``"random-fallback:<reason>"`` when this function bypassed the
          policy and picked a random replica. Distinct values let
          post-hoc trace analysis filter fallback rows out of per-policy
          aggregates -- otherwise a brief metrics-missing window
          silently biases the comparison toward random's distribution.

        For policies that don't need ``/metrics`` data (random, pd*,
        simple-session-affinity) we skip the snapshot entirely so the
        proxy can keep routing during a metrics-refresh outage. For
        metrics-using policies we filter ``live_metrics`` to
        currently-registered replicas; if any replica has no snapshot
        yet (cold start, /metrics timeout) we fall back to random
        rather than passing a partial view to the policy.

        Scope note: this function only catches *proxy-level* fallbacks
        (no metrics, no choice, exception). Some policy fns in
        ``policy/lb_aibrix.py`` have their own internal
        ``random.choice`` paths when their preconditions fail (e.g.
        empty candidate set after filtering). Those still report
        ``effective_policy == configured`` because the policy fn
        returned successfully -- catching them requires plumbing a
        return-mechanism out of the policy fn, which is a separate
        change.
        """
        configured = state["policy"]
        if not replica_urls:
            raise ValueError("no replicas configured")
        if len(replica_urls) == 1:
            # Tag explicitly so post-hoc analysis can spot single-replica
            # runs that snuck into a multi-replica comparison instead of
            # silently treating them as if the policy had picked.
            return replica_urls[0], configured, "single-replica"

        pdef: PolicyDef = POLICY_REGISTRY[normalize_policy(configured)]

        if pdef.needs_metrics:
            # Filter to replicas with a live snapshot. Both this code and the
            # background refresh run on the same asyncio thread so there's
            # no race; the dict comprehension just drops missing entries.
            metrics = {u: live_metrics[u] for u in replica_urls if u in live_metrics}
            if len(metrics) < len(replica_urls):
                missing = len(replica_urls) - len(metrics)
                _log(
                    f"live metrics missing for {missing} replica(s); "
                    f"falling back to random for this request"
                )
                return (
                    random.choice(replica_urls),
                    configured,
                    "random-fallback:missing-metrics",
                )
        else:
            metrics = {}

        target = pdef.fn(
            RouteContext(
                replica_urls=replica_urls,
                metrics=metrics,
                endpoints_queued_tokens=endpoints_queued_tokens,
                radix_trie=radix_trie,
                token_ids=token_ids,
                request_tokens=len(token_ids),
                hyperparameters=state["hyperparameters"],
            )
        )
        return target, configured, configured

    # ---------- JSON route handlers ----------
    #
    # Each handler returns ``(status, payload)``; the dispatcher reads
    # the body, calls the handler, and serializes the response. Body
    # parsing / 400-on-invalid-json is handled by ``_dispatch_json`` so
    # individual handlers can focus on validation and side effects.
    #
    # All handlers are nested closures because they read / mutate state
    # captured in this ``proxy()`` invocation -- ``replicas``,
    # ``live_metrics``, ``samples``, ``radix_trie``,
    # ``endpoints_queued_tokens``, ``state``. Lifting them to module
    # scope would require threading a ``ProxyContext`` through every
    # call. The naming convention matches the other request-shaped
    # closures already living here (``_handle_chat_completions``,
    # ``_handle_lifespan``).

    def _policy_payload(*, include_supported: bool) -> dict:
        policy = state["policy"]
        uses_hyperparameters = normalize_policy(policy) == "gorgo"
        payload = {
            "policy": policy,
            "uses_hyperparameters": uses_hyperparameters,
            "hyperparameters": state["hyperparameters"] if uses_hyperparameters else None,
        }
        if include_supported:
            payload["supported"] = sorted(ROUTING_POLICIES)
        return payload

    async def _handle_get_policy(_data) -> tuple[int, dict]:
        return 200, _policy_payload(include_supported=True)

    async def _handle_post_policy(data) -> tuple[int, dict]:
        raw = (data or {}).get("policy") or (data or {}).get("name")
        if not isinstance(raw, str):
            return 400, {"error": "body must include string policy or name"}
        name = normalize_policy(raw)
        if name not in ROUTING_POLICIES:
            return 400, {
                "error": f"unknown policy {raw!r}",
                "supported": sorted(ROUTING_POLICIES),
            }
        state["policy"] = name
        _log(f"routing policy set to {name!r}")
        return 200, _policy_payload(include_supported=False)

    async def _handle_get_replicas(_data) -> tuple[int, dict]:
        # Read-only: never mutate ``replica_urls`` from a GET. Used by
        # controllers (e.g. experiments/policy_matrix_app.py) to poll
        # readiness; if this re-synced from the global modal Dict it
        # would race against concurrent POST /replicas calls from sibling
        # proxies and clobber the per-policy pin -- the bug that made
        # all 9 matrix proxies converge on whichever 3 URLs the last
        # POST landed.
        registry = await _read_registry_async()
        return 200, {
            "replicas": list(replica_urls),
            "count": len(replica_urls),
            "registry": registry,
        }

    async def _handle_post_replicas(data) -> tuple[int, dict]:
        if isinstance(data, list):
            raw = data
        elif isinstance(data, dict):
            raw = data.get("replicas") or data.get("endpoints")
        else:
            raw = None
        if not isinstance(raw, list) or not all(isinstance(u, str) for u in raw):
            return 400, {
                "error": (
                    "body must be a JSON array of endpoint URLs "
                    'or an object like {"replicas": [...]}'
                )
            }

        seen: set[str] = set()
        normalized: list[str] = []
        invalid: list[str] = []
        for u in raw:
            u = u.strip().rstrip("/")
            if not u:
                continue
            if not (u.startswith("http://") or u.startswith("https://")):
                invalid.append(u)
                continue
            if u not in seen:
                seen.add(u)
                normalized.append(u)
        if invalid:
            return 400, {
                "error": "all endpoints must start with http:// or https://",
                "invalid": invalid,
            }

        # Update local state only. Do NOT mutate the global ``replicas``
        # modal Dict here: that Dict is shared across every proxy in the
        # environment, and the matrix experiment runs N proxies in
        # parallel each calling POST /replicas with their own per-policy
        # URLs. Writing back to the shared Dict caused each call to
        # clobber the previous one, and any sibling proxy that
        # subsequently re-synced (via the now-fixed GET /replicas)
        # would adopt whichever URLs the last writer left -- collapsing
        # all N policies onto the same 3 backends.
        added, removed = _sync_replicas_from_manual_urls(normalized)
        registry = await _read_registry_async()

        _log(f"replicas updated: +{len(added)} -{len(removed)} (total={len(replica_urls)})")
        return 200, {
            "replicas": list(replica_urls),
            "count": len(replica_urls),
            "registry": registry,
            "added": added,
            "removed": removed,
        }

    async def _handle_get_trie(_data) -> tuple[int, dict]:
        # Summary stats only -- the full trie is too large to serialize on
        # every request. ``coverage`` counts how many nodes are tagged with
        # each replica URL: a useful sanity check that routing is producing
        # the prefix-sharing shape we expect.
        coverage: dict[str, int] = {url: 0 for url in replica_urls}
        stack = [radix_trie.root]
        tagged_nodes = 0
        while stack:
            node = stack.pop()
            if node.replica_endpoints:
                tagged_nodes += 1
                for url in node.replica_endpoints:
                    coverage[url] = coverage.get(url, 0) + 1
            stack.extend(node.children.values())
        return 200, {
            "num_sequences": radix_trie.num_sequences,
            "total_tokens_inserted": radix_trie.total_tokens_inserted,
            "unique_token_count": radix_trie.unique_token_count(),
            "node_count": radix_trie.node_count(),
            "tagged_node_count": tagged_nodes,
            "replica_coverage": coverage,
        }

    async def _handle_get_replica_metrics(_data) -> tuple[int, dict]:
        now = time.monotonic()
        last = metrics_meta["last_refresh_monotonic"]
        return 200, {
            "refresh_interval_seconds": METRICS_REFRESH_INTERVAL_SECONDS,
            "last_refresh_age_seconds": (now - last) if last else None,
            "errors": metrics_meta["last_refresh_errors"],
            "metrics": {
                url: {
                    "num_running_reqs": m.num_running_reqs,
                    "num_queue_reqs": m.num_queue_reqs,
                    "num_used_tokens": m.num_used_tokens,
                    "latency_seconds": m.latency,
                    "network_rtt_seconds": m.network_rtt or None,
                    "gen_throughput": m.gen_throughput,
                    "utilization": m.utilization,
                }
                for url, m in live_metrics.items()
            },
            "endpoints_queued_tokens": endpoints_queued_tokens,
        }

    async def _handle_get_hyperparameters(_data) -> tuple[int, dict]:
        return 200, {
            "hyperparameters": state["hyperparameters"],
            "allowed_keys": sorted(ALLOWED_HYPERPARAM_KEYS),
            "defaults": dict(DEFAULT_GORGO_HYPERPARAMETERS),
        }

    async def _handle_write_hyperparameters(data, *, replace: bool) -> tuple[int, dict]:
        """Validate + apply a hyperparameter update against the
        structured store (see :mod:`policy.gorgo`).

        Two body shapes are accepted, both validated by
        ``policy.gorgo.validate_update``:

        * **Flat** -- ``{"t_prefill": X, "queued_tokens_weight": Y}``
          updates ``defaults`` only (backward-compat: this is what
          ``proxy/tuning.py`` and ``proxy/calibrate.py`` POST).
        * **Structured** -- ``{"defaults": {...}, "per_target": {url:
          {...}}}`` lets callers write per-replica overrides.

        ``replace=True`` (PUT) resets the store to factory defaults
        before applying the update; ``replace=False`` (POST/PATCH)
        layers the update on top of the existing store.
        """
        update, err = validate_update(data, known_targets=set(replica_urls))
        if err is not None:
            return 400, {
                "error": err,
                "allowed_keys": sorted(ALLOWED_HYPERPARAM_KEYS),
            }
        state["hyperparameters"] = merge_update(
            state["hyperparameters"], update or {}, replace=replace
        )
        _log(f"hyperparameters updated: {state['hyperparameters']}")
        return 200, {"hyperparameters": state["hyperparameters"]}

    async def _handle_post_hyperparameters(data) -> tuple[int, dict]:
        return await _handle_write_hyperparameters(data, replace=False)

    async def _handle_put_hyperparameters(data) -> tuple[int, dict]:
        return await _handle_write_hyperparameters(data, replace=True)

    async def _handle_post_flush(_data) -> tuple[int, dict]:
        radix_trie.clear()
        # Drop tuning samples too: post-flush per-token rates will look
        # different (cold KV cache, fresh queue depths) so mixing them
        # with pre-flush samples would bias the next auto-tune
        # recompute. The auto-tuner config (enabled / window / hop /
        # apply) is intentionally preserved -- only the buffered
        # samples + the per-window counter are reset.
        samples.clear()
        state["total_samples_appended"] = 0
        state["auto_tune"]["samples_since_last_apply"] = 0
        client = state["upstream_client"]
        replica_results: dict[str, dict] = {}
        if client is None:
            for u in replica_urls:
                replica_results[u] = {
                    "ok": False,
                    "error": "upstream client not yet initialized",
                }
        elif replica_urls:
            pairs = await asyncio.gather(
                *[_flush_upstream_replica(client, u) for u in replica_urls],
            )
            replica_results = dict(pairs)
        return 200, {"radix_trie_cleared": True, "replicas": replica_results}

    def _auto_tune_status() -> dict:
        """Snapshot of the live auto-tuner config + diagnostics.
        Shared by ``GET /tune``, ``GET /samples``, and the response of
        every ``POST /tune`` so callers always see what they just
        configured."""
        at = state["auto_tune"]
        return {
            "enabled": at["enabled"],
            "window_size": at["window_size"],
            "hop_size": at["hop_size"],
            "apply": at["apply"],
            "buffered_samples": len(samples),
            "samples_since_last_apply": at["samples_since_last_apply"],
            "samples_until_next_apply": (
                max(0, at["hop_size"] - at["samples_since_last_apply"]) if at["enabled"] else None
            ),
            "applied_count": at["applied_count"],
            "last_applied_at_monotonic": at["last_applied_at_monotonic"],
            "last_recommendation": at["last_recommendation"],
            "enabled_at_monotonic": at["enabled_at_monotonic"],
            "current_policy": state["policy"],
            "current_hyperparameters": state["hyperparameters"],
        }

    async def _handle_get_samples(_data) -> tuple[int, dict]:
        """Visibility into the tuning sample buffer. Returns the most
        recent samples (capped to keep the response small) plus the
        live auto-tuner status."""
        recent = list(samples)[-50:]
        return 200, {
            "buffered_samples": len(samples),
            "max_buffer_size": samples.maxlen,
            "total_samples_appended": state["total_samples_appended"],
            "auto_tune": _auto_tune_status(),
            "recent": recent,
        }

    async def _handle_get_tune(_data) -> tuple[int, dict]:
        """Live auto-tuner status (no side effects)."""
        return 200, {"auto_tune": _auto_tune_status()}

    async def _handle_post_tune(data) -> tuple[int, dict]:
        """Toggle / reconfigure the on-the-fly auto-tuner.

        The auto-tuner is *stateful*: once enabled it keeps
        recomputing ``t_prefill`` / ``queued_tokens_weight`` every
        ``hop_size`` new samples (after the first ``window_size``
        samples have buffered to fill the window) until disabled.
        Each ``POST /tune`` atomically merges the body into the live
        config; only the keys present in the body are touched.

        Body (all optional):
          * ``enabled``:     bool. When omitted, defaults to ``True``
            so a bare ``POST /tune {}`` turns the tuner on with the
            current config. Pass ``{"enabled": false}`` to disable.
          * ``window_size``: int. Trailing-sample window fed to
            ``recommend_hyperparameters``.
          * ``hop_size``:    int (>0). Recompute every N new samples.
            A small ``hop_size`` reacts faster to load shifts; a
            larger one is more stable.
          * ``apply``:       bool. When ``False`` the recommender
            still runs (and ``last_recommendation`` is updated) but
            ``state["hyperparameters"]`` is not mutated -- useful for
            shadow-mode comparison against the current settings.

        Enabling requires ``policy == 'gorgo'`` (the only policy that
        consumes the tuned scalars). Disabling works regardless of
        policy so a misconfigured tuner can always be turned off.
        """
        if not isinstance(data, dict):
            return 400, {"error": "body must be a JSON object"}

        at = state["auto_tune"]

        # Validate first; only mutate after every requested key passes.
        # Otherwise a bad ``hop_size`` after a good ``window_size``
        # would leave the tuner half-configured.
        new_window = at["window_size"]
        new_hop = at["hop_size"]
        new_apply = at["apply"]

        if "window_size" in data:
            try:
                new_window = int(data["window_size"])
            except (TypeError, ValueError):
                return 400, {"error": "window_size must be an integer"}
            if new_window <= 0:
                return 400, {"error": "window_size must be positive"}
        if "hop_size" in data:
            try:
                new_hop = int(data["hop_size"])
            except (TypeError, ValueError):
                return 400, {"error": "hop_size must be an integer"}
            if new_hop <= 0:
                return 400, {"error": "hop_size must be > 0"}
        if "apply" in data:
            new_apply = bool(data["apply"])

        # Default to enabling so a bare POST /tune {} turns it on. To
        # leave the toggle alone (e.g. just adjust window_size while
        # already running) callers can pass the current value back, but
        # in practice the explicit-default keeps the common case terse.
        new_enabled = bool(data.get("enabled", True))

        if new_enabled and normalize_policy(state["policy"]) != "gorgo":
            return 400, {
                "error": ("auto-tuning can only be enabled when the active policy is 'gorgo'"),
                "current_policy": state["policy"],
            }

        was_enabled = at["enabled"]
        at["window_size"] = new_window
        at["hop_size"] = new_hop
        at["apply"] = new_apply
        at["enabled"] = new_enabled

        if new_enabled and not was_enabled:
            # Fresh enable: zero the per-window counter so the first
            # recompute is measured from this moment, not from stale
            # samples that landed while the tuner was off.
            at["samples_since_last_apply"] = 0
            at["enabled_at_monotonic"] = time.monotonic()
            _log(f"auto-tune ENABLED window={new_window} hop={new_hop} apply={new_apply}")
        elif not new_enabled and was_enabled:
            _log("auto-tune DISABLED")
        elif new_enabled:
            # Reconfigured while running -- keep the existing counter
            # so we don't reset the "samples until next apply" clock
            # on every adjustment.
            _log(f"auto-tune RECONFIGURED window={new_window} hop={new_hop} apply={new_apply}")

        # Best-effort summary of the current trailing window if it's
        # already large enough; gives the caller something useful to
        # see immediately even before the next recompute fires.
        preview: dict | None = None
        if samples:
            window = list(samples)[-new_window:]
            preview = {
                "window_size_used": len(window),
                "recommendation": recommend_hyperparameters_per_target(window),
                "stats": summarize_samples(window),
            }

        return 200, {
            "auto_tune": _auto_tune_status(),
            "preview": preview,
        }

    async def _handle_post_trace_start(data) -> tuple[int, dict]:
        nonlocal metrics_trace_events, request_trace_events
        if not isinstance(data, dict):
            return 400, {"error": "body must be a JSON object"}
        tr = state["trace"]
        trace_id = data.get("trace_id")
        if trace_id is not None and not isinstance(trace_id, str):
            return 400, {"error": "trace_id must be a string"}
        try:
            max_events = int(data.get("max_events", tr["max_events"]))
        except (TypeError, ValueError):
            return 400, {"error": "max_events must be an integer"}
        if max_events <= 0:
            return 400, {"error": "max_events must be positive"}

        append = bool(data.get("append", False))
        if not append or max_events != tr["max_events"]:
            metrics_trace_events = deque(maxlen=max_events)
            request_trace_events = deque(maxlen=max_events)

        tr.update(
            {
                "enabled": True,
                "trace_id": trace_id
                or f"trace-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}",
                "sample_metrics": bool(data.get("sample_metrics", True)),
                "sample_requests": bool(data.get("sample_requests", True)),
                "max_events": max_events,
                "started_at": _now_wall_ts(),
                "stopped_at": None,
                "saved_paths": None,
            }
        )
        if not append:
            tr["dropped_metrics"] = 0
            tr["dropped_requests"] = 0
        _log(f"trace started id={tr['trace_id']} max_events={max_events}")
        return 200, {"trace": _trace_status_payload()}

    async def _handle_get_trace_status(_data) -> tuple[int, dict]:
        return 200, {"trace": _trace_status_payload()}

    async def _handle_post_trace_stop(_data) -> tuple[int, dict]:
        state["trace"]["enabled"] = False
        state["trace"]["stopped_at"] = _now_wall_ts()
        _log(f"trace stopped id={state['trace']['trace_id']}")
        return 200, {"trace": _trace_status_payload()}

    async def _handle_post_trace_save(_data) -> tuple[int, dict]:
        paths = _save_trace_to_volume()
        return 200, {"trace": _trace_status_payload(), "paths": paths}

    def _batch_tuning_public_state() -> dict:
        bt = state["batch_tuning"]
        payload = {k: v for k, v in bt.items() if k != "task"}
        payload["history"] = list(bt.get("history") or [])
        return payload

    def _parse_int(data: dict, key: str, default: int) -> int:
        try:
            return int(data.get(key, default))
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be an integer")

    def _parse_float(data: dict, key: str, default: float) -> float:
        try:
            return float(data.get(key, default))
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be a number")

    def _parse_optional_str(data: dict, key: str, default: str = "") -> str:
        value = data.get(key, default)
        if value is None:
            return ""
        if not isinstance(value, str):
            raise ValueError(f"{key} must be a string")
        return value

    def _parse_optional_bool(data: dict, key: str):
        value = data.get(key, None)
        if value is None or value == "":
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            s = value.strip().lower()
            if s in ("1", "true", "yes", "y", "on"):
                return True
            if s in ("0", "false", "no", "n", "off"):
                return False
        raise ValueError(f"{key} must be true/false/null")

    def _normalize_batch_tuning_config(data: dict) -> dict:
        if not isinstance(data, dict):
            raise ValueError("body must be a JSON object")
        metric = _parse_optional_str(data, "metric", "output_throughput")
        if metric not in SCORE_FUNCTIONS:
            raise ValueError(f"unknown metric {metric!r}; choices: {sorted(SCORE_FUNCTIONS)}")
        algorithm = _parse_optional_str(data, "algorithm", "hill-climb")
        if algorithm not in ("hill-climb", "gaussian-es"):
            raise ValueError("algorithm must be 'hill-climb' or 'gaussian-es'")
        ranges = validated_ranges(
            {
                "t_prefill": (
                    _parse_float(data, "t_prefill_min", 1e-5),
                    _parse_float(data, "t_prefill_max", 1.0),
                ),
                "queued_tokens_weight": (
                    _parse_float(data, "queued_tokens_weight_min", 1e-5),
                    _parse_float(data, "queued_tokens_weight_max", 1.0),
                ),
            }
        )
        return {
            "source": _parse_optional_str(data, "source", "glm5") or "glm5",
            "data_path": _parse_optional_str(data, "data_path", ""),
            "arrival_mode": _parse_optional_str(data, "arrival_mode", "bounded") or "bounded",
            "time_scale": _parse_float(data, "time_scale", 1.0),
            "start_time": _parse_optional_str(data, "start_time"),
            "end_time": _parse_optional_str(data, "end_time"),
            "offset": _parse_int(data, "offset", 0),
            "num_requests": (_parse_int(data, "num_requests", 0) or None),
            "concurrency": _parse_int(data, "concurrency", 16),
            "model": _parse_optional_str(data, "model", ""),
            "stream": _parse_optional_bool(data, "stream"),
            "max_tokens": (_parse_int(data, "max_tokens", 0) or None),
            "max_input_tokens": _parse_int(data, "max_input_tokens", 0),
            "metric": metric,
            "algorithm": algorithm,
            "max_steps": _parse_int(data, "max_steps", 16),
            "initial_step": _parse_float(data, "initial_step", 0.5),
            "min_step": _parse_float(data, "min_step", 0.05),
            "sigma": _parse_float(data, "sigma", 0.5),
            "sigma_min": _parse_float(data, "sigma_min", 0.02),
            "seed": (_parse_int(data, "seed", -1)),
            "relative_tolerance": _parse_float(data, "relative_tolerance", 0.005),
            "output_dir": _parse_optional_str(data, "output_dir", ""),
            "ranges": ranges,
        }

    async def _run_batch_tuning(run_id: str, config: dict) -> None:
        from proxy.workload_core import DEFAULT_MODEL, run_replay_async

        bt = state["batch_tuning"]
        run_started_at = datetime.now(timezone.utc)
        out_dir = (
            config["output_dir"]
            if config["output_dir"] and os.path.isabs(config["output_dir"])
            else os.path.join(
                "/results",
                config["output_dir"] or f"tune_{run_started_at.strftime('%Y%m%d_%H%M%S')}",
            )
        )
        os.makedirs(out_dir, exist_ok=True)
        summary_path = os.path.join(out_dir, "summary.json")
        bt["output_path"] = summary_path

        workload_kwargs = {
            "source": config["source"],
            "data_path": config["data_path"] or None,
            "start_time": config["start_time"] or None,
            "end_time": config["end_time"] or None,
            "offset": config["offset"],
            "num_requests": config["num_requests"],
            "concurrency": config["concurrency"],
            "model": config["model"] or DEFAULT_MODEL,
            "stream": config["stream"],
            "max_tokens": config["max_tokens"],
            "max_input_tokens": config["max_input_tokens"],
            "arrival_mode": config["arrival_mode"],
            "time_scale": config["time_scale"],
            "save_per_request": False,
        }
        metric = config["metric"]
        score_fn = SCORE_FUNCTIONS[metric]
        active_policy = state["policy"]
        current_hp = dict((state["hyperparameters"] or {}).get("defaults") or {})
        tuner = build_tuner(
            config["algorithm"],
            initial_params=current_hp,
            ranges=config["ranges"],
            max_steps=config["max_steps"],
            relative_tolerance=config["relative_tolerance"],
            initial_step=config["initial_step"],
            min_step=config["min_step"],
            sigma=config["sigma"],
            sigma_min=config["sigma_min"],
            seed=None if config["seed"] < 0 else config["seed"],
        )
        history: list[dict] = []
        baseline_score: float | None = None

        def persist_summary(finished_at: datetime | None = None) -> None:
            summary = build_summary(
                run_started_at=run_started_at,
                proxy_url="http://127.0.0.1:8000",
                workload_kwargs=workload_kwargs,
                metric=metric,
                tuner=tuner,
                baseline_score=baseline_score,
                history=history,
                active_policy=active_policy,
                ranges=config["ranges"],
                finished_at=finished_at,
                output_path=summary_path,
            )
            summary["trace"] = _trace_status_payload()
            tmp_path = f"{summary_path}.tmp"
            with open(tmp_path, "w") as f:
                json.dump(summary, f, indent=2)
            os.replace(tmp_path, summary_path)
            bench_results_volume.commit()

        try:
            _log(
                f"batch tuning {run_id} started algorithm={tuner.name} "
                f"metric={metric} max_steps={config['max_steps']}"
            )
            trial_idx = 0
            while True:
                candidate = tuner.propose()
                if candidate is None:
                    break
                if bt.get("stop_requested"):
                    raise asyncio.CancelledError()
                is_baseline = tuner.best_score is None
                bt["trial"] = trial_idx
                _log(f"batch tuning {run_id} trial {trial_idx}: evaluating {candidate}")

                status, payload = await _handle_post_flush({})
                if status >= 400:
                    raise RuntimeError(f"flush: {payload}")
                status, payload = await _handle_post_hyperparameters(candidate)
                if status >= 400:
                    raise RuntimeError(f"hyperparameters: {payload}")

                trial_output = os.path.join(out_dir, f"replay_trial_{trial_idx:03d}.json")
                stats = await run_replay_async(
                    proxy_url="http://127.0.0.1:8000",
                    output_path=trial_output,
                    run_id=f"{run_id}-trial-{trial_idx:03d}",
                    **workload_kwargs,
                )
                score = score_fn(stats)
                accepted = tuner.report(candidate, score)
                tuner_state = dict(tuner.state)
                if is_baseline:
                    baseline_score = score
                    _log(f"batch tuning {run_id} trial {trial_idx}: baseline score={score:.4f}")
                else:
                    tag = "ACCEPTED" if accepted else "rejected"
                    _log(
                        f"batch tuning {run_id} trial {trial_idx}: "
                        f"score={score:.4f} {tag} incumbent={tuner.best_score:.4f}"
                    )
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
                        "stats": stats,
                    }
                )
                bt["history"] = history
                bt["best_params"] = dict(tuner.best_params)
                bt["best_score"] = tuner.best_score
                bt["baseline_score"] = baseline_score
                persist_summary()
                trial_idx += 1

            if tuner.best_params:
                status, payload = await _handle_post_hyperparameters(tuner.best_params)
                if status >= 400:
                    raise RuntimeError(f"final hyperparameters: {payload}")
                status, payload = await _handle_post_flush({})
                if status >= 400:
                    raise RuntimeError(f"final flush: {payload}")

            finished_at = datetime.now(timezone.utc)
            persist_summary(finished_at=finished_at)
            bt["status"] = "succeeded"
            bt["finished_at"] = finished_at.isoformat().replace("+00:00", "Z")
            _log(f"batch tuning {run_id} succeeded output={summary_path}")
        except asyncio.CancelledError:
            finished_at = datetime.now(timezone.utc)
            bt["status"] = "cancelled"
            bt["finished_at"] = finished_at.isoformat().replace("+00:00", "Z")
            bt["error"] = "cancelled"
            try:
                persist_summary(finished_at=finished_at)
            except Exception as e:
                _log(f"batch tuning {run_id} summary persist after cancel failed: {e}")
            _log(f"batch tuning {run_id} cancelled")
        except Exception as e:
            finished_at = datetime.now(timezone.utc)
            bt["status"] = "failed"
            bt["finished_at"] = finished_at.isoformat().replace("+00:00", "Z")
            bt["error"] = f"{type(e).__name__}: {e}"
            _log(f"batch tuning {run_id} failed: {bt['error']}")
        finally:
            bt["running"] = False
            bt["task"] = None
            bt["stop_requested"] = False

    async def _handle_post_tuning_start(data) -> tuple[int, dict]:
        if not isinstance(data, dict):
            return 400, {"error": "body must be a JSON object"}
        bt = state["batch_tuning"]
        task = bt.get("task")
        if task is not None and not task.done():
            return 409, {
                "error": "batch tuning already running",
                "batch_tuning": _batch_tuning_public_state(),
            }
        if normalize_policy(state["policy"]) != "gorgo":
            return 400, {
                "error": "batch tuning requires active policy 'gorgo'",
                "current_policy": state["policy"],
            }
        if state["auto_tune"]["enabled"]:
            return 409, {"error": "disable live auto-tune before starting batch tuning"}
        if not replica_urls:
            return 400, {"error": "no replicas registered"}
        try:
            config = _normalize_batch_tuning_config(data)
        except ValueError as e:
            return 400, {"error": str(e)}

        run_id = (
            f"tune-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        )
        bt.update(
            {
                "running": True,
                "status": "running",
                "run_id": run_id,
                "started_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "finished_at": None,
                "config": config,
                "trial": None,
                "history": [],
                "best_params": None,
                "best_score": None,
                "baseline_score": None,
                "output_path": None,
                "error": None,
                "stop_requested": False,
            }
        )
        task = asyncio.create_task(_run_batch_tuning(run_id, config))
        bt["task"] = task
        return 202, {
            "run_id": run_id,
            "status": "running",
            "status_url": "/tuning/status",
            "batch_tuning": _batch_tuning_public_state(),
        }

    async def _handle_get_tuning_status(_data) -> tuple[int, dict]:
        return 200, {"batch_tuning": _batch_tuning_public_state()}

    async def _handle_post_tuning_stop(_data) -> tuple[int, dict]:
        bt = state["batch_tuning"]
        task = bt.get("task")
        if task is None or task.done():
            return 200, {"batch_tuning": _batch_tuning_public_state()}
        bt["stop_requested"] = True
        task.cancel()
        return 202, {"batch_tuning": _batch_tuning_public_state()}

    def _workload_public_state() -> dict:
        wr = state["workload_run"]
        return {k: v for k, v in wr.items() if k != "task"}

    def _normalize_workload_config(data: dict) -> dict:
        if not isinstance(data, dict):
            raise ValueError("body must be a JSON object")
        data_path = data.get("data_path") or data.get("trace_path")
        if not isinstance(data_path, str) or not data_path:
            raise ValueError("data_path or trace_path is required")
        return {
            "data_path": data_path,
            "run_id": _parse_optional_str(data, "run_id", ""),
            "concurrency": _parse_int(data, "concurrency", 16),
            "model": _parse_optional_str(data, "model", ""),
            "stream": _parse_optional_bool(data, "stream"),
            "max_tokens": (_parse_int(data, "max_tokens", 0) or None),
            "max_input_tokens": _parse_int(data, "max_input_tokens", 0),
            "arrival_mode": _parse_optional_str(data, "arrival_mode", "open-loop"),
            "time_scale": _parse_float(data, "time_scale", 1.0),
            "output_path": _parse_optional_str(data, "output_path", ""),
            "save_per_request": bool(data.get("save_per_request", True)),
            "start_at_wall_time": _parse_optional_str(data, "start_at_wall_time", ""),
        }

    async def _run_workload(run_id: str, config: dict) -> None:
        from proxy.workload_core import DEFAULT_MODEL, run_replay_async

        wr = state["workload_run"]
        try:
            start_at = config.get("start_at_wall_time")
            if start_at:
                target = datetime.fromisoformat(start_at.replace("Z", "+00:00"))
                if target.tzinfo is None:
                    target = target.replace(tzinfo=timezone.utc)
                now = datetime.now(timezone.utc)
                delay = (target.astimezone(timezone.utc) - now).total_seconds()
                if delay > 0:
                    wr["status"] = "scheduled"
                    wr["phase"] = "waiting-start-at-wall-time"
                    _log(f"workload {run_id} scheduled for {start_at} ({delay:.3f}s)")
                    await asyncio.sleep(delay)
                    wr["status"] = "running"
                    wr["phase"] = "starting"
            _log(f"workload {run_id} started path={config['data_path']}")
            output_path = config["output_path"] or f"/results/workload_runs/{run_id}.json"
            wr["phase"] = "replay-running"
            stats = await run_replay_async(
                proxy_url="http://127.0.0.1:8000",
                source="mooncake",
                data_path=config["data_path"],
                concurrency=config["concurrency"],
                model=config["model"] or DEFAULT_MODEL,
                stream=config["stream"],
                max_tokens=config["max_tokens"],
                max_input_tokens=config["max_input_tokens"],
                output_path=output_path,
                save_per_request=config["save_per_request"],
                run_id=run_id,
                arrival_mode=config["arrival_mode"],
                time_scale=config["time_scale"],
            )
            finished_at = datetime.now(timezone.utc)
            wr["status"] = "succeeded"
            wr["phase"] = "done"
            wr["finished_at"] = finished_at.isoformat().replace("+00:00", "Z")
            wr["stats"] = stats
            wr["output_path"] = stats.get("output_path")
            _log(f"workload {run_id} succeeded output={wr['output_path']}")
        except asyncio.CancelledError:
            finished_at = datetime.now(timezone.utc)
            wr["status"] = "cancelled"
            wr["phase"] = "cancelled"
            wr["finished_at"] = finished_at.isoformat().replace("+00:00", "Z")
            wr["error"] = "cancelled"
            _log(f"workload {run_id} cancelled")
        except Exception as e:
            finished_at = datetime.now(timezone.utc)
            wr["status"] = "failed"
            wr["phase"] = "failed"
            wr["finished_at"] = finished_at.isoformat().replace("+00:00", "Z")
            wr["error"] = f"{type(e).__name__}: {e}"
            _log(f"workload {run_id} failed: {wr['error']}")
        finally:
            wr["running"] = False
            wr["task"] = None
            wr["stop_requested"] = False

    async def _handle_post_workload_start(data) -> tuple[int, dict]:
        wr = state["workload_run"]
        task = wr.get("task")
        if task is not None and not task.done():
            return 409, {"error": "workload already running", "workload": _workload_public_state()}
        try:
            config = _normalize_workload_config(data)
        except ValueError as e:
            return 400, {"error": str(e)}
        run_id = (
            config["run_id"]
            or f"workload-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        )
        wr.update(
            {
                "running": True,
                "status": "running",
                "phase": "starting",
                "run_id": run_id,
                "started_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "finished_at": None,
                "config": config,
                "stats": None,
                "output_path": None,
                "error": None,
                "stop_requested": False,
            }
        )
        task = asyncio.create_task(_run_workload(run_id, config))
        wr["task"] = task
        return 202, {"run_id": run_id, "status": "running", "workload": _workload_public_state()}

    async def _handle_get_workload_status(_data) -> tuple[int, dict]:
        return 200, {"workload": _workload_public_state()}

    async def _handle_post_workload_stop(_data) -> tuple[int, dict]:
        wr = state["workload_run"]
        task = wr.get("task")
        if task is None or task.done():
            return 200, {"workload": _workload_public_state()}
        wr["stop_requested"] = True
        task.cancel()
        return 202, {"workload": _workload_public_state()}

    # JSON route table. The dispatcher distinguishes 405 (method exists
    # for path but not this verb) from 404 (path unknown) by membership
    # checks, so adding a new (method, path) entry here is the only edit
    # required to expose a new JSON endpoint.
    json_routes: dict[tuple[str, str], object] = {
        ("GET", "/policy"): _handle_get_policy,
        ("POST", "/policy"): _handle_post_policy,
        ("GET", "/replicas"): _handle_get_replicas,
        ("POST", "/replicas"): _handle_post_replicas,
        ("GET", "/trie"): _handle_get_trie,
        ("GET", "/replica_metrics"): _handle_get_replica_metrics,
        ("GET", "/hyperparameters"): _handle_get_hyperparameters,
        ("POST", "/hyperparameters"): _handle_post_hyperparameters,
        ("PATCH", "/hyperparameters"): _handle_post_hyperparameters,
        ("PUT", "/hyperparameters"): _handle_put_hyperparameters,
        ("POST", "/flush"): _handle_post_flush,
        ("GET", "/samples"): _handle_get_samples,
        ("GET", "/tune"): _handle_get_tune,
        ("POST", "/tune"): _handle_post_tune,
        ("POST", "/trace/start"): _handle_post_trace_start,
        ("GET", "/trace/status"): _handle_get_trace_status,
        ("POST", "/trace/stop"): _handle_post_trace_stop,
        ("POST", "/trace/save"): _handle_post_trace_save,
        ("POST", "/tuning/start"): _handle_post_tuning_start,
        ("GET", "/tuning/status"): _handle_get_tuning_status,
        ("POST", "/tuning/stop"): _handle_post_tuning_stop,
        ("POST", "/workload/start"): _handle_post_workload_start,
        ("GET", "/workload/status"): _handle_get_workload_status,
        ("POST", "/workload/stop"): _handle_post_workload_stop,
    }

    async def _dispatch_json(method: str, path: str, receive, send) -> bool:
        handler = json_routes.get((method, path))
        if handler is None:
            if any(p == path for (_, p) in json_routes):
                await _send_json(send, 405, {"error": "method not allowed"})
                return True
            return False  # let caller emit 404

        # Body-bearing methods may include a JSON payload; GET callers
        # pass an empty dict so handlers don't have to special-case.
        if method in ("POST", "PUT", "PATCH"):
            try:
                data = await _read_json_body(receive)
            except json.JSONDecodeError:
                await _send_json(send, 400, {"error": "invalid JSON body"})
                return True
        else:
            data = {}

        status, payload = await handler(data)
        await _send_json(send, status, payload)
        return True

    # ---------- Chat completions (streaming passthrough + tuning tap) ----------

    def _record_request_sample(
        *,
        target: str,
        ttft_ns: int | None,
        total_ns: int,
        prompt_tokens: int | None,
        completion_tokens: int | None,
    ) -> None:
        """Append a per-request sample shaped like
        :func:`proxy.measure.measure_chat_completion`'s output. Skipped
        silently when token counts are missing (calibrate-style safeguard
        against corrupted per-token rates).

        ``ping_seconds`` is the irreducible RTT subtracted from TTFT
        before fitting the prefill rate. Source order:
          1. ``snap.network_rtt`` (EWMA-smoothed dedicated probe of the
             replica's base URL; preferred -- isolates pure network RTT
             from SGLang's /metrics handler load).
          2. ``snap.latency`` (scrape RTT; fallback when the probe
             hasn't completed a successful round-trip yet, e.g. cold
             start). Inflates under load, hence only a fallback.
          3. ``0.0`` (no snapshot at all -- prefill rate becomes a
             slight overestimate rather than negative).
        Same role ``ping_once`` plays in ``proxy/calibrate.py``.
        """
        if (
            ttft_ns is None
            or prompt_tokens is None
            or prompt_tokens <= 0
            or completion_tokens is None
            or completion_tokens <= 0
        ):
            return

        snap = live_metrics.get(target)
        if snap is None:
            ping_rtt_s = 0.0
        elif snap.network_rtt > 0.0:
            ping_rtt_s = snap.network_rtt
        else:
            ping_rtt_s = snap.latency

        ttft_s = ttft_ns / NS_PER_S
        total_s = total_ns / NS_PER_S
        prefill_s = max(ttft_s - ping_rtt_s, 0.0)
        decode_s = max(total_s - ttft_s, 0.0)

        samples.append(
            {
                "ping_seconds": ping_rtt_s,
                "ttft_seconds": ttft_s,
                "total_seconds": total_s,
                "prefill_seconds": prefill_s,
                "decode_seconds": decode_s,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "prefill_rate_seconds_per_token": prefill_s / prompt_tokens,
                "decode_rate_seconds_per_token": decode_s / completion_tokens,
                "target": target,
                "recorded_at_monotonic": time.monotonic(),
            }
        )
        state["total_samples_appended"] += 1

        # Auto-tune hook: only fire on the same code path that produced
        # the sample so the recompute is naturally serialized with
        # appends (single asyncio thread, no lock needed). The fast path
        # is a single bool check when the tuner is disabled.
        at = state["auto_tune"]
        if not at["enabled"]:
            return
        at["samples_since_last_apply"] += 1
        if at["samples_since_last_apply"] < at["hop_size"]:
            return
        if len(samples) < at["window_size"]:
            # Don't fire until the buffer holds at least one full window;
            # earlier recomputes would over-weight whichever short prefix
            # of the run happens to have landed first.
            return
        if normalize_policy(state["policy"]) != "gorgo":
            # Auto-tune only writes ``t_prefill`` / ``queued_tokens_weight``;
            # under any other policy the writes are inert. Skip the work
            # and keep the counter pinned so a switch back to gorgo
            # immediately resumes recomputing on the next sample.
            at["samples_since_last_apply"] = at["hop_size"]
            return

        window = list(samples)[-at["window_size"] :]
        # Per-target recommendation: pooled ``defaults`` for unseen
        # replicas, plus per-replica overrides for any replica with
        # at least ``min_samples_per_target`` observations in the
        # window. Replicas that fall below the threshold simply
        # inherit ``defaults`` instead of getting a noisy single-
        # sample median.
        recommendation = recommend_hyperparameters_per_target(window)
        if at["apply"]:
            # Use the same merge primitive as ``POST /hyperparameters``
            # so layering rules stay identical between manual writes
            # and auto-tune writes (key-level merge: per-target keys
            # not in the recommendation are preserved).
            state["hyperparameters"] = merge_update(
                state["hyperparameters"], recommendation, replace=False
            )
        at["applied_count"] += 1
        at["last_applied_at_monotonic"] = time.monotonic()
        at["last_recommendation"] = recommendation
        at["samples_since_last_apply"] = 0
        _log(
            f"auto-tune #{at['applied_count']} "
            f"window={len(window)} defaults={recommendation['defaults']} "
            f"per_target={list(recommendation['per_target'])} "
            f"(apply={at['apply']})"
        )

    async def _handle_chat_completions(scope, receive, send) -> None:
        headers = {
            k.decode("latin1").lower(): v.decode("latin1") for k, v in (scope.get("headers") or [])
        }
        request_id = headers.get("x-gorgo-request-id") or f"proxy-{uuid.uuid4().hex}"
        try:
            data = await _read_json_body(receive)
        except json.JSONDecodeError:
            await _send_json(send, 400, {"error": "invalid JSON body"})
            return
        if not isinstance(data, dict):
            await _send_json(send, 400, {"error": "body must be a JSON object"})
            return

        messages = data.get("messages", [])
        if not isinstance(messages, list):
            await _send_json(send, 400, {"error": "'messages' must be a list"})
            return

        token_ids = tokenize_input(messages)
        request_tokens = len(token_ids)
        decision_monotonic = time.monotonic()
        decision_wall_ts = _now_wall_ts()
        metrics_seq_at_decision = metrics_meta.get("refresh_seq", 0)
        last_refresh = metrics_meta.get("last_refresh_monotonic") or 0.0
        metrics_age_seconds = (decision_monotonic - last_refresh) if last_refresh else None
        cached_by_replica = (
            radix_trie.cached_prefix_lengths(token_ids, replica_urls)
            if token_ids and state["trace"]["enabled"] and state["trace"]["sample_requests"]
            else {}
        )
        candidate_snapshot = None
        if state["trace"]["enabled"] and state["trace"]["sample_requests"]:
            candidate_snapshot = {}
            for u in replica_urls:
                snap = live_metrics.get(u)
                candidate_snapshot[u] = {
                    "latency_seconds": snap.latency if snap else None,
                    "num_running_reqs": snap.num_running_reqs if snap else None,
                    "num_queue_reqs": snap.num_queue_reqs if snap else None,
                    "num_used_tokens": snap.num_used_tokens if snap else None,
                    "utilization": snap.utilization if snap else None,
                    "gen_throughput": snap.gen_throughput if snap else None,
                    "queued_tokens": endpoints_queued_tokens.get(u, 0),
                    "cached_prefix_tokens": cached_by_replica.get(u, 0),
                }

        try:
            target, configured_policy, effective_policy = _select_endpoint(token_ids)
        except Exception as e:
            configured_policy = state["policy"]
            _log(f"policy {configured_policy!r} failed ({e}); falling back to random")
            if not replica_urls:
                await _send_json(send, 503, {"error": "no replicas registered"})
                return
            target = random.choice(replica_urls)
            # Include the configured policy name in the sentinel so the
            # same exception class raised from different policies is
            # distinguishable in the trace (e.g. ``KeyError`` from gorgo
            # vs prefix-cache). Don't include ``str(e)`` -- exception
            # messages can carry token ids and would explode group-by
            # cardinality in downstream analysis.
            effective_policy = f"random-fallback:exception:{configured_policy}:{type(e).__name__}"

        request_trace_event = {
            "kind": "request",
            "trace_id": state["trace"]["trace_id"],
            "request_id": request_id,
            "wall_ts": decision_wall_ts,
            "monotonic_s": decision_monotonic,
            # ``policy`` is the configured policy at decision time --
            # passed back from ``_select_endpoint`` (or captured locally
            # in the exception path) rather than re-read from
            # ``state["policy"]`` so a concurrent ``POST /policy``
            # mid-request can't produce a row whose ``policy`` and
            # ``effective_policy`` disagree just from a race.
            # ``effective_policy`` is what actually selected the target:
            # equal to ``policy`` when the policy fn ran, or
            # ``random-fallback:<reason>`` / ``single-replica`` when
            # ``_select_endpoint`` short-circuited. Filter on
            # ``effective_policy == policy`` to drop fallback rows from
            # per-policy aggregates.
            "policy": configured_policy,
            "effective_policy": effective_policy,
            "target": target,
            "request_tokens": request_tokens,
            "metrics_seq_at_decision": metrics_seq_at_decision,
            "metrics_age_seconds": metrics_age_seconds,
            "candidate_snapshot": candidate_snapshot,
            "status": None,
            "ttft_ns": None,
            "total_ns": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "error": None,
        }

        endpoints_queued_tokens[target] = endpoints_queued_tokens.get(target, 0) + request_tokens
        client = state["upstream_client"]
        if client is None:
            # Race between startup and the first inbound request; defensive.
            await _send_json(send, 503, {"error": "upstream client not yet initialized"})
            request_trace_event.update(
                {
                    "status": 503,
                    "total_ns": 0,
                    "error": "upstream client not yet initialized",
                }
            )
            _trace_append("request", request_trace_event)
            if target in endpoints_queued_tokens:
                endpoints_queued_tokens[target] = max(
                    0, endpoints_queued_tokens[target] - request_tokens
                )
            return

        # Serialize the original parsed JSON exactly once for the upstream
        # request. We forward bytes directly instead of letting httpx do
        # its own json= serialization to avoid a second round-trip through
        # the encoder. ``accept-encoding: identity`` tells the upstream
        # not to compress so we don't have to juggle ``content-encoding``
        # on the way out to the client.
        if data.get("stream") is True:
            stream_options = data.get("stream_options")
            if not isinstance(stream_options, dict):
                stream_options = {}
                data["stream_options"] = stream_options
            stream_options.setdefault("include_usage", True)
        upstream_body = json.dumps(data).encode()
        upstream_headers = {
            "accept-encoding": "identity",
            "content-type": "application/json",
        }
        headers_sent = False
        upstream_status = None
        ttft_ns = None
        total_ns = None
        prompt_tokens = None
        completion_tokens = None
        output_tokens = 0
        # Captured before client.stream(...) so TTFT measured by the SSE
        # tee includes request-send + response-headers latency, matching
        # the contract in proxy/measure.py::consume_sse_stream.
        request_start_ns = time.perf_counter_ns()
        try:
            async with client.stream(
                "POST",
                f"{target}/v1/chat/completions",
                content=upstream_body,
                headers=upstream_headers,
            ) as upstream:
                upstream_status = upstream.status_code
                # At this point the request body has been sent upstream
                # and the response headers have come back, so the replica
                # has ingested the prompt and (at minimum) started
                # prefill. Record the prefix on the trie so concurrent
                # requests arriving during our streaming phase can see
                # that ``target`` now caches this prefix.
                #
                # Only tag successful dispatches -- a 4xx/5xx likely
                # means the replica never got far enough to cache
                # anything meaningful.
                if token_ids and 200 <= upstream.status_code < 300:
                    try:
                        radix_trie.insert(token_ids, endpoint=target)
                    except Exception as e:
                        # Trie bookkeeping must never break forwarding.
                        _log(f"radix trie insert failed: {e}")

                response_headers = [
                    (k.lower().encode(), v.encode())
                    for k, v in upstream.headers.items()
                    if k.lower() not in _HOP_BY_HOP_HEADERS
                ]
                await send(
                    {
                        "type": "http.response.start",
                        "status": upstream.status_code,
                        "headers": response_headers,
                    }
                )
                headers_sent = True

                is_sse = upstream.headers.get("content-type", "").startswith("text/event-stream")

                async def _sink(chunk: bytes) -> None:
                    """Forward an upstream byte chunk to the ASGI client.
                    Used as the ``chunk_sink`` for ``consume_sse_stream``
                    so passthrough latency is unaffected by the parse loop."""
                    await send(
                        {
                            "type": "http.response.body",
                            "body": chunk,
                            "more_body": True,
                        }
                    )

                if is_sse and 200 <= upstream.status_code < 300:
                    # Tee path: parse SSE for tuning samples while the
                    # raw bytes flow straight through to the client.
                    (
                        ttft_ns,
                        output_tokens,
                        prompt_tokens,
                        completion_tokens,
                    ) = await consume_sse_stream(
                        upstream,
                        request_start_ns=request_start_ns,
                        chunk_sink=_sink,
                    )
                    total_ns = time.perf_counter_ns() - request_start_ns
                    _record_request_sample(
                        target=target,
                        ttft_ns=ttft_ns,
                        total_ns=total_ns,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=(
                            completion_tokens if completion_tokens is not None else output_tokens
                        ),
                    )
                else:
                    # Plain passthrough for non-SSE bodies (e.g. error
                    # JSON, or a caller that asked for stream=False).
                    async for chunk in upstream.aiter_raw():
                        if chunk:
                            await _sink(chunk)
                    total_ns = time.perf_counter_ns() - request_start_ns
                await send({"type": "http.response.body", "body": b"", "more_body": False})
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            request_trace_event["error"] = f"{type(e).__name__}: {e}"
            if not headers_sent:
                await _send_json(send, 502, {"error": f"upstream replica unreachable: {e}"})
                upstream_status = 502
        except httpx.HTTPError as e:
            request_trace_event["error"] = f"{type(e).__name__}: {e}"
            if headers_sent:
                # Already committed to a status; best we can do is close
                # the body cleanly so the client sees a truncated stream.
                _log(f"upstream stream error mid-response: {e}")
                try:
                    await send({"type": "http.response.body", "body": b"", "more_body": False})
                except Exception:
                    pass
            else:
                await _send_json(send, 502, {"error": f"upstream stream error: {e}"})
                upstream_status = 502
        finally:
            request_trace_event.update(
                {
                    "status": upstream_status,
                    "ttft_ns": ttft_ns,
                    "total_ns": total_ns
                    if total_ns is not None
                    else time.perf_counter_ns() - request_start_ns,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": (
                        completion_tokens if completion_tokens is not None else output_tokens
                    ),
                }
            )
            _trace_append("request", request_trace_event)
            # Only decrement if the replica is still registered; if it was
            # removed mid-request via /replicas POST, don't leak its key
            # back into the queue-tokens dict.
            if target in endpoints_queued_tokens:
                endpoints_queued_tokens[target] = max(
                    0, endpoints_queued_tokens[target] - request_tokens
                )

    # ---------- Lifespan ----------

    async def _handle_lifespan(receive, send) -> None:
        while True:
            msg = await receive()
            if msg["type"] == "lifespan.startup":
                state["upstream_client"] = _new_upstream_client()
                # Prime synchronously so the first inbound request doesn't
                # see empty live_metrics and fall back to random.
                try:
                    await _refresh_all(state["upstream_client"])
                except Exception as e:
                    _log(f"initial metrics refresh failed: {e}")
                state["metrics_task"] = asyncio.create_task(_metrics_refresh_loop())
                _log(
                    f"metrics refresh loop started "
                    f"(interval={METRICS_REFRESH_INTERVAL_SECONDS}s, "
                    f"{len(replica_urls)} replicas); "
                    f"upstream client: http2=True, "
                    f"max_connections={upstream_limits.max_connections}, "
                    f"max_keepalive_connections={upstream_limits.max_keepalive_connections}"
                )
                await send({"type": "lifespan.startup.complete"})
            elif msg["type"] == "lifespan.shutdown":
                batch_task = state["batch_tuning"].get("task")
                if batch_task is not None:
                    state["batch_tuning"]["stop_requested"] = True
                    batch_task.cancel()
                    try:
                        await batch_task
                    except (asyncio.CancelledError, Exception):
                        pass
                workload_task = state["workload_run"].get("task")
                if workload_task is not None:
                    state["workload_run"]["stop_requested"] = True
                    workload_task.cancel()
                    try:
                        await workload_task
                    except (asyncio.CancelledError, Exception):
                        pass
                task = state.get("metrics_task")
                if task is not None:
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass
                client = state.get("upstream_client")
                if client is not None:
                    try:
                        await client.aclose()
                    except Exception as e:
                        _log(f"upstream client close failed: {e}")
                    state["upstream_client"] = None
                await send({"type": "lifespan.shutdown.complete"})
                return

    # ---------- ASGI entry point ----------

    async def asgi_app(scope, receive, send):
        if scope["type"] == "lifespan":
            await _handle_lifespan(receive, send)
            return
        if scope["type"] != "http":
            return

        path = scope["path"]
        method = scope["method"]

        if path == "/v1/chat/completions" and method == "POST":
            await _handle_chat_completions(scope, receive, send)
            return

        if await _dispatch_json(method, path, receive, send):
            return

        await send(
            {
                "type": "http.response.start",
                "status": 404,
                "headers": [(b"content-type", b"text/plain")],
            }
        )
        await send({"type": "http.response.body", "body": b"Not found"})

    with modal.forward(8000) as tunnel:
        _log(f"proxy listening at {tunnel.url}")
        if registry_key:
            proxies[registry_key] = tunnel.url
            _log(f"proxy registered in modal dict key={registry_key!r}")
        # TODO: replace uvicorn with a faster reverse-proxy (e.g. nginx, envoy, or Rust-based)
        uvicorn.run(
            asgi_app,
            host="0.0.0.0",
            port=8000,
            log_config=_UVICORN_LOG_CONFIG,
        )
        if registry_key and proxies.get(registry_key) == tunnel.url:
            proxies[registry_key] = ""
