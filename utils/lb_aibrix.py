"""Load-balancing policies aligned with names in Aibrix
``pkg/plugins/gateway/algorithms`` (vLLM / aibrix), adapted to metrics the GORGO
proxy scrapes from SGLang ``/metrics``.

Policies are selected by kebab-case names (e.g. ``least-request``). Underscores
in ``/policy`` POST bodies are normalized to kebab-case.

Unavailable in this standalone proxy (no K8s cache / Redis / SLO queue) are
approximated: see docstrings on each function.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from utils.radix_trie import RadixTrie


def normalize_policy(name: str) -> str:
    return name.strip().replace("_", "-").lower()


class ReplicaSnapshot:
    """Metrics per replica from one /metrics scrape (SGLang)."""

    __slots__ = (
        "num_running_reqs",
        "num_queue_reqs",
        "num_used_tokens",
        "latency",
        "gen_throughput",
        "utilization",
    )

    def __init__(
        self,
        *,
        num_running_reqs: int,
        num_queue_reqs: int,
        num_used_tokens: int,
        latency: float,
        gen_throughput: float = 0.0,
        utilization: float = 0.0,
    ):
        self.num_running_reqs = num_running_reqs
        self.num_queue_reqs = num_queue_reqs
        self.num_used_tokens = num_used_tokens
        self.latency = latency
        self.gen_throughput = gen_throughput
        self.utilization = utilization

    def combined_load(self, queued_prompt_tokens: int, used_weight: float = 1.0) -> float:
        return (
            self.num_running_reqs
            + self.num_queue_reqs
            + used_weight * self.num_used_tokens
            + queued_prompt_tokens
        )


def _tie_break_min(candidates: list[str], score: Callable[[str], float]) -> str:
    best = min(score(u) for u in candidates)
    tied = [u for u in candidates if score(u) == best]
    return random.choice(tied)


def _tie_break_max(candidates: list[str], score: Callable[[str], float]) -> str:
    best = max(score(u) for u in candidates)
    tied = [u for u in candidates if score(u) == best]
    return random.choice(tied)


def route_random(replica_urls: list[str]) -> str:
    return random.choice(replica_urls)


def route_power_of_two(
    replica_urls: list[str],
    metrics: dict[str, ReplicaSnapshot],
    endpoints_queued_tokens: dict[str, int],
) -> str:
    """Aibrix ``power-of-two``: two random choices, pick lower load.

    Load = ``num_used_tokens + queued_prompt_tokens`` (GORGO's existing signal).
    """
    candidates = [u for u in replica_urls if u in metrics]
    if len(candidates) < 2:
        return candidates[0] if candidates else random.choice(replica_urls)
    a, b = random.sample(candidates, 2)
    la = metrics[a].num_used_tokens + endpoints_queued_tokens.get(a, 0)
    lb = metrics[b].num_used_tokens + endpoints_queued_tokens.get(b, 0)
    return a if la <= lb else b


def route_least_request(
    replica_urls: list[str],
    metrics: dict[str, ReplicaSnapshot],
) -> str:
    """``least-request``: minimize ``sglang:num_running_reqs``."""
    return _tie_break_min(
        [u for u in replica_urls if u in metrics],
        lambda u: float(metrics[u].num_running_reqs),
    )


def route_least_load(
    replica_urls: list[str],
    metrics: dict[str, ReplicaSnapshot],
    endpoints_queued_tokens: dict[str, int],
) -> str:
    """``least-load``: minimize running + queue + proxy queued tokens + used KV tokens."""
    return _tie_break_min(
        [u for u in replica_urls if u in metrics],
        lambda u: metrics[u].combined_load(endpoints_queued_tokens.get(u, 0)),
    )


def route_least_kv_cache(
    replica_urls: list[str],
    metrics: dict[str, ReplicaSnapshot],
) -> str:
    """``least-kv-cache``: minimize ``sglang:num_used_tokens``."""
    return _tie_break_min(
        [u for u in replica_urls if u in metrics],
        lambda u: float(metrics[u].num_used_tokens),
    )


def route_least_gpu_cache(
    replica_urls: list[str],
    metrics: dict[str, ReplicaSnapshot],
) -> str:
    """``least-gpu-cache``: same as KV for homogeneous SGLang (single pool)."""
    return route_least_kv_cache(replica_urls, metrics)


def route_least_latency(
    replica_urls: list[str],
    metrics: dict[str, ReplicaSnapshot],
) -> str:
    """``least-latency``: minimize last /metrics scrape RTT (proxy-side latency)."""
    return _tie_break_min(
        [u for u in replica_urls if u in metrics],
        lambda u: metrics[u].latency,
    )


def route_least_utilization(
    replica_urls: list[str],
    metrics: dict[str, ReplicaSnapshot],
) -> str:
    """``least-utilization``: minimize ``sglang:utilization``."""
    return _tie_break_min(
        [u for u in replica_urls if u in metrics],
        lambda u: metrics[u].utilization,
    )


def route_least_busy_time(
    replica_urls: list[str],
    metrics: dict[str, ReplicaSnapshot],
) -> str:
    """``least-busy-time``: Aibrix uses GPU busy ratio; we proxy with utilization."""
    return route_least_utilization(replica_urls, metrics)


def route_throughput(
    replica_urls: list[str],
    metrics: dict[str, ReplicaSnapshot],
) -> str:
    """``throughput``: prefer highest ``sglang:gen_throughput`` (tokens/s)."""
    return _tie_break_max(
        [u for u in replica_urls if u in metrics],
        lambda u: metrics[u].gen_throughput,
    )


def route_pack_load(
    replica_urls: list[str],
    metrics: dict[str, ReplicaSnapshot],
    endpoints_queued_tokens: dict[str, int],
) -> str:
    """``pack-load``: maximize load among replicas still under a soft cap (pack work).

    Aibrix uses pull-mode utilization + cap; we maximize ``combined_load`` capped
    at ``cap = median_load + 2*MAD`` when MAD>0 else +inf.
    """
    candidates = [u for u in replica_urls if u in metrics]
    if not candidates:
        return random.choice(replica_urls)
    loads = [metrics[u].combined_load(endpoints_queued_tokens.get(u, 0)) for u in candidates]
    med = statistics.median(loads)
    if len(loads) >= 2:
        sorted_loads = sorted(loads)
        deviations = [abs(x - med) for x in sorted_loads]
        mad = statistics.median(deviations) if deviations else 0.0
    else:
        mad = 0.0
    cap = med + 2.0 * mad + 1e-6
    under = [
        u for u in candidates if metrics[u].combined_load(endpoints_queued_tokens.get(u, 0)) <= cap
    ]
    pool = under if under else candidates
    return _tie_break_max(
        pool,
        lambda u: metrics[u].combined_load(endpoints_queued_tokens.get(u, 0)),
    )


def route_queue_router(
    replica_urls: list[str],
    metrics: dict[str, ReplicaSnapshot],
) -> str:
    """``queue-router``: Aibrix wraps a queue + backend; we route to min queue depth."""
    return _tie_break_min(
        [u for u in replica_urls if u in metrics],
        lambda u: float(metrics[u].num_queue_reqs),
    )


def route_prefix_cache(
    replica_urls: list[str],
    metrics: dict[str, ReplicaSnapshot],
    endpoints_queued_tokens: dict[str, int],
    radix_trie: RadixTrie,
    token_ids: list[int],
    *,
    imbalance_abs: int = 8,
    std_factor: float = 2.0,
) -> str:
    """Aibrix-style prefix cache routing (see algorithms README).

    If running-request imbalance > ``imbalance_abs``, use least-request.
    Else among replicas with best radix prefix match, pick running < mean + factor*std;
    if none qualify, least-request among matches; if no matches, least-request global.
    """
    candidates = [u for u in replica_urls if u in metrics]
    if not candidates:
        return random.choice(replica_urls)

    running = [metrics[u].num_running_reqs for u in candidates]
    if running and max(running) - min(running) > imbalance_abs:
        return route_least_request(candidates, metrics)

    if not token_ids:
        return route_least_load(candidates, metrics, endpoints_queued_tokens)

    cached = radix_trie.cached_prefix_lengths(token_ids, candidates)
    best_cached = max(cached.values()) if cached else 0
    if best_cached <= 0:
        return route_least_request(candidates, metrics)

    match_urls = [u for u in candidates if cached.get(u, 0) == best_cached]
    mean_r = statistics.mean(metrics[u].num_running_reqs for u in candidates)
    if len(candidates) >= 2:
        std_r = statistics.stdev(metrics[u].num_running_reqs for u in candidates)
    else:
        std_r = 0.0
    threshold = mean_r + std_factor * std_r

    qualified = [u for u in match_urls if float(metrics[u].num_running_reqs) <= threshold]
    pool = qualified if qualified else match_urls
    return _tie_break_min(pool, lambda u: float(metrics[u].num_running_reqs))


def route_gorgo(
    replica_urls: list[str],
    metrics: dict[str, ReplicaSnapshot],
    endpoints_queued_tokens: dict[str, int],
    radix_trie: RadixTrie,
    token_ids: list[int],
    request_tokens: int,
    hyperparameters: dict[str, float],
) -> str:
    """GORGO multi-objective (original ``gorgo`` policy)."""
    endpoints_cached_tokens = (
        radix_trie.cached_prefix_lengths(token_ids, replica_urls)
        if token_ids
        else {u: 0 for u in replica_urls}
    )
    scores: dict[str, float] = {}
    for u in replica_urls:
        if u not in metrics:
            continue
        m = metrics[u]
        cached = endpoints_cached_tokens.get(u, 0)
        effective_prefill = max(0, request_tokens - cached)
        prefill_cost = effective_prefill * hyperparameters["t_prefill"]
        queue_cost = (endpoints_queued_tokens.get(u, 0) + m.num_used_tokens) * hyperparameters[
            "queued_tokens_weight"
        ]
        scores[u] = m.latency + prefill_cost + queue_cost
    if not scores:
        return random.choice(replica_urls)
    return min(scores, key=scores.get)


def route_simple_session_affinity(
    replica_urls: list[str],
    token_ids: list[int],
) -> str:
    """Sticky routing from prompt token hash (no client IP in proxy)."""
    if not token_ids:
        return random.choice(replica_urls)
    h = hash(tuple(token_ids[:256]))
    return replica_urls[h % len(replica_urls)]


def route_vtc_basic(
    replica_urls: list[str],
    metrics: dict[str, ReplicaSnapshot],
    endpoints_queued_tokens: dict[str, int],
) -> str:
    """``vtc-basic`` needs per-client fairness; without user id we blend load + util."""
    return _tie_break_min(
        [u for u in replica_urls if u in metrics],
        lambda u: (
            metrics[u].combined_load(endpoints_queued_tokens.get(u, 0))
            + 10.0 * metrics[u].utilization
        ),
    )


def route_fallback(
    replica_urls: list[str],
    metrics: dict[str, ReplicaSnapshot],
) -> str:
    """Aibrix fallback defaults to least-request."""
    return route_least_request(replica_urls, metrics)


def route_slo_family(
    replica_urls: list[str],
    metrics: dict[str, ReplicaSnapshot],
    endpoints_queued_tokens: dict[str, int],
    variant: str,
) -> str:
    """SLO routers in Aibrix use queues; we map variants to load heuristics."""
    if variant == "slo-pack-load":
        return route_pack_load(replica_urls, metrics, endpoints_queued_tokens)
    if variant in ("slo-least-load", "slo-least-load-pulling", "slo"):
        return route_least_load(replica_urls, metrics, endpoints_queued_tokens)
    return route_least_load(replica_urls, metrics, endpoints_queued_tokens)


def route_pd_stub(replica_urls: list[str]) -> str:
    """Prefill/decode split not modeled; random."""
    return random.choice(replica_urls)


@dataclass(frozen=True, slots=True)
class RouteContext:
    """Uniform per-request inputs every routing policy can read.

    Policies use whichever fields they need; absent metrics are represented
    by an empty ``metrics`` dict (the proxy filters missing replicas before
    invoking the policy).
    """

    replica_urls: list[str]
    metrics: dict[str, ReplicaSnapshot]
    endpoints_queued_tokens: dict[str, int]
    radix_trie: RadixTrie
    token_ids: list[int]
    request_tokens: int
    hyperparameters: dict[str, float]


@dataclass(frozen=True, slots=True)
class PolicyDef:
    """Routing-policy descriptor.

    ``needs_metrics`` lets the proxy decide whether to scrape (and snapshot)
    ``/metrics`` before invoking ``fn``; policies that only need
    ``replica_urls`` / ``token_ids`` (e.g. ``random``, ``simple-session-
    affinity``) can route even when no live metrics are available yet.
    """

    name: str
    needs_metrics: bool
    fn: Callable[[RouteContext], str]


# Single source of truth: maps normalized (kebab-case) policy id to its
# descriptor. Adapter lambdas keep the underlying ``route_*`` functions'
# signatures intact so they remain easy to call directly from tests.
POLICY_REGISTRY: dict[str, PolicyDef] = {
    pdef.name: pdef
    for pdef in [
        # Core / legacy GORGO
        PolicyDef("random", False, lambda c: route_random(c.replica_urls)),
        PolicyDef(
            "gorgo",
            True,
            lambda c: route_gorgo(
                c.replica_urls,
                c.metrics,
                c.endpoints_queued_tokens,
                c.radix_trie,
                c.token_ids,
                c.request_tokens,
                c.hyperparameters,
            ),
        ),
        PolicyDef(
            "power-of-two",
            True,
            lambda c: route_power_of_two(c.replica_urls, c.metrics, c.endpoints_queued_tokens),
        ),
        # Aibrix gateway/algorithms
        PolicyDef("least-request", True, lambda c: route_least_request(c.replica_urls, c.metrics)),
        PolicyDef(
            "least-load",
            True,
            lambda c: route_least_load(c.replica_urls, c.metrics, c.endpoints_queued_tokens),
        ),
        PolicyDef(
            "least-kv-cache", True, lambda c: route_least_kv_cache(c.replica_urls, c.metrics)
        ),
        PolicyDef(
            "least-gpu-cache", True, lambda c: route_least_gpu_cache(c.replica_urls, c.metrics)
        ),
        PolicyDef("least-latency", True, lambda c: route_least_latency(c.replica_urls, c.metrics)),
        PolicyDef(
            "least-utilization",
            True,
            lambda c: route_least_utilization(c.replica_urls, c.metrics),
        ),
        PolicyDef(
            "least-busy-time", True, lambda c: route_least_busy_time(c.replica_urls, c.metrics)
        ),
        PolicyDef("throughput", True, lambda c: route_throughput(c.replica_urls, c.metrics)),
        PolicyDef(
            "pack-load",
            True,
            lambda c: route_pack_load(c.replica_urls, c.metrics, c.endpoints_queued_tokens),
        ),
        PolicyDef(
            "prefix-cache",
            True,
            lambda c: route_prefix_cache(
                c.replica_urls,
                c.metrics,
                c.endpoints_queued_tokens,
                c.radix_trie,
                c.token_ids,
            ),
        ),
        PolicyDef("queue-router", True, lambda c: route_queue_router(c.replica_urls, c.metrics)),
        PolicyDef(
            "simple-session-affinity",
            False,
            lambda c: route_simple_session_affinity(c.replica_urls, c.token_ids),
        ),
        PolicyDef(
            "vtc",
            True,
            lambda c: route_vtc_basic(c.replica_urls, c.metrics, c.endpoints_queued_tokens),
        ),
        PolicyDef(
            "vtc-basic",
            True,
            lambda c: route_vtc_basic(c.replica_urls, c.metrics, c.endpoints_queued_tokens),
        ),
        PolicyDef(
            "slo",
            True,
            lambda c: route_slo_family(c.replica_urls, c.metrics, c.endpoints_queued_tokens, "slo"),
        ),
        PolicyDef(
            "slo-pack-load",
            True,
            lambda c: route_slo_family(
                c.replica_urls, c.metrics, c.endpoints_queued_tokens, "slo-pack-load"
            ),
        ),
        PolicyDef(
            "slo-least-load",
            True,
            lambda c: route_slo_family(
                c.replica_urls, c.metrics, c.endpoints_queued_tokens, "slo-least-load"
            ),
        ),
        PolicyDef(
            "slo-least-load-pulling",
            True,
            lambda c: route_slo_family(
                c.replica_urls, c.metrics, c.endpoints_queued_tokens, "slo-least-load-pulling"
            ),
        ),
        PolicyDef("fallback", True, lambda c: route_fallback(c.replica_urls, c.metrics)),
        # Explicitly not supported without PD split / Redis tracker -> stubbed to random
        PolicyDef("pd", False, lambda c: route_pd_stub(c.replica_urls)),
        PolicyDef("pd-disaggregation", False, lambda c: route_pd_stub(c.replica_urls)),
    ]
}

# Frozen view exposed for ``/policy`` validation. Derived from the registry
# so adding/removing a PolicyDef is the single edit.
ROUTING_POLICIES: frozenset[str] = frozenset(POLICY_REGISTRY)


def get_policy(name: str) -> PolicyDef:
    """Look up a :class:`PolicyDef` by raw or normalized name. Raises
    ``ValueError`` for unknown policies (matches the old ``route()`` contract)."""
    p = normalize_policy(name)
    pdef = POLICY_REGISTRY.get(p)
    if pdef is None:
        raise ValueError(f"unknown routing policy: {name!r}")
    return pdef


def route(
    policy: str,
    replica_urls: list[str],
    metrics: dict[str, ReplicaSnapshot],
    endpoints_queued_tokens: dict[str, int],
    radix_trie: RadixTrie,
    token_ids: list[int],
    request_tokens: int,
    hyperparameters: dict[str, float],
) -> str:
    """Dispatch by normalized policy name. Thin wrapper over the registry."""
    if not replica_urls:
        raise ValueError("no replicas")
    pdef = get_policy(policy)
    return pdef.fn(
        RouteContext(
            replica_urls=replica_urls,
            metrics=metrics,
            endpoints_queued_tokens=endpoints_queued_tokens,
            radix_trie=radix_trie,
            token_ids=token_ids,
            request_tokens=request_tokens,
            hyperparameters=hyperparameters,
        )
    )
