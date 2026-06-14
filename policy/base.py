"""Shared base for routing policies.

This module owns the generic infrastructure every policy module
relies on:

* :class:`ReplicaSnapshot` -- per-replica metrics shape produced by
  ``proxy/modal_proxy.py``'s metrics scrape.
* :class:`RouteContext`    -- uniform per-request inputs every policy
  reads from (``replica_urls``, ``metrics``, etc.).
* :class:`PolicyDef`       -- registry descriptor (name + needs_metrics
  + callable).
* :data:`POLICY_REGISTRY`  -- the single source of truth for the kebab-
  case policy ids the ``/policy`` endpoint accepts. Composed lazily
  from :data:`policy.lb_aibrix.AIBRIX_POLICIES` and
  :data:`policy.gorgo.GORGO_POLICIES` plus the tiny ``random`` core
  policy that lives here.
* :func:`route_random`     -- baseline random pick. Lives here because
  it's used both as a public policy and as a fallback by other modules.

aibrix-derived policies live in :mod:`policy.lb_aibrix`; the GORGO
policy lives in :mod:`policy.gorgo`. To add a new policy family,
create a new submodule exposing a ``list[PolicyDef]`` and add it to
:func:`_ensure_registry` below.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, NamedTuple

if TYPE_CHECKING:
    from utils.radix_trie import RadixTrie


class RouteDecision(NamedTuple):
    """Return shape for every routing policy.

    ``target`` is the chosen replica URL.

    ``fallback_reason`` is ``None`` when the policy's primary scoring
    path produced ``target``. It is a short kebab-case string when the
    policy hit an internal random fallback (e.g. empty candidate set
    after filtering, missing token ids for a session-affinity hash) and
    chose by ``random.choice`` instead. Distinct values let post-hoc
    trace analysis tell apart "policy fired" from "policy bailed out
    and rolled dice" without changing the policy's name in the trace.
    Keep the vocabulary small and stable; downstream group-bys assume
    it.

    Conventions:
      * ``"empty-candidates"``           -- no replica passed the policy's
        per-snapshot filter (e.g. all metrics missing).
      * ``"insufficient-candidates"``    -- fewer candidates than the
        policy needs (e.g. ``power-of-two`` with 1 candidate).
      * ``"missing-token-ids"``          -- session-affinity-style
        policies given an empty token list.

    The proxy concatenates these onto the configured policy name when
    writing the trace's ``effective_policy`` field, e.g.
    ``"random-fallback:internal:least-request:empty-candidates"``.

    ``scores`` is an optional per-replica cost-model output dict
    (``{replica_url: float}``). Only populated by policies that compute
    explicit scores (currently just GORGO). Other policies leave it as
    ``None``. Used by the proxy to write ``candidate_scores`` into the
    request trace event so post-hoc analysis can explain *why* a target
    was chosen without re-running the policy.
    """

    target: str
    fallback_reason: str | None = None
    scores: dict[str, float] | None = None


def normalize_policy(name: str) -> str:
    """Canonicalize a policy name to kebab-case lower. ``/policy`` POSTs
    are normalized through this so callers can use ``power_of_two``,
    ``Power-Of-Two``, etc. interchangeably."""
    return name.strip().replace("_", "-").lower()


class ReplicaSnapshot:
    """Per-replica metrics from a single SGLang ``/metrics`` scrape.

    ``latency`` is the wall-clock RTT of the scrape itself; it includes
    SGLang's Prometheus handler time and serialization, so it's a noisy
    upper bound on pure network RTT (and inflates under load).

    ``network_rtt`` is the EWMA-smoothed RTT of a dedicated lightweight
    probe to the replica's base URL; preferred over ``latency`` when
    subtracting the irreducible network leg from TTFT samples. ``0.0``
    means the probe hasn't completed a successful round-trip yet
    (cold start) -- callers should fall back to ``latency`` in that case.
    """

    __slots__ = (
        "num_running_reqs",
        "num_queue_reqs",
        "num_used_tokens",
        "latency",
        "network_rtt",
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
        network_rtt: float = 0.0,
        gen_throughput: float = 0.0,
        utilization: float = 0.0,
    ):
        self.num_running_reqs = num_running_reqs
        self.num_queue_reqs = num_queue_reqs
        self.num_used_tokens = num_used_tokens
        self.latency = latency
        self.network_rtt = network_rtt
        self.gen_throughput = gen_throughput
        self.utilization = utilization

    def combined_load(self, queued_prompt_tokens: int, used_weight: float = 1.0) -> float:
        return (
            self.num_running_reqs
            + self.num_queue_reqs
            + used_weight * self.num_used_tokens
            + queued_prompt_tokens
        )


@dataclass(frozen=True, slots=True)
class RouteContext:
    """Uniform per-request inputs every routing policy can read.

    Policies use whichever fields they need; absent metrics are
    represented by an empty ``metrics`` dict (the proxy filters
    missing replicas before invoking the policy).

    ``endpoints_queued_tokens`` is a per-target *token* counter
    (incremented by ``request_tokens`` on dispatch, decremented on
    completion). Used by load-aware policies that score on tokens.

    ``endpoints_queued_uncached_tokens`` is the same lifecycle counter,
    but cache-aware: it increments by the number of uncached tokens for
    the request on the selected replica at dispatch time. New GORGO
    variants use this as a better TTFT load signal than raw prompt
    tokens.

    ``endpoints_inflight_requests`` is a per-target *request* counter
    with the same lifecycle. Lets request-counting policies (notably
    ``least-request``) bridge the staleness window of the SGLang
    metrics scrape: between scrapes ``num_running_reqs`` is frozen at
    its last value, but the proxy has been dispatching requests in the
    meantime; this counter captures those. ``route_least_request``
    scores by ``max(snap.num_running_reqs, inflight[u])`` so the score
    is correct in both regimes -- fresh metrics dominate when they're
    available, the local counter takes over during a stale window.

    ``hyperparameters`` carries the structured GORGO hyperparameter
    store (see :mod:`policy.gorgo` for its shape: ``{"defaults":
    {...}, "per_target": {url: {...}}}``). Non-GORGO policies don't
    read it.
    """

    replica_urls: list[str]
    metrics: dict[str, ReplicaSnapshot]
    endpoints_queued_tokens: dict[str, int]
    endpoints_queued_uncached_tokens: dict[str, int]
    endpoints_inflight_requests: dict[str, int]
    radix_trie: RadixTrie
    token_ids: list[int]
    request_tokens: int
    hyperparameters: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PolicyDef:
    """Routing-policy descriptor.

    ``needs_metrics`` lets the proxy decide whether to scrape (and
    snapshot) ``/metrics`` before invoking ``fn``; policies that only
    need ``replica_urls`` / ``token_ids`` (e.g. ``random``,
    ``simple-session-affinity``) can route even when no live metrics
    are available yet.
    """

    name: str
    needs_metrics: bool
    fn: Callable[[RouteContext], RouteDecision]


def route_random(replica_urls: list[str]) -> RouteDecision:
    """Uniform random pick. Used as a public policy *and* as the
    fallback every other module reaches for when its preconditions
    aren't met (e.g. no metrics yet)."""
    return RouteDecision(target=random.choice(replica_urls))


# ----- Registry assembly ----------------------------------------------------
#
# Composition is *lazy* on purpose. ``policy.gorgo`` and
# ``policy.lb_aibrix`` import ``PolicyDef`` / ``RouteContext`` from
# this module; if we eagerly imported them at the bottom of this
# file, importing ``policy.gorgo`` first would hit a half-loaded
# ``policy.base`` and fail (Python's classic circular-import
# trap). Building the registry on first access sidesteps the
# ordering entirely: by the time any caller asks for
# ``POLICY_REGISTRY`` the policy modules have finished loading.

_CORE_POLICIES: list[PolicyDef] = [
    PolicyDef("random", False, lambda c: route_random(c.replica_urls)),
]


def _build_registry(*policy_lists: list[PolicyDef]) -> dict[str, PolicyDef]:
    """Combine multiple lists of PolicyDef into a single name-keyed
    dict. Raises if two lists collide on a name (catching typos at
    import time rather than mysteriously routing the wrong policy)."""
    registry: dict[str, PolicyDef] = {}
    for plist in policy_lists:
        for pdef in plist:
            if pdef.name in registry:
                raise ValueError(
                    f"duplicate policy name {pdef.name!r} when assembling POLICY_REGISTRY"
                )
            registry[pdef.name] = pdef
    return registry


_POLICY_REGISTRY_CACHE: dict[str, PolicyDef] | None = None


def _ensure_registry() -> dict[str, PolicyDef]:
    global _POLICY_REGISTRY_CACHE
    if _POLICY_REGISTRY_CACHE is None:
        from policy.gorgo import GORGO_POLICIES
        from policy.lb_aibrix import AIBRIX_POLICIES

        _POLICY_REGISTRY_CACHE = _build_registry(
            _CORE_POLICIES,
            AIBRIX_POLICIES,
            GORGO_POLICIES,
        )
    return _POLICY_REGISTRY_CACHE


def __getattr__(name: str):
    """Lazy module attributes. ``POLICY_REGISTRY`` and
    ``ROUTING_POLICIES`` are built on first access so ``policy.base``
    can be imported by the sibling policy modules without circular
    fallout."""
    if name == "POLICY_REGISTRY":
        return _ensure_registry()
    if name == "ROUTING_POLICIES":
        return frozenset(_ensure_registry())
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_policy(name: str) -> PolicyDef:
    """Look up a :class:`PolicyDef` by raw or normalized name. Raises
    ``ValueError`` for unknown policies (matches the old ``route()``
    contract)."""
    p = normalize_policy(name)
    pdef = _ensure_registry().get(p)
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
    hyperparameters: dict[str, Any],
    endpoints_inflight_requests: dict[str, int] | None = None,
    endpoints_queued_uncached_tokens: dict[str, int] | None = None,
) -> str:
    """Dispatch by normalized policy name. Thin wrapper over the
    registry kept around for tests / scripts that don't want to
    construct a :class:`RouteContext` themselves.

    Returns the chosen URL only; any internal-fallback signal from the
    underlying :class:`RouteDecision` is dropped on the floor (callers
    that care should construct a context and call the policy fn
    directly). ``endpoints_inflight_requests`` defaults to an empty
    dict so old callers don't have to plumb the new field through;
    counter-bridging is only useful when the proxy is the one calling.
    """
    if not replica_urls:
        raise ValueError("no replicas")
    pdef = get_policy(policy)
    decision = pdef.fn(
        RouteContext(
            replica_urls=replica_urls,
            metrics=metrics,
            endpoints_queued_tokens=endpoints_queued_tokens,
            endpoints_queued_uncached_tokens=endpoints_queued_uncached_tokens or {},
            endpoints_inflight_requests=endpoints_inflight_requests or {},
            radix_trie=radix_trie,
            token_ids=token_ids,
            request_tokens=request_tokens,
            hyperparameters=hyperparameters,
        )
    )
    return decision.target
