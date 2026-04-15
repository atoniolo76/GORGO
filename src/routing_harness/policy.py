"""RoutingPolicy protocol + policy registry.

Policies are pluggable. To add one:
  1. Subclass RoutingPolicy and set `policy_id` class attr.
  2. Register with @register_policy("my-id").
  3. Pass the contract tests in tests/contract/test_policy_contract.py.

No policy should mutate cluster or cache state — the simulator owns
mutation. Policies read only.
"""

from __future__ import annotations

import inspect
from typing import Callable, Protocol, runtime_checkable

from .cluster import ClusterState
from .core import Decision, Request
from .kv_cache import KVCacheState


@runtime_checkable
class RoutingPolicy(Protocol):
    """Protocol every routing policy must satisfy.

    Implementations are expected to be pure functions of (request,
    cluster, kv_cache, config). Randomized policies must use the
    injected `rng` to stay deterministic under seeding.
    """

    policy_id: str

    def decide(
        self,
        request: Request,
        cluster: ClusterState,
        kv_cache: KVCacheState,
    ) -> Decision:
        """Return the pod(s) this request should be routed to.

        Must not raise on empty clusters — return a Decision pointing at
        a sentinel `"__none__"` pod_id if no routing is possible, so the
        simulator can record a failed-route metric rather than crash.
        """
        ...

    # Optional hook: the engine invokes `observe_completion` (if
    # defined) after each request finishes, passing the request, the
    # decision that was made, and the total tokens consumed
    # (prompt + decode). Fairness policies (VTC) and learning policies
    # use it to update internal state. Implementations are optional;
    # policies that do not define it are silently skipped.


_REGISTRY: dict[str, Callable[..., RoutingPolicy]] = {}


def register_policy(policy_id: str) -> Callable[[type], type]:
    """Class decorator: register a policy class under a stable id."""

    def _wrap(cls: type) -> type:
        if policy_id in _REGISTRY:
            raise ValueError(f"policy id already registered: {policy_id}")
        setattr(cls, "policy_id", policy_id)
        _REGISTRY[policy_id] = cls
        return cls

    return _wrap


def get_policy(policy_id: str, **kwargs) -> RoutingPolicy:
    """Instantiate a policy by id. kwargs are policy-specific config.

    Sweeps Cartesian-join over policy_id and policy.params, so a param
    that only applies to one policy (e.g. `block_size` on prefix-cache)
    leaks into the kwargs of unrelated policies. We drop kwargs the
    target policy does not accept rather than crashing the sweep.
    """
    if policy_id not in _REGISTRY:
        raise KeyError(
            f"unknown policy: {policy_id!r}. known: {sorted(_REGISTRY)}"
        )
    cls = _REGISTRY[policy_id]
    sig = inspect.signature(cls)
    accepted = {
        name for name, p in sig.parameters.items()
        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    }
    has_var_keyword = any(
        p.kind == p.VAR_KEYWORD for p in sig.parameters.values()
    )
    filtered = (
        kwargs if has_var_keyword
        else {k: v for k, v in kwargs.items() if k in accepted}
    )
    return cls(**filtered)


def list_policies() -> list[str]:
    return sorted(_REGISTRY)
