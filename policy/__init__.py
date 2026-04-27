"""Routing policies for the GORGO proxy.

Public surface re-exported from this package's submodules:

* :mod:`policy.base`       -- shared dataclasses, lazy registry, dispatch
* :mod:`policy.lb_aibrix`  -- aibrix-derived ``route_*`` policies
* :mod:`policy.gorgo`      -- the GORGO policy and its per-target
                              hyperparameter store

Most callers want one of:

    from policy import POLICY_REGISTRY, RouteContext, normalize_policy
    from policy.gorgo import make_default_store, effective_hyperparameters

Symbols specific to a single policy (gorgo's hyperparameter store
helpers, aibrix's individual ``route_*`` functions) are deliberately
*not* re-exported here -- they're imported directly from the
submodule that owns them so each call site documents which policy
family it's reaching into.
"""

from policy.base import (
    PolicyDef,
    ReplicaSnapshot,
    RouteContext,
    get_policy,
    normalize_policy,
    route,
    route_random,
)


def __getattr__(name: str):
    """Forward lazy attributes from :mod:`policy.base` so callers can
    write ``from policy import POLICY_REGISTRY`` without triggering
    the registry build at package-import time."""
    if name in {"POLICY_REGISTRY", "ROUTING_POLICIES"}:
        from policy import base

        return getattr(base, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PolicyDef",
    "POLICY_REGISTRY",
    "ROUTING_POLICIES",
    "ReplicaSnapshot",
    "RouteContext",
    "get_policy",
    "normalize_policy",
    "route",
    "route_random",
]
