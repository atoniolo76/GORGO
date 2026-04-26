"""Routing policy implementations.

Import this module to register all built-in policies. Tests rely on the
import side-effect of `get_all_policies()` to populate the registry.
"""

from __future__ import annotations

from . import (
    gorgo,
    least_busy_time,
    least_kv_cache,
    least_latency,
    least_request,
    pd,
    pd_preble,
    per_tenant_load_balance,
    prefix_cache,
    prefix_cache_preble,
    session_affinity,
    throughput,
)
from . import (
    random as random_policy,  # avoid shadowing stdlib
)


def all_policy_ids() -> list[str]:
    from ..policy import list_policies

    return list_policies()


__all__ = [
    "gorgo",
    "least_busy_time",
    "least_kv_cache",
    "least_latency",
    "least_request",
    "pd",
    "pd_preble",
    "per_tenant_load_balance",
    "prefix_cache",
    "prefix_cache_preble",
    "random_policy",
    "session_affinity",
    "throughput",
    "all_policy_ids",
]
