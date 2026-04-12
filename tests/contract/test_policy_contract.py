"""Contract tests that apply to every registered RoutingPolicy.

Parameterized over `list_policies()`, so adding a new policy
automatically exercises it against the shared contract.
"""

from __future__ import annotations

import pytest

from routing_harness import policies  # noqa: F401 — registers policies
from routing_harness.policy import RoutingPolicy, get_policy, list_policies
from tests.fixtures.tiny_trace import shared_prefix_trace


POLICY_IDS = list_policies()


def _instantiate(policy_id: str):
    # Randomized policies take a seed; others take nothing. The registry
    # passes kwargs through so missing kwargs default to the dataclass
    # defaults.
    kwargs = {"seed": 0} if policy_id == "random" else {}
    return get_policy(policy_id, **kwargs)


@pytest.mark.parametrize("policy_id", POLICY_IDS)
def test_policy_is_registered(policy_id):
    policy = _instantiate(policy_id)
    assert isinstance(policy, RoutingPolicy)
    assert policy.policy_id == policy_id


@pytest.mark.parametrize("policy_id", POLICY_IDS)
def test_policy_decides_on_nonempty_cluster(policy_id, cluster, kv_cache):
    policy = _instantiate(policy_id)
    req = shared_prefix_trace()[0]
    d = policy.decide(req, cluster, kv_cache)
    assert d.prefill_pod_id != "__none__"
    assert d.prefill_pod_id in cluster.pods
    assert d.decode_pod_id in cluster.pods
    assert isinstance(d.rationale, str) and d.rationale


@pytest.mark.parametrize("policy_id", POLICY_IDS)
def test_policy_handles_empty_cluster(policy_id, kv_cache):
    from routing_harness.cluster import ClusterState

    empty = ClusterState.from_specs([])
    policy = _instantiate(policy_id)
    req = shared_prefix_trace()[0]
    d = policy.decide(req, empty, kv_cache)
    assert d.prefill_pod_id == "__none__"


@pytest.mark.parametrize("policy_id", POLICY_IDS)
def test_policy_does_not_mutate_cluster(policy_id, cluster, kv_cache):
    policy = _instantiate(policy_id)
    snapshot = {
        pid: (p.active_prefill, p.active_decode, p.queued, p.ewma_latency_ms)
        for pid, p in cluster.pods.items()
    }
    for req in shared_prefix_trace():
        policy.decide(req, cluster, kv_cache)
    after = {
        pid: (p.active_prefill, p.active_decode, p.queued, p.ewma_latency_ms)
        for pid, p in cluster.pods.items()
    }
    assert snapshot == after
