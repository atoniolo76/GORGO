"""Simulator-level smoke for every registered policy (go-w9q).

Catches registry, config-wiring, and end-to-end runtime regressions
that unit and contract tests miss. Not a correctness deep-dive — that
is the job of the per-policy audits under tests/unit. This is a
liveness and wiring check: every policy_id in the registry must
instantiate via the standard config path, drive a 200-request
synthetic workload against a 3-pod colocated cluster without raising,
and produce sane summary metrics.

Complements (does not replace) the per-policy audits and the Modal
smokes (go-3j8).
"""

from __future__ import annotations

import math

import pytest

from routing_harness import policies  # noqa: F401 — registers policies
from routing_harness.core import Phase, PodSpec
from routing_harness.cost_model import (
    ComputeParams,
    NetworkParams,
    SchedulerParams,
)
from routing_harness.policy import list_policies
from routing_harness.simulator.engine import EngineConfig
from routing_harness.simulator.runner import run_single
from routing_harness.workload.synthetic import SyntheticParams, generate

# Policies whose expected behavior is to concentrate dispatch on a
# subset of pods — checking "every pod got at least one dispatch" is
# inappropriate for these. session-affinity hashes sessions to pods
# and a 12-session trace can land entirely on 1–2 pods; prefix-cache
# variants steer to the cache owner and may starve other pods on a
# Zipf-skewed family distribution.
CONCENTRATION_POLICIES = frozenset({"session-affinity", "prefix-cache", "prefix-cache-preble"})

# Policy-specific kwargs the standard config path would pass through.
# Mirrors what get_policy(...) does via the sweep machinery: kwargs the
# target dataclass does not accept are dropped, so we can pass a
# superset safely.
_POLICY_KWARGS: dict[str, dict] = {
    "random": {"seed": 0},
    "prefix-cache": {"block_size": 16},
    "prefix-cache-preble": {"block_size": 16},
    "pd": {"block_size": 16},
    "pd-preble": {"block_size": 16},
    "gorgo": {"block_size": 16},
}


def _kwargs_for(policy_id: str) -> dict:
    return _POLICY_KWARGS.get(policy_id, {})


@pytest.fixture(scope="module")
def colocated_3pod_topology() -> list[PodSpec]:
    # Three Phase.BOTH pods sized to comfortably hold a 200-request
    # synthetic trace without cache thrash dominating the run. PD and
    # PD-Preble degrade gracefully on a colocated cluster (see
    # PDPolicy docstring — both pools collapse), so the same topology
    # exercises every policy.
    return [
        PodSpec(
            pod_id=f"p{i}",
            role=Phase.BOTH,
            gpu_count=1,
            kv_cache_bytes=64 * 1024 * 1024,  # 64 MiB
            max_concurrent_prefill=4,
            max_concurrent_decode=16,
        )
        for i in range(3)
    ]


@pytest.fixture(scope="module")
def smoke_compute() -> ComputeParams:
    # Illustrative coefficients — chosen to be in the same ballpark as
    # configs/example_run.yaml but small enough that 200 requests run
    # in well under a second.
    return ComputeParams(
        prefill_ms_per_token=0.08,
        decode_ms_per_token=6.0,
        prefill_overhead_ms=5.0,
        decode_overhead_ms=2.0,
    )


@pytest.fixture(scope="module")
def smoke_network() -> NetworkParams:
    return NetworkParams(
        client_rtt_ms=5.0,
        inter_pod_rtt_ms=0.2,
        inter_pod_bandwidth_gbps=100.0,
        kv_bytes_per_token=1024,
        serialization_overhead_ms=0.5,
    )


@pytest.fixture(scope="module")
def smoke_scheduler() -> SchedulerParams:
    return SchedulerParams(base_routing_ms=0.2, per_pod_consideration_us=5.0)


@pytest.fixture(scope="module")
def smoke_engine_cfg() -> EngineConfig:
    return EngineConfig(kv_ewma_alpha=0.2, block_size=16, initial_warm_latency_ms=5.0)


@pytest.fixture(scope="module")
def smoke_trace():
    # 200 requests, seed=0, multi-family + multi-session so prefix and
    # session-aware policies have meaningful structure to react to.
    # shared_prefix_tokens > 0 forces real block-level prefix hashing
    # (rather than the opaque-key mode that pins capture_rate to 1.0
    # across all policies — see SyntheticParams docstring).
    params = SyntheticParams(
        n_requests=200,
        arrival_rate_qps=20.0,
        n_prefix_families=16,
        zipf_s=0.8,
        prompt_len_min=64,
        prompt_len_max=256,
        max_output_tokens=16,
        n_sessions=12,
        seed=0,
        shared_prefix_tokens=64,
    )
    return generate(params)


@pytest.mark.parametrize("policy_id", list_policies())
def test_policy_smoke(
    policy_id,
    colocated_3pod_topology,
    smoke_compute,
    smoke_network,
    smoke_scheduler,
    smoke_engine_cfg,
    smoke_trace,
    tmp_path,
):
    """Each registered policy runs end-to-end through run_single."""
    result = run_single(
        policy_id=policy_id,
        policy_kwargs=_kwargs_for(policy_id),
        topology=colocated_3pod_topology,
        trace=smoke_trace,
        compute=smoke_compute,
        network=smoke_network,
        scheduler=smoke_scheduler,
        engine_cfg=smoke_engine_cfg,
        output_root=tmp_path,
        run_meta={"test": "smoke", "policy": policy_id},
    )

    # (a) no exceptions — implicit if we got here.
    # (b) non-empty metrics output.
    metrics = result["metrics"]
    assert metrics, f"{policy_id}: empty metrics dict"
    assert metrics["n"] == len(smoke_trace.requests), (
        f"{policy_id}: metrics.n={metrics['n']} != trace size {len(smoke_trace.requests)}"
    )

    # (c) balance-sensitive policies: every pod got at least one
    # dispatch. Policies whose contract permits concentration
    # (session-sticky, prefix-locked) are exempt.
    per_pod = metrics["load"]["per_pod_busy_ms"]
    if policy_id not in CONCENTRATION_POLICIES:
        active_pods = {pid for pid, busy in per_pod.items() if busy > 0.0}
        assert len(active_pods) == 3, (
            f"{policy_id}: expected all 3 pods to receive dispatch, "
            f"got busy pods={active_pods} (per_pod_busy_ms={per_pod})"
        )
    else:
        # Even concentration policies must dispatch *somewhere*.
        assert any(busy > 0.0 for busy in per_pod.values()), (
            f"{policy_id}: no pod received any dispatch"
        )

    # (d) p95 latency is finite and > 0.
    p95 = metrics["latency_ms"]["p95"]
    assert math.isfinite(p95) and p95 > 0.0, f"{policy_id}: p95 latency invalid: {p95}"

    # (e) hit_rate in [0, 1].
    hit_rate = metrics["kv"]["hit_rate"]
    assert 0.0 <= hit_rate <= 1.0, f"{policy_id}: hit_rate={hit_rate} outside [0, 1]"


def test_smoke_covers_every_registered_policy():
    """Guard against accidental drift between the registry and the
    parametrize list — if a new policy is added, this confirms the
    smoke really exercises it (parametrize is evaluated at import
    time, so a stale policies import would silently skip new ones)."""
    registered = set(list_policies())
    expected = {
        "random",
        "least-request",
        "least-busy-time",
        "least-latency",
        "least-kv-cache",
        "throughput",
        "prefix-cache",
        "prefix-cache-preble",
        "session-affinity",
        "per-tenant-load-balance",
        "pd",
        "pd-preble",
        "gorgo",
    }
    assert registered == expected, (
        f"policy registry drifted from smoke baseline: "
        f"missing={expected - registered} extra={registered - expected}"
    )
