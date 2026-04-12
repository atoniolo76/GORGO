"""Shared pytest fixtures for the routing harness tests.

Fixtures are intentionally small and deterministic — tests must be fast
and must not depend on any external data.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from routing_harness.cluster import ClusterState
from routing_harness.core import Phase, PodSpec
from routing_harness.cost_model import (
    AnalyticCostModel,
    ComputeParams,
    NetworkParams,
    SchedulerParams,
)
from routing_harness.kv_cache import KVCacheState
from routing_harness.simulator.engine import EngineConfig


@pytest.fixture
def pod_specs() -> list[PodSpec]:
    return [
        PodSpec(
            pod_id=f"p{i}",
            role=Phase.BOTH,
            gpu_count=1,
            kv_cache_bytes=8 * 1024 * 1024,  # 8 MiB for tests
            max_concurrent_prefill=2,
            max_concurrent_decode=8,
        )
        for i in range(3)
    ]


@pytest.fixture
def pd_specs() -> list[PodSpec]:
    return [
        PodSpec("pf0", Phase.PREFILL, 1, 16 * 1024 * 1024, 4, 0, peer_ids=("dc0",)),
        PodSpec("dc0", Phase.DECODE, 1, 4 * 1024 * 1024, 0, 8, peer_ids=("pf0",)),
    ]


@pytest.fixture
def cluster(pod_specs) -> ClusterState:
    return ClusterState.from_specs(pod_specs)


@pytest.fixture
def kv_cache(pod_specs) -> KVCacheState:
    return KVCacheState.from_specs({s.pod_id: s.kv_cache_bytes for s in pod_specs})


@pytest.fixture
def compute_params() -> ComputeParams:
    return ComputeParams(
        prefill_ms_per_token=0.1,
        decode_ms_per_token=5.0,
        prefill_overhead_ms=4.0,
        decode_overhead_ms=1.0,
    )


@pytest.fixture
def network_params() -> NetworkParams:
    return NetworkParams(
        client_rtt_ms=5.0,
        inter_pod_rtt_ms=0.2,
        inter_pod_bandwidth_gbps=100.0,
        kv_bytes_per_token=1024,
        serialization_overhead_ms=0.5,
    )


@pytest.fixture
def scheduler_params() -> SchedulerParams:
    return SchedulerParams(base_routing_ms=0.2, per_pod_consideration_us=5.0)


@pytest.fixture
def cost_model(compute_params, network_params, scheduler_params) -> AnalyticCostModel:
    return AnalyticCostModel(
        compute=compute_params, network=network_params, scheduler=scheduler_params
    )


@pytest.fixture
def engine_cfg() -> EngineConfig:
    return EngineConfig(kv_ewma_alpha=0.3, block_size=8, initial_warm_latency_ms=4.0)


@pytest.fixture
def fixtures_root() -> Path:
    return Path(__file__).parent / "fixtures"
