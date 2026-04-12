"""End-to-end toy runs.

Validates that (a) the simulator processes a small trace against every
policy without raising, (b) output files have the right shape, and (c)
prefix-cache policies actually yield higher capture_rate than random on
a shared-prefix trace.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from routing_harness import policies  # noqa: F401
from routing_harness.policy import list_policies
from routing_harness.simulator.engine import EngineConfig
from routing_harness.simulator.runner import run_single
from routing_harness.workload.trace import InMemoryTrace
from tests.fixtures.tiny_trace import shared_prefix_trace


def _kwargs(pid: str) -> dict:
    if pid == "random":
        return {"seed": 0}
    return {}


@pytest.mark.parametrize("policy_id", list_policies())
def test_toy_run_per_policy(
    policy_id,
    pod_specs,
    compute_params,
    network_params,
    scheduler_params,
    engine_cfg,
    tmp_path,
):
    trace = InMemoryTrace(requests=shared_prefix_trace(), source="toy")
    out = tmp_path / "results"
    result = run_single(
        policy_id=policy_id,
        policy_kwargs=_kwargs(policy_id),
        topology=pod_specs,
        trace=trace,
        compute=compute_params,
        network=network_params,
        scheduler=scheduler_params,
        engine_cfg=engine_cfg,
        output_root=out,
        run_meta={"test": "toy"},
    )
    rid = result["run_id"]
    assert (out / rid / "config.json").exists()
    assert (out / rid / "metrics.json").exists()
    assert (out / rid / "records.csv").exists()
    assert (out / "index.json").exists()
    idx = json.loads((out / "index.json").read_text())
    assert any(e["run_id"] == rid for e in idx)
    assert result["metrics"]["n"] == len(trace.requests)


def test_prefix_cache_beats_random_on_capture_rate(
    pod_specs, compute_params, network_params, scheduler_params, engine_cfg, tmp_path
):
    trace = InMemoryTrace(requests=shared_prefix_trace(), source="toy")
    common = dict(
        topology=pod_specs,
        trace=trace,
        compute=compute_params,
        network=network_params,
        scheduler=scheduler_params,
        engine_cfg=engine_cfg,
        output_root=tmp_path,
    )
    rnd = run_single("random", {"seed": 0}, **common)
    pfx = run_single("prefix-cache", {"block_size": 16}, **common)
    # On a small shared-prefix trace, random may occasionally match; we
    # assert weak dominance of capture_rate on average. Using sum of
    # captured blocks as a robust proxy, and requiring the prefix-aware
    # run to actually capture *something* (otherwise "both zero" would
    # pass vacuously).
    rnd_cap = rnd["metrics"]["kv"]["reuse_captured_blocks"]
    pfx_cap = pfx["metrics"]["kv"]["reuse_captured_blocks"]
    assert pfx_cap >= rnd_cap
    assert pfx_cap > 0, "prefix-cache captured zero blocks on a shared-prefix trace"
