"""Cartesian sweep expansion.

Takes a SweepConfig and yields concrete RunConfig overrides. Dotted
paths specify where in the config the value lives, e.g.
  "policy.params.alpha"
  "network.client_rtt_ms"
  "workload.params.arrival_rate_qps"
"""

from __future__ import annotations

import copy
from dataclasses import asdict, is_dataclass
from itertools import product
from typing import Iterator

from .schema import RunConfig, SweepConfig, _build_topology


def _set_path(d: dict, path: str, value) -> None:
    keys = path.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur[k]
    cur[keys[-1]] = value


def _dc_to_plain(obj):
    if is_dataclass(obj):
        return {k: _dc_to_plain(v) for k, v in asdict(obj).items()}
    if isinstance(obj, (list, tuple)):
        return [_dc_to_plain(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _dc_to_plain(v) for k, v in obj.items()}
    return obj


def expand(sweep: SweepConfig) -> Iterator[tuple[dict, RunConfig]]:
    """Yield (axis_values, run_config) for every grid point."""
    keys = list(sweep.grid.keys())
    values = [sweep.grid[k] for k in keys]
    base_plain = _dc_to_plain(sweep.base)
    for combo in product(*values):
        snap = copy.deepcopy(base_plain)
        for k, v in zip(keys, combo):
            _set_path(snap, k, v)
        # Rebuild a RunConfig from plain dict
        from .schema import (
            ComputeConfig, EngineConfigSpec, NetworkConfig, PolicyConfig,
            RunConfig as RC, SchedulerConfig, WorkloadConfig,
        )
        rc = RC(
            name=snap["name"],
            policy=PolicyConfig(
                policy_id=snap["policy"]["policy_id"],
                params=dict(snap["policy"].get("params", {})),
            ),
            topology=_build_topology(snap["topology"]),
            compute=ComputeConfig(**snap["compute"]),
            network=NetworkConfig(**snap["network"]),
            scheduler=SchedulerConfig(**snap["scheduler"]),
            engine=EngineConfigSpec(**snap.get("engine", {})),
            workload=WorkloadConfig(
                kind=snap["workload"]["kind"],
                params=dict(snap["workload"].get("params", {})),
            ),
            seeds=tuple(int(s) for s in snap["seeds"]),
            output_dir=snap["output_dir"],
        )
        yield dict(zip(keys, combo)), rc
