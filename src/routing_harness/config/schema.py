"""Config schema: dataclass-backed, validated on load.

No silent defaults for *experiment-defining* fields. Harness-internal
knobs (e.g. EWMA alpha) may carry documented defaults.

Supported formats: YAML (primary) and JSON. YAML uses PyYAML. If PyYAML
is not installed, the JSON path still works; loaders raise a helpful
error on YAML files with a missing dependency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class PodConfig:
    pod_id: str
    role: str  # "prefill" | "decode" | "both"
    gpu_count: int
    kv_cache_bytes: int
    max_concurrent_prefill: int
    max_concurrent_decode: int
    peer_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class TopologyConfig:
    pods: tuple[PodConfig, ...]


@dataclass(frozen=True)
class ComputeConfig:
    prefill_ms_per_token: float
    decode_ms_per_token: float
    prefill_overhead_ms: float
    decode_overhead_ms: float


@dataclass(frozen=True)
class NetworkConfig:
    client_rtt_ms: float
    inter_pod_rtt_ms: float
    inter_pod_bandwidth_gbps: float
    kv_bytes_per_token: int
    serialization_overhead_ms: float


@dataclass(frozen=True)
class SchedulerConfig:
    base_routing_ms: float
    per_pod_consideration_us: float


@dataclass(frozen=True)
class EngineConfigSpec:
    kv_ewma_alpha: float = 0.2
    block_size: int = 16
    initial_warm_latency_ms: float = 5.0


@dataclass(frozen=True)
class WorkloadConfig:
    kind: str  # "synthetic" | "lmsys"
    params: dict


@dataclass(frozen=True)
class PolicyConfig:
    policy_id: str
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RunConfig:
    name: str
    policy: PolicyConfig
    topology: TopologyConfig
    compute: ComputeConfig
    network: NetworkConfig
    scheduler: SchedulerConfig
    engine: EngineConfigSpec
    workload: WorkloadConfig
    seeds: tuple[int, ...]
    output_dir: str


@dataclass(frozen=True)
class SweepConfig:
    name: str
    base: RunConfig
    grid: dict[str, list[Any]]  # dotted path -> list of values


def _require(d: dict, *keys: str, context: str) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise ConfigError(f"missing keys {missing!r} in {context}")


def _load_text(path: Path) -> dict:
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise ConfigError(
                "YAML config requires PyYAML; install it or use JSON"
            ) from exc
        return yaml.safe_load(text)
    if path.suffix == ".json":
        return json.loads(text)
    raise ConfigError(f"unsupported config extension: {path.suffix}")


def _build_topology(raw: dict) -> TopologyConfig:
    _require(raw, "pods", context="topology")
    pods: list[PodConfig] = []
    for p in raw["pods"]:
        _require(
            p,
            "pod_id", "role", "gpu_count", "kv_cache_bytes",
            "max_concurrent_prefill", "max_concurrent_decode",
            context="topology.pods[]",
        )
        pods.append(PodConfig(
            pod_id=p["pod_id"],
            role=p["role"],
            gpu_count=int(p["gpu_count"]),
            kv_cache_bytes=int(p["kv_cache_bytes"]),
            max_concurrent_prefill=int(p["max_concurrent_prefill"]),
            max_concurrent_decode=int(p["max_concurrent_decode"]),
            peer_ids=tuple(p.get("peer_ids", [])),
        ))
    return TopologyConfig(pods=tuple(pods))


def _build_dc(dc: type, raw: dict, context: str) -> Any:
    required = [f.name for f in fields(dc) if f.default is field().default and f.default_factory is field().default_factory]  # type: ignore[misc]
    _require(raw, *required, context=context)
    return dc(**{f.name: raw.get(f.name, getattr(dc, f.name, None)) for f in fields(dc) if f.name in raw or hasattr(dc, f.name)})


def load_run(path: str | Path) -> RunConfig:
    path = Path(path)
    raw = _load_text(path)
    _require(
        raw,
        "name", "policy", "topology", "compute", "network", "scheduler",
        "workload", "seeds", "output_dir",
        context=str(path),
    )
    policy_raw = raw["policy"]
    _require(policy_raw, "policy_id", context="policy")
    return RunConfig(
        name=raw["name"],
        policy=PolicyConfig(
            policy_id=policy_raw["policy_id"],
            params=dict(policy_raw.get("params", {})),
        ),
        topology=_build_topology(raw["topology"]),
        compute=ComputeConfig(**raw["compute"]),
        network=NetworkConfig(**raw["network"]),
        scheduler=SchedulerConfig(**raw["scheduler"]),
        engine=EngineConfigSpec(**raw.get("engine", {})),
        workload=WorkloadConfig(
            kind=raw["workload"]["kind"],
            params=dict(raw["workload"].get("params", {})),
        ),
        seeds=tuple(int(s) for s in raw["seeds"]),
        output_dir=raw["output_dir"],
    )


def load_sweep(path: str | Path) -> SweepConfig:
    path = Path(path)
    raw = _load_text(path)
    _require(raw, "name", "base", "grid", context=str(path))
    base_path = Path(raw["base"])
    if not base_path.is_absolute():
        base_path = (path.parent / base_path).resolve()
    return SweepConfig(
        name=raw["name"],
        base=load_run(base_path),
        grid={k: list(v) for k, v in raw["grid"].items()},
    )
