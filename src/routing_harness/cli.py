"""CLI: `routing-harness run|sweep|list-policies`.

Implementation-only: does not execute runs against real infrastructure.
All side effects (file writes) happen under the user-provided
output_dir. The CLI is thin — most logic is in `simulator.runner`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .config.schema import RunConfig, load_run, load_sweep
from .config.sweep import expand
from .core import PodSpec, Phase
from .cost_model import ComputeParams, NetworkParams, SchedulerParams
from .simulator.engine import EngineConfig
from .simulator.runner import run_single
from . import policies  # noqa: F401 — registers policies
from .policy import list_policies
from .workload import lmsys as lmsys_adapter
from .workload import synthetic as synthetic_gen


def _topology_to_specs(tc) -> list[PodSpec]:
    role_map = {"prefill": Phase.PREFILL, "decode": Phase.DECODE, "both": Phase.BOTH}
    return [
        PodSpec(
            pod_id=p.pod_id,
            role=role_map[p.role],
            gpu_count=p.gpu_count,
            kv_cache_bytes=p.kv_cache_bytes,
            max_concurrent_prefill=p.max_concurrent_prefill,
            max_concurrent_decode=p.max_concurrent_decode,
            peer_ids=p.peer_ids,
        )
        for p in tc.pods
    ]


def _build_trace(workload_cfg, seed: int):
    kind = workload_cfg.kind
    params = dict(workload_cfg.params)
    params["seed"] = seed
    if kind == "synthetic":
        sp = synthetic_gen.SyntheticParams(**params)
        return synthetic_gen.generate(sp)
    if kind == "lmsys":
        lmsys_cfg = lmsys_adapter.LmsysConfig(
            local_path=params["local_path"],
            max_conversations=params.get("max_conversations"),
            language_filter=tuple(params.get("language_filter", ("en",))),
            min_turns=params.get("min_turns", 1),
            max_turns=params.get("max_turns", 16),
            seed=seed,
        )
        trace_params = lmsys_adapter.TraceParams(
            arrival_rate_qps=params["arrival_rate_qps"],
            tokens_per_char=params.get("tokens_per_char", 0.25),
            max_output_tokens=params.get("max_output_tokens", 256),
            seed=seed,
        )
        return lmsys_adapter.build_trace(lmsys_cfg, trace_params)
    raise ValueError(f"unknown workload kind: {kind}")


def _execute(rc: RunConfig) -> list[dict]:
    out: list[dict] = []
    for seed in rc.seeds:
        topology = _topology_to_specs(rc.topology)
        trace = _build_trace(rc.workload, seed)
        result = run_single(
            policy_id=rc.policy.policy_id,
            policy_kwargs=rc.policy.params,
            topology=topology,
            trace=trace,
            compute=ComputeParams(**asdict(rc.compute)),
            network=NetworkParams(**asdict(rc.network)),
            scheduler=SchedulerParams(**asdict(rc.scheduler)),
            engine_cfg=EngineConfig(**asdict(rc.engine)),
            output_root=Path(rc.output_dir),
            run_meta={"name": rc.name, "seed": seed},
        )
        out.append(result)
    return out


def _cmd_run(args: argparse.Namespace) -> int:
    rc = load_run(args.config)
    results = _execute(rc)
    print(json.dumps([{"run_id": r["run_id"], "metrics": r["metrics"]} for r in results], indent=2))
    return 0


def _cmd_sweep(args: argparse.Namespace) -> int:
    sweep = load_sweep(args.config)
    all_results: list[dict] = []
    for axis, rc in expand(sweep):
        results = _execute(rc)
        for r in results:
            r["axis"] = axis
        all_results.extend(results)
    print(json.dumps([{"run_id": r["run_id"], "axis": r.get("axis"), "metrics": r["metrics"]} for r in all_results], indent=2))
    return 0


def _cmd_list_policies(_: argparse.Namespace) -> int:
    for pid in list_policies():
        print(pid)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="routing-harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run a single configured experiment")
    run.add_argument("--config", required=True)
    run.set_defaults(func=_cmd_run)

    sweep = sub.add_parser("sweep", help="run a parameter sweep")
    sweep.add_argument("--config", required=True)
    sweep.set_defaults(func=_cmd_sweep)

    lp = sub.add_parser("list-policies", help="list registered policy ids")
    lp.set_defaults(func=_cmd_list_policies)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
