"""Aggregate the gorgo comparison sweeps into pivot tables.

Reads the per-run output directories under
``research/data/gorgo_comparison/<workload>`` and pivots each run's
``metrics.json`` against the policy / qps / zipf axes captured in
``config.json``. Emits markdown-ready tables to stdout. Used to
back-fill ``research/reports/gorgo_comparison.md`` without hand-rolling
numbers.

Usage:
    uv run python scripts/aggregate_gorgo_sweep.py

The synthetic sweep mixes the comparison runs (4 policies × 4 qps × 3
zipfs = 48) with the hyperparameter grid (1 policy × 3 t_prefill × 3
queued_tokens_weight = 9). The aggregator splits them by inspecting
``policy_kwargs``: hyperparam-grid runs all share policy_id='gorgo'
but vary t_prefill / queued_tokens_weight off the defaults.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SYN = ROOT / "research" / "data" / "gorgo_comparison" / "synthetic"
SG = ROOT / "research" / "data" / "gorgo_comparison" / "sharegpt"

DEFAULT_T_PREFILL = 0.05
DEFAULT_QTW = 0.001


def _load_runs(workload_dir: Path) -> list[dict]:
    runs = []
    for run_dir in sorted(workload_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        cfg = json.loads((run_dir / "config.json").read_text())
        met = json.loads((run_dir / "metrics.json").read_text())
        runs.append({"run_id": run_dir.name, "config": cfg, "metrics": met})
    return runs


def _is_default_hp(kwargs: dict) -> bool:
    """A run is part of the comparison sweep (not the hyperparam grid)
    if it uses the default t_prefill and queued_tokens_weight, OR if
    it's not gorgo."""
    if kwargs.get("policy_id", "") != "gorgo" and "t_prefill" not in kwargs:
        return True
    return (
        abs(kwargs.get("t_prefill", DEFAULT_T_PREFILL) - DEFAULT_T_PREFILL) < 1e-12
        and abs(kwargs.get("queued_tokens_weight", DEFAULT_QTW) - DEFAULT_QTW) < 1e-12
    )


def _split_synthetic(runs: list[dict]) -> tuple[list[dict], list[dict]]:
    comp = []
    hp = []
    for r in runs:
        cfg = r["config"]
        kw = dict(cfg["policy_kwargs"])
        kw["policy_id"] = cfg["policy_id"]
        if cfg["policy_id"] == "gorgo" and not _is_default_hp(kw):
            hp.append(r)
        else:
            comp.append(r)
        # Default-hp gorgo runs at qps=8/zipf=1.1 also belong in the
        # hyperparam grid view (they fill the (0.05, 0.001) cell that
        # would otherwise be missing). Include the matching slice.
        if (
            cfg["policy_id"] == "gorgo"
            and _is_default_hp(kw)
            and _qps(r) == 8.0
            and _zipf(r) == 1.1
        ):
            hp.append(r)
    return comp, hp


def _qps(r: dict) -> float:
    params = r["config"]["trace"].get("params", {})
    if "arrival_rate_qps" in params:
        return params["arrival_rate_qps"]
    # sharegpt / lmsys nest under params.trace
    nested = params.get("trace", {})
    return nested.get("arrival_rate_qps", 0.0)


def _zipf(r: dict) -> float | None:
    params = r["config"]["trace"].get("params", {})
    return params.get("zipf_s")


def _policy(r: dict) -> str:
    return r["config"]["policy_id"]


def _p95(r: dict) -> float:
    return r["metrics"]["latency_ms"]["p95"]


def _hit(r: dict) -> float:
    return r["metrics"]["kv"]["hit_rate"]


def _skew(r: dict) -> float:
    return r["metrics"]["load"]["skew"]


def _format_p95(value: float) -> str:
    return f"{value:>7.0f}"


def _table_p95_by_policy_qps(runs: list[dict], header: str) -> str:
    by_pol_qps: dict[tuple[str, float], list[float]] = {}
    for r in runs:
        by_pol_qps.setdefault((_policy(r), _qps(r)), []).append(_p95(r))
    policies = sorted({_policy(r) for r in runs})
    qpss = sorted({_qps(r) for r in runs})
    lines = [header]
    qps_cells = " | ".join(f"qps={int(q):>3}" for q in qpss)
    head = f"| Policy                 | {qps_cells} | hit_rate | skew  |"
    sep_qps = "|".join("-" * 8 for _ in qpss)
    sep = "|" + "-" * 24 + "|" + sep_qps + "|" + "-" * 10 + "|" + "-" * 7 + "|"
    lines.append(head)
    lines.append(sep)
    for pol in policies:
        cells = []
        for q in qpss:
            vals = by_pol_qps.get((pol, q), [])
            cell = _format_p95(statistics.median(vals)) if vals else "  -"
            cells.append(cell)
        # Aggregate hit_rate/skew over all runs for this policy.
        pol_runs = [r for r in runs if _policy(r) == pol]
        hr = statistics.median(_hit(r) for r in pol_runs)
        sk = statistics.median(_skew(r) for r in pol_runs)
        lines.append(
            f"| {pol:<22} | "
            + " | ".join(cells)
            + f" | {hr:>8.3f} | {sk:>5.3f} |"
        )
    return "\n".join(lines)


def _table_hyperparam_grid(runs: list[dict]) -> str:
    """Median p95 by (t_prefill, queued_tokens_weight)."""
    by_hp: dict[tuple[float, float], list[float]] = {}
    for r in runs:
        kw = r["config"]["policy_kwargs"]
        key = (kw["t_prefill"], kw["queued_tokens_weight"])
        by_hp.setdefault(key, []).append(_p95(r))
    tps = sorted({k[0] for k in by_hp})
    qws = sorted({k[1] for k in by_hp})
    lines = ["| t_prefill \\\\ qtw |" + "|".join(f" qtw={qw:<7g} " for qw in qws) + "|"]
    lines.append("|" + "-" * 18 + "|" + "|".join("-" * 13 for _ in qws) + "|")
    for tp in tps:
        cells = []
        for qw in qws:
            vals = by_hp.get((tp, qw), [])
            cell = _format_p95(statistics.median(vals)) if vals else "  -"
            cells.append(cell)
        lines.append(f"| t_prefill={tp:<7g} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    syn_runs = _load_runs(SYN)
    sg_runs = _load_runs(SG)
    comp_runs, hp_runs = _split_synthetic(syn_runs)
    print("# Synthetic-potent: median p95 (ms) over 3 zipf values, single seed\n")
    print(_table_p95_by_policy_qps(comp_runs, ""))
    print("\n# ShareGPT: median p95 (ms), single seed\n")
    print(_table_p95_by_policy_qps(sg_runs, ""))
    print("\n# Gorgo hyperparam sensitivity (synthetic, qps=8, zipf=1.1, single seed)\n")
    print(_table_hyperparam_grid(hp_runs))
    # Numeric leadership check: where does gorgo win/lose vs each baseline?
    print(
        "\n# Leadership matrix (synthetic): gorgo p95 minus baseline p95, ms;"
        " negative = gorgo wins\n"
    )
    by_pol_qps_zipf: dict[tuple[str, float, float], float] = {}
    for r in comp_runs:
        z = _zipf(r) or 0.0
        by_pol_qps_zipf[(_policy(r), _qps(r), z)] = _p95(r)
    qpss = sorted({_qps(r) for r in comp_runs})
    zipfs = sorted({_zipf(r) for r in comp_runs if _zipf(r) is not None})
    others = ["prefix-cache-preble", "pd-preble", "least-kv-cache"]
    for o in others:
        print(f"\n## gorgo - {o}\n")
        head = "| zipf \\\\ qps | " + " | ".join(f"qps={int(q):>3}" for q in qpss) + " |"
        sep = "|" + "-" * 13 + "|" + "|".join("-" * 8 for _ in qpss) + "|"
        print(head)
        print(sep)
        for z in zipfs:
            cells = []
            for q in qpss:
                a = by_pol_qps_zipf.get(("gorgo", q, z))
                b = by_pol_qps_zipf.get((o, q, z))
                if a is None or b is None:
                    cells.append("    -")
                else:
                    cells.append(f"{a - b:>+6.0f}")
            print(f"| zipf={z:<6} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
