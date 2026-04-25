"""Aggregate PD-vs-colocated sweep stdout into the §6.4 table.

Inputs:
    --pd <path>        stdout JSON from `routing-harness sweep --config
                       configs/example_pd_potent_sweep.yaml`
    --colocated <path> stdout JSON from `routing-harness sweep --config
                       configs/example_colocated_potent_sweep.yaml`

Both files share schema:
    [{run_id, axis: {policy.policy_id, workload.params.arrival_rate_qps,
      workload.params.zipf_s}, metrics: {...}}]

Output: stdout markdown tables matched to the figure scaffolding in
`research/reports/routing-comparison.md` §6.4 — median p95 and hit_rate
across seeds, per (topology, policy, qps), with Zipf folded by median.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

_AXIS = {
    "policy": "policy.policy_id",
    "qps": "workload.params.arrival_rate_qps",
    "zipf": "workload.params.zipf_s",
}


def _load(path: str) -> list[dict]:
    return json.loads(Path(path).read_text())


def _key(run: dict, *names: str) -> tuple:
    return tuple(run["axis"][_AXIS[n]] for n in names)


def _median(xs: list[float]) -> float:
    return statistics.median(xs) if xs else float("nan")


def _agg(runs: list[dict], topology: str) -> list[dict]:
    """Median-across-seeds-and-zipf for each (policy, qps)."""
    by_cell: dict[tuple, list[dict]] = defaultdict(list)
    for r in runs:
        by_cell[_key(r, "policy", "qps")].append(r)
    rows = []
    for (policy, qps), cell in sorted(by_cell.items()):
        p95 = [r["metrics"]["latency_ms"]["p95"] for r in cell]
        p99 = [r["metrics"]["latency_ms"]["p99"] for r in cell]
        hit = [r["metrics"]["kv"]["hit_rate"] for r in cell]
        skew = [r["metrics"]["load"]["skew"] for r in cell]
        rows.append({
            "topology": topology,
            "policy": policy,
            "qps": qps,
            "p95": _median(p95),
            "p99": _median(p99),
            "hit_rate": _median(hit),
            "skew": _median(skew),
            "n_runs": len(cell),
        })
    return rows


def _md_table(rows: list[dict], qps_order: list[float]) -> str:
    """Format a compact median-p95-by-qps table for §6.4."""
    by_pol: dict[tuple[str, str], dict] = {}
    for r in rows:
        by_pol.setdefault((r["topology"], r["policy"]), {})[r["qps"]] = r
    header = (
        "| Topology | Policy | "
        + " | ".join(f"qps={int(q)}" for q in qps_order)
        + " | hit_rate | skew |"
    )
    sep = (
        "|----------|--------|"
        + "|".join(["--------"] * len(qps_order))
        + "|----------|------|"
    )
    lines = [header, sep]
    for (topo, pol), cells in sorted(by_pol.items()):
        row = f"| {topo:<8} | {pol:<22} | "
        row += " | ".join(
            f"{cells[q]['p95']:>6,.0f}" if q in cells else "  —   " for q in qps_order
        )
        anycell = next(iter(cells.values()))
        row += f" | {anycell['hit_rate']:.3f}    | {anycell['skew']:.2f} |"
        lines.append(row)
    return "\n".join(lines)


def _pairwise_pd_gain(rows: list[dict], qps_order: list[float]) -> str:
    """PD-pd / PD-pd-preble vs colocated-prefix-cache-preble margin table."""
    idx = {(r["topology"], r["policy"], r["qps"]): r for r in rows}
    coloc_best = "prefix-cache-preble"
    out = [
        "",
        "**Margin vs colocated `prefix-cache-preble` (negative = PD wins, ms):**",
        "",
        (
            "| QPS | PD `pd` p95 | PD `pd-preble` p95 | "
            "colocated `prefix-cache-preble` p95 | margin pd | margin pd-preble |"
        ),
        (
            "|-----|------------|--------------------|"
            "------------------------------------|----------|------------------|"
        ),
    ]
    for q in qps_order:
        coloc = idx.get(("colocated", coloc_best, q))
        pd = idx.get(("PD", "pd", q))
        pdp = idx.get(("PD", "pd-preble", q))
        if not (coloc and pd and pdp):
            continue
        out.append(
            f"| {int(q):>3} | {pd['p95']:>10,.0f} | {pdp['p95']:>18,.0f} | "
            f"{coloc['p95']:>34,.0f} | {pd['p95']-coloc['p95']:>+8,.0f} | "
            f"{pdp['p95']-coloc['p95']:>+16,.0f} |"
        )
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pd", required=True)
    ap.add_argument("--colocated", required=True)
    args = ap.parse_args()

    pd_runs = _load(args.pd)
    coloc_runs = _load(args.colocated)
    rows = _agg(pd_runs, "PD") + _agg(coloc_runs, "colocated")

    qps_order = sorted({r["qps"] for r in rows})
    print("## §6.4 — PD-disaggregated vs colocated (potent synthetic, matched 4 GPUs)")
    print()
    print(_md_table(rows, qps_order))
    print(_pairwise_pd_gain(rows, qps_order))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
