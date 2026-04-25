"""Aggregate v4 colocated sweep stdout into the §6.1-6.3 tables.

Input: stdout JSON from
  `routing-harness sweep --config configs/example_sweep.yaml`

Schema per row:
  {run_id, axis: {policy.policy_id, workload.params.arrival_rate_qps,
   workload.params.zipf_s}, metrics: {...}}

Output: stdout markdown matching `research/reports/routing-comparison.md`
§6.1 (headline p95 across qps), §6.2 (Preble vs prefix-cache margin),
§6.3 (skew comparison). Median across seeds and Zipf values per
(policy, qps).
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

_HEADLINE_POLICIES = [
    "random",
    "least-request",
    "prefix-cache",
    "prefix-cache-preble",
    "least-busy-time",
]


def _median(xs):
    return statistics.median(xs) if xs else float("nan")


def _agg(runs):
    by_cell = defaultdict(list)
    for r in runs:
        key = (r["axis"][_AXIS["policy"]], r["axis"][_AXIS["qps"]])
        by_cell[key].append(r)
    rows = {}
    for (policy, qps), cell in by_cell.items():
        rows[(policy, qps)] = {
            "p95": _median([r["metrics"]["latency_ms"]["p95"] for r in cell]),
            "p99": _median([r["metrics"]["latency_ms"]["p99"] for r in cell]),
            "p50": _median([r["metrics"]["latency_ms"]["p50"] for r in cell]),
            "hit_rate": _median([r["metrics"]["kv"]["hit_rate"] for r in cell]),
            "skew": _median([r["metrics"]["load"]["skew"] for r in cell]),
            "n_runs": len(cell),
        }
    return rows


def _headline_table(rows, qps_order, policies):
    lines = [
        "### 6.1 Headline table (median p95 across seeds and Zipf values)",
        "",
        "| Policy              | "
        + " | ".join(f"qps={int(q)}" for q in qps_order)
        + " | hit_rate | skew  |",
        "|---------------------|"
        + "|".join(["--------"] * len(qps_order))
        + "|----------|-------|",
    ]
    for pol in policies:
        cells = [rows.get((pol, q)) for q in qps_order]
        if not all(cells):
            continue
        anycell = next(c for c in cells if c)
        bold = pol == "prefix-cache-preble"
        name = f"**{pol}**" if bold else pol
        p95s_disp = " | ".join(
            (f"**{c['p95']:>6,.0f}**" if bold else f"{c['p95']:>6,.0f}")
            for c in cells
        )
        skew_disp = (
            f"**{anycell['skew']:.3f}**" if bold else f"{anycell['skew']:.3f}"
        )
        lines.append(
            f"| {name:<19} | {p95s_disp} | {anycell['hit_rate']:.3f}    | {skew_disp} |"
        )
    return "\n".join(lines)


def _margin_table(rows, qps_order):
    lines = [
        "",
        "### 6.2 Preble vs prefix-cache: p95 margin",
        "",
        "| QPS | Margin (ms)  | Interpretation                               |",
        "|-----|-------------|----------------------------------------------|",
    ]
    for q in qps_order:
        a = rows.get(("prefix-cache-preble", q))
        b = rows.get(("prefix-cache", q))
        if not (a and b):
            continue
        margin = a["p95"] - b["p95"]
        sign = "−" if margin < 0 else "+"
        lines.append(
            f"| {int(q):>3} | **{sign}{abs(margin):>5,.0f}**     |"
            "  (interpret manually)                          |"
        )
    return "\n".join(lines)


def _skew_summary(rows, qps_order):
    lines = [
        "",
        "### 6.3 Hotspot mitigation",
        "",
        "Median skew across qps × seeds × zipf:",
        "",
        "| Policy              | skew  | hit_rate |",
        "|---------------------|-------|----------|",
    ]
    seen = set()
    for q in qps_order:
        for pol in _HEADLINE_POLICIES:
            if pol in seen:
                continue
            seen.add(pol)
            row = rows.get((pol, q))
            if row:
                lines.append(
                    f"| {pol:<19} | {row['skew']:.3f} | {row['hit_rate']:.3f}    |"
                )
        break
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="stdout JSON from routing-harness sweep")
    args = ap.parse_args()

    runs = json.loads(Path(args.input).read_text())
    rows = _agg(runs)
    qps_order = sorted({q for (_, q) in rows.keys()})

    print(f"Total runs: {len(runs)} (cells: {len(rows)}, qps: {qps_order})")
    print()
    print(_headline_table(rows, qps_order, _HEADLINE_POLICIES))
    print(_margin_table(rows, qps_order))
    print(_skew_summary(rows, qps_order))
    print()
    print("### Full grid (median p95 ms per policy × qps)")
    print()
    all_policies = sorted({p for (p, _) in rows.keys()})
    print(
        "| Policy | "
        + " | ".join(f"qps={int(q)}" for q in qps_order)
        + " | hit_rate | skew  |"
    )
    print(
        "|--------|"
        + "|".join(["--------"] * len(qps_order))
        + "|----------|-------|"
    )
    for pol in all_policies:
        cells = [rows.get((pol, q)) for q in qps_order]
        if not all(cells):
            continue
        any_c = next(c for c in cells if c)
        p95s = " | ".join(f"{c['p95']:>6,.0f}" for c in cells)
        print(
            f"| {pol} | {p95s} | {any_c['hit_rate']:.3f} | {any_c['skew']:.3f} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
