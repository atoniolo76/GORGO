"""Analyze experiment results and produce CSV + summary markdown.

Reads per-policy workload result JSONs from results/workload_runs/
and produces:
  1. A CSV with one row per policy (TTFT/E2E/throughput stats)
  2. A markdown summary table
  3. Auto-tune hyperparameter comparison (if applicable)

Usage:
    python scripts/analyze_results.py --prefix abstract_night_000_glm5_0030_to_0100 --label "GLM5 W1"
    python scripts/analyze_results.py --prefix abstract_night_w2_000_glm5_0100_to_0130 --label "GLM5 W2"

    # Compare two runs side by side:
    python scripts/analyze_results.py \
        --prefix abstract_night_000_glm5_0030_to_0100 \
        --prefix2 abstract_night_w2_000_glm5_0100_to_0130 \
        --label "GLM5 W1" --label2 "GLM5 W2"
"""

from __future__ import annotations

import argparse
import csv
import json
import glob
import os
from pathlib import Path


def _load_policies(results_dir: str, prefix: str) -> list[dict]:
    policies = []
    for f in sorted(glob.glob(os.path.join(results_dir, f"{prefix}_*.json"))):
        label = os.path.basename(f).replace(prefix + "_", "").replace(".json", "")
        d = json.load(open(f))
        s = d.get("stats") or {}
        t = s.get("ttft_seconds") or {}
        e = s.get("request_e2e_seconds") or {}
        itl = s.get("itl_ms") or {}
        inp = s.get("input_tokens") or {}
        out = s.get("output_tokens") or {}
        policies.append(
            {
                "policy": label,
                "ok": s.get("ok", 0),
                "fail": s.get("fail", 0),
                "sent": s.get("sent", 0),
                "success_rate": s.get("success_rate", 0),
                "elapsed_s": s.get("elapsed_seconds", 0),
                "rps": s.get("request_throughput_rps", 0),
                "input_tok_s": s.get("input_token_throughput", 0),
                "output_tok_s": s.get("output_token_throughput", 0),
                "ttft_p50": t.get("p50", 0),
                "ttft_p95": t.get("p95", 0),
                "ttft_p99": t.get("p99", 0),
                "ttft_max": t.get("max", 0),
                "ttft_avg": t.get("avg", 0),
                "e2e_p50": e.get("p50", 0),
                "e2e_p95": e.get("p95", 0),
                "e2e_p99": e.get("p99", 0),
                "itl_p50": itl.get("p50", 0),
                "itl_p95": itl.get("p95", 0),
                "avg_input_tokens": inp.get("avg", 0),
                "avg_output_tokens": out.get("avg", 0),
                "total_input_tokens": s.get("total_input_tokens", 0),
                "partial": s.get("partial", False),
            }
        )
    return policies


def _write_csv(policies: list[dict], path: Path) -> None:
    if not policies:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(policies[0].keys()))
        w.writeheader()
        w.writerows(policies)
    print(f"  CSV: {path}")


def _write_markdown(policies: list[dict], label: str, path: Path) -> None:
    if not policies:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    policies_sorted = sorted(policies, key=lambda p: p["ttft_p95"])
    lines = [
        f"# {label}\n",
        f"| Policy | OK | Succ% | TTFT p50 | p95 | p99 | max | E2E p95 | E2E p99 | req/s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for p in policies_sorted:
        lines.append(
            f"| {p['policy']} | {p['ok']:,} | {p['success_rate'] * 100:.1f}% "
            f"| {p['ttft_p50']:.3f}s | {p['ttft_p95']:.3f}s | {p['ttft_p99']:.3f}s "
            f"| {p['ttft_max']:.2f}s | {p['e2e_p95']:.3f}s | {p['e2e_p99']:.3f}s "
            f"| {p['rps']:.2f} |"
        )
    best = policies_sorted[0]
    worst = policies_sorted[-1]
    lines.append(
        f"\np95 spread: {best['ttft_p95']:.3f}s ({best['policy']}) → "
        f"{worst['ttft_p95']:.3f}s ({worst['policy']}) = "
        f"{worst['ttft_p95'] / best['ttft_p95']:.2f}×\n"
    )
    path.write_text("\n".join(lines) + "\n")
    print(f"  Markdown: {path}")


def _write_comparison(p1: list[dict], p2: list[dict], label1: str, label2: str, path: Path) -> None:
    d1 = {p["policy"]: p for p in p1}
    d2 = {p["policy"]: p for p in p2}
    common = [p for p in d1 if p in d2]
    if not common:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {label1} vs {label2}\n",
        f"| Policy | {label1} p95 | {label2} p95 | Δ p95 | {label1} p99 | {label2} p99 | Δ p99 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pol in sorted(common):
        a, b = d1[pol], d2[pol]
        dp95 = (b["ttft_p95"] - a["ttft_p95"]) / a["ttft_p95"] * 100 if a["ttft_p95"] else 0
        dp99 = (b["ttft_p99"] - a["ttft_p99"]) / a["ttft_p99"] * 100 if a["ttft_p99"] else 0
        lines.append(
            f"| {pol} | {a['ttft_p95']:.3f}s | {b['ttft_p95']:.3f}s | {dp95:+.1f}% "
            f"| {a['ttft_p99']:.3f}s | {b['ttft_p99']:.3f}s | {dp99:+.1f}% |"
        )
    path.write_text("\n".join(lines) + "\n")
    print(f"  Comparison: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True, help="Workload result file prefix")
    parser.add_argument("--prefix2", default="", help="Second prefix for comparison")
    parser.add_argument("--label", default="Run 1", help="Label for first run")
    parser.add_argument("--label2", default="Run 2", help="Label for second run")
    parser.add_argument(
        "--results-dir", default="results/workload_runs", help="Directory with result JSONs"
    )
    parser.add_argument("--out-dir", default="results/analysis", help="Output directory for CSV/MD")
    args = parser.parse_args()

    out = Path(args.out_dir)

    p1 = _load_policies(args.results_dir, args.prefix)
    if not p1:
        print(f"No results found for prefix {args.prefix!r} in {args.results_dir}")
        return

    slug1 = args.prefix.replace("/", "_")
    print(f"\n{args.label}: {len(p1)} policies")
    _write_csv(p1, out / f"{slug1}.csv")
    _write_markdown(p1, args.label, out / f"{slug1}.md")

    if args.prefix2:
        p2 = _load_policies(args.results_dir, args.prefix2)
        if p2:
            slug2 = args.prefix2.replace("/", "_")
            print(f"\n{args.label2}: {len(p2)} policies")
            _write_csv(p2, out / f"{slug2}.csv")
            _write_markdown(p2, args.label2, out / f"{slug2}.md")
            _write_comparison(p1, p2, args.label, args.label2, out / f"{slug1}_vs_{slug2}.md")

    # Print summary table to stdout
    for p in sorted(p1, key=lambda x: x["ttft_p95"]):
        print(
            f"  {p['policy']:<28} p95={p['ttft_p95']:.3f}s  p99={p['ttft_p99']:.3f}s  e2e_p95={p['e2e_p95']:.3f}s"
        )


if __name__ == "__main__":
    main()
