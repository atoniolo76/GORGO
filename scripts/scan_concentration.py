"""Scan every gorgo proxy-trace on the bench-results volume and report the
routing concentration (max share of requests sent to a single replica).

Reads the volume directly (no local downloads). Flags runs where one replica
got ~100% of traffic -- the reward-hacking / single-replica-concentration
signature.

    modal run --env=alessio-dev scripts/scan_concentration.py::main
"""

from __future__ import annotations

import json
import os
from collections import Counter

import modal

from app import app, bench_results_volume

image = modal.Image.debian_slim().add_local_python_source("app")


@app.function(image=image, volumes={"/results": bench_results_volume}, timeout=3600)
def scan():
    base = "/results/proxy_traces"
    rows = []
    for d in sorted(os.listdir(base)):
        if "gorgo" not in d:
            continue
        req = os.path.join(base, d, "requests.jsonl")
        if not os.path.exists(req):
            continue
        counter: Counter = Counter()
        n = 0
        try:
            with open(req) as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        e = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if e.get("kind") != "request":
                        continue
                    tgt = e.get("target")
                    if not tgt:
                        continue
                    counter[tgt] += 1
                    n += 1
        except OSError:
            continue
        if n == 0:
            continue
        top = counter.most_common(1)[0]
        max_share = top[1] / n
        rows.append(
            {
                "run": d,
                "n": n,
                "n_replicas": len(counter),
                "max_share": max_share,
            }
        )

    rows.sort(key=lambda r: -r["max_share"])
    print(f"\n{'=' * 90}")
    print(
        f"ROUTING CONCENTRATION ACROSS {len(rows)} GORGO PROXY TRACES (sorted by max single-replica share)"
    )
    print(f"{'=' * 90}")
    print(f"{'max_share':>10} {'n_repl':>7} {'n_req':>8}  run")
    for r in rows:
        flag = (
            "  <== 100%"
            if r["max_share"] >= 0.999
            else ("  <== >=99%" if r["max_share"] >= 0.99 else "")
        )
        print(f"{r['max_share'] * 100:9.1f}% {r['n_replicas']:>7} {r['n']:>8}  {r['run']}{flag}")

    hundred = [r for r in rows if r["max_share"] >= 0.999]
    print(f"\n{len(hundred)} run(s) at ~100% concentration:")
    for r in hundred:
        print(f"  {r['run']}  (n={r['n']})")
    return rows


def _conc(requests_path: str) -> tuple[float, int, int]:
    if not requests_path or not os.path.exists(requests_path):
        return (0.0, 0, 0)
    c: Counter = Counter()
    n = 0
    with open(requests_path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("kind") != "request" or not e.get("target"):
                continue
            c[e["target"]] += 1
            n += 1
    if n == 0:
        return (0.0, 0, 0)
    return (c.most_common(1)[0][1] / n, len(c), n)


def _metrics(stats: dict) -> dict:
    t = stats.get("ttft_seconds") or {}
    e = stats.get("request_e2e_seconds") or {}
    itl = stats.get("itl_ms") or {}
    return {
        "ttft_p50": t.get("p50"),
        "ttft_p95": t.get("p95"),
        "ttft_p99": t.get("p99"),
        "e2e_p50": e.get("p50"),
        "e2e_p95": e.get("p95"),
        "e2e_p99": e.get("p99"),
        "itl_avg_ms": itl.get("avg"),
        "decode_tok_s": stats.get("output_token_throughput"),
    }


# lower-is-better for all except decode_tok_s
_LOWER_BETTER = {"ttft_p50", "ttft_p95", "ttft_p99", "e2e_p50", "e2e_p95", "e2e_p99", "itl_avg_ms"}


@app.function(image=image, volumes={"/results": bench_results_volume}, timeout=3600)
def scan_runs():
    import glob

    matrices = glob.glob("/results/policy_matrix_sweep/**/*sweep_matrix.json", recursive=True)
    print(f"[scan_runs] {len(matrices)} sweep matrices found")
    out = []
    for mp in matrices:
        try:
            m = json.load(open(mp))
        except (json.JSONDecodeError, OSError):
            continue
        for tr in m.get("results") or []:
            man = tr.get("manifest") or {}
            pols = man.get("results") or []
            gorgo = None
            for r in pols:
                if isinstance(r, dict) and "gorgo" in (r.get("label") or r.get("policy") or ""):
                    gorgo = r
                    break
            if not gorgo:
                continue
            paths = (gorgo.get("trace") or {}).get("paths") or {}
            share, nrep, nreq = _conc(paths.get("requests_path", ""))
            if share < 0.999:
                continue
            # collect all-policy metrics
            rows = []
            for r in pols:
                if not isinstance(r, dict) or r.get("error"):
                    continue
                lbl = r.get("label") or r.get("policy")
                st = (r.get("workload") or {}).get("stats") or {}
                rows.append((lbl, _metrics(st)))
            hp = gorgo.get("hyperparameters") or (gorgo.get("auto_tune") or {}).get(
                "hyperparameters"
            )
            out.append(
                {
                    "matrix_path": mp,
                    "run_id": man.get("run_id"),
                    "gorgo_label": gorgo.get("label"),
                    "gorgo_hp": hp,
                    "concentration": share,
                    "n_replicas": nrep,
                    "n_req": nreq,
                    "policies": rows,
                }
            )

    print(f"\n{'#' * 90}")
    print(f"RUNS WITH ~100% GORGO CONCENTRATION: {len(out)}")
    print(f"{'#' * 90}")
    for o in out:
        print(
            f"\n=== {o['run_id']}  (gorgo={o['gorgo_label']}, conc={o['concentration'] * 100:.1f}%, "
            f"replicas={o['n_replicas']}, n={o['n_req']}) ==="
        )
        print(f"    gorgo hyperparams: {json.dumps(o['gorgo_hp'])}")
        metrics = [
            "ttft_p50",
            "ttft_p95",
            "ttft_p99",
            "e2e_p50",
            "e2e_p95",
            "e2e_p99",
            "itl_avg_ms",
            "decode_tok_s",
        ]
        # header
        print(f"    {'policy':26s} " + " ".join(f"{mk:>10}" for mk in metrics))
        for lbl, mvals in o["policies"]:

            def fmt(mk):
                v = mvals.get(mk)
                if v is None:
                    return "-"
                return (
                    f"{v * 1000:.0f}"
                    if mk in {"ttft_p50", "ttft_p95", "ttft_p99"}
                    else (f"{v:.2f}" if mk in {"e2e_p50", "e2e_p95", "e2e_p99"} else f"{v:.1f}")
                )

            print(f"    {lbl:26s} " + " ".join(f"{fmt(mk):>10}" for mk in metrics))
        # improvement of gorgo over next-best per metric
        gorgo_row = next((mv for lbl, mv in o["policies"] if "gorgo" in lbl), None)
        others = [mv for lbl, mv in o["policies"] if "gorgo" not in lbl]
        print(f"    {'IMPROVEMENT vs next-best:':26s}")
        for mk in metrics:
            gv = gorgo_row.get(mk) if gorgo_row else None
            ovals = [mv.get(mk) for mv in others if mv.get(mk) is not None]
            if gv is None or not ovals:
                continue
            if mk in _LOWER_BETTER:
                nb = min(ovals)
                impr = (nb - gv) / nb * 100
            else:
                nb = max(ovals)
                impr = (gv - nb) / nb * 100
            tag = "BETTER" if impr > 0 else "WORSE"
            print(
                f"        {mk:12s}: gorgo {'beats' if impr > 0 else 'loses to'} next-best by {impr:+.1f}%  ({tag})"
            )
    return out


@app.local_entrypoint()
def main(runs: bool = True):
    if runs:
        scan_runs.remote()
    else:
        rows = scan.remote()
        print(json.dumps(rows, indent=2))
