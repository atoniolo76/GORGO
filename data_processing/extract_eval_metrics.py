"""Extract E2E + ITL + TTFT percentiles for the apr6/apr7 held-out eval runs.

Scans the GORGO-bench-results volume for the per-policy result JSONs of
``glm5_c64_eval_ts2_apr6`` and ``glm5_c64_eval_ts3_apr7`` and prints
TTFT/E2E/ITL p50/p95/p99 per policy. Falls back to recomputing from the
per-request ``requests.jsonl`` trace if a harvested JSON lacks the metrics.

Usage::

    modal run --env=alessio-dev data_processing/extract_eval_metrics.py::main
"""

from __future__ import annotations

import json

import modal

from app import app, bench_results_volume

image = modal.Image.debian_slim().add_local_python_source("app")

RUN_IDS = ["glm5_c64_eval_ts2_apr6", "glm5_c64_eval_ts3_apr7", "glm5_c64_eval_ts2_apr7"]


def _find(o, key):
    if isinstance(o, dict):
        if key in o and isinstance(o[key], dict) and "p50" in o[key]:
            return o[key]
        for v in o.values():
            r = _find(v, key)
            if r:
                return r
    return None


@app.function(
    image=image, memory=1024 * 8, timeout=3600, volumes={"/results": bench_results_volume}
)
def extract() -> dict:
    import os

    # Locate per-policy result JSONs for each run id.
    matches: dict[str, list[str]] = {r: [] for r in RUN_IDS}
    for root, _dirs, files in os.walk("/results"):
        for f in files:
            if not f.endswith(".json"):
                continue
            for rid in RUN_IDS:
                if rid in f or rid in root:
                    matches[rid].append(os.path.join(root, f))

    out: dict[str, dict] = {}
    for rid, paths in matches.items():
        out[rid] = {}
        # de-dupe, keep per-policy result files (skip sweep_matrix aggregates)
        for p in sorted(set(paths)):
            base = os.path.basename(p)
            if "sweep_matrix" in base:
                continue
            try:
                o = json.load(open(p))
            except Exception as e:  # noqa: BLE001
                continue
            if not isinstance(o, dict) or "config" not in o:
                continue
            label = (o.get("config", {}).get("proxy", {}) or {}).get("label")
            policy = label or (o.get("config", {}).get("proxy", {}) or {}).get("policy") or base
            t = _find(o, "request_ttft_seconds") or _find(o, "ttft_seconds")
            e = _find(o, "request_e2e_seconds")
            i = _find(o, "itl_ms")
            rec = {"path": p}
            rec["n"] = (e or {}).get("n")
            if t:
                rec["ttft_ms"] = {k: round(t[k] * 1000) for k in ("p50", "p95", "p99")}
            if e:
                rec["e2e_s"] = {k: round(e[k], 2) for k in ("p50", "p95", "p99")}
            if i:
                rec["itl_ms"] = {k: round(i[k], 1) for k in ("p50", "p95", "p99")}
            out[rid][policy] = rec
        print(f"[{rid}] {len(out[rid])} policies", flush=True)
        for pol, rec in out[rid].items():
            print(
                f"  {pol}: {json.dumps({k: v for k, v in rec.items() if k != 'path'})}", flush=True
            )
    return out


def _pct(vals, p):
    if not vals:
        return None
    vals = sorted(vals)
    i = min(len(vals) - 1, max(0, int(round(p * (len(vals) - 1)))))
    return vals[i]


@app.function(
    image=image, memory=1024 * 8, timeout=3600, volumes={"/results": bench_results_volume}
)
def recompute_loadsweep(run_substr: str = "glm5_c64_loadsweep_apr7_ts3") -> dict:
    """Recompute TTFT/E2E/ITL percentiles from requests.jsonl (status==200)."""
    import os

    # Find requests.jsonl files under dirs matching the run.
    found: list[str] = []
    for root, _dirs, files in os.walk("/results"):
        if run_substr in root and "requests.jsonl" in files:
            found.append(os.path.join(root, "requests.jsonl"))

    out: dict[str, dict] = {}
    for path in sorted(found):
        # policy = trailing dir segment after the window tag
        dirname = os.path.basename(os.path.dirname(path))
        policy = dirname.split("_1945_to_2015_")[-1]
        ttft, e2e, itl = [], [], []
        with open(path) as f:
            for line in f:
                if '"kind":"request"' not in line and '"kind": "request"' not in line:
                    continue
                try:
                    e = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if e.get("kind") != "request" or e.get("status") != 200:
                    continue
                tn = e.get("ttft_ns")
                tot = e.get("total_ns")
                ct = e.get("completion_tokens") or 0
                if tn is None or tot is None:
                    continue
                ttft.append(tn / 1e9)
                e2e.append(tot / 1e9)
                if ct > 1:
                    itl.append((tot - tn) / 1e6 / (ct - 1))
        out[policy] = {
            "n": len(ttft),
            "ttft_ms": {
                k: round(_pct(ttft, p) * 1000)
                for k, p in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99))
            },
            "e2e_s": {
                k: round(_pct(e2e, p), 2) for k, p in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99))
            },
            "itl_ms": {
                k: round(_pct(itl, p), 1) for k, p in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99))
            },
        }
        print(f"  {policy}: " + json.dumps(out[policy]), flush=True)
    return out


@app.function(
    image=image, memory=1024 * 8, timeout=3600, volumes={"/results": bench_results_volume}
)
def recompute_run(run_substr: str) -> dict:
    """Recompute TTFT/E2E/ITL p50/p95/p99 from requests.jsonl (status==200).

    Generic over any run: matches dirs whose path contains ``run_substr`` and
    keys results by the per-policy directory name (full dirname, so nothing is
    lost when the window tag varies).
    """
    import os

    found: list[str] = []
    for root, _dirs, files in os.walk("/results"):
        if run_substr in root and "requests.jsonl" in files:
            found.append(os.path.join(root, "requests.jsonl"))

    out: dict[str, dict] = {}
    for path in sorted(found):
        policy = os.path.basename(os.path.dirname(path))
        ttft, e2e, itl = [], [], []
        with open(path) as f:
            for line in f:
                if '"kind":"request"' not in line and '"kind": "request"' not in line:
                    continue
                try:
                    e = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if e.get("kind") != "request" or e.get("status") != 200:
                    continue
                tn = e.get("ttft_ns")
                tot = e.get("total_ns")
                ct = e.get("completion_tokens") or 0
                if tn is None or tot is None:
                    continue
                ttft.append(tn / 1e9)
                e2e.append(tot / 1e9)
                if ct > 1:
                    itl.append((tot - tn) / 1e6 / (ct - 1))
        out[policy] = {
            "path": path,
            "n": len(ttft),
            "ttft_ms": {
                k: round(_pct(ttft, p) * 1000)
                for k, p in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99))
            },
            "e2e_s": {
                k: round(_pct(e2e, p), 2) for k, p in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99))
            },
            "itl_ms": {
                k: round(_pct(itl, p), 1) for k, p in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99))
            },
        }
        print(
            f"  {policy} (n={out[policy]['n']}): "
            + json.dumps({k: v for k, v in out[policy].items() if k not in ("path", "n")}),
            flush=True,
        )
    if not out:
        print(f"  no requests.jsonl found for run_substr={run_substr!r}", flush=True)
    return out


@app.local_entrypoint()
def hacked(
    run_substr: str = "glm5_c64_eval_p95ttft_diurnal_v2",
    out_path: str = "results/decoded_v9/hacked_ablation_metrics.json",
):
    import os

    result = recompute_run.remote(run_substr)
    print(json.dumps(result, indent=2))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {out_path}")


@app.local_entrypoint()
def loadsweep(out_path: str = "results/decoded_v9/apr7_ts3_loadsweep_metrics.json"):
    import os

    result = recompute_loadsweep.remote()
    print(json.dumps(result, indent=2))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {out_path}")


@app.local_entrypoint()
def main(out_path: str = "results/decoded_v9/eval_full_metrics.json"):
    import os

    result = extract.remote()
    print(json.dumps(result, indent=2))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {out_path}")
