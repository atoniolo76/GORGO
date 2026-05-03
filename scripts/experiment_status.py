"""Status summary for a policy matrix experiment run."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import proxies, replicas


def _status_for_proxy(key: str, url: str, timeout: float) -> str:
    base = url.rstrip("/")
    try:
        w = httpx.get(base + "/workload/status", timeout=timeout).json()["workload"]
        t = httpx.get(base + "/trace/status", timeout=timeout).json()["trace"]
        return (
            f"{key:45} "
            f"workload={str(w.get('status')):10} "
            f"running={str(w.get('running')):5} "
            f"req_events={str(t.get('request_events')):>6} "
            f"metrics_events={str(t.get('metrics_events')):>6} "
            f"saved={t.get('saved_paths') is not None} "
            f"output={w.get('output_path')}"
        )
    except Exception as e:
        return f"{key:45} ERROR {type(e).__name__}: {e}"


def _volume_ls(path: str, env: str) -> str:
    try:
        out = subprocess.check_output(
            ["modal", "volume", "ls", f"--env={env}", "GORGO-bench-results", path],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=20,
        )
        return out.strip()
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-id", default="neurips_h100_matrix_v2")
    p.add_argument("--env", default="alessio-dev")
    p.add_argument("--timeout", type=float, default=10.0)
    p.add_argument(
        "--result-path",
        default="/policy_matrix_sweep/moon_neurips_one_trace_v2",
        help="Path inside GORGO-bench-results volume to list.",
    )
    args = p.parse_args()

    rep_rows = sorted(
        (k, v) for k, v in replicas.items() if str(k).startswith(args.experiment_id) and v
    )
    proxy_rows = sorted(
        (k, v) for k, v in proxies.items() if str(k).startswith(args.experiment_id) and v
    )

    print(f"experiment_id: {args.experiment_id}")
    print(f"replicas_registered: {len(rep_rows)}")
    print(f"proxies_registered: {len(proxy_rows)}")
    print()
    print("proxy workload/trace status:")
    for key, url in proxy_rows:
        print("  " + _status_for_proxy(str(key), url, args.timeout))

    print()
    print(f"bench-results {args.result_path}:")
    listing = _volume_ls(args.result_path, args.env)
    print(listing or "  <empty or not found>")


if __name__ == "__main__":
    main()
