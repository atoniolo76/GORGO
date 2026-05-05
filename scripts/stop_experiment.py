"""Gracefully stop a running experiment and save partial results.

Hits every proxy registered under the experiment_id in the GORGO-proxies
Modal Dict with POST /workload/stop (cancel the running workload) then
POST /trace/stop + POST /trace/save (persist whatever events landed).

The matrix controller's _run_one_policy will see the workload transition
to status=cancelled, collect the partial stats, stop+save the trace, and
write the manifest as usual -- so you get a complete (if truncated)
result set.

Usage:
    python scripts/stop_experiment.py --experiment-id bench_lmsys_20M_v4

To just check status without stopping:
    python scripts/experiment_status.py --experiment-id bench_lmsys_20M_v4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import proxies


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-id", required=True)
    p.add_argument("--env", default="alessio-dev")
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument(
        "--skip-save",
        action="store_true",
        help="Stop workloads but don't save traces (faster, no volume write).",
    )
    args = p.parse_args()

    proxy_rows = sorted(
        (k, v) for k, v in proxies.items() if str(k).startswith(args.experiment_id) and v
    )
    if not proxy_rows:
        print(f"no proxies found for experiment_id={args.experiment_id!r}")
        return

    print(f"found {len(proxy_rows)} proxies for {args.experiment_id}")
    for key, url in proxy_rows:
        base = url.rstrip("/")
        label = str(key).replace(f"{args.experiment_id}-", "")
        try:
            # Stop the workload
            w = httpx.post(base + "/workload/stop", timeout=args.timeout).json()
            ws = (w.get("workload") or {}).get("status", "?")
            print(f"  {label:<30} /workload/stop -> status={ws}")

            if not args.skip_save:
                # Stop trace capture
                httpx.post(base + "/trace/stop", json={}, timeout=args.timeout)
                # Save trace to volume
                s = httpx.post(base + "/trace/save", json={}, timeout=args.timeout).json()
                paths = s.get("paths") or {}
                print(f"  {label:<30} /trace/save -> {paths.get('manifest_path', '(none)')}")
            else:
                print(f"  {label:<30} (skipped trace save)")
        except Exception as e:
            print(f"  {label:<30} ERROR: {type(e).__name__}: {e}")

    print("\nDone. The matrix controller will see cancelled workloads and write the manifest.")
    print("Results will appear in the output_dir once the controller finishes.")


if __name__ == "__main__":
    main()
