"""Run the same Mooncake trace against multiple policy/proxy pairs.

This is intentionally a lightweight orchestrator: each proxy owns its local
workload execution via ``POST /workload/start``. The script only configures
policies, starts/stops traces, starts workloads at roughly the same time, polls
status, saves traces, and writes a matrix manifest.

Spec shape:

{
  "trace_path": "/data/mooncake_traces/.../with_bodies/trace.jsonl",
  "run_id": "moon_seed0",
  "concurrency": 64,
  "stream": true,
  "max_tokens": 1024,
  "policies": [
    {"name": "gorgo", "proxy_url": "https://..."},
    {"name": "least-load", "proxy_url": "https://..."}
  ]
}
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx


TERMINAL = {"succeeded", "failed", "cancelled"}


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s).strip("_")


def _start_at_wall_time(spec: dict) -> str:
    explicit = spec.get("start_at_wall_time")
    if explicit:
        return explicit
    delay = float(spec.get("start_delay_seconds", 30.0))
    return (
        (datetime.now(timezone.utc) + timedelta(seconds=delay)).isoformat().replace("+00:00", "Z")
    )


async def _post_json(client: httpx.AsyncClient, path: str, payload: dict | None = None) -> dict:
    r = await client.post(path, json=payload or {})
    r.raise_for_status()
    return r.json()


async def _get_json(client: httpx.AsyncClient, path: str) -> dict:
    r = await client.get(path)
    r.raise_for_status()
    return r.json()


async def run_one_policy(global_spec: dict, policy_spec: dict) -> dict:
    name = policy_spec["name"]
    label = policy_spec.get("label") or name
    proxy_url = policy_spec["proxy_url"].rstrip("/")
    run_id = f"{global_spec.get('run_id', 'policy_matrix')}_{_slug(label)}"
    trace_id = policy_spec.get("trace_id") or run_id
    output_path = global_spec.get(
        "output_path_template", "/results/workload_runs/{run_id}.json"
    ).format(
        run_id=run_id,
        policy=_slug(name),
    )

    timeout = httpx.Timeout(connect=20.0, read=60.0, write=30.0, pool=20.0)
    async with httpx.AsyncClient(base_url=proxy_url, timeout=timeout) as client:
        await _post_json(client, "/policy", {"policy": name})
        if policy_spec.get("hyperparameters"):
            await _post_json(client, "/hyperparameters", policy_spec["hyperparameters"])
        await _post_json(client, "/flush")
        await _post_json(
            client,
            "/trace/start",
            {
                "trace_id": trace_id,
                "sample_metrics": True,
                "sample_requests": True,
                "max_events": int(global_spec.get("max_trace_events", 200_000)),
            },
        )
        start_doc = await _post_json(
            client,
            "/workload/start",
            {
                "data_path": global_spec["trace_path"],
                "run_id": run_id,
                "concurrency": int(global_spec.get("concurrency", 16)),
                "model": global_spec.get("model", ""),
                "stream": global_spec.get("stream", True),
                "max_tokens": int(global_spec.get("max_tokens", 0)),
                "max_input_tokens": int(global_spec.get("max_input_tokens", 0)),
                "arrival_mode": global_spec.get("arrival_mode", "open-loop"),
                "time_scale": float(global_spec.get("time_scale", 1.0)),
                "output_path": output_path,
                "save_per_request": bool(global_spec.get("save_per_request", True)),
                "start_at_wall_time": global_spec["start_at_wall_time"],
            },
        )

        started = time.time()
        poll_interval = float(global_spec.get("poll_interval_seconds", 5.0))
        while True:
            status_doc = await _get_json(client, "/workload/status")
            workload = status_doc.get("workload") or {}
            if workload.get("status") in TERMINAL:
                break
            await asyncio.sleep(poll_interval)

        await _post_json(client, "/trace/stop")
        trace_doc = await _post_json(client, "/trace/save")

    return {
        "policy": name,
        "label": label,
        "proxy_url": proxy_url,
        "run_id": run_id,
        "trace_id": trace_id,
        "started_doc": start_doc,
        "workload": workload,
        "trace": trace_doc,
        "elapsed_seconds": time.time() - started,
    }


async def main_async(spec: dict) -> dict:
    spec = dict(spec)
    spec["start_at_wall_time"] = _start_at_wall_time(spec)
    tasks = [run_one_policy(spec, p) for p in spec["policies"]]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    normalized = []
    for policy, result in zip(spec["policies"], results):
        if isinstance(result, Exception):
            normalized.append(
                {
                    "policy": policy.get("name"),
                    "proxy_url": policy.get("proxy_url"),
                    "error": f"{type(result).__name__}: {result}",
                }
            )
        else:
            normalized.append(result)
    manifest = {
        "run_id": spec.get("run_id"),
        "trace_path": spec.get("trace_path"),
        "start_at_wall_time": spec.get("start_at_wall_time"),
        "policies": [p.get("label") or p.get("name") for p in spec["policies"]],
        "results": normalized,
    }
    out_path = Path(
        spec.get("manifest_path", f"results/policy_matrix/{spec.get('run_id', 'run')}.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True, type=Path)
    args = p.parse_args()
    spec = json.loads(args.spec.read_text())
    manifest = asyncio.run(main_async(spec))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
