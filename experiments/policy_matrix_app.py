"""Single-app policy matrix experiment controller.

This app avoids shelling out to ``modal run`` for engines/proxies. Region/GPU
settings are decorator-time in this Modal SDK, so we expose one engine function
per benchmark region and select the right one in the controller.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import modal

from app import (
    ENVIRONMENT_NAME,
    app,
    bench_results_volume,
    completions_volume,
    hf_datasets_volume,
    lmsys_chat_1m_volume,
    proxies,
    replicas,
)
from engine.modal_sglang import (
    CONTEXT_LENGTH,
    HF_REPO_ID,
    MODEL_REVISION,
    N_GPUS,
    PORT,
    SCALEDOWN_WINDOW_SECONDS,
    WAIT_READY_TIMEOUT,
    sglang_image,
    wait_ready,
)


PROXY_IMAGE = (
    modal.Image.debian_slim()
    .pip_install("httpx[http2]", "uvicorn", "tiktoken", "pyarrow", "datasets>=3.0")
    .add_local_python_source("app", "engine", "proxy", "policy", "utils")
)


def _serve_model(registry_key: str, tp_size: int | None = None) -> None:
    """Launch SGLang on this container.

    ``tp_size`` overrides the imported ``N_GPUS`` for the ``--tp`` arg.
    Set explicitly per-engine when the @app.function uses a different
    GPU count than the engine module's default (e.g. L40S:2 fleets).
    """
    os.environ["SGLANG_JIT_DEEPGEMM_FAST_WARMUP"] = "1"
    tp = tp_size if tp_size is not None else N_GPUS
    cmd = [
        "python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        HF_REPO_ID,
        "--revision",
        MODEL_REVISION,
        "--served-model-name",
        HF_REPO_ID,
        "--host",
        "0.0.0.0",
        "--port",
        f"{PORT}",
        "--tp",
        f"{tp}",
        "--cuda-graph-max-bs",
        f"{10 * 2}",
        "--enable-metrics",
        "--decode-log-interval",
        "100",
        "--mem-fraction",
        "0.8",
        "--context-length",
        f"{CONTEXT_LENGTH}",
    ]
    with modal.forward(PORT) as tunnel:
        print(f"tunnel.url        = {tunnel.url}")
        print(f"tunnel.tls_socket = {tunnel.tls_socket}")
        process = subprocess.Popen(cmd)
        try:
            wait_ready(process)
            replicas[registry_key] = tunnel.url
            print(replicas[registry_key])
            process.wait()
        finally:
            if replicas.get(registry_key) == tunnel.url:
                replicas[registry_key] = ""
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


@app.function(
    image=sglang_image,
    timeout=24 * 60 * 60,
    region="CANADA-2",
    gpu="H100",
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    volumes={
        "/root/.cache/huggingface": modal.Volume.from_name(
            "Qwen3.5-35B-A3B-FP8-huggingface-cache",
            create_if_missing=True,
            environment_name=ENVIRONMENT_NAME,
        )
    },
)
def engine_canada(registry_key: str) -> None:
    _serve_model(registry_key)


@app.function(
    image=sglang_image,
    timeout=24 * 60 * 60,
    region="sines-2",
    gpu="H100",
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    volumes={
        "/root/.cache/huggingface": modal.Volume.from_name(
            "Qwen3.5-35B-A3B-FP8-huggingface-cache",
            create_if_missing=True,
            environment_name=ENVIRONMENT_NAME,
        )
    },
)
def engine_sines(registry_key: str) -> None:
    _serve_model(registry_key)


@app.function(
    image=sglang_image,
    timeout=24 * 60 * 60,
    region="us-west4",
    gpu="H100",
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    volumes={
        "/root/.cache/huggingface": modal.Volume.from_name(
            "Qwen3.5-35B-A3B-FP8-huggingface-cache",
            create_if_missing=True,
            environment_name=ENVIRONMENT_NAME,
        )
    },
)
def engine_us_west4(registry_key: str) -> None:
    _serve_model(registry_key)


# ---- L40S:2 engines (tp=2; 35B FP8 weights ~35GB so 1xL40S is too tight) ----

_L40S_HF_VOLUME = {
    "/root/.cache/huggingface": modal.Volume.from_name(
        "Qwen3.5-35B-A3B-FP8-huggingface-cache",
        create_if_missing=True,
        environment_name=ENVIRONMENT_NAME,
    )
}


@app.function(
    image=sglang_image,
    timeout=24 * 60 * 60,
    region="ap-seoul-1",
    gpu="L40S:2",
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    volumes=_L40S_HF_VOLUME,
)
def engine_ap_seoul(registry_key: str) -> None:
    _serve_model(registry_key, tp_size=2)


@app.function(
    image=sglang_image,
    timeout=24 * 60 * 60,
    region="eu-frankfurt-1",
    gpu="L40S:2",
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    volumes=_L40S_HF_VOLUME,
)
def engine_eu_frankfurt(registry_key: str) -> None:
    _serve_model(registry_key, tp_size=2)


@app.function(
    image=sglang_image,
    timeout=24 * 60 * 60,
    region="us-ashburn-1",
    gpu="L40S:2",
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    volumes=_L40S_HF_VOLUME,
)
def engine_us_ashburn(registry_key: str) -> None:
    _serve_model(registry_key, tp_size=2)


@app.function(
    image=PROXY_IMAGE,
    region="us-east",
    timeout=24 * 60 * 60,
    volumes={
        "/data": completions_volume,
        "/lmsys": lmsys_chat_1m_volume,
        "/datasets": hf_datasets_volume,
        "/results": bench_results_volume,
    },
)
def proxy_runner(registry_key: str) -> None:
    from proxy.modal_proxy import proxy

    proxy.local(registry_key=registry_key)


ENGINE_BY_REGION = {
    "CANADA-2": engine_canada,
    "sines-2": engine_sines,
    "us-west4": engine_us_west4,
    "ap-seoul-1": engine_ap_seoul,
    "eu-frankfurt-1": engine_eu_frankfurt,
    "us-ashburn-1": engine_us_ashburn,
}

TERMINAL_WORKLOAD_STATUS = {"succeeded", "failed", "cancelled"}


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


def _with_bodies_path(path: str) -> str:
    p = Path(path)
    if p.parent.name == "with_bodies":
        return path
    return str(p.parent / "with_bodies" / p.name)


def _trace_stem(path: str) -> str:
    return Path(path).stem


def _unique_output_dir(base: str, experiment_id: str) -> str:
    """Return a collision-resistant result directory under ``base``.

    If ``base`` already appears to include the experiment id, still append a
    UTC timestamp so reruns do not overwrite previous manifests. This keeps
    volume layout predictable while making every controller invocation unique.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    root = Path(base)
    return str(root / f"{experiment_id}_{ts}")


def _label(policy: dict) -> str:
    return policy.get("label") or policy["name"]


def _homogeneity_config(spec: dict) -> dict:
    """Resolve the optional pre-flight replica homogeneity check config.

    Disabled by default to preserve existing behavior. When enabled, the
    controller hits each replica directly (bypassing the proxy/policy)
    with N small streaming chat-completions and records per-replica
    TTFT. This catches the Modal-cold-start variance that was the
    confound behind the spurious least-request advantage in
    ``moon_neurips_main_000_quick200`` -- one of three shared backends
    happened to be ~10x faster than the others, and the policy that
    accidentally herded onto it looked like a winner.

    Knobs:
      ``enabled``: master switch (default ``False``).
      ``warmup_requests_per_replica``: per-replica request count.
        First ``warmup_requests`` of those are discarded as warm-up;
        the remainder are kept for stats.
      ``warmup_requests``: how many of the first per-replica requests
        to discard. Must be < ``warmup_requests_per_replica``. The
        engine's own ``wait_ready`` already issues one warmup chat
        before declaring ready, but the *first* probe here pays
        TCP/TLS handshake on the controller -> tunnel route, so a
        discard >= 1 is recommended; the default 2 leaves a safety
        margin for any first-batch CUDA-graph capture SGLang does.
      ``max_tokens``: cap on output tokens per probe (small to keep
        the check fast; we only care about TTFT).
      ``max_ttft_ratio``: per-pool max-P50 / min-P50 TTFT ratio above
        which ``on_violation`` fires. ``0`` disables the gate (still
        records stats in the manifest).
      ``on_violation``: ``"warn"`` (default; logs and continues) or
        ``"abort"`` (raises ``RuntimeError``). Use ``"abort"`` when
        you want a clean fail-fast for paper-grade comparisons.
      ``request_timeout_seconds``: per-probe upstream timeout.
    """
    cfg = dict(spec.get("replica_homogeneity_check") or {})
    out = {
        "enabled": bool(cfg.get("enabled", False)),
        "warmup_requests_per_replica": int(cfg.get("warmup_requests_per_replica", 6)),
        "warmup_requests": int(cfg.get("warmup_requests", 2)),
        "max_tokens": int(cfg.get("max_tokens", 16)),
        "max_ttft_ratio": float(cfg.get("max_ttft_ratio", 0.0)),
        "on_violation": str(cfg.get("on_violation", "warn")).lower(),
        "request_timeout_seconds": float(cfg.get("request_timeout_seconds", 60.0)),
    }
    if out["on_violation"] not in ("warn", "abort"):
        raise ValueError(
            f"replica_homogeneity_check.on_violation must be 'warn' or 'abort', "
            f"got {out['on_violation']!r}"
        )
    if out["warmup_requests"] >= out["warmup_requests_per_replica"]:
        raise ValueError(
            "replica_homogeneity_check.warmup_requests must be strictly less than "
            "warmup_requests_per_replica (otherwise no measurement requests remain)"
        )
    return out


async def _measure_replica_ttft(
    client: httpx.AsyncClient,
    replica_url: str,
    *,
    total_requests: int,
    warmup_requests: int,
    max_tokens: int,
    request_timeout_seconds: float,
) -> dict:
    """Send ``total_requests`` short streaming chats directly to a
    replica and return TTFT stats over the post-warmup tail.

    Sequential per-replica (concurrency=1) so warm-up effects show up
    cleanly in the discarded prefix instead of contaminating the
    measurement window. Hits ``{replica_url}/v1/chat/completions``
    directly -- no proxy, no policy, no metrics interaction -- so the
    measurement is independent of any routing logic under test.
    """
    body = {
        "model": HF_REPO_ID,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    payload = json.dumps(body).encode()
    headers = {"accept-encoding": "identity", "content-type": "application/json"}
    measurement_ttfts_ms: list[float] = []
    measurement_total_ms: list[float] = []
    error_count = 0
    last_error: str | None = None
    for i in range(total_requests):
        started_ns = time.perf_counter_ns()
        ttft_ns: int | None = None
        try:
            async with client.stream(
                "POST",
                f"{replica_url.rstrip('/')}/v1/chat/completions",
                content=payload,
                headers=headers,
                timeout=request_timeout_seconds,
            ) as resp:
                if resp.status_code != 200:
                    error_count += 1
                    last_error = f"status={resp.status_code}"
                    await resp.aread()
                    continue
                buffer = bytearray()
                buffer_first_chunk_ns: int | None = None
                async for chunk in resp.aiter_raw():
                    chunk_ns = time.perf_counter_ns()
                    if not buffer:
                        buffer_first_chunk_ns = chunk_ns
                    buffer.extend(chunk)
                    while True:
                        idx = buffer.find(b"\n\n")
                        if idx < 0:
                            break
                        event = bytes(buffer[:idx])
                        del buffer[: idx + 2]
                        event_arrival_ns = buffer_first_chunk_ns
                        buffer_first_chunk_ns = chunk_ns if buffer else None
                        if ttft_ns is not None:
                            continue
                        for line in event.split(b"\n"):
                            if not line.startswith(b"data:"):
                                continue
                            data = line[len(b"data:") :].strip()
                            if not data or data == b"[DONE]":
                                continue
                            try:
                                obj = json.loads(data)
                            except json.JSONDecodeError:
                                continue
                            choices = obj.get("choices") or []
                            if not choices:
                                continue
                            delta = choices[0].get("delta") or {}
                            if delta.get("content"):
                                ttft_ns = (event_arrival_ns or chunk_ns) - started_ns
                                break
                    if ttft_ns is not None:
                        break
                # Drain the rest so the connection returns to the pool.
                async for _ in resp.aiter_raw():
                    pass
        except Exception as e:
            error_count += 1
            last_error = f"{type(e).__name__}: {e}"
            continue
        total_ns = time.perf_counter_ns() - started_ns
        if i < warmup_requests:
            continue
        if ttft_ns is None:
            error_count += 1
            last_error = "no content event received"
            continue
        measurement_ttfts_ms.append(ttft_ns / 1e6)
        measurement_total_ms.append(total_ns / 1e6)

    def _percentile(xs: list[float], p: float) -> float | None:
        if not xs:
            return None
        s = sorted(xs)
        k = max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))
        return s[k]

    return {
        "replica_url": replica_url,
        "requests_total": total_requests,
        "warmup_discarded": warmup_requests,
        "measured_n": len(measurement_ttfts_ms),
        "errors": error_count,
        "last_error": last_error,
        "ttft_ms_p50": _percentile(measurement_ttfts_ms, 0.50),
        "ttft_ms_p95": _percentile(measurement_ttfts_ms, 0.95),
        "ttft_ms_max": max(measurement_ttfts_ms) if measurement_ttfts_ms else None,
        "total_ms_p50": _percentile(measurement_total_ms, 0.50),
    }


async def _check_replica_homogeneity(fleet_manifest: dict, cfg: dict) -> dict:
    """Pre-flight per-pool TTFT measurement and homogeneity gate.

    Hits each replica directly so the result is independent of the
    routing policy that will later run against the same pool. Returns
    a manifest the caller can stash inside ``fleet_manifest``; raises
    ``RuntimeError`` when ``on_violation == "abort"`` and any pool's
    P50 TTFT spread exceeds ``max_ttft_ratio``.

    What this catches: level-shift heterogeneity -- one replica is
    routinely slower than its siblings (cold/wrong-tier instance,
    noisy neighbor, model still loading after ``wait_ready`` returned).
    This was the confound behind the spurious least-request advantage
    in ``moon_neurips_main_000_quick200``, where one of three shared
    backends was ~10x faster than the others.

    What this does NOT catch: prefill-shape heterogeneity. The probe
    uses a tiny ``"ping"`` prompt (TTFT dominated by RTT + minimal
    prefill), while real Mooncake-replay workloads carry up to 24k
    input tokens (TTFT dominated by full-prompt prefill). A replica
    that's CUDA-graph-warm for tiny prompts but not for 24k can pass
    this gate and still skew the workload. "Ratio green" means
    "replicas are equivalent at the probe shape," not "end-to-end TTFT
    will be uniform."

    On abort: this raises mid-experiment with the engine fleet still
    spawned. They idle until Modal's ``scaledown_window`` expires
    (set in each ``engine_*`` decorator) -- not a regression vs the
    success path, which also doesn't tear them down, but worth knowing
    when you're paying for GPUs.
    """
    if not cfg["enabled"]:
        return {"enabled": False}
    print(
        f"[homogeneity] checking {sum(len(p['replica_urls']) for p in fleet_manifest['fleets'])} "
        f"replicas across {len(fleet_manifest['fleets'])} pools "
        f"(probes={cfg['warmup_requests_per_replica']}, "
        f"warmup_discarded={cfg['warmup_requests']})",
        flush=True,
    )
    started = time.time()
    pools_report: list[dict] = []
    violations: list[str] = []
    async with httpx.AsyncClient() as client:
        for pool in fleet_manifest["fleets"]:
            label = pool["label"]
            replica_urls = pool["replica_urls"]
            per_replica = await asyncio.gather(
                *[
                    _measure_replica_ttft(
                        client,
                        url,
                        total_requests=cfg["warmup_requests_per_replica"],
                        warmup_requests=cfg["warmup_requests"],
                        max_tokens=cfg["max_tokens"],
                        request_timeout_seconds=cfg["request_timeout_seconds"],
                    )
                    for url in replica_urls
                ]
            )
            valid_p50s = [r["ttft_ms_p50"] for r in per_replica if r["ttft_ms_p50"] is not None]
            ratio: float | None = None
            if len(valid_p50s) >= 2:
                lo = min(valid_p50s)
                hi = max(valid_p50s)
                if lo > 0:
                    ratio = hi / lo
            measured_ns = sorted({r["measured_n"] for r in per_replica})
            uneven_sample = len(measured_ns) > 1
            pool_report = {
                "label": label,
                "policy": pool["policy"],
                "replica_count": len(replica_urls),
                "replicas": per_replica,
                "ttft_ms_p50_min": min(valid_p50s) if valid_p50s else None,
                "ttft_ms_p50_max": max(valid_p50s) if valid_p50s else None,
                "max_ttft_ratio_observed": ratio,
                "uneven_sample_sizes": uneven_sample,
            }
            pools_report.append(pool_report)
            print(
                f"[homogeneity] pool={label!r} "
                f"per_replica_p50_ms="
                f"{[round(r['ttft_ms_p50']) if r['ttft_ms_p50'] else None for r in per_replica]} "
                f"ratio={ratio if ratio is not None else 'n/a'}",
                flush=True,
            )
            if uneven_sample:
                # Per-replica error counts diverged, so the ratio is
                # comparing an unequal number of samples per replica.
                # Surface it so the operator knows the gate verdict
                # rests on noisier-than-expected stats.
                print(
                    f"[homogeneity][warn] pool {label!r} uneven measurement counts "
                    f"per replica: {measured_ns} (some probes failed)",
                    flush=True,
                )
            if cfg["max_ttft_ratio"] > 0 and ratio is not None and ratio > cfg["max_ttft_ratio"]:
                violations.append(
                    f"pool {label!r} ratio={ratio:.2f} > max_ttft_ratio={cfg['max_ttft_ratio']}"
                )
    elapsed = time.time() - started
    print(
        f"[homogeneity] done in {elapsed:.1f}s "
        f"violations={len(violations)} action={cfg['on_violation']!r}",
        flush=True,
    )
    report = {
        "enabled": True,
        "config": cfg,
        "elapsed_seconds": round(elapsed, 2),
        "pools": pools_report,
        "violations": violations,
    }
    if violations:
        if cfg["on_violation"] == "abort":
            print(
                f"[homogeneity][abort] {len(fleet_manifest['fleets'])} pools spawned; "
                "engine fleet will idle until each engine_* function's scaledown_window "
                "expires (no controller-side teardown today; matches the success path)",
                flush=True,
            )
            raise RuntimeError(
                "replica homogeneity check failed: "
                + "; ".join(violations)
                + ". Set replica_homogeneity_check.on_violation='warn' to ignore."
            )
        for v in violations:
            print(f"[homogeneity][warn] {v}", flush=True)
    return report


async def _wait_for_keys(dct, keys: list[str], timeout_s: float) -> dict[str, str]:
    deadline = time.time() + timeout_s
    pending = set(keys)
    found: dict[str, str] = {}
    while pending and time.time() < deadline:
        for key in list(pending):
            value = await dct.get.aio(key)
            if isinstance(value, str) and value:
                found[key] = value
                pending.remove(key)
                print(f"[ready] {key} -> {value}", flush=True)
        if pending:
            await asyncio.sleep(5)
    if pending:
        raise TimeoutError(f"timed out waiting for {sorted(pending)}")
    return found


async def _post_json(url: str, path: str, payload: dict | list) -> dict:
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(url.rstrip("/") + path, json=payload)
        r.raise_for_status()
        return r.json()


async def _get_json(url: str, path: str) -> dict:
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.get(url.rstrip("/") + path)
        r.raise_for_status()
        return r.json()


async def _wait_for_proxy_metrics(
    url: str,
    *,
    deadline_seconds: float = 10 * 60,
    poll_interval_seconds: float = 1.0,
) -> dict:
    started = time.time()
    deadline = started + deadline_seconds
    last_state: dict | None = None

    while time.time() < deadline:
        replicas_doc = await _get_json(url, "/replicas")
        replica_count = len(replicas_doc.get("replicas") or [])
        metrics_doc = await _get_json(url, "/replica_metrics")
        metrics = metrics_doc.get("metrics") or {}
        errors = metrics_doc.get("errors") or {}
        last_state = {
            "replica_count": replica_count,
            "metrics_count": len(metrics),
            "errors": errors,
            "last_refresh_age_seconds": metrics_doc.get("last_refresh_age_seconds"),
        }
        if replica_count > 0 and len(metrics) == replica_count and not errors:
            print(
                f"[metrics-ready] {url} replicas={replica_count} "
                f"waited={time.time() - started:.1f}s",
                flush=True,
            )
            return last_state
        await asyncio.sleep(poll_interval_seconds)

    raise TimeoutError(f"timed out waiting for proxy metrics at {url}: {last_state}")


def _workload_payload(
    *,
    spec: dict,
    run_id: str,
    trace_path: str,
    output_path: str,
    start_at_wall_time: str | None = None,
) -> dict:
    payload = {
        "data_path": trace_path,
        "run_id": run_id,
        "concurrency": int(spec.get("concurrency", 16)),
        "model": spec.get("model", ""),
        "stream": spec.get("stream", True),
        "max_tokens": int(spec.get("max_tokens", 0)),
        "max_input_tokens": int(spec.get("max_input_tokens", 0)),
        "arrival_mode": spec.get("arrival_mode", "open-loop"),
        "time_scale": float(spec.get("time_scale", 1.0)),
        "output_path": output_path,
        "save_per_request": bool(spec.get("save_per_request", True)),
    }
    if start_at_wall_time:
        payload["start_at_wall_time"] = start_at_wall_time
    return payload


async def _wait_for_workload(url: str, poll_interval: float) -> dict:
    while True:
        status_doc = await _get_json(url, "/workload/status")
        workload = status_doc.get("workload") or {}
        if workload.get("status") in TERMINAL_WORKLOAD_STATUS:
            return workload
        await asyncio.sleep(poll_interval)


def _auto_tune_payload(policy_spec: dict) -> dict | None:
    auto = policy_spec.get("auto_tune")
    if not auto:
        return None
    payload: dict = {
        "enabled": True,
        "window_size": int(auto.get("window_size", 200)),
        "hop_size": int(auto.get("hop_size", 50)),
        "apply": bool(auto.get("apply", True)),
    }
    # Forward the full auto-tune knob set so spec-driven `mode` /
    # `objective_metric` (and any future fields) reach the proxy. The
    # earlier short list silently dropped these, which made
    # ``gorgo-online-es`` fall back to the default ``fit`` mode.
    if "mode" in auto:
        payload["mode"] = str(auto["mode"])
    if "objective_metric" in auto:
        payload["objective_metric"] = str(auto["objective_metric"])
    return payload


async def _run_one_policy(global_spec: dict, policy_spec: dict) -> dict:
    name = policy_spec["name"]
    label = policy_spec.get("label") or name
    url = policy_spec["proxy_url"].rstrip("/")
    run_id = f"{global_spec.get('run_id', 'policy_matrix')}_{_slug(label)}"
    trace_id = policy_spec.get("trace_id") or run_id
    output_path = global_spec.get(
        "output_path_template", "/results/workload_runs/{run_id}.json"
    ).format(
        run_id=run_id,
        policy=_slug(name),
    )
    await _post_json(url, "/policy", {"policy": name})
    if policy_spec.get("hyperparameters"):
        await _post_json(url, "/hyperparameters", policy_spec["hyperparameters"])
    await _post_json(url, "/flush", {})
    auto_tune_config = _auto_tune_payload(policy_spec)
    if auto_tune_config:
        await _post_json(url, "/tune", auto_tune_config)
    metrics_ready = await _wait_for_proxy_metrics(
        url,
        deadline_seconds=float(global_spec.get("metrics_ready_timeout_seconds", 10 * 60)),
        poll_interval_seconds=float(global_spec.get("metrics_ready_poll_interval_seconds", 1.0)),
    )
    await _post_json(
        url,
        "/trace/start",
        {
            "trace_id": trace_id,
            "sample_metrics": True,
            "sample_requests": True,
            "max_events": int(global_spec.get("max_trace_events", 200_000)),
        },
    )
    await _post_json(
        url,
        "/workload/start",
        _workload_payload(
            spec=global_spec,
            run_id=run_id,
            trace_path=global_spec["trace_path"],
            output_path=output_path,
            start_at_wall_time=global_spec["start_at_wall_time"],
        ),
    )
    started = time.time()
    workload = await _wait_for_workload(url, float(global_spec.get("poll_interval_seconds", 5.0)))
    final_hyperparameters = None
    if auto_tune_config:
        await _post_json(url, "/tune", {"enabled": False})
        hps = await _get_json(url, "/hyperparameters")
        final_hyperparameters = hps.get("hyperparameters")
    await _post_json(url, "/trace/stop", {})
    trace_doc = await _post_json(url, "/trace/save", {})
    return {
        "policy": name,
        "label": label,
        "proxy_url": url,
        "run_id": run_id,
        "trace_id": trace_id,
        "auto_tune": (
            {"config": auto_tune_config, "hyperparameters": final_hyperparameters}
            if auto_tune_config
            else None
        ),
        "workload": workload,
        "trace": trace_doc,
        "metrics_ready": metrics_ready,
        "elapsed_seconds": time.time() - started,
    }


async def _run_policy_matrix(base_spec: dict) -> dict:
    spec = dict(base_spec)
    spec["start_at_wall_time"] = _start_at_wall_time(spec)
    results = await asyncio.gather(
        *[_run_one_policy(spec, p) for p in spec["policies"]],
        return_exceptions=True,
    )
    normalized = []
    for policy, result in zip(spec["policies"], results):
        if isinstance(result, Exception):
            normalized.append(
                {
                    "policy": policy.get("name"),
                    "label": policy.get("label") or policy.get("name"),
                    "proxy_url": policy.get("proxy_url"),
                    "error": f"{type(result).__name__}: {result}",
                }
            )
        else:
            normalized.append(result)
    return {
        "run_id": spec.get("run_id"),
        "trace_path": spec.get("trace_path"),
        "start_at_wall_time": spec.get("start_at_wall_time"),
        "policies": [p.get("label") or p.get("name") for p in spec["policies"]],
        "results": normalized,
    }


async def _run_sweep_matrix(
    *,
    base_spec: dict,
    sweep_manifest: dict,
    start_index: int,
    top_k: int,
    output_dir: Path,
) -> dict:
    top = sweep_manifest.get("top") or []
    selected = top[start_index : start_index + top_k]
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for local_idx, item in enumerate(selected):
        idx = start_index + local_idx
        result = item.get("result") or {}
        trace_path = result.get("output_path")
        if not trace_path:
            results.append({"ok": False, "index": idx, "error": "missing result.output_path"})
            continue
        body_trace = _with_bodies_path(trace_path)
        trace_name = _trace_stem(trace_path)
        spec = deepcopy(base_spec)
        base_run_id = base_spec.get("run_id", "policy_matrix")
        spec["trace_path"] = body_trace
        spec["run_id"] = f"{base_run_id}_{idx:03d}_{trace_name}"
        try:
            manifest = await _run_policy_matrix(spec)
            matrix_path = output_dir / f"{spec['run_id']}.json"
            matrix_path.write_text(json.dumps(manifest, indent=2))
            results.append(
                {
                    "ok": True,
                    "index": idx,
                    "trace_path": body_trace,
                    "source_trace": trace_path,
                    "matrix_manifest_path": str(matrix_path),
                    "manifest": manifest,
                }
            )
        except Exception as e:
            results.append(
                {
                    "ok": False,
                    "index": idx,
                    "trace_path": body_trace,
                    "source_trace": trace_path,
                    "error": f"{type(e).__name__}: {e}",
                }
            )
    aggregate = {
        "base_run_id": base_spec.get("run_id"),
        "sweep_manifest_path": sweep_manifest.get("_path"),
        "start_index": start_index,
        "top_k": top_k,
        "results": results,
    }
    aggregate_path = output_dir / f"{base_spec.get('run_id', 'policy_matrix')}_sweep_matrix.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2))
    aggregate["aggregate_manifest_path"] = str(aggregate_path)
    return aggregate


@app.function(
    image=modal.Image.debian_slim()
    .pip_install("httpx")
    .add_local_python_source("app", "engine", "proxy", "policy", "scripts", "utils"),
    volumes={"/results": bench_results_volume},
    timeout=24 * 60 * 60,
)
def run_policy_matrix_experiment(
    base_spec: dict,
    sweep_manifest: dict,
    experiment_id: str = "neurips_h100_matrix",
    start_index: int = 1,
    top_k: int = 1,
    output_dir: str = "/results/policy_matrix_sweep/moon_neurips_one_trace",
    engine_timeout_s: float = 45 * 60,
    proxy_timeout_s: float = 10 * 60,
) -> dict:
    return asyncio.run(
        _run_policy_matrix_experiment(
            base_spec=base_spec,
            sweep_manifest=sweep_manifest,
            experiment_id=experiment_id,
            start_index=start_index,
            top_k=top_k,
            output_dir=output_dir,
            engine_timeout_s=engine_timeout_s,
            proxy_timeout_s=proxy_timeout_s,
        )
    )


async def _run_policy_matrix_experiment(
    *,
    base_spec: dict,
    sweep_manifest: dict,
    experiment_id: str,
    start_index: int,
    top_k: int,
    output_dir: str,
    engine_timeout_s: float,
    proxy_timeout_s: float,
) -> dict:
    policies = base_spec["policies"]
    regions = base_spec.get("fleet_regions") or ["CANADA-2", "sines-2", "us-west4"]
    output_dir = _unique_output_dir(output_dir, experiment_id)

    engine_calls = []
    engine_keys = []
    for policy in policies:
        label = _label(policy)
        for region in regions:
            fn = ENGINE_BY_REGION[region]
            key = f"{experiment_id}-{label}-{region}"
            engine_keys.append(key)
            engine_calls.append(await fn.spawn.aio(key))

    replica_urls = await _wait_for_keys(replicas, engine_keys, engine_timeout_s)

    proxy_keys = []
    proxy_calls = []
    for policy in policies:
        label = _label(policy)
        key = f"{experiment_id}-{label}"
        proxy_keys.append(key)
        proxy_calls.append(await proxy_runner.spawn.aio(key))

    proxy_urls = await _wait_for_keys(proxies, proxy_keys, proxy_timeout_s)

    launched_spec = deepcopy(base_spec)
    launched_spec["policies"] = []
    fleet_manifest = {"experiment_id": experiment_id, "regions": regions, "fleets": []}

    for policy in policies:
        label = _label(policy)
        policy_replica_keys = [f"{experiment_id}-{label}-{region}" for region in regions]
        policy_replica_urls = [replica_urls[k] for k in policy_replica_keys]
        proxy_key = f"{experiment_id}-{label}"
        proxy_url = proxy_urls[proxy_key]
        await _post_json(proxy_url, "/replicas", {"replicas": policy_replica_urls})
        await _post_json(proxy_url, "/policy", {"policy": policy["name"]})
        if policy.get("hyperparameters"):
            await _post_json(proxy_url, "/hyperparameters", policy["hyperparameters"])
        p = deepcopy(policy)
        p["proxy_url"] = proxy_url
        launched_spec["policies"].append(p)
        fleet_manifest["fleets"].append(
            {
                "label": label,
                "policy": policy["name"],
                "proxy_key": proxy_key,
                "proxy_url": proxy_url,
                "replica_keys": policy_replica_keys,
                "replica_urls": policy_replica_urls,
            }
        )

    homogeneity_cfg = _homogeneity_config(base_spec)
    homogeneity_report = await _check_replica_homogeneity(fleet_manifest, homogeneity_cfg)
    fleet_manifest["homogeneity_check"] = homogeneity_report

    matrix = await _run_sweep_matrix(
        base_spec=launched_spec,
        sweep_manifest=sweep_manifest,
        start_index=start_index,
        top_k=top_k,
        output_dir=Path(output_dir),
    )
    return {"output_dir": output_dir, "fleet": fleet_manifest, "matrix": matrix}


@app.local_entrypoint()
def main(
    base_spec_path: str = "specs/policy_matrix_neurips_main.json",
    sweep_manifest_path: str = "specs/manifest.json",
    experiment_id: str = "neurips_h100_matrix",
    start_index: int = 1,
    top_k: int = 1,
    output_dir: str = "/results/policy_matrix_sweep/moon_neurips_one_trace",
):
    base_spec = json.loads(Path(base_spec_path).read_text())
    sweep_manifest = json.loads(Path(sweep_manifest_path).read_text())
    result = run_policy_matrix_experiment.remote(
        base_spec,
        sweep_manifest,
        experiment_id=experiment_id,
        start_index=start_index,
        top_k=top_k,
        output_dir=output_dir,
    )
    print(json.dumps(result, indent=2))
