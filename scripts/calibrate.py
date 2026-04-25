"""Cost-model calibration sweep on Modal (A100-80GB, GORGO env).

Implements Phase 2 of docs/calibration_plan.md (bead go-8cm). Launches
vLLM with Llama-3-8B-Instruct on a single A100-80GB in the
``arcadia-research`` / ``GORGO`` Modal environment, runs three
micro-benchmarks from §3.1 / §3.2 / §3.3 of the plan, and writes raw
per-request measurements as JSONL to a Modal Volume. The local
entrypoint then downloads the raw files to
``research/data/calibration/<timestamp>/`` and fits the five
``ComputeParams`` coefficients, emitting ``configs/calibrated_a100.yaml``
and a ``fit_summary.json``.

Run:

    MODAL_PROFILE=arcadia-research modal run --env=GORGO \\
        scripts/calibrate.py [--seed 0] [--function-timeout 14400]

``MODAL_PROFILE`` is prefixed per-invocation rather than via
``modal profile activate`` because profile activation is global process
state and is not safe under concurrent polecat sessions.

The ``hf_token_rome`` Modal secret is attached to the GPU function and
exposes ``HF_TOKEN_ROME`` inside the container; we map that to
``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` so vLLM can pull gated
Llama-3 weights.

The function hard-caps at 4h (14 400 s) per plan §7, bounding worst-case
spend at ~$14 even if the sweep hangs.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent

# --- Sweep parameters (mirror docs/calibration_plan.md §3) ------------------

DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
DEFAULT_GPU = "A100-80GB"
DEFAULT_MAX_MODEL_LEN = 4096
VLLM_VERSION = "0.6.6"

# §3.1 prefill
PREFILL_PROMPT_LENS = [64, 128, 256, 512, 1024, 2048, 4096]
PREFILL_REPEATS = 20

# §3.2 decode @ batch=1
DECODE_PROMPT_LEN = 128
DECODE_OUTPUT_LENS = [32, 64, 128, 256, 512, 1024]
DECODE_REPEATS = 15

# §3.3 decode batching
BATCH_PROMPT_LEN = 128
BATCH_OUTPUT_LEN = 128
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]
BATCH_REPEATS = 5

# vLLM startup may take several minutes to download weights (~16 GiB for
# Llama-3-8B). Cap the wait so we fail loud rather than burning budget.
VLLM_READY_TIMEOUT_S = 900


# --- Modal app / image / volumes --------------------------------------------

app = modal.App("gorgo-calibrate")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git")
    .pip_install(
        f"vllm=={VLLM_VERSION}",
        "httpx>=0.27",
        "numpy>=1.26",
        "PyYAML>=6.0",
        "transformers>=4.44",
    )
)

# Persistent HF cache so re-runs don't re-download the model.
hf_cache = modal.Volume.from_name("gorgo-hf-cache", create_if_missing=True)
# Raw calibration outputs live here; the local entrypoint reads them back.
calibration_vol = modal.Volume.from_name("gorgo-calibration", create_if_missing=True)


# --- Remote GPU function ----------------------------------------------------


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    timeout=14400,  # 4h hard cap per plan §7
    secrets=[modal.Secret.from_name("hf_token_rome")],
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/vol": calibration_vol,
    },
)
def run_calibration(
    model: str = DEFAULT_MODEL,
    seed: int = 0,
    timestamp: str = "",
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
) -> dict:
    """Launch vLLM, run the three sweeps, persist JSONL to the volume.

    Returns ``{"timestamp": ..., "run_dir": ..., "files": {name: bytes}}``
    so the local entrypoint can mirror the volume into the worktree
    without a separate download step.
    """
    import os

    token = os.environ.get("HF_TOKEN_ROME") or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "Neither HF_TOKEN_ROME nor HF_TOKEN is set in the container. "
            "Confirm the hf_token_rome Modal secret is attached to this "
            "function and is in the GORGO environment."
        )
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token

    if not timestamp:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path("/vol/calibration") / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== calibration run_dir={run_dir} seed={seed} ===", flush=True)

    vllm_cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--max-model-len",
        str(max_model_len),
        "--disable-log-requests",
        # Prefix caching must be off so §3.1's prefill sweep charges
        # the full prompt on every request; vLLM 0.6.x uses argparse
        # BooleanOptionalAction for this flag so the --no- form
        # disables it. Chunked prefill defaults to False in vLLM 0.6.x
        # for max_model_len <= 32k, so we rely on the default rather
        # than passing a flag with inconsistent CLI semantics.
        "--no-enable-prefix-caching",
        "--seed",
        str(seed),
        "--gpu-memory-utilization",
        "0.9",
    ]
    print(f"=== launching vLLM: {' '.join(vllm_cmd)} ===", flush=True)
    vllm_proc = subprocess.Popen(
        vllm_cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    try:
        _wait_for_vllm_ready("http://127.0.0.1:8000", VLLM_READY_TIMEOUT_S)

        metadata = {
            "model": model,
            "gpu": DEFAULT_GPU,
            "seed": seed,
            "timestamp": timestamp,
            "vllm_version": VLLM_VERSION,
            "max_model_len": max_model_len,
            "plan_path": "docs/calibration_plan.md",
            "bead": "go-8cm",
        }
        (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

        print("=== §3.1 prefill sweep ===", flush=True)
        _run_prefill_sweep(run_dir, model, seed)
        calibration_vol.commit()

        print("=== §3.2 decode@b=1 sweep ===", flush=True)
        _run_decode_single_sweep(run_dir, model, seed)
        calibration_vol.commit()

        print("=== §3.3 decode-batch sweep ===", flush=True)
        _run_decode_batch_sweep(run_dir, model, seed)
        calibration_vol.commit()

    finally:
        print("=== terminating vLLM ===", flush=True)
        vllm_proc.terminate()
        try:
            vllm_proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            vllm_proc.kill()

    files: dict[str, bytes] = {}
    for f in sorted(run_dir.rglob("*")):
        if f.is_file():
            files[f.relative_to(run_dir).as_posix()] = f.read_bytes()
    total = sum(len(v) for v in files.values())
    print(
        f"=== returning {len(files)} files, {total:,} bytes from {run_dir} ===",
        flush=True,
    )
    return {
        "timestamp": timestamp,
        "run_dir": str(run_dir),
        "files": files,
    }


# --- vLLM client helpers (run inside the Modal container) -------------------


def _wait_for_vllm_ready(base_url: str, timeout_s: int) -> None:
    import httpx

    deadline = time.monotonic() + timeout_s
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{base_url}/health", timeout=5.0)
            if r.status_code == 200:
                print(f"=== vLLM ready at {base_url} ===", flush=True)
                return
        except Exception as e:  # noqa: BLE001
            last_err = e
        time.sleep(5.0)
    raise RuntimeError(f"vLLM did not become ready within {timeout_s}s (last error: {last_err})")


def _random_prompt_ids(rng: random.Random, n_tokens: int) -> list[int]:
    """Random token IDs in a safe mid-vocab range.

    Llama-3 tokenizer has ~128k vocab; IDs in [1000, 100000) avoid
    special / reserved tokens and stay well away from the top of the
    table. We pass this list as the ``prompt`` field of
    ``/v1/completions`` — vLLM accepts a list of ints as a pre-
    tokenized prompt, which removes any tokenizer variance from the
    wall-clock measurement.
    """
    return [rng.randint(1000, 100000) for _ in range(n_tokens)]


def _measure_request(
    base_url: str,
    model: str,
    prompt_ids: list[int],
    max_tokens: int,
    seed: int,
) -> dict:
    """Issue a single streamed completion and return timing metrics."""
    import httpx

    body = {
        "model": model,
        "prompt": prompt_ids,
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "stream": True,
        "ignore_eos": True,
        "seed": seed,
    }
    t0 = time.perf_counter()
    ttft_ms: float | None = None
    chunks = 0
    with httpx.stream(
        "POST",
        f"{base_url}/v1/completions",
        json=body,
        timeout=httpx.Timeout(connect=30.0, read=600.0, write=30.0, pool=30.0),
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - t0) * 1000.0
            chunks += 1
    total_ms = (time.perf_counter() - t0) * 1000.0
    if ttft_ms is None:
        # Server produced no streamed chunk (error / empty). Treat total
        # as ttft so the row is at least recognizable as degenerate.
        ttft_ms = total_ms
    return {
        "prompt_len": len(prompt_ids),
        "max_tokens": max_tokens,
        "ttft_ms": ttft_ms,
        "total_ms": total_ms,
        "decode_ms": max(0.0, total_ms - ttft_ms),
        "chunks": chunks,
    }


def _run_prefill_sweep(run_dir: Path, model: str, seed: int) -> None:
    """§3.1 prefill sweep: vary prompt_len at max_tokens=1, batch=1."""
    rng = random.Random(seed)
    out_path = run_dir / "prefill.jsonl"
    with out_path.open("w") as fh:
        for prompt_len in PREFILL_PROMPT_LENS:
            for rep in range(PREFILL_REPEATS):
                prompt_ids = _random_prompt_ids(rng, prompt_len)
                m = _measure_request(
                    "http://127.0.0.1:8000",
                    model,
                    prompt_ids,
                    max_tokens=1,
                    seed=seed,
                )
                m.update(
                    phase="prefill",
                    batch_size=1,
                    rep=rep,
                    seed=seed,
                )
                fh.write(json.dumps(m) + "\n")
                fh.flush()
            print(
                f"  prefill prompt_len={prompt_len}: {PREFILL_REPEATS} reps done",
                flush=True,
            )


def _run_decode_single_sweep(run_dir: Path, model: str, seed: int) -> None:
    """§3.2 decode sweep: fixed prompt, vary output_len, batch=1."""
    rng = random.Random(seed + 1)
    out_path = run_dir / "decode_single.jsonl"
    with out_path.open("w") as fh:
        for out_len in DECODE_OUTPUT_LENS:
            for rep in range(DECODE_REPEATS):
                prompt_ids = _random_prompt_ids(rng, DECODE_PROMPT_LEN)
                m = _measure_request(
                    "http://127.0.0.1:8000",
                    model,
                    prompt_ids,
                    max_tokens=out_len,
                    seed=seed,
                )
                m.update(
                    phase="decode_single",
                    batch_size=1,
                    rep=rep,
                    seed=seed,
                )
                fh.write(json.dumps(m) + "\n")
                fh.flush()
            print(
                f"  decode_single output_len={out_len}: {DECODE_REPEATS} reps done",
                flush=True,
            )


def _run_decode_batch_sweep(run_dir: Path, model: str, seed: int) -> None:
    """§3.3 batching sweep: concurrent requests, measure mean tpot vs batch."""
    import asyncio

    import httpx

    out_path = run_dir / "decode_batch.jsonl"
    rng = random.Random(seed + 2)

    async def _one_request(client: httpx.AsyncClient, prompt_ids: list[int], seed_i: int) -> dict:
        body = {
            "model": model,
            "prompt": prompt_ids,
            "max_tokens": BATCH_OUTPUT_LEN,
            "temperature": 1.0,
            "stream": True,
            "ignore_eos": True,
            "seed": seed_i,
        }
        t0 = time.perf_counter()
        ttft_ms: float | None = None
        chunks = 0
        async with client.stream("POST", "/v1/completions", json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t0) * 1000.0
                chunks += 1
        total_ms = (time.perf_counter() - t0) * 1000.0
        if ttft_ms is None:
            ttft_ms = total_ms
        return {
            "prompt_len": len(prompt_ids),
            "max_tokens": BATCH_OUTPUT_LEN,
            "ttft_ms": ttft_ms,
            "total_ms": total_ms,
            "decode_ms": max(0.0, total_ms - ttft_ms),
            "chunks": chunks,
        }

    async def _run_one_batch(client: httpx.AsyncClient, batch_size: int, rep: int) -> list[dict]:
        # Fresh random prefixes per request so prefix caching (even if
        # it were enabled upstream) cannot collapse the batch.
        prompts = [_random_prompt_ids(rng, BATCH_PROMPT_LEN) for _ in range(batch_size)]
        seeds = [seed + 1000 * (rep + 1) + i for i in range(batch_size)]
        tasks = [_one_request(client, p, s) for p, s in zip(prompts, seeds)]
        results = await asyncio.gather(*tasks)
        for i, r in enumerate(results):
            r.update(
                phase="decode_batch",
                batch_size=batch_size,
                rep=rep,
                batch_index=i,
                seed=seeds[i],
            )
        return results

    async def _run_all() -> None:
        async with httpx.AsyncClient(
            base_url="http://127.0.0.1:8000",
            timeout=httpx.Timeout(connect=30.0, read=600.0, write=30.0, pool=30.0),
            limits=httpx.Limits(
                max_connections=max(BATCH_SIZES) + 8,
                max_keepalive_connections=max(BATCH_SIZES) + 8,
            ),
        ) as client:
            with out_path.open("w") as fh:
                for batch_size in BATCH_SIZES:
                    for rep in range(BATCH_REPEATS):
                        results = await _run_one_batch(client, batch_size, rep)
                        for r in results:
                            fh.write(json.dumps(r) + "\n")
                        fh.flush()
                    print(
                        f"  decode_batch batch_size={batch_size}: {BATCH_REPEATS} reps done",
                        flush=True,
                    )

    asyncio.run(_run_all())


# --- Local fit / config emission -------------------------------------------


def _linreg(x: list[float], y: list[float]) -> tuple[float, float, float]:
    """Ordinary least squares y = a + b*x. Returns (a, b, R²)."""
    import numpy as np

    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    n = len(xa)
    if n < 2:
        raise ValueError("need >= 2 points to fit")
    x_mean = xa.mean()
    y_mean = ya.mean()
    sxx = float(((xa - x_mean) ** 2).sum())
    sxy = float(((xa - x_mean) * (ya - y_mean)).sum())
    if sxx == 0.0:
        raise ValueError("all x identical, can't fit a slope")
    b = float(sxy / sxx)
    a = float(y_mean - b * x_mean)
    y_pred = a + b * xa
    ss_res = float(((ya - y_pred) ** 2).sum())
    ss_tot = float(((ya - y_mean) ** 2).sum())
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0.0 else 1.0
    return a, b, r2


def _residual_se(y_obs: list[float], y_pred: list[float]) -> float:
    import numpy as np

    y_obs_a = np.asarray(y_obs, dtype=float)
    y_pred_a = np.asarray(y_pred, dtype=float)
    n = len(y_obs_a)
    if n < 2:
        return 0.0
    return float(((y_obs_a - y_pred_a) ** 2).sum() / max(1, (n - 2))) ** 0.5


def _fit_batch_k(
    batch_sizes: list[int],
    mean_tpot: list[float],
    decode_ms_per_token: float,
) -> tuple[float, float]:
    """Fit k in ``tpot(b) = decode_ms_per_token / (1 + k*log(1 + (b-1)))``.

    Returns (k, residual_standard_error). Uses scipy's non-linear least
    squares with a sensible bracket; falls back to a 1-D grid search if
    scipy is unavailable.
    """
    import math

    import numpy as np

    b_arr = np.asarray(batch_sizes, dtype=float)
    t_arr = np.asarray(mean_tpot, dtype=float)

    def model(k: float) -> np.ndarray:
        return decode_ms_per_token / (1.0 + k * np.log(1.0 + np.maximum(0.0, b_arr - 1.0)))

    try:
        from scipy.optimize import minimize_scalar

        res = minimize_scalar(
            lambda k: float(((model(k) - t_arr) ** 2).sum()),
            bounds=(0.0, 5.0),
            method="bounded",
        )
        k_hat = float(res.x)
    except Exception:  # noqa: BLE001
        # Fallback: dense 1-D grid over the plausible band.
        grid = np.linspace(0.0, 5.0, 5001)
        sse = np.array([float(((model(k) - t_arr) ** 2).sum()) for k in grid])
        k_hat = float(grid[int(sse.argmin())])

    pred = decode_ms_per_token / (1.0 + k_hat * np.log(1.0 + np.maximum(0.0, b_arr - 1.0)))
    rse = _residual_se(mean_tpot, pred.tolist())
    if not math.isfinite(k_hat):
        raise RuntimeError("batch-k fit returned non-finite value")
    return k_hat, rse


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _fit_from_run(run_dir: Path) -> dict:
    """Fit the five ComputeParams coefficients from the raw JSONL files."""
    prefill = _read_jsonl(run_dir / "prefill.jsonl")
    decode_single = _read_jsonl(run_dir / "decode_single.jsonl")
    decode_batch = _read_jsonl(run_dir / "decode_batch.jsonl")

    # §3.1: linear regression TTFT vs prompt_len.
    prefill_x = [float(r["prompt_len"]) for r in prefill]
    prefill_y = [float(r["ttft_ms"]) for r in prefill]
    prefill_overhead_ms, prefill_ms_per_token, prefill_r2 = _linreg(prefill_x, prefill_y)
    prefill_pred = [prefill_overhead_ms + prefill_ms_per_token * x for x in prefill_x]
    prefill_rse = _residual_se(prefill_y, prefill_pred)
    prefill_ttft_mean = sum(prefill_y) / max(1, len(prefill_y))

    # §3.2: linear regression total-decode-time vs output_len. For
    # max_tokens=N, total_decode_ms ≈ total_ms - ttft_ms.
    decode_x = [float(r["max_tokens"]) for r in decode_single]
    decode_y = [float(r["decode_ms"]) for r in decode_single]
    decode_overhead_ms, decode_ms_per_token, decode_r2 = _linreg(decode_x, decode_y)

    # §3.3: aggregate per-batch mean tpot, then fit k.
    #
    # tpot is the per-output-token decode time: decode_ms / (max_tokens - 1).
    # (We subtract 1 because the first token is already accounted for in
    # ttft; the "inter-token latency" is averaged over the remaining
    # N-1 gaps between tokens.)
    per_batch: dict[int, list[float]] = {}
    for r in decode_batch:
        bs = int(r["batch_size"])
        N = int(r["max_tokens"])
        if N < 2:
            continue
        tpot = float(r["decode_ms"]) / (N - 1)
        per_batch.setdefault(bs, []).append(tpot)
    batch_sizes_sorted = sorted(per_batch)
    mean_tpot = [sum(per_batch[b]) / len(per_batch[b]) for b in batch_sizes_sorted]
    k_hat, batch_rse = _fit_batch_k(batch_sizes_sorted, mean_tpot, decode_ms_per_token)
    batch_tpot_mean = sum(mean_tpot) / max(1, len(mean_tpot))

    # Acceptance-criteria gates per plan §6.
    gates = {
        "prefill_r2_min_0_97": prefill_r2 >= 0.97,
        "prefill_rse_pct": (prefill_rse / prefill_ttft_mean)
        if prefill_ttft_mean > 0.0
        else float("inf"),
        "prefill_rse_le_10pct": (prefill_rse / prefill_ttft_mean) <= 0.10
        if prefill_ttft_mean > 0.0
        else False,
        "decode_r2_min_0_98": decode_r2 >= 0.98,
        "batch_rse_pct": (batch_rse / batch_tpot_mean) if batch_tpot_mean > 0.0 else float("inf"),
        "batch_rse_le_15pct": (batch_rse / batch_tpot_mean) <= 0.15
        if batch_tpot_mean > 0.0
        else False,
        "k_in_plausible_band": 0.1 <= k_hat <= 2.0,
    }

    return {
        "compute_params": {
            "prefill_ms_per_token": prefill_ms_per_token,
            "decode_ms_per_token": decode_ms_per_token,
            "prefill_overhead_ms": prefill_overhead_ms,
            "decode_overhead_ms": decode_overhead_ms,
            "decode_batch_k": k_hat,
        },
        "fit_diagnostics": {
            "prefill_r2": prefill_r2,
            "prefill_residual_se_ms": prefill_rse,
            "prefill_mean_ttft_ms": prefill_ttft_mean,
            "decode_r2": decode_r2,
            "decode_batch_residual_se_ms": batch_rse,
            "decode_batch_mean_tpot_ms": batch_tpot_mean,
            "batch_mean_tpot_by_size": dict(zip([str(b) for b in batch_sizes_sorted], mean_tpot)),
        },
        "acceptance_gates": gates,
        "n_requests": {
            "prefill": len(prefill),
            "decode_single": len(decode_single),
            "decode_batch": len(decode_batch),
        },
    }


def _emit_yaml_config(out_path: Path, summary: dict, metadata: dict) -> None:
    """Write a RunConfig-compatible YAML with calibrated compute params.

    Network / scheduler sections are copied from configs/example_run.yaml
    verbatim and annotated as not calibrated here (plan §5 item 2).
    """
    import yaml

    cp = summary["compute_params"]
    config = {
        "name": "calibrated-a100-llama3-8b",
        "policy": {"policy_id": "prefix-cache", "params": {"block_size": 16}},
        "topology": {
            "pods": [
                {
                    "pod_id": f"p{i}",
                    "role": "both",
                    "gpu_count": 1,
                    "kv_cache_bytes": 4294967296,
                    "max_concurrent_prefill": 4,
                    "max_concurrent_decode": 16,
                }
                for i in range(4)
            ]
        },
        "compute": {
            "prefill_ms_per_token": round(float(cp["prefill_ms_per_token"]), 6),
            "decode_ms_per_token": round(float(cp["decode_ms_per_token"]), 6),
            "prefill_overhead_ms": round(float(cp["prefill_overhead_ms"]), 6),
            "decode_overhead_ms": round(float(cp["decode_overhead_ms"]), 6),
            "decode_batch_k": round(float(cp["decode_batch_k"]), 6),
        },
        "network": {
            "client_rtt_ms": 5.0,
            "inter_pod_rtt_ms": 0.2,
            "inter_pod_bandwidth_gbps": 100.0,
            "kv_bytes_per_token": 131072,
            "serialization_overhead_ms": 0.5,
        },
        "scheduler": {
            "base_routing_ms": 0.2,
            "per_pod_consideration_us": 5.0,
        },
        "engine": {
            "kv_ewma_alpha": 0.2,
            "block_size": 16,
            "initial_warm_latency_ms": 5.0,
        },
        "workload": {
            "kind": "synthetic",
            "params": {
                "n_requests": 2000,
                "arrival_rate_qps": 8.0,
                "n_prefix_families": 256,
                "zipf_s": 1.1,
                "prompt_len_min": 256,
                "prompt_len_max": 2048,
                "max_output_tokens": 128,
                "n_sessions": 200,
                "shared_prefix_tokens": 1024,
            },
        },
        "seeds": [0, 1, 2],
        "output_dir": "results/calibrated-a100-llama3-8b",
    }
    header = (
        "# Calibrated RunConfig: {model} on {gpu} (vLLM {vllm}), "
        "calibration run {ts} (bead go-8cm).\n"
        "# Source: see research/data/calibration/{ts}/ and fit_summary.json.\n"
        "# NOTE: only the `compute:` block is calibrated here. network:/\n"
        "# scheduler:/workload: are copied from configs/example_run.yaml\n"
        "# and are NOT fitted by this calibration — they are deployment-\n"
        "# specific and out of scope per plan §0 / §5.\n"
    ).format(
        model=metadata["model"],
        gpu=metadata["gpu"],
        vllm=metadata["vllm_version"],
        ts=metadata["timestamp"],
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        fh.write(header)
        yaml.safe_dump(config, fh, sort_keys=False)


# --- Local entrypoint -------------------------------------------------------


@app.local_entrypoint()
def main(
    seed: int = 0,
    model: str = DEFAULT_MODEL,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    timestamp: str = "",
    skip_run: bool = False,
    run_dir: str = "",
    write_config: bool = True,
) -> None:
    """Run the sweep on Modal and fit locally.

    ``--skip-run --run-dir=research/data/calibration/<ts>`` re-fits an
    existing local run without re-running the sweep (useful for iterating
    on the fit logic without burning GPU-hours).
    """
    repo_root = REPO_ROOT
    if skip_run:
        if not run_dir:
            raise SystemExit("--skip-run requires --run-dir")
        local_run_dir = Path(run_dir)
        if not local_run_dir.is_absolute():
            local_run_dir = repo_root / local_run_dir
        if not local_run_dir.exists():
            raise SystemExit(f"run_dir does not exist: {local_run_dir}")
        ts = local_run_dir.name
    else:
        ts = timestamp or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        print(
            f"submitting run_calibration(model={model!r}, seed={seed}, timestamp={ts!r}) to Modal …"
        )
        result = run_calibration.remote(
            model=model, seed=seed, timestamp=ts, max_model_len=max_model_len
        )
        ts = result["timestamp"]
        local_run_dir = repo_root / "research" / "data" / "calibration" / ts
        local_run_dir.mkdir(parents=True, exist_ok=True)
        for rel, content in result["files"].items():
            out = local_run_dir / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(content)
            print(f"wrote {out} ({len(content):,} bytes)")

    metadata_path = local_run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())

    print(f"=== fitting coefficients from {local_run_dir} ===")
    summary = _fit_from_run(local_run_dir)
    summary["metadata"] = metadata

    summary_path = local_run_dir / "fit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {summary_path}")

    cp = summary["compute_params"]
    print("Fitted ComputeParams:")
    for k, v in cp.items():
        print(f"  {k}: {v:.6f}")
    print("Acceptance gates:")
    for k, v in summary["acceptance_gates"].items():
        print(f"  {k}: {v}")

    if write_config:
        config_path = repo_root / "configs" / "calibrated_a100.yaml"
        _emit_yaml_config(config_path, summary, metadata)
        print(f"wrote {config_path}")


@app.local_entrypoint()
def fetch(timestamp: str, write_config: bool = True) -> None:
    """Pull a previously-written run from the Volume and fit it.

    Useful if ``main`` was launched with ``modal run --detach`` (so the
    remote function keeps running after the local entrypoint returns)
    or if the local entrypoint disconnected mid-sweep. The sweep writes
    to ``/vol/calibration/<timestamp>/`` inside the container; this
    entrypoint mirrors that into
    ``research/data/calibration/<timestamp>/`` via the sync Volume API
    and runs the same local fit as ``main``.

        MODAL_PROFILE=arcadia-research modal run --env=GORGO \\
            scripts/calibrate.py::fetch --timestamp=20260425T...
    """
    if not timestamp:
        raise SystemExit("fetch requires --timestamp=<ts>")
    calibration_vol.reload()
    remote_root = f"calibration/{timestamp}"
    entries = calibration_vol.listdir(remote_root, recursive=True)
    files = [e for e in entries if e.type.name == "FILE"]
    if not files:
        raise SystemExit(
            f"no files under {remote_root} in volume 'gorgo-calibration' "
            "(sweep not complete or timestamp wrong?)"
        )
    local_run_dir = REPO_ROOT / "research" / "data" / "calibration" / timestamp
    local_run_dir.mkdir(parents=True, exist_ok=True)
    for entry in files:
        rel = (
            entry.path[len(remote_root) + 1 :]
            if entry.path.startswith(remote_root + "/")
            else entry.path
        )
        out = local_run_dir / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"".join(calibration_vol.read_file(entry.path)))
        print(f"wrote {out} ({entry.size:,} bytes)")

    metadata = json.loads((local_run_dir / "metadata.json").read_text())
    summary = _fit_from_run(local_run_dir)
    summary["metadata"] = metadata
    (local_run_dir / "fit_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {local_run_dir / 'fit_summary.json'}")

    cp = summary["compute_params"]
    print("Fitted ComputeParams:")
    for k, v in cp.items():
        print(f"  {k}: {v:.6f}")
    print("Acceptance gates:")
    for k, v in summary["acceptance_gates"].items():
        print(f"  {k}: {v}")

    if write_config:
        config_path = REPO_ROOT / "configs" / "calibrated_a100.yaml"
        _emit_yaml_config(config_path, summary, metadata)
        print(f"wrote {config_path}")


if __name__ == "__main__":
    # Supports ``python scripts/calibrate.py --skip-run --run-dir=...`` for
    # a pure local fit without invoking Modal at all.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--run-dir", type=str, default="")
    parser.add_argument("--no-write-config", action="store_true")
    args = parser.parse_args()
    if not args.skip_run:
        raise SystemExit(
            "Direct invocation is only supported with --skip-run. "
            "To launch the Modal sweep, use:\n"
            "    MODAL_PROFILE=arcadia-research modal run --env=GORGO "
            "scripts/calibrate.py"
        )
    local = Path(args.run_dir)
    if not local.is_absolute():
        local = REPO_ROOT / local
    if not local.exists():
        raise SystemExit(f"run_dir does not exist: {local}")
    metadata = json.loads((local / "metadata.json").read_text())
    summary = _fit_from_run(local)
    summary["metadata"] = metadata
    (local / "fit_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {local / 'fit_summary.json'}")
    if not args.no_write_config:
        _emit_yaml_config(REPO_ROOT / "configs" / "calibrated_a100.yaml", summary, metadata)
