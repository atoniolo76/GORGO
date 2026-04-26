"""Calibrate the GORGO policy hyperparameters from a single idle replica.

Produces an *initial estimate* for ``t_prefill`` and
``queued_tokens_weight`` -- the two knobs in
``utils/lb_aibrix.py::route_gorgo``::

    score(u) = latency(u)
             + t_prefill            * effective_prefill_tokens
             + queued_tokens_weight * (queued + used_tokens)

Both knobs have units of *seconds per token*, so we measure them
directly.

Per-sample procedure
--------------------

1. Ping the replica ``K`` times back-to-back; take the median RTT
   (filters single-ping jitter without being too costly).
2. Send one *streaming* chat completion with ``max_tokens=N`` (large),
   using a prompt sampled from the GLM 5.1 dataset (long realistic
   prompts make the rate-per-token estimate stable -- with short
   prompts the fixed overheads dominate).
3. Parse the SSE stream with chunk-arrival-precise timing (same logic
   as ``proxy/workload.py``) so TTFT reflects the wire-arrival of the
   first ``delta.content`` event, not the moment our parser happened
   to finish a line.
4. Pull ``prompt_tokens`` / ``completion_tokens`` from the final
   ``usage`` event (we set ``stream_options.include_usage=True``).
5. Decompose::

       TTFT          = ping_rtt + prefill_seconds
       total_seconds = TTFT     + decode_seconds

   so

       t_prefill  ~= (TTFT - ping_rtt)       / prompt_tokens
       t_decode   ~= (total - TTFT)           / completion_tokens

The script reports per-sample rates plus aggregate statistics (median,
mean, min, max, p50/p95/p99). The recommended initial values are the
*medians* (robust to outliers); a linear-regression fit
``y = a + b*x`` is also reported so you can sanity-check that the
slope (= t_prefill) is consistent with the median-of-rates and that
the intercept (= fixed per-request overhead) is small.

Sequential measurements only. The replica is expected to be idle
during the run -- concurrent load would smear all three terms (ping,
prefill, decode) in ways that defeat the decomposition. To enforce
this, the calibrator hits a replica *directly* by URL, bypassing the
proxy.

Usage::

    modal run proxy/calibrate.py --replica-url https://replica.modal.host \\
        --start-time 2026-04-01T12:00:00 --num-samples 32

If you don't know the replica URL, point at the proxy and the
calibrator will pick the first replica it sees::

    modal run proxy/calibrate.py --proxy-url https://proxy.modal.host
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime, timezone

import modal

from app import app, bench_results_volume, completions_volume
from proxy.workload import (
    DEFAULT_MODEL,
    _iter_bodies,
    _parse_iso,
    _select_files,
)

REGION = os.getenv("REGION", "us-east")

image = (
    modal.Image.debian_slim()
    .pip_install("httpx[http2]", "pyarrow")
    .add_local_python_source("app", "proxy", "utils")
)

NS_PER_S = 1_000_000_000


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(len(s) * p))]


def _stats(xs: list[float]) -> dict:
    if not xs:
        return {
            "n": 0,
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }
    s = sorted(xs)
    return {
        "n": len(xs),
        "mean": sum(xs) / len(xs),
        "median": s[len(s) // 2],
        "min": s[0],
        "max": s[-1],
        "p50": _percentile(xs, 0.50),
        "p95": _percentile(xs, 0.95),
        "p99": _percentile(xs, 0.99),
    }


def _ols(xs: list[float], ys: list[float]) -> dict:
    """Plain OLS for ``y = a + b*x``. Returns ``{a, b, r2, n}``."""
    n = len(xs)
    if n < 2:
        return {"a": 0.0, "b": 0.0, "r2": 0.0, "n": n}
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0:
        return {"a": my, "b": 0.0, "r2": 0.0, "n": n}
    b = sxy / sxx
    a = my - b * mx
    syy = sum((y - my) ** 2 for y in ys)
    r2 = 1.0 - sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / syy if syy > 0 else 1.0
    return {"a": a, "b": b, "r2": r2, "n": n}


@app.function(
    image=image,
    region=REGION,
    timeout=24 * 60 * 60,
    volumes={
        "/data": completions_volume,
        "/results": bench_results_volume,
    },
)
def calibrate(
    *,
    replica_url: str | None = None,
    proxy_url: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    offset: int = 0,
    num_samples: int = 32,
    warmup_samples: int = 4,
    decode_max_tokens: int = 256,
    min_prompt_chars: int = 2000,
    pings_per_sample: int = 3,
    model: str | None = DEFAULT_MODEL,
    output_path: str | None = None,
) -> dict:
    """Estimate ``t_prefill`` and ``queued_tokens_weight`` from a single replica.

    Args:
        replica_url: Direct URL of the SGLang replica to probe (bypasses
            the proxy). Mutually exclusive with ``proxy_url``.
        proxy_url: If ``replica_url`` is not given, the calibrator will
            fetch ``GET /replicas`` from this proxy and use the first one.
        start_time/end_time/offset: GLM dataset slice for prompt sourcing
            (mirrors ``proxy/workload.py``). Use a window with realistic
            production-shaped requests.
        num_samples: Total number of measurement requests (after warmup).
        warmup_samples: Discard this many initial samples; first requests
            after a cold cache often show prefill artifacts.
        decode_max_tokens: ``max_tokens`` requested per probe. Larger
            values give a better decode-rate estimate but cost more
            wall-clock time. 256 is usually plenty.
        min_prompt_chars: Reject prompts shorter than this many chars.
            Heuristic stand-in for "long prompts" without tokenizing
            client-side; 2000 chars ~= 500 tokens.
        pings_per_sample: ``GET /v1/models`` probes per sample, median
            taken. 3 is enough to filter outliers without being slow.
        model: Override the request body's ``model`` field. ``None``
            leaves the GLM dataset value in place (likely rejected by
            SGLang -- usually you want the default).
        output_path: Where to save the calibration JSON inside the
            ``GORGO-bench-results`` volume. ``None`` (default) ->
            ``calibrate_<UTC-timestamp>.json``.

    Returns:
        Dict with raw per-sample data + aggregate stats + recommended
        initial hyperparameter values, plus the resolved
        ``output_path``.
    """
    import httpx

    if replica_url is None and proxy_url is None:
        raise SystemExit("either --replica-url or --proxy-url is required")
    if replica_url is not None and proxy_url is not None:
        raise SystemExit("specify only one of --replica-url / --proxy-url")

    proxy_url = proxy_url.rstrip("/") if proxy_url else None
    if replica_url is None:
        with httpx.Client(timeout=15.0) as bootstrap:
            r = bootstrap.get(f"{proxy_url}/replicas")
            r.raise_for_status()
            replicas = r.json().get("replicas") or []
        if not replicas:
            raise SystemExit(f"proxy {proxy_url!r} has no replicas registered")
        replica_url = replicas[0]
        print(f"[calibrate] using replica {replica_url} (auto-discovered via proxy)")
    replica_url = replica_url.rstrip("/")

    start_dt = _parse_iso(start_time)
    end_dt = _parse_iso(end_time)

    completions_volume.reload()
    files = _select_files("/data", start_dt, end_dt)
    if not files:
        raise SystemExit(
            f"no parquet files match the requested time range "
            f"(start={start_time!r}, end={end_time!r})"
        )

    run_started_at = datetime.now(timezone.utc)
    print(
        f"[calibrate] replica={replica_url} samples={num_samples} "
        f"warmup={warmup_samples} max_tokens={decode_max_tokens} "
        f"min_prompt_chars={min_prompt_chars}"
    )

    async def amain() -> dict:
        # ``read=None`` so a long generation can't trip the timeout;
        # connect/write tight so dead replicas fail fast.
        timeout = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0)
        async with httpx.AsyncClient(base_url=replica_url, http2=True, timeout=timeout) as client:
            # TODO(atoniolo76): why can't we just ping the IP/URL directly?
            async def ping_once() -> float:
                """Median of ``pings_per_sample`` /v1/models RTTs (seconds).

                /v1/models is the cheapest universally-OpenAI-compat
                endpoint -- /health is not standardized across servers.
                """
                rtts = []
                for _ in range(pings_per_sample):
                    t0 = time.perf_counter_ns()
                    r = await client.get("/v1/models")
                    r.raise_for_status()
                    rtts.append((time.perf_counter_ns() - t0) / NS_PER_S)
                rtts.sort()
                return rtts[len(rtts) // 2]

            # TODO(atoniolo76): move this along with other nested functions outside of amain()
            async def measure_one(body: dict) -> dict | None:
                """One streaming completion + paired ping. Returns ``None``
                on transport / parse errors; the caller skips and tries
                the next prompt.
                """
                try:
                    ping_rtt = await ping_once()
                except httpx.HTTPError as e:
                    print(f"[calibrate]   ping failed: {e}; skipping")
                    return None

                request_start_ns = time.perf_counter_ns()
                ttft_ns: int | None = None
                output_tokens = 0
                prompt_tokens: int | None = None
                completion_tokens: int | None = None
                try:
                    # TODO(atoniolo76): maybe we can move this SSE event parsing logic to a helper function used in workload.py
                    async with client.stream(
                        "POST",
                        "/v1/chat/completions",
                        json=body,
                        headers={"accept-encoding": "identity"},
                    ) as resp:
                        if resp.status_code != 200:
                            await resp.aread()
                            print(f"[calibrate]   replica returned {resp.status_code}; skipping")
                            return None
                        # Same chunk-arrival-precise SSE loop as
                        # workload.py::send_one; kept inline so the
                        # calibration script doesn't depend on workload
                        # internals.
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
                                event_bytes = bytes(buffer[:idx])
                                del buffer[: idx + 2]
                                event_arrival_ns = buffer_first_chunk_ns
                                buffer_first_chunk_ns = chunk_ns if buffer else None
                                for line in event_bytes.split(b"\n"):
                                    if not line.startswith(b"data:"):
                                        continue
                                    payload = line[len(b"data:") :].strip()
                                    if not payload or payload == b"[DONE]":
                                        continue
                                    try:
                                        obj = json.loads(payload)
                                    except json.JSONDecodeError:
                                        continue
                                    usage = obj.get("usage")
                                    if isinstance(usage, dict):
                                        pt = usage.get("prompt_tokens")
                                        ct = usage.get("completion_tokens")
                                        if isinstance(pt, int):
                                            prompt_tokens = pt
                                        if isinstance(ct, int):
                                            completion_tokens = ct
                                    choices = obj.get("choices") or []
                                    if not choices:
                                        continue
                                    delta = choices[0].get("delta") or {}
                                    content = delta.get("content")
                                    if content:
                                        if ttft_ns is None and event_arrival_ns is not None:
                                            ttft_ns = event_arrival_ns - request_start_ns
                                        output_tokens += 1
                except httpx.HTTPError as e:
                    print(f"[calibrate]   request failed: {e}; skipping")
                    return None

                total_ns = time.perf_counter_ns() - request_start_ns
                if ttft_ns is None or prompt_tokens is None or prompt_tokens <= 0:
                    return None
                final_completion_tokens = (
                    completion_tokens if completion_tokens is not None else output_tokens
                )
                if final_completion_tokens <= 0:
                    return None

                ttft_s = ttft_ns / NS_PER_S
                total_s = total_ns / NS_PER_S
                # ``ping_rtt`` represents the irreducible round-trip; it
                # could in principle exceed the observed prefill (e.g.
                # network blip). Clamp at 0 to avoid negative rates.
                prefill_s = max(ttft_s - ping_rtt, 0.0)
                decode_s = max(total_s - ttft_s, 0.0)

                return {
                    "ping_seconds": ping_rtt,
                    "ttft_seconds": ttft_s,
                    "total_seconds": total_s,
                    "prefill_seconds": prefill_s,
                    "decode_seconds": decode_s,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": final_completion_tokens,
                    "prefill_rate_seconds_per_token": prefill_s / prompt_tokens,
                    "decode_rate_seconds_per_token": decode_s / final_completion_tokens,
                }

            # Stream prompts from the GLM dataset, keep only sufficiently
            # long ones, run sequentially. ``stream=True`` and
            # ``max_tokens=decode_max_tokens`` are forced so each probe
            # produces a clean (TTFT, total) pair.
            samples: list[dict] = []
            warmup_seen = 0
            attempted = 0
            t_run_start = time.perf_counter()

            for _ts, body in _iter_bodies(
                "/data",
                files,
                start_dt=start_dt,
                end_dt=end_dt,
                offset=offset,
                num_requests=None,
                model_override=model,
                stream_override=True,
                max_tokens_override=decode_max_tokens,
            ):
                # Cheap client-side filter for "long enough" prompts; we
                # don't have a tokenizer in this image (and don't want
                # to pip install one for calibration) so a char-count
                # threshold stands in.
                msgs = body.get("messages") or []
                # TODO(atoniolo76): we are getting all the messages as if none of the previous prompts have been KV_cached...
                total_chars = sum(
                    len(m.get("content") or "") if isinstance(m, dict) else 0 for m in msgs
                )
                if total_chars < min_prompt_chars:
                    continue

                attempted += 1
                sample = await measure_one(body)
                if sample is None:
                    continue

                if warmup_seen < warmup_samples:
                    warmup_seen += 1
                    print(
                        f"[calibrate]   warmup {warmup_seen}/{warmup_samples}: "
                        f"prompt_tokens={sample['prompt_tokens']} "
                        f"prefill={sample['prefill_seconds']:.3f}s "
                        f"decode={sample['decode_seconds']:.3f}s"
                    )
                    continue

                samples.append(sample)
                print(
                    f"[calibrate]   sample {len(samples)}/{num_samples}: "
                    f"pt={sample['prompt_tokens']} ct={sample['completion_tokens']} "
                    f"ping={sample['ping_seconds'] * 1000:.1f}ms "
                    f"prefill={sample['prefill_seconds']:.3f}s "
                    f"decode={sample['decode_seconds']:.3f}s "
                    f"t_prefill={sample['prefill_rate_seconds_per_token'] * 1000:.3f}ms/tok "
                    f"t_decode={sample['decode_rate_seconds_per_token'] * 1000:.3f}ms/tok",
                    flush=True,
                )

                if len(samples) >= num_samples:
                    break

            elapsed = time.perf_counter() - t_run_start
            print(
                f"[calibrate] collected {len(samples)} samples in {elapsed:.1f}s "
                f"(attempted {attempted})"
            )
            return {
                "samples": samples,
                "elapsed_seconds": elapsed,
                "attempted": attempted,
            }

    inner = asyncio.run(amain())
    samples = inner["samples"]

    if not samples:
        raise SystemExit("no successful calibration samples collected")

    pings = [s["ping_seconds"] for s in samples]
    prefill_rates = [s["prefill_rate_seconds_per_token"] for s in samples]
    decode_rates = [s["decode_rate_seconds_per_token"] for s in samples]
    prompt_tokens = [float(s["prompt_tokens"]) for s in samples]
    prefill_secs = [s["prefill_seconds"] for s in samples]
    completion_tokens = [float(s["completion_tokens"]) for s in samples]
    decode_secs = [s["decode_seconds"] for s in samples]

    # Linear-regression cross-check. The slope is a different (and often
    # cleaner) estimator of the per-token cost than median-of-ratios:
    # short prompts get less weight because their ratios are dominated
    # by fixed overhead, but in OLS they contribute primarily to the
    # intercept rather than the slope.
    prefill_fit = _ols(prompt_tokens, prefill_secs)
    decode_fit = _ols(completion_tokens, decode_secs)

    # TODO(atoniolo76): move these stats into a helper function

    ping_stats = _stats(pings)
    prefill_rate_stats = _stats(prefill_rates)
    decode_rate_stats = _stats(decode_rates)

    recommended = {
        "t_prefill": prefill_rate_stats["median"],
        "queued_tokens_weight": decode_rate_stats["median"],
    }

    print()
    print(
        f"[calibrate] ping (s)            median={ping_stats['median']:.4f} "
        f"p95={ping_stats['p95']:.4f} (n={ping_stats['n']})"
    )
    print(
        f"[calibrate] t_prefill (s/tok)   median={prefill_rate_stats['median']:.6f} "
        f"mean={prefill_rate_stats['mean']:.6f} "
        f"p95={prefill_rate_stats['p95']:.6f} "
        f"(slope={prefill_fit['b']:.6f} intercept={prefill_fit['a']:.4f}s "
        f"r2={prefill_fit['r2']:.3f})"
    )
    print(
        f"[calibrate] t_decode  (s/tok)   median={decode_rate_stats['median']:.6f} "
        f"mean={decode_rate_stats['mean']:.6f} "
        f"p95={decode_rate_stats['p95']:.6f} "
        f"(slope={decode_fit['b']:.6f} intercept={decode_fit['a']:.4f}s "
        f"r2={decode_fit['r2']:.3f})"
    )
    print(f"[calibrate] recommended initial hyperparameters: {recommended}")

    if output_path is None:
        ts = run_started_at.strftime("%Y%m%d_%H%M%S")
        resolved_output_path = f"/results/calibrate_{ts}.json"
    else:
        resolved_output_path = (
            output_path if os.path.isabs(output_path) else os.path.join("/results", output_path)
        )
    os.makedirs(os.path.dirname(resolved_output_path), exist_ok=True)

    doc = {
        "config": {
            "replica_url": replica_url,
            "proxy_url": proxy_url,
            "model": model,
            "num_samples": num_samples,
            "warmup_samples": warmup_samples,
            "decode_max_tokens": decode_max_tokens,
            "min_prompt_chars": min_prompt_chars,
            "pings_per_sample": pings_per_sample,
            "data_slice": {
                "start_time": start_time,
                "end_time": end_time,
                "offset": offset,
                "selected_files": files,
            },
            "region": REGION,
            "run_started_at": run_started_at.isoformat().replace("+00:00", "Z"),
        },
        "samples": samples,
        "stats": {
            "ping_seconds": ping_stats,
            "prefill_rate_seconds_per_token": prefill_rate_stats,
            "decode_rate_seconds_per_token": decode_rate_stats,
            "prefill_ols_fit": prefill_fit,
            "decode_ols_fit": decode_fit,
            "elapsed_seconds": inner["elapsed_seconds"],
            "attempted": inner["attempted"],
        },
        "recommended_hyperparameters": recommended,
    }
    with open(resolved_output_path, "w") as f:
        json.dump(doc, f)
    bench_results_volume.commit()
    print(f"[calibrate] saved results to volume GORGO-bench-results at {resolved_output_path}")

    # TODO(atoniolo76): why do we unpack when returning? should we let the client unpack?
    return {**doc, "output_path": resolved_output_path}


@app.local_entrypoint()
def main(
    replica_url: str = "",
    proxy_url: str = "",
    start_time: str = "",
    end_time: str = "",
    offset: int = 0,
    num_samples: int = 32,
    warmup_samples: int = 4,
    decode_max_tokens: int = 256,
    min_prompt_chars: int = 2000,
    pings_per_sample: int = 3,
    model: str = DEFAULT_MODEL,
    output_path: str = "",
):
    """CLI wrapper for ``calibrate``.

    Pass exactly one of ``--replica-url`` / ``--proxy-url``. Sentinel
    mapping matches the rest of the proxy/ scripts:

      empty string for replica_url/proxy_url/start_time/end_time/model/output_path
    """
    calibrate.remote(
        # TODO(atoniolo76): can we get rid of proxy url here since we haven't added a separate calibrate function to the proxy?
        replica_url=replica_url or None,
        proxy_url=proxy_url or None,
        start_time=start_time or None,
        end_time=end_time or None,
        offset=offset,
        num_samples=num_samples,
        warmup_samples=warmup_samples,
        decode_max_tokens=decode_max_tokens,
        min_prompt_chars=min_prompt_chars,
        pings_per_sample=pings_per_sample,
        model=model or None,
        output_path=output_path or None,
    )
