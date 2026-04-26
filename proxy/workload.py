"""Replay GLM 5.1 traffic against the GORGO proxy.

Runs as a Modal function pinned to the same region as the proxy (and the
engine), with the ``GORGO-glm5-completions`` volume mounted read-only at
``/data``. Streams parquet rows in chronological order, applies optional
time-range / offset / count filters, and dispatches concurrently against the
proxy's ``/v1/chat/completions`` endpoint via a single pooled HTTP/2 client.

The inter-request gap from the original timeline is *not* preserved -- the
``concurrency`` knob alone determines how fast the dataset is consumed.

Usage::

    modal run proxy/workload.py --proxy-url https://...modal.host \\
        --start-time 2026-04-01T12:00:00 \\
        --end-time   2026-04-01T13:00:00 \\
        --concurrency 32

All knobs are also kwargs on ``replay`` for programmatic invocation.
"""

from __future__ import annotations

import asyncio
import json
from nt import error
import os
import time
from datetime import datetime, timezone

import modal

from app import app, bench_results_volume, completions_volume

# We want to launch the workload client in the same region as the proxy server
# in order to minimize the variable latency of crossing regions. REGION strings
# can also contain a zone like 1.
REGION = os.getenv("REGION", "us-east-1")

image = (
    modal.Image.debian_slim().pip_install("httpx[http2]", "pyarrow").add_local_python_source("app")
)

DEFAULT_MODEL = "Qwen/Qwen3.5-35B-A3B-FP8"


def _parse_iso(s: str | None) -> datetime | None:
    """Parse an ISO 8601 string into a naive (tz-stripped) datetime."""
    if not s:
        return None
    s = s.strip()
    if not s:
        return None
    d = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return d.replace(tzinfo=None) if d.tzinfo else d


def _to_naive_dt(ts) -> datetime | None:
    """ClickHouse timestamps come back from pyarrow as datetimes most of the
    time; defensively handle ISO strings too."""
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts.replace(tzinfo=None) if ts.tzinfo else ts
    if isinstance(ts, str):
        try:
            d = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            return None
        return d.replace(tzinfo=None) if d.tzinfo else d
    return None


def _select_files(
    data_dir: str,
    start_dt: datetime | None,
    end_dt: datetime | None,
) -> list[str]:
    """Pick parquets in ``[start_dt, end_dt)`` (filename-timestamp wise),
    plus the file immediately before ``start_dt`` because chunks may straddle
    the boundary -- the row-level timestamp filter handles the rest."""
    names = sorted(
        f for f in os.listdir(data_dir) if f.startswith("llm_responses_") and f.endswith(".parquet")
    )

    dated: list[tuple[datetime, str]] = []
    for name in names:
        stem = name[len("llm_responses_") : -len(".parquet")]
        try:
            ts = datetime.strptime(stem, "%Y%m%d_%H%M%S")
        except ValueError:
            continue
        dated.append((ts, name))
    dated.sort()

    selected: list[str] = []
    last_before_start: str | None = None
    for ts, name in dated:
        if start_dt is not None and ts < start_dt:
            last_before_start = name
            continue
        if end_dt is not None and ts >= end_dt:
            break
        selected.append(name)

    if start_dt is not None and last_before_start is not None:
        selected.insert(0, last_before_start)

    return selected


def _iter_bodies(
    data_dir: str,
    files: list[str],
    *,
    start_dt: datetime | None,
    end_dt: datetime | None,
    offset: int,
    num_requests: int | None,
    model_override: str | None,
    stream_override: bool | None,
    max_tokens_override: int | None,
):
    """Yield (timestamp, body_dict) chronologically, applying time-range +
    offset + limit. Streamed via ``iter_batches`` so memory stays bounded
    even on a multi-day window."""
    import pyarrow.parquet as pq

    skipped = 0
    yielded = 0
    for filename in files:
        path = os.path.join(data_dir, filename)
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=4096, columns=["timestamp", "request"]):
            ts_col = batch.column("timestamp").to_pylist()
            req_col = batch.column("request").to_pylist()
            rows = sorted(zip(ts_col, req_col), key=lambda r: r[0] or datetime.min)
            for ts, raw in rows:
                ts_dt = _to_naive_dt(ts)
                if ts_dt is None:
                    continue
                if start_dt is not None and ts_dt < start_dt:
                    continue
                if end_dt is not None and ts_dt >= end_dt:
                    return
                try:
                    body = json.loads(raw) if isinstance(raw, str) else raw
                except (TypeError, ValueError):
                    continue
                if not isinstance(body, dict):
                    continue
                msgs = body.get("messages")
                if not isinstance(msgs, list) or not msgs:
                    continue
                if skipped < offset:
                    skipped += 1
                    continue
                # TODO(atoniolo76): what are these model/stream/max_tokens fields?
                if model_override is not None:
                    body["model"] = model_override
                if stream_override is not None:
                    body["stream"] = stream_override
                if max_tokens_override is not None:
                    body["max_tokens"] = max_tokens_override
                # Ask the server for a final ``usage`` event on streaming
                # requests so we can report accurate prompt / completion
                # token counts instead of counting SSE delta events.
                # TODO(atoniolo76): not sure what the point is here either? what is "server"?
                if body.get("stream") is True:
                    so = body.get("stream_options")
                    if not isinstance(so, dict):
                        so = {}
                    so["include_usage"] = True
                    body["stream_options"] = so
                yield ts_dt, body
                yielded += 1
                if num_requests is not None and yielded >= num_requests:
                    return


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(len(s) * p))]


async def _fetch_proxy_info(client) -> dict:
    """Snapshot the proxy's routing config + the served model, so each
    saved benchmark JSON records exactly what was being served when it
    ran. Best-effort: anything that fails is captured under ``errors``
    instead of bringing the run down.

    We pull policy + hyperparameters straight from ``GET /policy`` on the
    proxy; the replica list from ``GET /replicas``; and the model id from
    one of the replicas' ``GET /v1/models`` (the proxy doesn't proxy that
    OpenAI-compat route, but the workload is co-located in the same
    region as the replicas so a direct hop is fine).
    """
    info: dict = {"errors": {}}

    async def _get_json(url: str) -> dict | None:
        try:
            r = await client.get(url, timeout=10.0)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            info["errors"][url] = repr(e)
            return None

    policy = await _get_json("/policy")
    if policy is not None:
        info["policy"] = policy.get("policy")
        info["supported_policies"] = policy.get("supported")
        info["proxy_hyperparameters"] = policy.get("hyperparameters")

    reps = await _get_json("/replicas")
    if reps is not None:
        info["replicas"] = reps.get("replicas") or []
        info["replica_count"] = reps.get("count")

    replica_urls_local = info.get("replicas") or []
    if replica_urls_local:
        replica = replica_urls_local[0]
        models_doc = await _get_json(f"{replica}/v1/models")
        if models_doc is not None:
            models = [
                m.get("id")
                for m in (models_doc.get("data") or [])
                if isinstance(m, dict) and m.get("id")
            ]
            info["served_models"] = models
            info["served_model"] = models[0] if models else None
            info["model_source_replica"] = replica

    if not info["errors"]:
        del info["errors"]
    return info


@app.function(
    image=image,
    region=REGION,
    timeout=24 * 60 * 60,
    volumes={
        "/data": completions_volume,
        "/results": bench_results_volume,
    },
)
def replay(
    proxy_url: str,
    *,
    start_time: str | None = None,
    end_time: str | None = None,
    offset: int = 0,
    num_requests: int | None = None,
    concurrency: int = 16,
    model: str | None = DEFAULT_MODEL,
    stream: bool | None = None,
    max_tokens: int | None = None,
    output_path: str | None = None,
    save_per_request: bool = True,
) -> dict:
    """Replay chat-completions traffic from the GLM 5.1 dataset.

    Args:
        proxy_url: Base URL of the proxy (the ``modal.forward`` tunnel URL
            printed by ``modal run proxy/modal_proxy.py``). Trailing slashes
            are stripped.
        start_time / end_time: Half-open ``[start, end)`` filter on the row
            ``timestamp`` column (ISO 8601, e.g. ``2026-04-01T12:00:00``).
            ``None`` on either side means unbounded.
        offset: Skip this many in-range requests before sending.
        num_requests: Cap on requests sent (after ``offset``). ``None`` means
            consume everything in ``[start, end)``.
        concurrency: Number of in-flight requests at the proxy.
        model: Replace each request's ``model`` field. Defaults to the
            served-model-name in ``engine/modal_sglang.py``; the original GLM
            model name in the dataset would be rejected by SGLang. Pass
            ``None`` to leave the dataset value alone.
        stream: Override the ``stream`` flag on every request. ``None`` =
            pass through whatever the dataset row had.
        max_tokens: Override ``max_tokens``. ``None`` = pass through.
        output_path: Where to write the JSON results doc inside the
            ``GORGO-bench-results`` volume. Relative paths are resolved
            under ``/results``. ``None`` (default) -> auto-generated
            ``replay_<UTC-timestamp>.json``.
        save_per_request: Include the per-request rows (status, timings in
            ns, token counts) alongside the aggregate stats. Set to
            ``False`` for tiny output files when only the summary matters.

    Returns:
        Summary dict with ``sent`` / ``ok`` / ``fail`` / ``elapsed_seconds``
        / ``throughput_rps`` / latency percentiles plus the resolved
        ``output_path`` of the saved JSON doc.
    """
    import httpx

    proxy_url = proxy_url.rstrip("/")
    start_dt = _parse_iso(start_time)
    end_dt = _parse_iso(end_time)

    # Snapshot of the resolved invocation, embedded in the saved JSON so
    # each result file is self-describing (no need to cross-reference the
    # CLI that produced it).
    run_started_at = datetime.now(timezone.utc)
    config = {
        "proxy_url": proxy_url,
        "start_time": start_time,
        "end_time": end_time,
        "offset": offset,
        "num_requests": num_requests,
        "concurrency": concurrency,
        "model": model,
        "stream": stream,
        "max_tokens": max_tokens,
        "region": REGION,
        "run_started_at": run_started_at.isoformat().replace("+00:00", "Z"),
    }

    # Make sure we observe parquets written by the most recent
    # ``download_db.py`` run; volumes are eventually consistent until a
    # remote write triggers a refresh.
    # TODO(atoniolo76): what is the performance cost of this? probably a non-issue
    # completions_volume.reload()

    files = _select_files("/data", start_dt, end_dt)
    if not files:
        raise SystemExit(
            f"no parquet files match the requested time range "
            f"(start={start_time!r}, end={end_time!r})"
        )
    config["selected_files"] = files

    range_desc = "".join(
        [
            f" from {start_time}" if start_time else "",
            f" until {end_time}" if end_time else "",
        ]
    )
    print(f"[workload] {len(files)} parquet file(s) in range{range_desc}")
    print(
        f"[workload] dispatching: proxy={proxy_url} concurrency={concurrency} "
        f"offset={offset} limit={num_requests if num_requests is not None else 'all-in-range'}"
    )

    async def amain() -> tuple[dict, list[dict], dict]:
        timeout = httpx.Timeout(
            connect=15.0, read=None, write=30.0, pool=10.0
        )  # read=None so longer requests don't timeout
        limits = httpx.Limits(
            max_connections=concurrency
            * 2,  # double the maximum concurrency so requests will never queue for connections
            max_keepalive_connections=concurrency * 2,
            keepalive_expiry=None,
        )

        queue: asyncio.Queue = asyncio.Queue(maxsize=concurrency * 2)
        results: list[dict] = []
        sent = 0
        done = 0
        t_start = time.perf_counter()
        last_log = t_start

        async with httpx.AsyncClient(
            base_url=proxy_url,
            http2=True,
            timeout=timeout,
            limits=limits,
        ) as client:
            # Snapshot the currently policy info and print to console
            proxy_info = await _fetch_proxy_info(client)
            print(
                f"[workload]   proxy: policy={proxy_info.get('policy')!r} "
                f"model={proxy_info.get('served_model')!r} "
                f"replicas={proxy_info.get('replica_count')}"
            )
            if proxy_info.get("proxy_hyperparameters"):
                print(f"[workload]   proxy hyperparameters: {proxy_info['proxy_hyperparameters']}")

            async def send_one(body: dict) -> dict:
                """Send one request and capture streaming-aware timings.

                Uses ``aiter_raw`` so each network chunk is timestamped on
                arrival (``perf_counter_ns``) instead of after Python's line
                buffer fills. Each parsed SSE event is tagged with the
                arrival time of the chunk that delivered its first byte, so
                TTFT reflects wire-arrival of the first content event --
                not the moment our parser happened to finish a line.

                TTFT is set on the first non-empty
                ``choices[0].delta.content`` event; SGLang's leading role
                marker (``{"delta": {"role": "assistant"}}``) is correctly
                ignored.
                """
                request_start_ns = time.perf_counter_ns()
                ttft_ns: int | None = None
                output_tokens = 0
                # Server-reported counts; preferred when present.
                usage_prompt_tokens: int | None = None
                usage_completion_tokens: int | None = None
                try:
                    async with client.stream(
                        "POST",
                        "/v1/chat/completions",
                        json=body,
                        headers={"accept-encoding": "identity"},
                    ) as resp:
                        is_sse = resp.headers.get("content-type", "").startswith(
                            "text/event-stream"
                        )
                        if is_sse:
                            buffer = bytearray()
                            # Wire-arrival time of the chunk that delivered
                            # the first byte currently sitting in ``buffer``
                            # (i.e. the first byte of the in-progress event).
                            buffer_first_chunk_ns: int | None = None
                            async for chunk in resp.aiter_raw():
                                chunk_ns = time.perf_counter_ns()
                                if not buffer:
                                    buffer_first_chunk_ns = chunk_ns
                                buffer.extend(chunk)
                                # Process every event whose terminating
                                # blank line is now in the buffer.
                                # TODO(atoniolo76): this could be a good place for a unit test. Also this code is a bit messy
                                while True:
                                    idx = buffer.find(b"\n\n")
                                    if idx < 0:
                                        break
                                    event_bytes = bytes(buffer[:idx])
                                    del buffer[: idx + 2]
                                    event_arrival_ns = buffer_first_chunk_ns
                                    # Whatever's left in ``buffer`` is the
                                    # start of the next event; its first
                                    # byte arrived in the chunk we're still
                                    # processing.
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
                                        # Final ``usage`` event (sent when
                                        # ``stream_options.include_usage`` is
                                        # set) carries accurate token
                                        # counts; capture and keep going.
                                        usage = obj.get("usage")
                                        if isinstance(usage, dict):
                                            pt = usage.get("prompt_tokens")
                                            ct = usage.get("completion_tokens")
                                            if isinstance(pt, int):
                                                usage_prompt_tokens = pt
                                            if isinstance(ct, int):
                                                usage_completion_tokens = ct
                                        choices = obj.get("choices") or []
                                        if not choices:
                                            continue
                                        delta = choices[0].get("delta") or {}
                                        content = delta.get("content")
                                        if content:
                                            if ttft_ns is None and event_arrival_ns is not None:
                                                ttft_ns = event_arrival_ns - request_start_ns
                                            # One ``delta.content`` event
                                            # ~= one decoded token in
                                            # SGLang's streaming output.
                                            output_tokens += 1
                        # TODO(atoniolo76): we should reject this path and error since we cannot get accurate TTFT metrics from it
                        else:
                            # Non-streaming: drain the body, no per-token
                            # timing available. We can still salvage the
                            # final ``usage.completion_tokens`` if present.
                            # buf = bytearray()
                            # async for chunk in resp.aiter_raw():
                            #     buf.extend(chunk)
                            # try:
                            #     final = json.loads(buf.decode())
                            #     usage = final.get("usage") or {}
                            #     pt = usage.get("prompt_tokens")
                            #     ct = usage.get("completion_tokens")
                            #     if isinstance(pt, int):
                            #         usage_prompt_tokens = pt
                            #     if isinstance(ct, int):
                            #         usage_completion_tokens = ct
                            # except (UnicodeDecodeError, json.JSONDecodeError):
                            #     pass
                            return httpx.HTTPError
                        # Prefer the server's count when available; fall
                        # back to our SSE-event count for legacy servers
                        # that don't honor ``stream_options.include_usage``.
                        final_output_tokens = (
                            usage_completion_tokens
                            if usage_completion_tokens is not None
                            else output_tokens
                        )
                        return {
                            "status": resp.status_code,
                            "ttft_ns": ttft_ns,
                            "total_ns": time.perf_counter_ns() - request_start_ns,
                            "input_tokens": usage_prompt_tokens,
                            "output_tokens": final_output_tokens,
                            "is_sse": is_sse,
                        }
                except httpx.HTTPError:
                    # TODO(atoniolo76): can we return an incomplete status here?
                    return {
                        "status": 0,
                        "ttft_ns": ttft_ns,
                        "total_ns": time.perf_counter_ns() - request_start_ns,
                        "input_tokens": usage_prompt_tokens,
                        "output_tokens": (
                            usage_completion_tokens
                            if usage_completion_tokens is not None
                            else output_tokens
                        ),
                        "is_sse": False,
                    }

            # TODO(atoniolo76): can we unnest these nested functions?
            async def worker() -> None:
                nonlocal done, last_log
                while True:
                    item = await queue.get()
                    if item is None:
                        return
                    _, body = item
                    res = await send_one(body)
                    results.append(res)
                    done += 1
                    now = time.perf_counter()
                    if now - last_log >= 5.0:
                        elapsed = now - t_start
                        ok_n = sum(1 for r in results if 200 <= r["status"] < 300)
                        rate = done / elapsed if elapsed > 0 else 0.0
                        # TODO(atoniolo76): can we log this to a cleaner output format than the console?
                        print(
                            f"[workload]   sent={sent} done={done} "
                            f"ok={ok_n} fail={done - ok_n} "
                            f"elapsed={elapsed:.1f}s rate={rate:.1f} req/s",
                            flush=True,
                        )
                        last_log = now

            # Spawn n concurrent workers: have each pull tasks from the request queue
            workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
            try:
                # _iter_bodies will load x requests into memory. can we run a worker thread that will stream requests into a shared
                # queue and have other worker threads pull from that? we could copy the requests into a shared memory buffer
                # and have it periodically flushed. this should go outside of amain() at least. then, we could read from volume
                # and keep a minimum number of requests in the queue at all times. this would allow us to expand to bigger time-scales
                # when the entirety of a weeks' worth of data cannot fit into memory of this machine. alternatively, we could increase
                # memory; however, I'm not sure if this will have performance bottlenecks that could cause request sending to slow
                for ts, body in _iter_bodies(
                    "/data",
                    files,
                    start_dt=start_dt,
                    end_dt=end_dt,
                    offset=offset,
                    num_requests=num_requests,
                    model_override=model,
                    stream_override=stream,
                    max_tokens_override=max_tokens,
                ):
                    await queue.put((ts, body))
                    sent += 1
            finally:
                for _ in range(concurrency):
                    await queue.put(None)
                await asyncio.gather(*workers)

        elapsed = max(time.perf_counter() - t_start, 1e-9)
        ok_results = [r for r in results if 200 <= r["status"] < 300]
        ok = len(ok_results)
        fail = len(results) - ok

        # All per-request timings come back in nanoseconds; convert to s/ms
        # at the boundary so display formatting stays cheap.
        NS_PER_S = 1_000_000_000
        NS_PER_MS = 1_000_000

        ttfts = [r["ttft_ns"] / NS_PER_S for r in ok_results if r["ttft_ns"] is not None]
        totals = [r["total_ns"] / NS_PER_S for r in ok_results]
        input_tokens = [
            r["input_tokens"]
            for r in ok_results
            if r["input_tokens"] is not None and r["input_tokens"] > 0
        ]
        output_tokens = [r["output_tokens"] for r in ok_results if r["output_tokens"] > 0]

        # Inter-token latency and decode rate are only meaningful on SSE
        # responses with at least 2 emitted tokens (so we have a real decode
        # window between TTFT and end-of-stream).
        sse_decoded = [
            r
            for r in ok_results
            if r["is_sse"] and r["ttft_ns"] is not None and r["output_tokens"] >= 2
        ]
        itls_ms = [
            (r["total_ns"] - r["ttft_ns"]) / NS_PER_MS / (r["output_tokens"] - 1)
            for r in sse_decoded
        ]
        decode_rates = [
            (r["output_tokens"] - 1) * NS_PER_S / max(r["total_ns"] - r["ttft_ns"], 1)
            for r in sse_decoded
        ]

        # Aggregate token throughput is the headline number for inference
        # benchmarks: total tokens emitted (or consumed) by the server,
        # divided by wall-clock dispatch time. Uses ok-only token totals so
        # failed requests don't inflate or deflate the rate.
        total_input_tokens = sum(input_tokens)
        total_output_tokens = sum(output_tokens)
        input_throughput = total_input_tokens / elapsed
        output_throughput = total_output_tokens / elapsed
        total_throughput = (total_input_tokens + total_output_tokens) / elapsed

        # Per-status counts, useful for distinguishing "503 from queue
        # overflow" vs "0 from upstream connection drops".
        status_breakdown: dict[int, int] = {}
        for r in results:
            status_breakdown[r["status"]] = status_breakdown.get(r["status"], 0) + 1
        fail_breakdown = {s: c for s, c in status_breakdown.items() if not 200 <= s < 300}

        # TODO(atoniolo76): move this into a helper function
        def _stats(xs: list[float]) -> dict:
            if not xs:
                return {
                    "avg": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "p50": 0.0,
                    "p95": 0.0,
                    "p99": 0.0,
                    "n": 0,
                }
            return {
                "avg": sum(xs) / len(xs),
                "min": min(xs),
                "max": max(xs),
                "p50": _percentile(xs, 0.50),
                "p95": _percentile(xs, 0.95),
                "p99": _percentile(xs, 0.99),
                "n": len(xs),
            }

        ttft_s = _stats(ttfts)
        total_s = _stats(totals)
        itl_s = _stats(itls_ms)
        decode_s = _stats(decode_rates)
        in_tok_s = _stats([float(x) for x in input_tokens])
        out_tok_s = _stats([float(x) for x in output_tokens])

        success_rate = (ok / len(results)) if results else 0.0

        # TODO(atoniolo76): this logging can also be moved into stats. this way if we ever want to put data in an exportable file
        # we can just modify the stats function and the output data format like a .csv that can be searched easily via an agents

        print()
        print(f"[workload] done in {elapsed:.1f}s")
        print(
            f"[workload]   sent={sent} ok={ok} fail={fail} success_rate={success_rate * 100:.1f}%"
        )
        if fail_breakdown:
            print(f"[workload]   failures by status: {dict(sorted(fail_breakdown.items()))}")
        print(f"[workload]   request throughput={len(results) / elapsed:.2f} req/s")
        print(
            f"[workload]   token throughput   "
            f"input={input_throughput:,.1f} tok/s  "
            f"output={output_throughput:,.1f} tok/s  "
            f"total={total_throughput:,.1f} tok/s"
        )
        if ttft_s["n"]:
            print(
                f"[workload]   TTFT (s)         avg={ttft_s['avg']:.3f} "
                f"p50={ttft_s['p50']:.3f} p95={ttft_s['p95']:.3f} p99={ttft_s['p99']:.3f} "
                f"(n={ttft_s['n']})"
            )
        if total_s["n"]:
            print(
                f"[workload]   request E2E (s)  avg={total_s['avg']:.2f} "
                f"p50={total_s['p50']:.2f} p95={total_s['p95']:.2f} p99={total_s['p99']:.2f} "
                f"(n={total_s['n']})"
            )
        if itl_s["n"]:
            print(
                f"[workload]   ITL (ms)         avg={itl_s['avg']:.1f} "
                f"p50={itl_s['p50']:.1f} p95={itl_s['p95']:.1f} p99={itl_s['p99']:.1f} "
                f"(n={itl_s['n']})"
            )
        if decode_s["n"]:
            print(
                f"[workload]   decode (tok/s)   avg={decode_s['avg']:.1f} "
                f"p50={decode_s['p50']:.1f} p95={decode_s['p95']:.1f} p99={decode_s['p99']:.1f} "
                f"(n={decode_s['n']})"
            )
        if in_tok_s["n"]:
            print(
                f"[workload]   input tokens     avg={in_tok_s['avg']:.0f} "
                f"p50={in_tok_s['p50']:.0f} p95={in_tok_s['p95']:.0f} p99={in_tok_s['p99']:.0f} "
                f"(n={in_tok_s['n']})"
            )
        if out_tok_s["n"]:
            print(
                f"[workload]   output tokens    avg={out_tok_s['avg']:.0f} "
                f"p50={out_tok_s['p50']:.0f} p95={out_tok_s['p95']:.0f} p99={out_tok_s['p99']:.0f} "
                f"(n={out_tok_s['n']})"
            )

        stats = {
            "sent": sent,
            "ok": ok,
            "fail": fail,
            "success_rate": success_rate,
            "status_breakdown": status_breakdown,
            "elapsed_seconds": elapsed,
            "request_throughput_rps": len(results) / elapsed,
            "input_token_throughput": input_throughput,
            "output_token_throughput": output_throughput,
            "total_token_throughput": total_throughput,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "ttft_seconds": ttft_s,
            "request_e2e_seconds": total_s,
            "itl_ms": itl_s,
            "decode_tokens_per_second": decode_s,
            "input_tokens": in_tok_s,
            "output_tokens": out_tok_s,
        }
        return stats, results, proxy_info

    # this is the main loop
    stats, raw_results, proxy_info = asyncio.run(amain())
    config["proxy"] = proxy_info

    # Resolve where to drop the JSON: auto-name under /results when the
    # caller didn't pin a path, otherwise honor it (relative paths are
    # rooted at /results so callers don't have to know the mount point).
    if output_path is None:
        ts = run_started_at.strftime("%Y%m%d_%H%M%S")
        resolved_output_path = f"/results/replay_{ts}.json"
    else:
        resolved_output_path = (
            output_path if os.path.isabs(output_path) else os.path.join("/results", output_path)
        )
    os.makedirs(os.path.dirname(resolved_output_path), exist_ok=True)

    output_doc: dict = {
        "config": config,
        "stats": stats,
    }
    if save_per_request:
        # Keep nanosecond ints as-is (JSON-friendly and lossless); analysis
        # code can divide by 1e9 / 1e6 at read time.
        output_doc["requests"] = [
            {
                "status": r["status"],
                "ttft_ns": r["ttft_ns"],
                "total_ns": r["total_ns"],
                "input_tokens": r["input_tokens"],
                "output_tokens": r["output_tokens"],
                "is_sse": r["is_sse"],
            }
            for r in raw_results
        ]

    # this may be a massive file: containing metadata about every request that was made.
    with open(resolved_output_path, "w") as f:
        json.dump(output_doc, f)
    bench_results_volume.commit()
    print(f"[workload]   saved results to volume GORGO-bench-results at {resolved_output_path}")

    stats["output_path"] = resolved_output_path
    return stats


@app.local_entrypoint()
def main(
    proxy_url: str,
    start_time: str = "",
    end_time: str = "",
    offset: int = 0,
    num_requests: int = 0,
    concurrency: int = 16,
    model: str = DEFAULT_MODEL,
    stream: str = "",
    max_tokens: int = 0,
    output_path: str = "",
    save_per_request: bool = True,
):
    """CLI wrapper for ``replay``. Sentinel values map to ``None`` because
    Modal local_entrypoints don't accept ``Optional`` natively:

      empty string for start_time / end_time / model / stream / output_path
      0 for num_requests / max_tokens
    """
    stream_arg: bool | None
    s = stream.strip().lower()
    if s == "":
        stream_arg = None
    elif s in ("1", "true", "yes"):
        stream_arg = True
    elif s in ("0", "false", "no"):
        stream_arg = False
    else:
        raise SystemExit(f"invalid --stream={stream!r}; expected true/false")

    replay.remote(
        proxy_url=proxy_url,
        start_time=start_time or None,
        end_time=end_time or None,
        offset=offset,
        num_requests=num_requests or None,
        concurrency=concurrency,
        model=model or None,
        stream=stream_arg,
        max_tokens=max_tokens or None,
        output_path=output_path or None,
        save_per_request=save_per_request,
    )
