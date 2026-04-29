"""Replay chat-completions traffic against the GORGO proxy.

Runs as a Modal function pinned to the same region as the proxy (and the
engine). Three data sources are supported via the ``--source`` flag:

- ``glm5`` (default): the GLM 5.1 ClickHouse export. Parquet shards under
  ``GORGO-glm5-completions`` (mounted at ``/data``) with ``timestamp`` plus a
  JSON ``request`` column. Supports ISO ``--start-time`` / ``--end-time``
  filtering against the row timestamps.
- ``hf``: any Hugging Face ``save_to_disk`` chat dataset (LMSYS-Chat-1M,
  WildChat-4.8M, etc.). Reads the first non-empty among the ``conversation``
  / ``messages`` / ``conversations`` columns and assembles an OpenAI-style
  chat body. ``--preset lmsys`` or ``--preset wildchat`` fills in default
  disk paths under the mounted dataset volumes; ``--data-path`` overrides.
  HF rows have no native timestamp, so ``--start-time`` / ``--end-time`` are
  ignored for this source.

Sources stream rows lazily and feed a bounded asyncio queue, so memory stays
``O(concurrency)`` regardless of dataset size. The inter-request gap from the
original timeline is *not* preserved -- the ``concurrency`` knob alone
determines how fast the dataset is consumed.

Usage::

    # GLM 5.1 (default source).
    modal run proxy/workload.py --proxy-url https://...modal.host \\
        --start-time 2026-04-01T12:00:00 \\
        --end-time   2026-04-01T13:00:00 \\
        --concurrency 32

    # LMSYS-Chat-1M.
    modal run proxy/workload.py --proxy-url https://... \\
        --source hf --preset lmsys --num-requests 1000

    # WildChat-4.8M from a custom path.
    modal run proxy/workload.py --proxy-url https://... \\
        --source hf --data-path /datasets/datasets/allenai__WildChat-4.8M

All knobs are also kwargs on ``replay`` for programmatic invocation.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime, timezone
from typing import Iterator

import modal

from app import (
    app,
    bench_results_volume,
    completions_volume,
    hf_datasets_volume,
    lmsys_chat_1m_volume,
)
from proxy.measure import consume_sse_stream

# We want to launch the workload client in the same region as the proxy server
# in order to minimize the variable latency of crossing regions. REGION strings
# can also contain a zone like 1.
REGION = os.getenv("REGION", "us-east-1")

image = (
    modal.Image.debian_slim()
    .pip_install("httpx[http2]", "pyarrow", "datasets>=3.0")
    .add_local_python_source("app", "proxy")
)

DEFAULT_MODEL = "Qwen/Qwen3.5-35B-A3B-FP8"

# HF chat datasets store the conversation under one of these columns; we
# pick the first non-empty match. Order matters: ``conversation`` is the
# canonical role/content list on LMSYS / WildChat; ``messages`` and
# ``conversations`` are fallbacks for other shapes.
HF_MESSAGE_COLUMNS = ("conversation", "messages", "conversations")

# Roles SGLang's chat-completions endpoint accepts. Anything else is mapped
# to ``user`` so the request still validates.
_VALID_ROLES = frozenset({"system", "user", "assistant", "tool", "function"})

# ``--source`` choices.
SOURCE_GLM5 = "glm5"
SOURCE_HF = "hf"
SUPPORTED_SOURCES = (SOURCE_GLM5, SOURCE_HF)

# Defaults wired up to volumes mounted on ``replay``. Match the layouts used
# by ``data_processing/build_hf_prefix_trie.py``.
GLM5_DEFAULT_PATH = "/data"
HF_PRESETS = {
    "lmsys": "/datasets/datasets/lmsys__lmsys-chat-1m",
    "wildchat": "/datasets/datasets/allenai__WildChat-4.8M",
}


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


# A "row source" is any iterator that yields ``(timestamp_or_none, chat_completion_body)``
# tuples. ``_iter_bodies`` layers offset / limit / overrides / stream_options
# injection on top of one. Both built-in sources (GLM 5.1 parquet, HF
# ``save_to_disk``) follow this contract; adding a new dataset only requires
# implementing another generator.
RowSource = Iterator[tuple["datetime | None", dict]]


def _iter_glm5_rows(
    data_dir: str,
    *,
    start_dt: datetime | None,
    end_dt: datetime | None,
) -> RowSource:
    """Yield ``(timestamp, chat_completion_body)`` from GLM 5.1 parquet shards.

    Streams via ``iter_batches`` so memory stays bounded even on a multi-day
    window. The optional time range is applied at the row level; rows with
    missing or unparseable timestamps are dropped.
    """
    import pyarrow.parquet as pq

    files = _select_files(data_dir, start_dt, end_dt)
    if not files:
        raise SystemExit(f"no GLM5 parquet files match the requested time range under {data_dir!r}")
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
                yield ts_dt, body


def _hf_content_to_str(content) -> str:
    """Coerce an OpenAI-style ``content`` value to a plain string.

    HF chat datasets are usually plain strings, but some shapes (e.g.
    multimodal blocks ``[{"type": "text", "text": "..."}]``) need flattening.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text") or block.get("content") or ""
                if isinstance(text, str) and text:
                    parts.append(text)
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def _hf_normalize_messages(raw) -> list[dict]:
    """Normalize a HF row's conversation column into a chat-completions
    ``messages`` list. Drops empty messages; coerces unrecognized roles to
    ``user``."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if not isinstance(raw, list):
        return []
    out: list[dict] = []
    for msg in raw:
        if isinstance(msg, str):
            text = msg
            role = "user"
        elif isinstance(msg, dict):
            role = str(msg.get("role") or "user").lower()
            text = _hf_content_to_str(msg.get("content"))
        else:
            continue
        if not text:
            continue
        if role not in _VALID_ROLES:
            role = "user"
        out.append({"role": role, "content": text})
    return out


def _iter_hf_rows(data_path: str) -> RowSource:
    """Yield ``(None, chat_completion_body)`` from a HF ``save_to_disk`` chat dataset.

    Picks the first non-empty among :data:`HF_MESSAGE_COLUMNS` as the source
    of messages, matching the convention used by
    ``data_processing/build_hf_prefix_trie.py``. Iteration is lazy
    (Arrow-backed), so memory stays bounded.
    """
    from datasets import Dataset, DatasetDict, load_from_disk

    dsd = load_from_disk(data_path)
    if isinstance(dsd, DatasetDict):
        # Match build_hf_prefix_trie's split selection.
        if "train" in dsd:
            ds = dsd["train"]
        else:
            ds = dsd[next(iter(dsd))]
    elif isinstance(dsd, Dataset):
        ds = dsd
    else:
        raise SystemExit(f"unsupported HF dataset object at {data_path!r}: {type(dsd)!r}")

    columns = set(ds.column_names)
    msg_col: str | None = None
    for k in HF_MESSAGE_COLUMNS:
        if k in columns:
            msg_col = k
            break
    if msg_col is None:
        raise SystemExit(
            f"HF dataset at {data_path!r} has none of {HF_MESSAGE_COLUMNS}; "
            f"available columns: {sorted(columns)}"
        )

    print(
        f"[workload] hf dataset at {data_path!r}: {len(ds)} rows, using column {msg_col!r}",
        flush=True,
    )
    for row in ds:
        msgs = _hf_normalize_messages(row[msg_col])
        if not msgs:
            continue
        yield None, {"messages": msgs}


def _build_row_source(
    *,
    source: str,
    data_path: str | None,
    preset: str | None,
    start_dt: datetime | None,
    end_dt: datetime | None,
) -> tuple[RowSource, str]:
    """Resolve ``--source`` / ``--preset`` / ``--data-path`` into a concrete
    :data:`RowSource` plus the path it actually reads from. Raises
    :class:`SystemExit` for invalid combinations so the Modal CLI prints a
    clean error."""
    if source == SOURCE_GLM5:
        if preset:
            raise SystemExit(f"--preset is only valid with --source hf (got {preset!r})")
        path = data_path or GLM5_DEFAULT_PATH
        return _iter_glm5_rows(path, start_dt=start_dt, end_dt=end_dt), path
    if source == SOURCE_HF:
        if preset is not None:
            if preset not in HF_PRESETS:
                raise SystemExit(
                    f"unknown --preset {preset!r}; expected one of {sorted(HF_PRESETS)}"
                )
            path = data_path or HF_PRESETS[preset]
        elif data_path:
            path = data_path
        else:
            raise SystemExit("--source hf requires either --preset {lmsys|wildchat} or --data-path")
        if start_dt is not None or end_dt is not None:
            print(
                "[workload] note: --start-time / --end-time ignored for --source hf "
                "(no per-row timestamps)",
                flush=True,
            )
        return _iter_hf_rows(path), path
    raise SystemExit(f"unknown --source {source!r}; expected one of {list(SUPPORTED_SOURCES)}")


def _iter_bodies(
    rows: RowSource,
    *,
    offset: int,
    num_requests: int | None,
    model_override: str | None,
    stream_override: bool | None,
    max_tokens_override: int | None,
) -> RowSource:
    """Layer offset / limit / per-request field overrides / stream_options
    injection on top of any :data:`RowSource`.

    Splitting source enumeration from this transformer step lets the GLM5
    and HF readers stay simple generators -- all CLI knobs are honored
    uniformly here.
    """
    skipped = 0
    yielded = 0
    for ts_dt, body in rows:
        if not isinstance(body, dict):
            continue
        msgs = body.get("messages")
        if not isinstance(msgs, list) or not msgs:
            continue
        if skipped < offset:
            skipped += 1
            continue
        # OpenAI chat-completions request fields the caller can override on
        # every replayed row:
        #   model: served model name (must match the SGLang replica; the
        #     GLM dataset's original ``model`` value is rejected by SGLang,
        #     hence the override). HF rows usually have no ``model`` set.
        #   stream: SSE vs. single JSON response.
        #   max_tokens: cap on generated tokens per request.
        if model_override is not None:
            body["model"] = model_override
        if stream_override is not None:
            body["stream"] = stream_override
        if max_tokens_override is not None:
            body["max_tokens"] = max_tokens_override
        # On SSE requests, ask the upstream SGLang replica to emit a final
        # ``data: {... "usage": {...}}`` event by setting
        # ``stream_options.include_usage = True``. That gives us exact
        # prompt / completion token counts instead of having to approximate
        # ``completion_tokens`` by counting ``delta.content`` SSE events.
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


NS_PER_S = 1_000_000_000
NS_PER_MS = 1_000_000


def _stats(xs: list[float]) -> dict:
    """avg/min/max/p50/p95/p99/n summary; zero-filled when ``xs`` is empty."""
    if not xs:
        return {"avg": 0.0, "min": 0.0, "max": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "n": 0}
    return {
        "avg": sum(xs) / len(xs),
        "min": min(xs),
        "max": max(xs),
        "p50": _percentile(xs, 0.50),
        "p95": _percentile(xs, 0.95),
        "p99": _percentile(xs, 0.99),
        "n": len(xs),
    }


class _NonStreamingResponse(Exception):
    """Upstream returned a non-SSE response. We rely on SSE chunk arrival
    timestamps for TTFT/ITL, so a non-streaming reply can't be measured
    correctly and is treated as a failed request."""


async def _send_one(client, body: dict) -> dict:
    """Send one chat-completions request and capture streaming-aware timings.

    Always expects an SSE response. Non-streaming replies cannot produce a
    meaningful TTFT/ITL pair under this measurement strategy and are rejected
    via :class:`_NonStreamingResponse` (recorded as a failed request).
    """
    # Capture the timer BEFORE opening the stream so TTFT includes
    # request-send and response-header latency. ``consume_sse_stream`` is
    # explicit about this contract.
    request_start_ns = time.perf_counter_ns()
    ttft_ns: int | None = None
    output_tokens = 0
    usage_prompt_tokens: int | None = None
    usage_completion_tokens: int | None = None
    try:
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json=body,
            headers={"accept-encoding": "identity"},
        ) as resp:
            is_sse = resp.headers.get("content-type", "").startswith("text/event-stream")
            if not is_sse:
                raise _NonStreamingResponse(
                    f"upstream returned content-type "
                    f"{resp.headers.get('content-type')!r}; "
                    "workload requires stream=True for accurate TTFT/ITL"
                )
            (
                ttft_ns,
                output_tokens,
                usage_prompt_tokens,
                usage_completion_tokens,
            ) = await consume_sse_stream(resp, request_start_ns=request_start_ns)
            final_output_tokens = (
                usage_completion_tokens if usage_completion_tokens is not None else output_tokens
            )
            return {
                "status": resp.status_code,
                "ttft_ns": ttft_ns,
                "total_ns": time.perf_counter_ns() - request_start_ns,
                "input_tokens": usage_prompt_tokens,
                "output_tokens": final_output_tokens,
                "is_sse": True,
                "error": None,
            }
    except _NonStreamingResponse as e:
        return {
            "status": 0,
            "ttft_ns": None,
            "total_ns": time.perf_counter_ns() - request_start_ns,
            "input_tokens": usage_prompt_tokens,
            "output_tokens": (
                usage_completion_tokens if usage_completion_tokens is not None else output_tokens
            ),
            "is_sse": False,
            "error": f"non_streaming_response: {e}",
        }
    except Exception as e:
        # Captures httpx.HTTPError plus anything raised while we were
        # parsing the SSE stream. Status 0 means "no upstream HTTP status
        # was committed"; the ``error`` field disambiguates the cause.
        return {
            "status": 0,
            "ttft_ns": ttft_ns,
            "total_ns": time.perf_counter_ns() - request_start_ns,
            "input_tokens": usage_prompt_tokens,
            "output_tokens": (
                usage_completion_tokens if usage_completion_tokens is not None else output_tokens
            ),
            "is_sse": False,
            "error": f"{type(e).__name__}: {e}",
        }


def _print_summary(stats: dict, fail_breakdown: dict[int, int]) -> None:
    """Render the per-run summary lines from a fully-populated ``stats``
    dict. Kept separate from the dict construction so future output sinks
    (CSV, JSONL, dashboards) can consume the same source of truth."""
    elapsed = stats["elapsed_seconds"]
    sent = stats["sent"]
    ok = stats["ok"]
    fail = stats["fail"]
    success_rate = stats["success_rate"]
    ttft_s = stats["ttft_seconds"]
    total_s = stats["request_e2e_seconds"]
    itl_s = stats["itl_ms"]
    decode_s = stats["decode_tokens_per_second"]
    in_tok_s = stats["input_tokens"]
    out_tok_s = stats["output_tokens"]

    print()
    print(f"[workload] done in {elapsed:.1f}s")
    print(f"[workload]   sent={sent} ok={ok} fail={fail} success_rate={success_rate * 100:.1f}%")
    if fail_breakdown:
        print(f"[workload]   failures by status: {dict(sorted(fail_breakdown.items()))}")
    print(f"[workload]   request throughput={stats['request_throughput_rps']:.2f} req/s")
    print(
        f"[workload]   token throughput   "
        f"input={stats['input_token_throughput']:,.1f} tok/s  "
        f"output={stats['output_token_throughput']:,.1f} tok/s  "
        f"total={stats['total_token_throughput']:,.1f} tok/s"
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


@app.function(
    image=image,
    region=REGION,
    timeout=24 * 60 * 60,
    volumes={
        "/data": completions_volume,
        "/lmsys": lmsys_chat_1m_volume,
        "/datasets": hf_datasets_volume,
        "/results": bench_results_volume,
    },
)
def replay(
    proxy_url: str,
    *,
    source: str = SOURCE_GLM5,
    preset: str | None = None,
    data_path: str | None = None,
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
    """Replay chat-completions traffic against the GORGO proxy.

    Args:
        proxy_url: Base URL of the proxy (the ``modal.forward`` tunnel URL
            printed by ``modal run proxy/modal_proxy.py``). Trailing slashes
            are stripped.
        source: ``"glm5"`` (default) replays the GLM 5.1 ClickHouse export
            from ``/data``; ``"hf"`` replays a Hugging Face ``save_to_disk``
            chat dataset.
        preset: For ``source="hf"``, fills in a default ``data_path`` from
            :data:`HF_PRESETS` (currently ``lmsys`` and ``wildchat``). Mutually
            exclusive with passing a custom ``data_path`` only in the sense
            that an explicit ``data_path`` always wins.
        data_path: Override the dataset disk path. For ``glm5`` defaults to
            ``/data``; for ``hf`` is required unless ``preset`` is set.
        start_time / end_time: Half-open ``[start, end)`` filter on the row
            ``timestamp`` column (ISO 8601, e.g. ``2026-04-01T12:00:00``).
            ``None`` on either side means unbounded. Only honored for
            ``source="glm5"``; ignored (with a warning) for ``hf``.
        offset: Skip this many requests before sending.
        num_requests: Cap on requests sent (after ``offset``). ``None`` means
            consume the entire source.
        concurrency: Number of in-flight requests at the proxy.
        model: Replace each request's ``model`` field. Defaults to the
            served-model-name in ``engine/modal_sglang.py``. HF datasets
            don't carry a ``model`` field, so the override is what populates
            it. Pass ``None`` to leave the original alone.
        stream: Override the ``stream`` flag on every request. ``None`` =
            pass through (HF rows default to non-streaming, so passing
            ``True`` is recommended for accurate TTFT measurements).
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
        "source": source,
        "preset": preset,
        "data_path": data_path,
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

    # Refresh the volume backing whichever source we're about to read.
    # Modal volumes are eventually consistent; ``reload()`` is a single
    # round-trip and cheap relative to the replay run. The other volumes
    # don't need refreshing because we only read from one per run.
    if source == SOURCE_GLM5:
        completions_volume.reload()
    elif source == SOURCE_HF:
        # Both LMSYS and WildChat datasets are read-only here; reload both
        # since presets may resolve to either volume.
        lmsys_chat_1m_volume.reload()
        hf_datasets_volume.reload()

    rows, resolved_path = _build_row_source(
        source=source,
        data_path=data_path,
        preset=preset,
        start_dt=start_dt,
        end_dt=end_dt,
    )
    config["resolved_data_path"] = resolved_path

    range_desc = ""
    if source == SOURCE_GLM5:
        range_desc = "".join(
            [
                f" from {start_time}" if start_time else "",
                f" until {end_time}" if end_time else "",
            ]
        )
    print(
        f"[workload] source={source} path={resolved_path}{range_desc}"
        + (f" (preset={preset})" if preset else "")
    )
    print(
        f"[workload] dispatching: proxy={proxy_url} concurrency={concurrency} "
        f"offset={offset} limit={num_requests if num_requests is not None else 'all'}"
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

            progress_log: list[dict] = []

            async def worker() -> None:
                nonlocal done, last_log
                while True:
                    item = await queue.get()
                    if item is None:
                        return
                    _, body = item
                    res = await _send_one(client, body)
                    results.append(res)
                    done += 1
                    now = time.perf_counter()
                    if now - last_log >= 5.0:
                        elapsed = now - t_start
                        ok_n = sum(1 for r in results if 200 <= r["status"] < 300)
                        rate = done / elapsed if elapsed > 0 else 0.0
                        progress = {
                            "event": "progress",
                            "elapsed_seconds": round(elapsed, 3),
                            "sent": sent,
                            "done": done,
                            "ok": ok_n,
                            "fail": done - ok_n,
                            "rate_rps": round(rate, 2),
                        }
                        progress_log.append(progress)
                        # Emit as a single-line JSON record so log scrapers
                        # can grep ``"event": "progress"`` and parse with
                        # ``jq``; no second human-friendly line needed.
                        print(json.dumps(progress), flush=True)
                        last_log = now

            workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
            try:
                # ``_iter_bodies`` wraps a per-source generator (parquet
                # ``iter_batches`` for GLM5, Arrow-backed row iteration for
                # HF), so rows are pulled lazily; combined with ``queue``
                # (bounded at ``concurrency * 2``) the pipeline back-pressures
                # from the workers all the way down to the dataset reader.
                # Memory stays O(concurrency) regardless of dataset size.
                for ts, body in _iter_bodies(
                    rows,
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

        ttft_s = _stats(ttfts)
        total_s = _stats(totals)
        itl_s = _stats(itls_ms)
        decode_s = _stats(decode_rates)
        in_tok_s = _stats([float(x) for x in input_tokens])
        out_tok_s = _stats([float(x) for x in output_tokens])

        success_rate = (ok / len(results)) if results else 0.0

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
        _print_summary(stats, fail_breakdown)
        return stats, results, proxy_info, progress_log

    stats, raw_results, proxy_info, progress_log = asyncio.run(amain())
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
        "progress": progress_log,
    }
    if save_per_request:
        # Keep nanosecond ints as-is (JSON-friendly and lossless); analysis
        # code can divide by 1e9 / 1e6 at read time. ``error`` is None on
        # success, otherwise a short string identifying the failure reason.
        output_doc["requests"] = [
            {
                "status": r["status"],
                "ttft_ns": r["ttft_ns"],
                "total_ns": r["total_ns"],
                "input_tokens": r["input_tokens"],
                "output_tokens": r["output_tokens"],
                "is_sse": r["is_sse"],
                "error": r.get("error"),
            }
            for r in raw_results
        ]

    # Can be a sizeable file when ``save_per_request`` is on (one row per
    # replayed request); the volume is the right place for it.
    with open(resolved_output_path, "w") as f:
        json.dump(output_doc, f)
    bench_results_volume.commit()
    print(f"[workload]   saved results to volume GORGO-bench-results at {resolved_output_path}")

    stats["output_path"] = resolved_output_path
    return stats


@app.local_entrypoint()
def main(
    proxy_url: str,
    source: str = SOURCE_GLM5,
    preset: str = "",
    data_path: str = "",
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

      empty string for preset / data_path / start_time / end_time /
        model / stream / output_path
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
        source=source,
        preset=preset or None,
        data_path=data_path or None,
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
