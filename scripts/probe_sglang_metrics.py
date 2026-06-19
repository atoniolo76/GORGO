"""One-shot probe to confirm what an SGLang replica exposes for queue-delay calibration.

Answers three questions empirically against a *live* replica:

  1. Which queue/latency metrics exist and their Prometheus types
     (gauge vs histogram) -- esp. ``queue_time_seconds``,
     ``avg_request_queue_latency``, ``num_queue_reqs``, ``num_used_tokens``.
  2. Whether per-request timing is returned in the response ``meta_info``
     (native ``/generate``) or the OpenAI ``/v1/chat/completions`` body --
     i.e. can we get an engine-measured queue/prefill split per request.
  3. When the ``queue_time_seconds`` observation is recorded, by scraping
     ``_count``/``_sum`` immediately before and after a single request and
     watching the deltas (count should += number of requests that left the
     queue in between).
  4. (Q4, added) Whether the *streaming* ``/v1/chat/completions`` response
     surfaces per-request ``meta_info`` / ``metadata`` timing fields under
     load, and exactly which request flag is required to make them appear
     (vs. the native ``/generate`` endpoint). See ``meta_info_probe``.

Usage:
    python scripts/probe_sglang_metrics.py \
        --base-url https://<replica-host> \
        --model Qwen/Qwen3.5-35B-A3B-FP8 \
        --concurrency 8 --max-tokens 16

Notes:
    - Needs a *running* replica (the matrix engines are ephemeral; spin one
      up or point at a deployed engine first).
    - Read-only except for the tiny generation requests used to probe the
      response shape, bump the histogram, and drive nonzero queueing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics

import httpx

# ---------------------------------------------------------------------------
# Q4: per-request meta_info / metadata timing fields we want to confirm are
# surfaced on the *streaming* chat endpoint under load. SGLang's engine
# measures these internally; the open question is which request flag (if any)
# makes them ride along on the OpenAI-compatible chat stream vs. only on the
# native /generate stream.
# ---------------------------------------------------------------------------

# Engine-measured timing fields (seconds) we explicitly look for in meta_info.
META_TIMING_FIELDS = (
    "queue_time",
    "prefill_waiting_latency",
    "prefill_launch_latency",
    "e2e_latency",
)

# Token-accounting fields that ride along in meta_info / usage.
META_USAGE_FIELDS = (
    "cached_tokens",
    "completion_tokens",
    "prompt_tokens",
    "total_tokens",
)

# Metrics we specifically care about for queue-delay calibration.
QUEUE_PATTERNS = (
    "queue",
    "num_running_reqs",
    "num_used_tokens",
    "gen_throughput",
    "token_usage",
    "utilization",
    "cache_hit",
    "time_to_first_token",
    "e2e_request_latency",
    "inter_token",
    "time_per_output_token",
    "prompt_tokens_total",
    "generation_tokens_total",
)


def _scrape(client: httpx.Client, base: str) -> str:
    r = client.get(f"{base}/metrics", timeout=20.0)
    r.raise_for_status()
    return r.text


def _types_and_help(text: str) -> dict[str, dict[str, str]]:
    """Map metric base-name -> {'type': ..., 'help': ...} from # TYPE/# HELP lines."""
    info: dict[str, dict[str, str]] = {}
    for line in text.splitlines():
        if line.startswith("# TYPE "):
            _, _, name, mtype = line.split(maxsplit=3)
            info.setdefault(name, {})["type"] = mtype
        elif line.startswith("# HELP "):
            parts = line.split(maxsplit=3)
            if len(parts) == 4:
                info.setdefault(parts[2], {})["help"] = parts[3]
    return info


def _values(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        try:
            out[parts[0]] = float(parts[1])  # keep full name incl. labels
        except ValueError:
            continue
    return out


def _has_sample_timestamps(text: str) -> bool:
    """Prometheus allows an optional 3rd field (epoch ms) per sample.
    If present, SGLang would be revealing its own record time."""
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        # name{labels} value [timestamp]
        body = re.sub(r"\{[^}]*\}", "", line).split()
        if len(body) >= 3:
            return True
    return False


# ---------------------------------------------------------------------------
# Q4 helpers: streaming meta_info / metadata discovery.
# ---------------------------------------------------------------------------


def _find_meta_info(obj: object) -> dict | None:
    """Recursively search a parsed JSON object for a ``meta_info`` / ``metadata``
    dict. SGLang has, across versions, attached engine timing under either key
    and at either the top level, inside ``choices[*]``, or alongside ``usage``,
    so we walk the whole structure and return the first dict-valued match."""
    if isinstance(obj, dict):
        for key in ("meta_info", "metadata"):
            val = obj.get(key)
            if isinstance(val, dict) and val:
                return val
        for val in obj.values():
            found = _find_meta_info(val)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_meta_info(item)
            if found is not None:
                return found
    return None


def _present_timing_fields(meta: dict) -> dict[str, object]:
    """Pull the fields we care about out of a meta_info dict: the named timing
    fields, any ``*_ts`` wall-clock timestamps, and token-usage counters."""
    present: dict[str, object] = {}
    for field in META_TIMING_FIELDS:
        if field in meta:
            present[field] = meta[field]
    for field in META_USAGE_FIELDS:
        if field in meta:
            present[field] = meta[field]
    # Any wall-clock timestamp fields (engine record times) end in ``_ts``.
    for key, val in meta.items():
        if key.endswith("_ts"):
            present[key] = val
    return present


def _candidate_bodies(model: str, prompt: str, max_tokens: int) -> list[dict]:
    """The ordered set of request bodies we try to surface meta_info. Kept as
    plain dicts so the exact flags are printed verbatim in the output and we
    learn which one works against a live replica."""
    chat_messages = [{"role": "user", "content": prompt}]
    return [
        {
            "label": "chat + include_usage",
            "endpoint": "/v1/chat/completions",
            "kind": "chat",
            "body": {
                "model": model,
                "messages": chat_messages,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        },
        {
            "label": "chat + include_usage + return_logprob (SGLang extension)",
            "endpoint": "/v1/chat/completions",
            "kind": "chat",
            "body": {
                "model": model,
                "messages": chat_messages,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "stream": True,
                "stream_options": {"include_usage": True},
                "return_logprob": True,
                "return_text_in_logprobs": True,
            },
        },
        {
            "label": "native /generate + stream + return_logprob",
            "endpoint": "/generate",
            "kind": "generate",
            "body": {
                "text": prompt,
                "sampling_params": {"max_new_tokens": max_tokens, "temperature": 0.0},
                "stream": True,
                "return_logprob": True,
            },
        },
    ]


async def _stream_meta_info(
    client: httpx.AsyncClient,
    endpoint: str,
    body: dict,
    *,
    timeout: float = 120.0,
) -> tuple[dict | None, int, str | None]:
    """Fire one streaming request, parse the SSE ``data:`` payloads, and return
    ``(meta_info_or_None, num_events, error_or_None)``.

    Both the OpenAI chat stream and the native ``/generate`` stream emit
    ``data: {json}\\n\\n`` SSE frames terminated by ``data: [DONE]``; the final
    frame(s) carry the cumulative ``meta_info``/``usage`` we want, so we keep
    the last meta_info seen across the whole stream."""
    last_meta: dict | None = None
    n_events = 0
    try:
        async with client.stream(
            "POST",
            endpoint,
            json=body,
            headers={"accept-encoding": "identity"},
            timeout=timeout,
        ) as resp:
            if resp.status_code != 200:
                detail = (await resp.aread()).decode("utf-8", "replace")[:300]
                return None, 0, f"HTTP {resp.status_code}: {detail}"
            async for raw in resp.aiter_lines():
                line = raw.strip()
                if not line or not line.startswith("data:"):
                    continue
                payload = line[len("data:") :].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                n_events += 1
                meta = _find_meta_info(obj)
                if meta is not None:
                    last_meta = meta
    except httpx.HTTPError as e:
        return last_meta, n_events, f"{type(e).__name__}: {e}"
    return last_meta, n_events, None


async def _detect_model(client: httpx.AsyncClient) -> str | None:
    """Best-effort model id discovery via ``/v1/models`` then ``/get_model_info``."""
    try:
        r = await client.get("/v1/models", timeout=20.0)
        if r.status_code == 200:
            data = r.json().get("data") or []
            if data and isinstance(data[0], dict) and data[0].get("id"):
                return str(data[0]["id"])
    except httpx.HTTPError:
        pass
    try:
        r = await client.get("/get_model_info", timeout=20.0)
        if r.status_code == 200:
            info = r.json()
            for key in ("model_path", "model", "served_model_name"):
                if isinstance(info, dict) and info.get(key):
                    return str(info[key])
    except httpx.HTTPError:
        pass
    return None


def _summarize_distribution(metas: list[dict]) -> dict[str, dict[str, float]]:
    """Across a batch of meta_info dicts, compute min/median/max for every
    numeric timing/usage field that appeared in at least one response."""
    by_field: dict[str, list[float]] = {}
    for meta in metas:
        for key, val in _present_timing_fields(meta).items():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                by_field.setdefault(key, []).append(float(val))
    summary: dict[str, dict[str, float]] = {}
    for field, vals in by_field.items():
        vals_sorted = sorted(vals)
        summary[field] = {
            "n": len(vals_sorted),
            "min": vals_sorted[0],
            "median": statistics.median(vals_sorted),
            "max": vals_sorted[-1],
        }
    return summary


async def meta_info_probe(
    base: str,
    *,
    model: str | None,
    prompt: str,
    max_tokens: int,
    concurrency: int,
) -> None:
    """Q4 driver: confirm whether streaming chat surfaces per-request meta_info,
    discover the required flag, and report the timing-field distribution across
    a concurrent batch that drives nonzero queueing."""
    print()
    print("=" * 70)
    print("Q4. STREAMING meta_info / metadata SURFACING (under load)")
    print("=" * 70)

    async with httpx.AsyncClient(base_url=base) as client:
        if not model:
            model = await _detect_model(client)
            print(f"  auto-detected model: {model!r}")
        else:
            print(f"  using model: {model!r}")
        if not model:
            model = "default"
            print("  WARNING: no model id detected; falling back to 'default'")

        candidates = _candidate_bodies(model, prompt, max_tokens)

        # ---- Step 1+2: single-shot, find which body/endpoint surfaces it ----
        print()
        print("-- single-request attempts (in order) --")
        working: dict | None = None
        for cand in candidates:
            print(f"\n[{cand['label']}] POST {cand['endpoint']}")
            print("  body:", json.dumps(cand["body"]))
            meta, n_events, err = await _stream_meta_info(client, cand["endpoint"], cand["body"])
            if err:
                print(f"  -> error: {err}")
                continue
            if meta is None:
                print(
                    f"  -> meta_info NOT found on {cand['endpoint']} "
                    f"with body {json.dumps(cand['body'])} "
                    f"(parsed {n_events} SSE events)"
                )
                continue
            present = _present_timing_fields(meta)
            print(f"  -> meta_info FOUND ({n_events} SSE events). keys:", sorted(meta.keys()))
            print("     timing/usage fields present:", json.dumps(present, indent=2))
            missing = [f for f in META_TIMING_FIELDS if f not in meta]
            if missing:
                print("     timing fields MISSING:", missing)
            if working is None:
                working = cand

        if working is None:
            print()
            print("RESULT: no attempted body surfaced meta_info on any endpoint.")
            print("        See the per-attempt 'meta_info NOT found' lines above.")
            return

        print()
        print(f"RESULT: meta_info surfaced via [{working['label']}] on {working['endpoint']}")

        # ---- Step 3: concurrent batch to drive nonzero queueing -------------
        if concurrency < 1:
            return
        print()
        print(f"-- concurrent batch: {concurrency} requests via [{working['label']}] --")
        results = await asyncio.gather(
            *(
                _stream_meta_info(client, working["endpoint"], working["body"])
                for _ in range(concurrency)
            ),
            return_exceptions=True,
        )
        metas: list[dict] = []
        errors = 0
        for res in results:
            if isinstance(res, BaseException):
                errors += 1
                continue
            meta, _n, err = res
            if err or meta is None:
                errors += 1
                continue
            metas.append(meta)
        print(f"  {len(metas)}/{concurrency} requests returned meta_info ({errors} failed/empty)")
        if not metas:
            print("  no meta_info collected from the concurrent batch.")
            return
        summary = _summarize_distribution(metas)
        print("  timing/usage distribution across batch (min/median/max):")
        for field in sorted(summary):
            s = summary[field]
            print(
                f"    {field:28} n={int(s['n']):<3} "
                f"min={s['min']:.6g} median={s['median']:.6g} max={s['max']:.6g}"
            )
        nonzero_queue = [
            m
            for m in metas
            if isinstance(m.get("queue_time"), (int, float)) and m["queue_time"] > 0
        ]
        print(
            f"  requests with nonzero queue_time: {len(nonzero_queue)}/{len(metas)}"
            + ("" if "queue_time" in summary else "  (queue_time field absent)")
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--base-url",
        "--url",
        dest="base_url",
        required=True,
        help="Replica base URL (no trailing slash); --url is a legacy alias",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="Model id for the chat endpoint; auto-detected via /v1/models or "
        "/get_model_info if omitted",
    )
    ap.add_argument("--prompt", default="Say hello in one word.")
    ap.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Number of concurrent requests for the Q4 load probe (drives nonzero queueing)",
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=16,
        help="Max output tokens per probe request (kept small to bound prefill/decode)",
    )
    args = ap.parse_args()
    base = args.base_url.rstrip("/")

    with httpx.Client() as client:
        # ---- Q1: metric inventory ------------------------------------------
        text = _scrape(client, base)
        info = _types_and_help(text)
        print("=" * 70)
        print("Q1. QUEUE / LATENCY METRICS PRESENT (name -- type)")
        print("=" * 70)
        names = sorted(info)
        hit = [n for n in names if any(p in n for p in QUEUE_PATTERNS)]
        for n in hit:
            t = info[n].get("type", "?")
            h = info[n].get("help", "")
            print(f"  {n:48} {t:10} {h}")
        for key in ("sglang:queue_time_seconds", "sglang:avg_request_queue_latency"):
            print(f"  --> {key}: {'PRESENT' if key in info else 'ABSENT'}")

        print()
        print("Q3a. sample timestamps embedded in /metrics? ", _has_sample_timestamps(text))

        # ---- Q3: count/sum deltas around one request -----------------------
        v0 = _values(text)

        def g(d, k):
            return d.get(k)

        # ---- Q2: response meta_info shape ----------------------------------
        print()
        print("=" * 70)
        print("Q2. PER-REQUEST TIMING IN RESPONSE")
        print("=" * 70)
        meta = None
        try:
            gen = client.post(
                f"{base}/generate",
                json={
                    "text": args.prompt,
                    "sampling_params": {"max_new_tokens": 8, "temperature": 0.0},
                },
                timeout=120.0,
            )
            gen.raise_for_status()
            body = gen.json()
            meta = body.get("meta_info")
            print("native /generate meta_info keys:")
            print("  ", sorted(meta.keys()) if isinstance(meta, dict) else meta)
            if isinstance(meta, dict):
                timing = {
                    k: v
                    for k, v in meta.items()
                    if any(t in k.lower() for t in ("time", "latency", "queue", "ttft", "prefill"))
                }
                print("  timing-ish fields:", json.dumps(timing, indent=2))
        except Exception as e:
            print("  /generate failed:", repr(e))

        try:
            chat = client.post(
                f"{base}/v1/chat/completions",
                json={
                    "model": args.model or "default",
                    "messages": [{"role": "user", "content": args.prompt}],
                    "max_tokens": 8,
                    "stream": False,
                },
                timeout=120.0,
            )
            chat.raise_for_status()
            cb = chat.json()
            print("OpenAI /v1/chat/completions top-level keys:", sorted(cb.keys()))
            print("  usage:", json.dumps(cb.get("usage", {}), indent=2))
        except Exception as e:
            print("  /v1/chat/completions failed:", repr(e))

        # ---- Q3: re-scrape and diff queue histogram ------------------------
        text2 = _scrape(client, base)
        v1 = _values(text2)
        print()
        print("=" * 70)
        print("Q3b. queue_time_seconds deltas around 1 request")
        print("=" * 70)
        for stem in (
            "sglang:queue_time_seconds_count",
            "sglang:queue_time_seconds_sum",
            "sglang:time_to_first_token_seconds_count",
        ):
            # sum across any label sets
            def _sum_for(d, s):
                return sum(val for k, val in d.items() if k.split("{")[0] == s)

            before = _sum_for(v0, stem)
            after = _sum_for(v1, stem)
            print(f"  {stem:48} {before} -> {after}   (Δ={after - before})")
        # current live gauges for context
        for stem in ("sglang:num_queue_reqs", "sglang:num_running_reqs", "sglang:num_used_tokens"):

            def _sum_for(d, s):
                return sum(val for k, val in d.items() if k.split("{")[0] == s)

            print(f"  [gauge] {stem:42} {_sum_for(v1, stem)}")

    # ---- Q4: streaming meta_info surfacing under load (async) --------------
    asyncio.run(
        meta_info_probe(
            base,
            model=args.model,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            concurrency=args.concurrency,
        )
    )


if __name__ == "__main__":
    main()
