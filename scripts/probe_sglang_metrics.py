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

Usage:
    python scripts/probe_sglang_metrics.py \
        --url https://<replica-host> \
        --model Qwen/Qwen3.5-35B-A3B-FP8

Notes:
    - Needs a *running* replica (the matrix engines are ephemeral; spin one
      up or point at a deployed engine first).
    - Read-only except for a single tiny generation request used to probe
      the response shape and bump the histogram by one.
"""

from __future__ import annotations

import argparse
import json
import re

import httpx

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="Replica base URL (no trailing slash)")
    ap.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B-FP8")
    ap.add_argument("--prompt", default="Say hello in one word.")
    args = ap.parse_args()
    base = args.url.rstrip("/")

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
                    "model": args.model,
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


if __name__ == "__main__":
    main()
