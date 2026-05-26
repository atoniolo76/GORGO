# Proxy Validation Runbook

This runbook lists incremental validation tests for the Modal proxy, model
replica registration, request routing, workload feeding, and tuning. Each test
has a command and the expected result. Run tests in order; if an early test
fails, debug that layer before continuing.

Use this shell setup for the curl-based tests:

```bash
export MODEL="Qwen/Qwen3.5-35B-A3B-FP8"
export PROXY_URL="https://ta-01kqbvvhc7p54t9aysq8kn2mkg-8000-hu9jm0appp9c91vkcc2f0z18q.w.modal.host"
```

## 0. Launch Proxy And Replica

Start or restart the proxy:

```bash
REGION=us-east modal run --env=GORGO proxy/modal_proxy.py::proxy
```

The proxy reads non-empty values from the shared Modal replicas dict on startup
and during its metrics refresh loop. Restarting the proxy should not require
restarting already-running model replicas.

The proxy refreshes replica `/metrics` every 30 seconds by default to keep model
logs readable. Override with `METRICS_REFRESH_INTERVAL_SECONDS` if a test needs
faster or slower polling:

```bash
REGION=us-east METRICS_REFRESH_INTERVAL_SECONDS=10 modal run --env=GORGO proxy/modal_proxy.py::proxy
```

Start one model replica if no live model is already registered:

```bash
REGION=us-west GPU_TYPE=H100 MODEL_ORG=Qwen MODEL_NAME=Qwen3.5-35B-A3B-FP8 modal run --env=GORGO engine/modal_sglang.py
```

Wait for the model replica log line:

```text
The server is fired up and ready to roll!
```

Copy the proxy tunnel URL from the proxy logs and export it:

```bash
export PROXY_URL="https://ta-01kqbvvhc7p54t9aysq8kn2mkg-8000-hu9jm0appp9c91vkcc2f0z18q.w.modal.host"
```

Expected result:

- The proxy logs a public tunnel URL.
- The model replica logs `The server is fired up and ready to roll!`.
- After successful model startup, `engine/modal_sglang.py` writes
  `replicas[REGION] = tunnel.url`.
- When a model function exits, it sets `replicas[REGION] = ""`.
- The proxy never clears the shared dict at startup; it reads non-empty values.

## 1. Proxy Alive And Replica Registered

Run:

```bash
curl -s "$PROXY_URL/replicas" | jq
```

Expected result:

```json
{
  "replicas": [
    "https://..."
  ],
  "count": 1,
  "registry": {
    "us-west": "https://..."
  }
}
```

Observed passing result:

```json
{
  "replicas": [
    "https://ta-01kqbwmec88ypg6z24jzbc1ksp-8000-po59rv3y5ayqf7u6db0tg5r59.w.modal.host"
  ],
  "count": 1
}
```

Observed passing two-replica result:

```json
{
  "replicas": [
    "https://ta-01kqc02pta4egjvfc8kjk4125v-8000-tkq5siyhklqxhh96n4ao5edeq.w.modal.host",
    "https://ta-01kqc02vktwdjeyzv04s3577y6-8000-r3fam2n2bfq8ifob167cvtd5c.w.modal.host"
  ],
  "count": 2,
  "registry": {
    "us-east": "",
    "us-west": "https://ta-01kqc02pta4egjvfc8kjk4125v-8000-tkq5siyhklqxhh96n4ao5edeq.w.modal.host",
    "eu": "https://ta-01kqc02vktwdjeyzv04s3577y6-8000-r3fam2n2bfq8ifob167cvtd5c.w.modal.host"
  }
}
```

If this does not work, stop here. Request routing cannot work until the proxy
is reachable and can report registered replicas. If curl prints no body, inspect
the tunnel and HTTP status:

```bash
curl -sv "$PROXY_URL/replicas"
```

The `registry` field is the raw Modal Dict key/value view. Empty values mean a
region key exists but is inactive; only non-empty URL values are included in
`replicas`. This is useful for debugging region-key overwrites or stale cleanup
from exiting model functions.

## 2. Replica Metrics Scrape

Run:

```bash
curl -s "$PROXY_URL/replica_metrics" | jq
```

Expected result:

- `metrics` contains the registered replica URL as a key.
- The replica entry includes fields like `num_running_reqs`, `num_queue_reqs`,
  `num_used_tokens`, `latency_seconds`, `gen_throughput`, and `utilization`.
- `errors` is empty for the registered replica after warmup.
- `last_refresh_age_seconds` is non-null.

If the replica appears under `errors` briefly during startup, wait a few seconds
and retry. If it stays in `errors`, metrics-aware policies will fall back or
behave poorly.

With the default 30-second polling interval, wait up to one interval after a
fresh model registration before expecting the latest metrics snapshot to update.

Observed passing result:

```json
{
  "refresh_interval_seconds": 1.0,
  "last_refresh_age_seconds": 0.20332650899999294,
  "errors": {},
  "metrics": {
    "https://ta-01kqbwmec88ypg6z24jzbc1ksp-8000-po59rv3y5ayqf7u6db0tg5r59.w.modal.host": {
      "num_running_reqs": 0,
      "num_queue_reqs": 0,
      "num_used_tokens": 0,
      "latency_seconds": 0.07480443099998979,
      "gen_throughput": 0.0,
      "utilization": 0.0
    }
  },
  "endpoints_queued_tokens": {
    "https://ta-01kqbwmec88ypg6z24jzbc1ksp-8000-po59rv3y5ayqf7u6db0tg5r59.w.modal.host": 0
  }
}
```

## 3. Basic Non-Streaming Routing

Run:

```bash
curl -s "$PROXY_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi in one sentence.\"}],\"max_tokens\":16}" | jq
```

Expected result:

- Response is valid JSON.
- Response has OpenAI-compatible fields such as `id`, `object`, `model`, and
  `choices`.
- `choices[0].message.content` contains a short model response.
- No `502` or `503` is returned.

Observed passing result:

```json
{
  "id": "1384fa0c32084dffa66a9071ca7fe3c5",
  "object": "chat.completion",
  "created": 1777442363,
  "model": "Qwen/Qwen3.5-35B-A3B-FP8",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Thinking Process:\n\n1.  **Analyze the Request:**\n    *",
        "reasoning_content": null,
        "tool_calls": null
      },
      "logprobs": null,
      "finish_reason": "length",
      "matched_stop": null
    }
  ],
  "usage": {
    "prompt_tokens": 16,
    "total_tokens": 32,
    "completion_tokens": 16,
    "prompt_tokens_details": null,
    "reasoning_tokens": 0
  },
  "metadata": {
    "weight_version": "default"
  }
}
```

Then inspect proxy state:

```bash
curl -s "$PROXY_URL/trie" | jq
curl -s "$PROXY_URL/samples" | jq
```

Expected result:

- `/trie` returns counters like `num_sequences`, `node_count`, and
  `replica_coverage`.
- `/samples` returns a valid sample buffer response. For non-streaming
  completions, `buffered_samples` is expected to remain `0` because tuning
  samples are currently recorded only from the SSE streaming path.

Observed passing result:

```json
{
  "num_sequences": 1,
  "total_tokens_inserted": 6,
  "unique_token_count": 6,
  "node_count": 2,
  "tagged_node_count": 1,
  "replica_coverage": {
    "https://ta-01kqbwmec88ypg6z24jzbc1ksp-8000-po59rv3y5ayqf7u6db0tg5r59.w.modal.host": 1
  }
}
```

```json
{
  "buffered_samples": 0,
  "max_buffer_size": 1000,
  "total_samples_appended": 0,
  "auto_tune": {
    "enabled": false,
    "window_size": 100,
    "hop_size": 50,
    "apply": true,
    "buffered_samples": 0,
    "samples_since_last_apply": 0,
    "samples_until_next_apply": null,
    "applied_count": 0,
    "last_applied_at_monotonic": null,
    "last_recommendation": null,
    "enabled_at_monotonic": null,
    "current_policy": "random",
    "current_hyperparameters": {
      "defaults": {
        "prefill_weight": 1.0,
        "load_weight": 1.0
      },
      "per_target": {}
    }
  },
  "recent": []
}
```

## 4. Streaming Routing

Run:

```bash
curl -N "$PROXY_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Count from one to five.\"}],\"max_tokens\":32,\"stream\":true}"
```

Expected result:

- Curl prints SSE chunks of the form `data: {...}`.
- The final event is `data: [DONE]`.
- The stream does not truncate or hang.
- The proxy forwards `stream_options.include_usage=true` on streaming requests
  when the caller does not provide it, so the final SSE usage event contains
  token counts for tuning samples.

Inspect samples and trie state:

```bash
curl -s "$PROXY_URL/samples" | jq
curl -s "$PROXY_URL/trie" | jq
```

Expected result:

- `/samples` includes at least one recent sample with `ttft_seconds`,
  `total_seconds`, token counts, and `target`.
- `/trie` counters increase compared with the pre-request state.

If `/samples` remains empty after a successful stream, check whether the
upstream sent a final usage event. Missing prompt/completion token counts cause
the proxy to skip recording the sample.

Observed passing result:

```json
{
  "buffered_samples": 1,
  "max_buffer_size": 1000,
  "total_samples_appended": 1,
  "auto_tune": {
    "enabled": false,
    "window_size": 100,
    "hop_size": 50,
    "apply": true,
    "buffered_samples": 1,
    "samples_since_last_apply": 0,
    "samples_until_next_apply": null,
    "applied_count": 0,
    "last_applied_at_monotonic": null,
    "last_recommendation": null,
    "enabled_at_monotonic": null,
    "current_policy": "random",
    "current_hyperparameters": {
      "defaults": {
        "prefill_weight": 1.0,
        "load_weight": 1.0
      },
      "per_target": {}
    }
  },
  "recent": [
    {
      "ping_seconds": 0.22245550799999592,
      "ttft_seconds": 0.187165193,
      "total_seconds": 0.354140163,
      "prefill_seconds": 0.0,
      "decode_seconds": 0.16697497,
      "prompt_tokens": 16,
      "completion_tokens": 32,
      "prefill_rate_seconds_per_token": 0.0,
      "decode_rate_seconds_per_token": 0.0052179678125,
      "target": "https://ta-01kqbwmec88ypg6z24jzbc1ksp-8000-po59rv3y5ayqf7u6db0tg5r59.w.modal.host",
      "recorded_at_monotonic": 193.708817365
    }
  ]
}
```

## 5. Control Plane Basics

Run:

```bash
curl -s "$PROXY_URL/policy" | jq
```

Expected result:

- Response includes the active `policy`.
- Response includes `supported`.
- Response includes `uses_hyperparameters`.
- Response includes current `hyperparameters` only when the active policy uses
  them. For non-GORGO policies, `hyperparameters` should be `null`.

Switch to a simple baseline policy:

```bash
curl -s -X POST "$PROXY_URL/policy" \
  -H "Content-Type: application/json" \
  -d '{"policy":"random"}' | jq
```

Expected result: response reports `"policy": "random"`,
`"uses_hyperparameters": false`, and `"hyperparameters": null`.

Try a low-risk metrics-aware policy:

```bash
curl -s -X POST "$PROXY_URL/policy" \
  -H "Content-Type: application/json" \
  -d '{"policy":"least-request"}' | jq
```

Expected result:

```json
{
  "policy": "least-request",
  "uses_hyperparameters": false,
  "hyperparameters": null
}
```

With one replica, this validates handler correctness, not load-balancing
quality. This test is currently passing for policy switching, but the response
shape changed after this run to hide GORGO hyperparameters for non-GORGO
policies.

## 6. Flush Path

Current checkpoint: this is the next test to rerun. A plain `curl
"$PROXY_URL/flush"` is expected to fail because it sends `GET /flush`; the route
is `POST /flush`.

Run:

```bash
curl -s -X POST "$PROXY_URL/flush" | jq
```

Expected result:

- Response has `"radix_trie_cleared": true`.
- Response includes a `replicas` object keyed by registered replica URL.
- Each replica result has an `ok` boolean and usually a `status_code`.
- An upstream flush error can be acceptable if the model is busy, but the proxy
  should not crash.

Confirm trie reset:

```bash
curl -s "$PROXY_URL/trie" | jq
```

Expected result: trie counters are reset or lower than before the flush.

## 7. Tiny Real Workload

Run:

```bash
modal run proxy/workload.py --proxy-url "$PROXY_URL" \
  --source hf --preset lmsys --num-requests 5 --stream true
```

If the workload needs the Modal environment explicitly, use:

```bash
modal run --env=GORGO proxy/workload.py --proxy-url "$PROXY_URL" \
  --source hf --preset lmsys --num-requests 5 --stream true
```

Observed failure before fixing the preset path:

```text
FileNotFoundError: Directory /lmsys/lmsys-chat-1m not found
```

Root cause: the LMSYS dataset is stored in the `GORGO-hf-datasets` volume at:

```text
/datasets/datasets/lmsys__lmsys-chat-1m
```

The `lmsys` preset now points at that path. If testing an older proxy/workload
image before the preset fix, pass the path explicitly:

```bash
modal run --env=GORGO proxy/workload.py --proxy-url "$PROXY_URL" \
  --source hf --data-path /datasets/datasets/lmsys__lmsys-chat-1m \
  --num-requests 5 --stream true
```

Expected result:

- The workload completes without proxy errors.
- The workload reports completed requests rather than connection failures.

Observed passing result:

```text
[workload] source=hf path=/datasets/datasets/lmsys__lmsys-chat-1m (preset=lmsys)
[workload] dispatching: proxy=https://ta-01kqbxdbv7g6tjcsqgv7wfphc4-8000-b0g7nbfag0qrbd5mapcbsgf3s.w.modal.host concurrency=16 offset=0 limit=5
[workload]   proxy: policy='least-request' model='Qwen/Qwen3.5-35B-A3B-FP8' replicas=1
[workload] hf dataset at '/datasets/datasets/lmsys__lmsys-chat-1m': 1000000 rows, using column 'conversation'
{"event": "progress", "elapsed_seconds": 14.855, "sent": 5, "done": 1, "ok": 1, "fail": 0, "rate_rps": 0.07}
{"event": "progress", "elapsed_seconds": 27.479, "sent": 5, "done": 3, "ok": 3, "fail": 0, "rate_rps": 0.11}

[workload] done in 31.3s
[workload]   sent=5 ok=5 fail=0 success_rate=100.0%
[workload]   request throughput=0.16 req/s
[workload]   token throughput   input=83.7 tok/s  output=378.8 tok/s  total=462.5 tok/s
[workload]   TTFT (s)         avg=1.699 p50=1.855 p95=2.173 p99=2.173 (n=5)
[workload]   request E2E (s)  avg=18.66 p50=21.90 p95=25.73 p99=25.73 (n=5)
[workload]   ITL (ms)         avg=7.3 p50=7.1 p95=8.0 p99=8.0 (n=5)
[workload]   decode (tok/s)   avg=137.5 p50=140.8 p95=144.1 p99=144.1 (n=5)
[workload]   input tokens     avg=524 p50=262 p95=1632 p99=1632 (n=5)
[workload]   output tokens    avg=2372 p50=2843 p95=3560 p99=3560 (n=5)
[workload]   saved results to volume GORGO-bench-results at /results/replay_20260429_061612.json
```

Observed passing two-replica `least-request` result:

```text
[workload] source=hf path=/datasets/datasets/lmsys__lmsys-chat-1m (preset=lmsys)
[workload] dispatching: proxy=https://ta-01kqc02kanwqhc5x3nzz4kgfq1-8000-997ytmjgkkqlx4mun8se0f90d.w.modal.host concurrency=16 offset=0 limit=5
[workload]   proxy: policy='least-request' model='Qwen/Qwen3.5-35B-A3B-FP8' replicas=2
[workload] hf dataset at '/datasets/datasets/lmsys__lmsys-chat-1m': 1000000 rows, using column 'conversation'
{"event": "progress", "elapsed_seconds": 11.237, "sent": 5, "done": 1, "ok": 1, "fail": 0, "rate_rps": 0.09}
{"event": "progress", "elapsed_seconds": 21.167, "sent": 5, "done": 2, "ok": 2, "fail": 0, "rate_rps": 0.09}
{"event": "progress", "elapsed_seconds": 39.73, "sent": 5, "done": 5, "ok": 5, "fail": 0, "rate_rps": 0.13}

[workload] done in 39.7s
[workload]   sent=5 ok=5 fail=0 success_rate=100.0%
[workload]   request throughput=0.13 req/s
[workload]   token throughput   input=66.0 tok/s  output=379.2 tok/s  total=445.2 tok/s
[workload]   TTFT (s)         avg=2.407 p50=2.276 p95=3.381 p99=3.381 (n=5)
[workload]   request E2E (s)  avg=20.20 p50=17.81 p95=36.11 p99=36.11 (n=5)
[workload]   ITL (ms)         avg=6.1 p50=6.0 p95=6.9 p99=6.9 (n=5)
[workload]   decode (tok/s)   avg=165.4 p50=167.3 p95=185.5 p99=185.5 (n=5)
[workload]   input tokens     avg=524 p50=262 p95=1632 p99=1632 (n=5)
[workload]   output tokens    avg=3013 p50=2298 p95=5941 p99=5941 (n=5)
[workload]   saved results to volume GORGO-bench-results at /results/replay_20260429_065530.json
```

Inspect accumulated samples and metrics:

```bash
curl -s "$PROXY_URL/samples" | jq
curl -s "$PROXY_URL/replica_metrics" | jq
```

Expected result:

- `/samples` has new entries.
- `/replica_metrics` still has the replica in `metrics`, not persistent
  `errors`.

If this passes, scale incrementally:

```bash
modal run proxy/workload.py --proxy-url "$PROXY_URL" \
  --source hf --preset lmsys --num-requests 25 --stream true

modal run proxy/workload.py --proxy-url "$PROXY_URL" \
  --source hf --preset lmsys --num-requests 100 --stream true
```

## 8. GORGO Policy And Hyperparameters

Run:

```bash
curl -s "$PROXY_URL/hyperparameters" | jq
```

Expected result:

- Response includes `hyperparameters`, `allowed_keys`, and `defaults`.

Switch to the `gorgo` policy:

```bash
curl -s -X POST "$PROXY_URL/policy" \
  -H "Content-Type: application/json" \
  -d '{"policy":"gorgo"}' | jq
```

Expected result: response reports `"policy": "gorgo"`.

Apply a small manual hyperparameter update:

```bash
curl -s -X POST "$PROXY_URL/hyperparameters" \
  -H "Content-Type: application/json" \
  -d '{"prefill_weight":0.8,"load_weight":1.5}' | jq
```

Expected result:

- Response includes updated `hyperparameters`.
- Updated values appear under the default hyperparameter store.

Run one streaming completion again:

```bash
curl -N "$PROXY_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Give a short routing smoke test response.\"}],\"max_tokens\":32,\"stream\":true}"
```

Expected result: `gorgo` routes successfully with configured hyperparameters.

Observed passing result after flush and tiny workload:

```text
[workload] source=hf path=/datasets/datasets/lmsys__lmsys-chat-1m (preset=lmsys)
[workload] dispatching: proxy=https://ta-01kqbxdbv7g6tjcsqgv7wfphc4-8000-b0g7nbfag0qrbd5mapcbsgf3s.w.modal.host concurrency=16 offset=0 limit=5
[workload]   proxy: policy='gorgo' model='Qwen/Qwen3.5-35B-A3B-FP8' replicas=1
[workload]   proxy hyperparameters: {'defaults': {'prefill_weight': 0.8, 'load_weight': 1.5}, 'per_target': {}}
[workload] hf dataset at '/datasets/datasets/lmsys__lmsys-chat-1m': 1000000 rows, using column 'conversation'
{"event": "progress", "elapsed_seconds": 13.42, "sent": 5, "done": 1, "ok": 1, "fail": 0, "rate_rps": 0.07}
{"event": "progress", "elapsed_seconds": 19.199, "sent": 5, "done": 2, "ok": 2, "fail": 0, "rate_rps": 0.1}
{"event": "progress", "elapsed_seconds": 29.414, "sent": 5, "done": 4, "ok": 4, "fail": 0, "rate_rps": 0.14}
{"event": "progress", "elapsed_seconds": 49.612, "sent": 5, "done": 5, "ok": 5, "fail": 0, "rate_rps": 0.1}

[workload] done in 49.6s
[workload]   sent=5 ok=5 fail=0 success_rate=100.0%
[workload]   request throughput=0.10 req/s
[workload]   token throughput   input=52.8 tok/s  output=338.9 tok/s  total=391.7 tok/s
[workload]   TTFT (s)         avg=0.782 p50=0.729 p95=1.031 p99=1.031 (n=5)
[workload]   request E2E (s)  avg=22.58 p50=19.00 p95=45.17 p99=45.17 (n=5)
[workload]   ITL (ms)         avg=6.8 p50=7.0 p95=7.3 p99=7.3 (n=5)
[workload]   decode (tok/s)   avg=147.6 p50=143.6 p95=168.8 p99=168.8 (n=5)
[workload]   input tokens     avg=524 p50=262 p95=1632 p99=1632 (n=5)
[workload]   output tokens    avg=3363 p50=2624 p95=7453 p99=7453 (n=5)
[workload]   saved results to volume GORGO-bench-results at /results/replay_20260429_062139.json
```

Observed passing two-replica `gorgo` result:

```text
[workload] source=hf path=/datasets/datasets/lmsys__lmsys-chat-1m (preset=lmsys)
[workload] dispatching: proxy=https://ta-01kqc02kanwqhc5x3nzz4kgfq1-8000-997ytmjgkkqlx4mun8se0f90d.w.modal.host concurrency=16 offset=0 limit=5
[workload]   proxy: policy='gorgo' model='Qwen/Qwen3.5-35B-A3B-FP8' replicas=2
[workload]   proxy hyperparameters: {'defaults': {'prefill_weight': 1.0, 'load_weight': 1.0}, 'per_target': {}}
[workload] hf dataset at '/datasets/datasets/lmsys__lmsys-chat-1m': 1000000 rows, using column 'conversation'
{"event": "progress", "elapsed_seconds": 11.152, "sent": 5, "done": 1, "ok": 1, "fail": 0, "rate_rps": 0.09}
{"event": "progress", "elapsed_seconds": 25.442, "sent": 5, "done": 3, "ok": 3, "fail": 0, "rate_rps": 0.12}
{"event": "progress", "elapsed_seconds": 35.876, "sent": 5, "done": 5, "ok": 5, "fail": 0, "rate_rps": 0.14}

[workload] done in 35.9s
[workload]   sent=5 ok=5 fail=0 success_rate=100.0%
[workload]   request throughput=0.14 req/s
[workload]   token throughput   input=73.1 tok/s  output=447.3 tok/s  total=520.4 tok/s
[workload]   TTFT (s)         avg=1.169 p50=1.568 p95=1.831 p99=1.831 (n=5)
[workload]   request E2E (s)  avg=19.32 p50=21.69 p95=32.12 p99=32.12 (n=5)
[workload]   ITL (ms)         avg=5.9 p50=6.2 p95=6.2 p99=6.2 (n=5)
[workload]   decode (tok/s)   avg=171.9 p50=160.6 p95=199.4 p99=199.4 (n=5)
[workload]   input tokens     avg=524 p50=262 p95=1632 p99=1632 (n=5)
[workload]   output tokens    avg=3210 p50=3427 p95=5466 p99=5466 (n=5)
[workload]   saved results to volume GORGO-bench-results at /results/replay_20260429_065732.json
```

Comparison note: the two-replica smoke comparison is directionally encouraging
for GORGO on this tiny run (`TTFT avg 1.169s` vs `2.407s`, `p95 1.831s` vs
`3.381s`, with 5/5 success for both), but `n=5` is too small to claim a
statistically meaningful routing-quality win. Treat it as validation that both
multi-replica policies route successfully; use larger repeated runs for policy
claims.

## 9. Tuning Smoke Test

Run:

```bash
curl -s "$PROXY_URL/tune" | jq
```

Expected result:

- Response includes `auto_tune`.
- `enabled` is initially `false` unless already configured.

Enable tuning in shadow mode first:

```bash
curl -s -X POST "$PROXY_URL/tune" \
  -H "Content-Type: application/json" \
  -d '{"enabled":true,"window_size":5,"hop_size":2,"apply":false}' | jq
```

Expected result:

- Response has `auto_tune.enabled: true`.
- Response has `auto_tune.apply: false`.
- If enabling fails, confirm the active policy is `gorgo`.

Observed passing result:

```json
{
  "auto_tune": {
    "enabled": true,
    "window_size": 5,
    "hop_size": 2,
    "apply": false,
    "buffered_samples": 5,
    "samples_since_last_apply": 0,
    "samples_until_next_apply": 2,
    "applied_count": 0,
    "last_applied_at_monotonic": null,
    "last_recommendation": null,
    "enabled_at_monotonic": 521.74053396,
    "current_policy": "gorgo",
    "current_hyperparameters": {
      "defaults": {
        "prefill_weight": 1.0,
        "load_weight": 1.0
      },
      "per_target": {}
    }
  },
  "preview": {
    "window_size_used": 5,
    "recommendation": {
      "defaults": {
        "prefill_weight": 0.0038264109391304576,
        "load_weight": 0.006223890801575723
      },
      "per_target": {}
    }
  }
}
```

Run a tiny workload:

```bash
modal run proxy/workload.py --proxy-url "$PROXY_URL" \
  --source hf --preset lmsys --num-requests 10 --stream true
```

Inspect tuner state:

```bash
curl -s "$PROXY_URL/tune" | jq
curl -s "$PROXY_URL/samples" | jq
```

Expected result:

- `buffered_samples` increases.
- `last_recommendation` eventually appears after enough samples.
- In shadow mode, recommendations are produced without mutating live
  hyperparameters.

Observed passing proxy logs during the tiny workload:

```text
[proxy] auto-tune #1 window=5 defaults={'prefill_weight': 0.000839044623188541, 'load_weight': 0.008014929686309102} per_target=[] (apply=False)
[proxy] auto-tune #2 window=5 defaults={'prefill_weight': 0.002569623382429061, 'load_weight': 0.007499608868239356} per_target=[] (apply=False)
[proxy] auto-tune #3 window=5 defaults={'prefill_weight': 0.002976764047619033, 'load_weight': 0.007236269025833169} per_target=[] (apply=False)
```

This confirms the tuner recomputed recommendations from live traffic while
leaving live hyperparameters unchanged because `apply=false`.

Observed passing shadow-tuning workload result:

```text
[workload] source=hf path=/datasets/datasets/lmsys__lmsys-chat-1m (preset=lmsys)
[workload] dispatching: proxy=https://ta-01kqc02kanwqhc5x3nzz4kgfq1-8000-997ytmjgkkqlx4mun8se0f90d.w.modal.host concurrency=16 offset=0 limit=10
[workload]   proxy: policy='gorgo' model='Qwen/Qwen3.5-35B-A3B-FP8' replicas=2
[workload]   proxy hyperparameters: {'defaults': {'prefill_weight': 1.0, 'load_weight': 1.0}, 'per_target': {}}
[workload] hf dataset at '/datasets/datasets/lmsys__lmsys-chat-1m': 1000000 rows, using column 'conversation'
{"event": "progress", "elapsed_seconds": 12.035, "sent": 10, "done": 1, "ok": 1, "fail": 0, "rate_rps": 0.08}
{"event": "progress", "elapsed_seconds": 22.148, "sent": 10, "done": 2, "ok": 2, "fail": 0, "rate_rps": 0.09}
{"event": "progress", "elapsed_seconds": 30.295, "sent": 10, "done": 6, "ok": 6, "fail": 0, "rate_rps": 0.2}
{"event": "progress", "elapsed_seconds": 40.243, "sent": 10, "done": 7, "ok": 7, "fail": 0, "rate_rps": 0.17}
{"event": "progress", "elapsed_seconds": 46.543, "sent": 10, "done": 8, "ok": 8, "fail": 0, "rate_rps": 0.17}
{"event": "progress", "elapsed_seconds": 59.148, "sent": 10, "done": 10, "ok": 10, "fail": 0, "rate_rps": 0.17}

[workload] done in 59.1s
[workload]   sent=10 ok=10 fail=0 success_rate=100.0%
[workload]   request throughput=0.17 req/s
[workload]   token throughput   input=84.6 tok/s  output=755.9 tok/s  total=840.5 tok/s
[workload]   TTFT (s)         avg=0.904 p50=0.866 p95=1.576 p99=1.576 (n=10)
[workload]   request E2E (s)  avg=30.59 p50=27.40 p95=56.25 p99=56.25 (n=10)
[workload]   ITL (ms)         avg=7.1 p50=7.5 p95=8.4 p99=8.4 (n=10)
[workload]   decode (tok/s)   avg=145.1 p50=138.2 p95=191.7 p99=191.7 (n=10)
[workload]   input tokens     avg=500 p50=345 p95=1632 p99=1632 (n=10)
[workload]   output tokens    avg=4471 p50=3787 p95=8681 p99=8681 (n=10)
[workload]   saved results to volume GORGO-bench-results at /results/replay_20260429_070000.json
```

## 10. Apply-Mode Tuning Smoke Test

After shadow mode works, enable apply mode:

```bash
curl -s -X POST "$PROXY_URL/tune" \
  -H "Content-Type: application/json" \
  -d '{"apply":true}' | jq
```

Expected result: response has `auto_tune.apply: true`.

Run another tiny workload and confirm hyperparameters are updated:

```bash
curl -s -X POST "$PROXY_URL/tune" \
  -H "Content-Type: application/json" \
  -d '{"apply":true}' | jq

modal run --env=GORGO proxy/workload.py --proxy-url "$PROXY_URL" \
  --source hf --preset lmsys --num-requests 10 --stream true

curl -s "$PROXY_URL/tune" | jq
curl -s "$PROXY_URL/hyperparameters" | jq
```

Expected result:

- `auto_tune.apply` is `true`.
- `applied_count` increments after enough new samples.
- `last_recommendation` is non-null.
- `/hyperparameters` changes from the previous static defaults to the tuned
  recommendation.
- Traffic still succeeds while the tuner mutates live GORGO parameters.

After this passes, tuning has been validated in both shadow mode and apply mode.

## 11. Logical Next Step: Outcome-Based Tuner TBD

The logical next step is validating the actual outcome-based tuner path. Exact
commands are TBD, but useful starting points:

- Define the target outcome metric before running larger comparisons (for
  example TTFT, p95 TTFT, E2E latency, throughput, or a weighted objective).
- Run repeated baseline workloads for `least-request` and current `gorgo` with
  the same replica pool, request count, concurrency, dataset offset, and flush
  behavior.
- Run the outcome-based tuner on the same workload slice and confirm it changes
  routing parameters in the expected direction.
- Compare before/after runs using saved files in `GORGO-bench-results`.
- Promote only changes that improve the chosen outcome across repeated runs, not
  just one noisy sample.

## 12. Lower Priority / Deadline Optional

Reset hyperparameters to factory defaults:

```bash
curl -s -X PUT "$PROXY_URL/hyperparameters" \
  -H "Content-Type: application/json" \
  -d '{}' | jq
```

Manually replace the replica set:

```bash
curl -s -X POST "$PROXY_URL/replicas" \
  -H "Content-Type: application/json" \
  -d '{"replicas":["https://replica-a.modal.run"]}' | jq
```

Manual `/replicas` replacement is useful for debugging stale endpoints, but the
normal path is for `engine/modal_sglang.py` to auto-register its tunnel URL.

Other lower-priority tests:

- Large workloads and long soak tests.
- High concurrency.
- Killing replicas mid-run.
- Same-region multi-replica routing.
- Tuning convergence quality.
- Policy-to-policy performance comparisons.
- HTTP/2 and connection reuse benchmarking.
