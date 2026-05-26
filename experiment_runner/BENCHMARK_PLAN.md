# GORGO Experiment Plan

## Overview

This document describes the experimental setup for evaluating GORGO against baseline routing policies across three production chat-completion datasets. Each dataset is replayed as a Mooncake FAST '25 trace against isolated per-policy fleets of SGLang replicas deployed across three geographic regions.

## Infrastructure

| Parameter | Value |
| --- | --- |
| **Model** | Qwen/Qwen3.5-35B-A3B-FP8 |
| **Replicas per policy** | 3 × L40S:2 (tensor-parallel 2, 96 GB VRAM per replica) |
| **Regions** | `ap-seoul-1` (Asia), `eu-frankfurt-1` (Europe), `us-ashburn-1` (US East) |
| **Proxy** | 1 per policy in `us-east-1`, running the GORGO routing proxy (`proxy/modal_proxy.py`) |
| **Context length** | 32,768 tokens |
| **Max output tokens** | 128 |
| **Concurrency** | 32 in-flight requests per proxy |
| **Arrival mode** | Open-loop (Poisson-scheduled for HF datasets, real timestamps for GLM5) |

Each policy runs on its own dedicated fleet so routing decisions are fully isolated — no cross-policy interference on shared replicas.

## Routing Policies

### Baselines

| Label | Policy | Routing signal | Description |
| --- | --- | --- | --- |
| `random` | `random` | None | Uniform random. Lower bound on routing quality. |
| `least-request` | `least-request` | `max(num_running_reqs, proxy_inflight)` | Routes to the replica with fewest in-flight requests. Industry standard (NGINX `least_conn`). Uses a proxy-side in-flight counter to bridge staleness between SGLang metrics scrapes. |
| `least-load` | `least-load` | `num_running + num_queue + num_used_tokens` | Token-weighted load balance. Routes away from replicas with high KV-cache occupancy. |
| `prefix-cache` | `prefix-cache` | Radix trie prefix match length | Routes to the replica with the longest cached prefix for the incoming prompt. Maximizes KV-cache hit rate but ignores load. |
| `simple-session-affinity` | `simple-session-affinity` | Hash of first 256 token IDs | Sticky routing — same prompt prefix always hits the same replica. Maximizes intra-user cache reuse. *(GLM5 and WildChat only; excluded from LMSYS which has no user identity.)* |

### GORGO Variants

All three variants use the same additive cost model:

```
score(replica) = network_rtt
               + prefill_weight × (input_tokens − cached_prefix_tokens)
               + load_weight × (queued_tokens + used_tokens)
```

Where:
- **`network_rtt`** — EWMA-smoothed round-trip time from a dedicated lightweight probe (`GET /` to each replica), isolating pure network latency from SGLang's `/metrics` handler load.
- **`prefill_weight`** — per-uncached-token prefill cost. Fitted against `(TTFT − network_rtt) / uncached_tokens` so the rate represents actual prefill work, not an amortized rate diluted by cache hits.
- **`cached_prefix_tokens`** — looked up from the proxy's local radix trie at routing time.
- **`queued_tokens`** — proxy-side in-flight token counter, updated on every dispatch/completion.

| Label | Auto-tune | Description |
| --- | --- | --- |
| `gorgo-static` | Off | Fixed hyperparameters. On W1 runs, starts from manual values (`prefill_weight=0.07`, `load_weight=0.06`). On W2 runs, starts from the W1 hillclimb-learned values. Tests the cost model's value independent of online tuning. |
| `gorgo-autotune` | `fit` mode | Median-of-rates per-target fit. Every 16 new samples, recomputes `prefill_weight` and `load_weight` per replica from the last 64 observations. Adapts to per-replica RTT and hardware differences. |
| `gorgo-hillclimb` | `online-es` mode | Gaussian (1+1)-Evolution Strategy with Rechenberg's 1/5 success rule. Directly minimizes `neg_p95_ttft` over the rolling 64-sample window by perturbing hyperparameters in log-space. Treats the cost model weights as abstract knobs — no physical interpretation needed. |

## Datasets

| Dataset | Source | Total rows | Avg tokens/req | Intra-user reuse | Cross-user reuse | Global reuse |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| **GLM-5.1** | GLM production traffic | 411,000 | ~4,000–17,000 | 53.7% | 1.6% | 55.3% |
| **LMSYS-Chat-1M** | Chatbot Arena | 1,000,000 | ~470 | N/A (no user identity) | 9.0% | 9.0% |
| **WildChat-4.8M** | ChatGPT proxy | 3,200,000 | ~2,900 | 5.3% | 29.1% | 34.4% |

Reuse statistics from `data_processing/prefix_trie_results/*/stats.json`. These are dataset-level numbers over the full corpus; trace-level reuse depends on the time window selected.

## Trace Generation

Traces are built with `data_processing/build_mooncake_trace.py` using `--selection-mode chronological` (no curation) and `--include-bodies` (replayable). Each trace is pre-cleaned at build time via `--max-input-tokens 24000` to filter requests that exceed the model's context window.

### GLM5 traces

Two consecutive 30-minute windows from April 1, 2026 night traffic:

| Window | Time range | Purpose | Output path |
| --- | --- | --- | --- |
| W1 | 00:30–01:00 UTC | Tuning window (hyperparameter discovery) | `abstract_night_traces/.../glm5_0030_to_0100.jsonl` |
| W2 | 01:00–01:30 UTC | Evaluation window (fresh, unseen data) | `abstract_night_traces/.../glm5_0100_to_0130.jsonl` |

GLM5 uses real parquet timestamps with `--time-scale 1.0` so the trace replays at original arrival speed.

### LMSYS and WildChat traces

Two non-overlapping windows from each dataset using `--skip-rows` for the second window:

| Dataset | Window | Rows | Arrival rate | Notes |
| --- | --- | ---: | ---: | --- |
| LMSYS W1 | rows 0–108k | 108,000 | 60 req/s (Poisson) | No real timestamps; synthetic arrivals |
| LMSYS W2 | rows 108k–216k | 108,000 | 60 req/s (Poisson) | Same seed for reproducibility |
| WildChat W1 | rows 0–20.5k | 20,520 | 11.4 req/s (Poisson) | `--force-synthetic-arrivals` (real timestamps span days) |
| WildChat W2 | rows 20.5k–41k | 20,520 | 11.4 req/s (Poisson) | Same |

## How the Experiment Controller Works

`experiments/policy_matrix_app.py` is a single Modal app that orchestrates the entire experiment lifecycle:

```
1. LAUNCH FLEET
   For each policy × region, spawn an SGLang engine (L40S:2) and register
   its tunnel URL in the GORGO-replicas Modal Dict. Then spawn one proxy
   per policy and register it in GORGO-proxies. Wait for all to come online.

2. CONFIGURE
   For each policy's proxy:
     POST /replicas  → assign its 3-replica fleet
     POST /policy    → set the routing policy
     POST /hyperparameters → set starter (or learned) weights
     POST /tune      → enable auto-tune if configured in the spec

3. PRE-FLIGHT CHECKS
   • Homogeneity check: probe each replica directly with 16 streaming
     requests (bypassing the proxy) to detect cold-start asymmetry.
   • Metrics-ready gate: poll GET /replica_metrics on each proxy until
     all 3 replicas report live metrics with no errors.

4. RUN WORKLOAD
   All policies start simultaneously (synchronized via start_at_wall_time):
     POST /trace/start  → begin capturing routing decisions + metrics
     POST /workload/start → replay the Mooncake trace via localhost
   The proxy dispatches requests through its own routing policy against
   its own replica fleet. Policies run in parallel via asyncio.gather.

5. COLLECT
   When the workload finishes (or is cancelled via stop_experiment.py):
     POST /trace/stop + /trace/save → persist traces to GORGO-bench-results
     GET /hyperparameters → capture final auto-tuned values
   Write the per-policy results + sweep manifest to the volume.
```

The controller reads two files: a **spec** (policies, regions, concurrency, auto-tune config) and a **manifest** (which trace JSONL to replay). Same spec can be reused with different manifests to test the same policies on different datasets.

## Running Experiments

### Build traces

```bash
# GLM5 W1
modal run --env=alessio-dev data_processing/build_mooncake_trace.py::main \
  --source glm5 --start-time 2026-04-01T00:30:00 --end-time 2026-04-01T01:00:00 \
  --num-requests 200000 --selection-mode chronological --include-bodies \
  --max-input-tokens 24000 --time-scale 1.0 \
  --output-path mooncake_traces/abstract_night_traces/with_bodies/glm5_0030_to_0100.jsonl

# GLM5 W2
# (same command, change --start-time/--end-time to 01:00-01:30, change output path)
```

### Launch experiment

```bash
modal run --detach --env=alessio-dev experiments/policy_matrix_app.py::main \
  --base-spec-path specs/policy_matrix_abstract_night.json \
  --sweep-manifest-path specs/manifest_glm5_0030_0100.json \
  --experiment-id abstract_night_glm5_w1_v1 \
  --start-index 0 --top-k 1 \
  --output-dir /results/policy_matrix_sweep/abstract_night/glm5_w1_v1
```

For W2 runs, use `specs/policy_matrix_abstract_night_w2.json` (seeds GORGO variants with W1-learned hyperparameters) and swap the manifest to `specs/manifest_glm5_0100_0130.json`.

### Monitor

```bash
source .venv/bin/activate
python scripts/experiment_status.py --experiment-id abstract_night_glm5_w1_v1 --env alessio-dev
```

### Early stop with partial results

```bash
python scripts/stop_experiment.py --experiment-id abstract_night_glm5_w1_v1
```

Cancels all running workloads, saves traces to volume, and computes partial TTFT/E2E stats from whatever requests completed. The controller writes the manifest as usual.

## Metrics Collected

| Category | Metrics |
| --- | --- |
| **Latency** | TTFT p50/p95/p99/max, E2E p50/p95/p99, ITL p50/p95 |
| **Throughput** | req/s, input tok/s, output tok/s, wall time |
| **Reliability** | Success rate, fallback rate (% of requests where the policy bailed to random) |
| **Routing** | Per-replica request share, routing concentration |
| **Cache** | Per-request `cached_prefix_tokens` from the proxy trace |
| **Auto-tune** | Final learned hyperparameters, online-ES sigma/score trajectory |
| **Fleet health** | Per-replica network RTT (EWMA), homogeneity check results |

## Output Artifacts

Results are written to the `GORGO-bench-results` Modal volume:

```
/results/policy_matrix_sweep/abstract_night/<experiment_id>_<timestamp>/
  <run_id>_sweep_matrix.json              ← aggregate manifest
  <run_id>_<trace_stem>.json              ← per-policy results with stats

/results/workload_runs/
  <run_id>_<trace_stem>_<policy>.json     ← detailed per-request results

/results/proxy_traces/
  <run_id>_<trace_stem>_<policy>/
    metrics.jsonl                         ← per-scrape replica metrics + network RTT
    requests.jsonl                        ← per-request routing decisions + outcomes
    manifest.json                         ← trace metadata + fallback summary
```

Pull results locally:

```bash
modal volume get --env=alessio-dev --force GORGO-bench-results \
  /policy_matrix_sweep/abstract_night/<experiment_id> results/policy_matrix_sweep/abstract_night/

modal volume get --env=alessio-dev --force GORGO-bench-results \
  /workload_runs results/
```

## Spec Files

| File | Purpose |
| --- | --- |
| `specs/policy_matrix_abstract_night.json` | GLM5 W1 spec — 8 policies, manual GORGO starter hyperparameters |
| `specs/policy_matrix_abstract_night_w2.json` | GLM5 W2 spec — 8 policies, GORGO seeded with W1-learned values |
| `specs/policy_matrix_abstract_night_lmsys.json` | LMSYS spec — 7 policies (no session-affinity) |
| `specs/policy_matrix_abstract_night_wildchat.json` | WildChat spec — 8 policies |
| `specs/manifest_glm5_0030_0100.json` | GLM5 W1 trace manifest |
| `specs/manifest_glm5_0100_0130.json` | GLM5 W2 trace manifest |
| `specs/manifest_lmsys_window1.json` | LMSYS W1 trace manifest |
| `specs/manifest_lmsys_window2.json` | LMSYS W2 trace manifest |
| `specs/manifest_wildchat_window1.json` | WildChat W1 trace manifest |
| `specs/manifest_wildchat_window2.json` | WildChat W2 trace manifest |
