# c=64 Experiment Plan

Qwen3.5-35B-A3B-FP8 on 2×L40S, 3 regions (Seoul, Frankfurt, Ashburn),
open-loop at concurrency 64. All data is Apr 2 GLM5 production traces.

## What we're measuring

Does optimizing GORGO's routing weights for different TTFT percentiles
(p50, p95, p99) produce meaningfully different policies, and do the
learned weights generalize to held-out traffic?

## Cost model

```
score(u) = rtt_weight     × rtt_ms(u)
         + prefill_weight × (prefill_rate(u) × uncached_tokens(u)
                           + queue_rate(u)   × queued_tokens(u))
```

Four parameters, two families:

| Parameter | Type | Units | How it's set |
|-----------|------|-------|-------------|
| `prefill_rate` | physical rate | ms/tok | Calibrated idle at startup (per-replica) |
| `queue_rate` | physical rate | ms/tok | Fitted continuously from live traffic residuals (per-replica) |
| `prefill_weight` | tuning weight | dimensionless | Searched by online-ES during tuning; frozen for eval |
| `rtt_weight` | tuning weight | dimensionless | Searched by online-ES during tuning; frozen for eval |

The rates are measured, the weights are learned. The ES tuner sees a
stable cost surface because the underlying rates are pinned/fitted
independently.

## Experiment flow

### What happens inside every run (~43 min)

```
 0:00  Fleet startup        18 engines (2×L40S) + 6 proxies spin up
10:00  Homogeneity check    Probe each replica for TTFT variance
10:20  Calibration          Gorgo pools only: 16 probes/replica, cache flushed,
                            idle → pin prefill_rate per-replica (ms/tok)
12:00  Workload starts      Trace replayed at c=64 open-loop
       ├─ rate fitter       Continuously fits queue_rate from live residuals
       └─ ES tuner          Searches (prefill_weight, rtt_weight) [tuning only]
42:00  Teardown             Flush traces, commit volume, stop fleet
```

### 9-run matrix

**Phase 1 — Tuning** (3 runs, sequential, ~2.2 hrs)

Replay T1 (Apr 2 night 00:30–01:00, ~2,753 requests). The ES tuner
learns `prefill_weight` and `rtt_weight` that minimize each metric.

| Run | Spec | Metric | Experiment ID |
|-----|------|--------|---------------|
| TUNE-p50 | `tuning/policy_matrix_c64_tuning_p50ttft.json` | `neg_p50_ttft` | `glm5_c64_tuning_p50ttft_v1` |
| TUNE-p95 | `tuning/policy_matrix_c64_tuning_p95ttft.json` | `neg_p95_ttft` | `glm5_c64_tuning_p95ttft_v1` |
| TUNE-p99 | `tuning/policy_matrix_c64_tuning_p99ttft.json` | `neg_p99_ttft` | `glm5_c64_tuning_p99ttft_v1` |

All use manifest `manifests/manifest_glm5_apr2_0030_0100.json`.

After each: extract `best_params` → fill into the corresponding eval spec.
Only `prefill_weight` and `rtt_weight` transfer. Rates are re-measured
on the eval fleet.

**Phase 2 — Evaluation** (6 runs, parallelizable, ~43 min)

Freeze the learned weights and replay on two held-out traces:

| Run | Spec | Manifest | Tests |
|-----|------|----------|-------|
| EVAL-p50-temporal | `eval/policy_matrix_c64_eval_p50ttft.json` | `manifests/manifest_glm5_apr2_0100_0130.json` | Temporal generalization |
| EVAL-p50-diurnal | `eval/policy_matrix_c64_eval_p50ttft.json` | `manifests/manifest_glm5_apr2_1230_1300.json` | Diurnal generalization |
| EVAL-p95-temporal | `eval/policy_matrix_c64_eval_p95ttft.json` | `manifests/manifest_glm5_apr2_0100_0130.json` | Temporal generalization |
| EVAL-p95-diurnal | `eval/policy_matrix_c64_eval_p95ttft.json` | `manifests/manifest_glm5_apr2_1230_1300.json` | Diurnal generalization |
| EVAL-p99-temporal | `eval/policy_matrix_c64_eval_p99ttft.json` | `manifests/manifest_glm5_apr2_0100_0130.json` | Temporal generalization |
| EVAL-p99-diurnal | `eval/policy_matrix_c64_eval_p99ttft.json` | `manifests/manifest_glm5_apr2_1230_1300.json` | Diurnal generalization |

Each eval runs all 6 policies in parallel (random, least-request,
least-load, prefix-cache, session-affinity, gorgo-static) for
head-to-head comparison on the same trace.

### Timing

| Schedule | Wall time | Peak GPUs |
|----------|-----------|-----------|
| All sequential | ~6.5 hrs | 36 |
| Tuning sequential → evals parallel | **~2.9 hrs** | 216 |
| Everything parallel | ~86 min | 216 |

## Traces

All on `GORGO-glm5-completions` volume under
`/data/mooncake_traces/abstract_night_traces/with_bodies/`.
Manifests in `specs/c64/manifests/`.

| ID | File | Window | Role | Rows | Users | Avg input | KV reuse (global / intra-user / cross-user) |
|----|------|--------|------|------|-------|-----------|---------------------------------------------|
| T1 | `glm5_apr2_0030_to_0100.jsonl` | Apr 2 00:30–01:00 | Tuning | 2,754 | 180 | 3,393 | 72.7% / 71.1% / 1.6% |
| T2 | `glm5_apr2_0100_to_0130.jsonl` | Apr 2 01:00–01:30 | Eval (temporal) | 3,262 | 194 | 4,482 | 76.3% / 75.2% / 1.0% |
| T3 | `glm5_apr2_1230_to_1300.jsonl` | Apr 2 12:30–13:00 | Eval (diurnal) | 4,323 | 237 | 4,046 | 74.0% / 71.8% / 2.2% |

Reuse is >97% intra-user (multi-turn conversation prefix sharing).
Cross-user shared prefixes are negligible (0.4–2.2%).

Token distribution: p50 input is 8 tokens (most requests are tiny),
p95 is ~17–19k tokens (the heavy tail that drives prefill cost).

## Commands

```bash
# TUNE-p50
modal run --detach --env=alessio-dev experiment_runner/policy_matrix_app.py::main \
  --base-spec-path specs/c64/tuning/policy_matrix_c64_tuning_p50ttft.json \
  --sweep-manifest-path specs/c64/manifests/manifest_glm5_apr2_0030_0100.json \
  --experiment-id glm5_c64_tuning_p50ttft_v1 \
  --start-index 0 --top-k 1 \
  --output-dir /results/policy_matrix_sweep/c64/glm5_c64_tuning_p50ttft_v1

# TUNE-p95
modal run --detach --env=alessio-dev experiment_runner/policy_matrix_app.py::main \
  --base-spec-path specs/c64/tuning/policy_matrix_c64_tuning_p95ttft.json \
  --sweep-manifest-path specs/c64/manifests/manifest_glm5_apr2_0030_0100.json \
  --experiment-id glm5_c64_tuning_p95ttft_v1 \
  --start-index 0 --top-k 1 \
  --output-dir /results/policy_matrix_sweep/c64/glm5_c64_tuning_p95ttft_v1

# TUNE-p99
modal run --detach --env=alessio-dev experiment_runner/policy_matrix_app.py::main \
  --base-spec-path specs/c64/tuning/policy_matrix_c64_tuning_p99ttft.json \
  --sweep-manifest-path specs/c64/manifests/manifest_glm5_apr2_0030_0100.json \
  --experiment-id glm5_c64_tuning_p99ttft_v1 \
  --start-index 0 --top-k 1 \
  --output-dir /results/policy_matrix_sweep/c64/glm5_c64_tuning_p99ttft_v1

# --- After extracting learned weights into eval specs ---

# EVAL-p50-temporal
modal run --detach --env=alessio-dev experiment_runner/policy_matrix_app.py::main \
  --base-spec-path specs/c64/eval/policy_matrix_c64_eval_p50ttft.json \
  --sweep-manifest-path specs/c64/manifests/manifest_glm5_apr2_0100_0130.json \
  --experiment-id glm5_c64_eval_p50ttft_temporal_v1 \
  --start-index 0 --top-k 1 \
  --output-dir /results/policy_matrix_sweep/c64/glm5_c64_eval_p50ttft_temporal_v1

# EVAL-p50-diurnal
modal run --detach --env=alessio-dev experiment_runner/policy_matrix_app.py::main \
  --base-spec-path specs/c64/eval/policy_matrix_c64_eval_p50ttft.json \
  --sweep-manifest-path specs/c64/manifests/manifest_glm5_apr2_1230_1300.json \
  --experiment-id glm5_c64_eval_p50ttft_diurnal_v1 \
  --start-index 0 --top-k 1 \
  --output-dir /results/policy_matrix_sweep/c64/glm5_c64_eval_p50ttft_diurnal_v1

# EVAL-p95-temporal
modal run --detach --env=alessio-dev experiment_runner/policy_matrix_app.py::main \
  --base-spec-path specs/c64/eval/policy_matrix_c64_eval_p95ttft.json \
  --sweep-manifest-path specs/c64/manifests/manifest_glm5_apr2_0100_0130.json \
  --experiment-id glm5_c64_eval_p95ttft_temporal_v1 \
  --start-index 0 --top-k 1 \
  --output-dir /results/policy_matrix_sweep/c64/glm5_c64_eval_p95ttft_temporal_v1

# EVAL-p95-diurnal
modal run --detach --env=alessio-dev experiment_runner/policy_matrix_app.py::main \
  --base-spec-path specs/c64/eval/policy_matrix_c64_eval_p95ttft.json \
  --sweep-manifest-path specs/c64/manifests/manifest_glm5_apr2_1230_1300.json \
  --experiment-id glm5_c64_eval_p95ttft_diurnal_v1 \
  --start-index 0 --top-k 1 \
  --output-dir /results/policy_matrix_sweep/c64/glm5_c64_eval_p95ttft_diurnal_v1

# EVAL-p99-temporal
modal run --detach --env=alessio-dev experiment_runner/policy_matrix_app.py::main \
  --base-spec-path specs/c64/eval/policy_matrix_c64_eval_p99ttft.json \
  --sweep-manifest-path specs/c64/manifests/manifest_glm5_apr2_0100_0130.json \
  --experiment-id glm5_c64_eval_p99ttft_temporal_v1 \
  --start-index 0 --top-k 1 \
  --output-dir /results/policy_matrix_sweep/c64/glm5_c64_eval_p99ttft_temporal_v1

# EVAL-p99-diurnal
modal run --detach --env=alessio-dev experiment_runner/policy_matrix_app.py::main \
  --base-spec-path specs/c64/eval/policy_matrix_c64_eval_p99ttft.json \
  --sweep-manifest-path specs/c64/manifests/manifest_glm5_apr2_1230_1300.json \
  --experiment-id glm5_c64_eval_p99ttft_diurnal_v1 \
  --start-index 0 --top-k 1 \
  --output-dir /results/policy_matrix_sweep/c64/glm5_c64_eval_p99ttft_diurnal_v1
```

## File map

```
specs/c64/
  EXPERIMENTS.md                              ← this file
  tuning/
    policy_matrix_c64_tuning_p50ttft.json     ES searches prefill_weight + rtt_weight, metric=neg_p50_ttft
    policy_matrix_c64_tuning_p95ttft.json     same, metric=neg_p95_ttft
    policy_matrix_c64_tuning_p99ttft.json     same, metric=neg_p99_ttft
    policy_matrix_c64_tuning.json             original v8 spec (reference)
  eval/
    policy_matrix_c64_eval_p50ttft.json       frozen weights from p50 tuning (TODO: fill after tuning)
    policy_matrix_c64_eval_p95ttft.json       frozen weights from p95 tuning (TODO: fill after tuning)
    policy_matrix_c64_eval_p99ttft.json       frozen weights from p99 tuning (TODO: fill after tuning)
    policy_matrix_c64_eval.json               original v8 eval (reference)
  manifests/
    manifest_glm5_apr2_0030_0100.json         T1: tuning trace
    manifest_glm5_apr2_0100_0130.json         T2: eval (temporal)
    manifest_glm5_apr2_1230_1300.json         T3: eval (diurnal)
    manifest_glm5_0030_0100.json              Apr 1 reference (not used)
    manifest_glm5_0100_0130.json              Apr 1 reference (not used)
    manifest_glm5_apr1_midday.json            Apr 1 reference (not used)
```

## Code changes (this session)

| File | What changed |
|------|-------------|
| `policy/gorgo.py` | Added `queue_rate` to schema; split scoring into `own_prefill + queue_delay` |
| `proxy/measure.py` | Added `_fit_queue_rate()` residual regression; extended `recommend_rates()` |
| `proxy/modal_proxy.py` | Plumbed `queued_tokens_at_dispatch`; rate fitting inside ES path; ES no longer clobbers `per_target` |
| `experiment_runner/policy_matrix_app.py` | Added `_calibrate_gorgo_fleet()` integrated into startup |
| `data_processing/build_mooncake_trace.py` | Added `analyze_trace_reuse()` utility |

## Prior runs (reference)

| Experiment ID | Trace | Metric | Learned |
|---------------|-------|--------|---------|
| `glm5_c64_tuning_v8` | Apr 1 00:30–01:00 | `neg_p95_ttft` | `prefill_weight=0.060, rtt_weight=4.132` |
| `glm5_c64_eval_v8` | Apr 2 00:30–01:00 | static eval | gorgo-static best TTFT, worst E2E |

## Trace analysis tool

```bash
modal run --env=alessio-dev data_processing/build_mooncake_trace.py::analyze_trace_reuse \
  --trace-paths-csv "/data/mooncake_traces/abstract_night_traces/with_bodies/glm5_apr2_0030_to_0100.jsonl,/data/mooncake_traces/abstract_night_traces/with_bodies/glm5_apr2_0100_to_0130.jsonl,/data/mooncake_traces/abstract_night_traces/with_bodies/glm5_apr2_1230_to_1300.jsonl"
```
