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
         + prefill_weight × uncached_tokens(u)
         + load_weight    × queued_tokens(u)
```

Three weights, all learned by ES:

| Parameter | What it controls | Range |
|-----------|-----------------|-------|
| `rtt_weight` | ms of score per ms of network RTT | (1e-5, 50.0) |
| `prefill_weight` | ms of score per uncached prompt token (absorbs hardware speed) | (1e-5, 5.0) |
| `load_weight` | ms of score per queued token on the replica (absorbs queue drain rate) | (1e-5, 5.0) |

No physical rates (`prefill_rate`, `queue_rate`). No calibration phase.
The ES absorbs hardware speed and queue dynamics into the weights.
`prefill_weight` and `load_weight` are decoupled so the tuner can
independently balance "prefer cached replicas" vs "avoid loaded replicas."

## Experiment flow

### What happens inside every run (~42 min)

```
 0:00  Fleet startup        18 engines (2×L40S) + 6 proxies spin up
10:00  Homogeneity check    Probe each replica for TTFT variance
10:20  Workload starts      Trace replayed at c=64 open-loop
       └─ ES tuner          Searches (prefill_weight, load_weight, rtt_weight)
                            [tuning only; eval runs use frozen weights]
40:00  Teardown             Flush traces, commit volume, stop fleet
```

### 9-run matrix

**Phase 1 — Tuning** (3 runs, sequential, ~2.1 hrs)

Replay T1 (Apr 2 night 00:30–01:00, ~2,753 requests). The ES tuner
learns `prefill_weight`, `load_weight`, and `rtt_weight` that minimize
each metric.

| Run | Spec | Metric | Experiment ID |
|-----|------|--------|---------------|
| TUNE-p50 | `tuning/policy_matrix_c64_tuning_p50ttft.json` | `neg_p50_ttft` | `glm5_c64_tuning_p50ttft_v1` |
| TUNE-p95 | `tuning/policy_matrix_c64_tuning_p95ttft.json` | `neg_p95_ttft` | `glm5_c64_tuning_p95ttft_v1` |
| TUNE-p99 | `tuning/policy_matrix_c64_tuning_p99ttft.json` | `neg_p99_ttft` | `glm5_c64_tuning_p99ttft_v1` |

All use manifest `manifests/manifest_glm5_apr2_0030_0100.json`.

After each: extract `best_params` → fill into the corresponding eval spec.

**Phase 2 — Evaluation** (6 runs, parallelizable, ~42 min)

Freeze the learned weights and replay on two held-out traces:

| Run | Spec | Manifest | Tests |
|-----|------|----------|-------|
| EVAL-p50-temporal | `eval/policy_matrix_c64_eval_p50ttft.json` | `manifests/manifest_glm5_apr2_0100_0130.json` | Temporal generalization |
| EVAL-p50-diurnal | `eval/policy_matrix_c64_eval_p50ttft.json` | `manifests/manifest_glm5_apr2_1230_1300.json` | Diurnal generalization |
| EVAL-p95-temporal | `eval/policy_matrix_c64_eval_p95ttft.json` | `manifests/manifest_glm5_apr2_0100_0130.json` | Temporal generalization |
| EVAL-p95-diurnal | `eval/policy_matrix_c64_eval_p95ttft.json` | `manifests/manifest_glm5_apr2_1230_1300.json` | Diurnal generalization |
| EVAL-p99-temporal | `eval/policy_matrix_c64_eval_p99ttft.json` | `manifests/manifest_glm5_apr2_0100_0130.json` | Temporal generalization |
| EVAL-p99-diurnal | `eval/policy_matrix_c64_eval_p99ttft.json` | `manifests/manifest_glm5_apr2_1230_1300.json` | Diurnal generalization |

Each eval runs all 6 policies in parallel for head-to-head comparison.

## Traces

All on `GORGO-glm5-completions` volume under
`/data/mooncake_traces/abstract_night_traces/with_bodies/`.
Manifests in `specs/c64/manifests/`.

| ID | File | Window | Role | Rows | Users | Avg input | KV reuse (global / intra / cross) |
|----|------|--------|------|------|-------|-----------|-----------------------------------|
| T1 | `glm5_apr2_0030_to_0100.jsonl` | Apr 2 00:30–01:00 | Tuning | 2,754 | 180 | 3,393 | 72.7% / 71.1% / 1.6% |
| T2 | `glm5_apr2_0100_to_0130.jsonl` | Apr 2 01:00–01:30 | Eval (temporal) | 3,262 | 194 | 4,482 | 76.3% / 75.2% / 1.0% |
| T3 | `glm5_apr2_1230_to_1300.jsonl` | Apr 2 12:30–13:00 | Eval (diurnal) | 4,323 | 237 | 4,046 | 74.0% / 71.8% / 2.2% |

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
    policy_matrix_c64_tuning_p50ttft.json     ES searches 3 weights, metric=neg_p50_ttft
    policy_matrix_c64_tuning_p95ttft.json     same, metric=neg_p95_ttft
    policy_matrix_c64_tuning_p99ttft.json     same, metric=neg_p99_ttft
    policy_matrix_c64_tuning.json             original v8 spec (reference)
  eval/
    policy_matrix_c64_eval_p50ttft.json       frozen weights from p50 tuning (TODO)
    policy_matrix_c64_eval_p95ttft.json       frozen weights from p95 tuning (TODO)
    policy_matrix_c64_eval_p99ttft.json       frozen weights from p99 tuning (TODO)
    policy_matrix_c64_eval.json               original v8 eval (reference)
  manifests/
    manifest_glm5_apr2_0030_0100.json         T1: tuning trace
    manifest_glm5_apr2_0100_0130.json         T2: eval (temporal)
    manifest_glm5_apr2_1230_1300.json         T3: eval (diurnal)
    manifest_glm5_0030_0100.json              Apr 1 reference (not used)
    manifest_glm5_0100_0130.json              Apr 1 reference (not used)
    manifest_glm5_apr1_midday.json            Apr 1 reference (not used)
```

## Code changes (3-weight model)

| File | What changed |
|------|-------------|
| `policy/gorgo.py` | 3-weight model: `rtt_weight × rtt_ms + prefill_weight × uncached + load_weight × queued`. No rates. |
| `proxy/modal_proxy.py` | Removed rate fitting from online-es path. ES searches 3 weights. |
| `proxy/measure.py` | Removed `_fit_queue_rate()`. `recommend_rates` retained as diagnostic only. |
| `proxy/tuning.py` | Added `load_weight` to `HYPERPARAM_RANGES`. `rtt_weight` range widened to 50.0. |
| `experiment_runner/policy_matrix_app.py` | Removed `_calibrate_gorgo_fleet()`. No calibration phase. |

## Trace analysis tool

```bash
modal run --env=alessio-dev data_processing/build_mooncake_trace.py::analyze_trace_reuse \
  --trace-paths-csv "/data/mooncake_traces/abstract_night_traces/with_bodies/glm5_apr2_0030_to_0100.jsonl,/data/mooncake_traces/abstract_night_traces/with_bodies/glm5_apr2_0100_to_0130.jsonl,/data/mooncake_traces/abstract_night_traces/with_bodies/glm5_apr2_1230_to_1300.jsonl"
```

## Pareto frontier simulator

```bash
modal run --env=alessio-dev data_processing/build_mooncake_trace.py::simulate_pareto_sweep \
  --trace-path "/data/mooncake_traces/abstract_night_traces/with_bodies/glm5_apr2_0030_to_0100.jsonl" \
  --replicas-json '[{"region":"us-ashburn-1","rtt_ms":32,"prefill_rate":0.093},{"region":"eu-frankfurt-1","rtt_ms":364,"prefill_rate":0.073},{"region":"ap-seoul-1","rtt_ms":602,"prefill_rate":0.130}]' \
  --prefill-weights-csv "1.0" \
  --rtt-weights-csv "0.1,0.5,1.0,2.0,5.0,10.0" \
  --queue-weights-csv "0.001,0.005,0.01,0.02,0.05,0.1,0.2,0.5"
```
