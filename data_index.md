# GORGO Data Index

Where everything lives across the Modal volumes and local repo.

## Modal Volumes

### `GORGO-glm5-completions` (environment: `alessio-dev`)

Raw production data and derived traces.

#### Raw Data (private — do not redistribute)

```
/llm_responses_202604*.parquet     ← ClickHouse export, Apr 1-7 2026
                                     Columns: uuid, timestamp, request_metadata.token_hash,
                                     request (full JSON), response (full JSON)
                                     ~411k requests, ~4,984 users
```

#### Tokenized Cache

```
/tokenized_llm_responses_202604/   ← Per-file tokenized parquets
                                     Columns: token_hash, prompt_ids, per_message, ...
                                     Produced by build_eval_dataset.py::tokenize_main
```

#### Prefix Trie Stats

```
/prefix_tries_llm_responses_202604/   ← Pickled radix tries (global + per-user)
/prefix_trie_stats_llm_responses_202604.json  ← Aggregate reuse stats
                                                53.7% intra-user, 1.6% cross-user, 55.3% global
```

#### Metadata Traces (shareable — no message content)

Privacy-safe per-request routing metadata. Contains per-message roles and
token counts, prefix-aware block hashes, but zero text content.

```
/mooncake_traces/metadata/
  glm5_metadata_full_7day.jsonl      ← Full 7-day window (Apr 1-8, ~411k requests)
  glm5_metadata_apr2_0030_to_0100.jsonl  ← W1 tuning window (nighttime)
  glm5_metadata_apr2_0100_to_0130.jsonl  ← W2a eval window (nighttime)
  glm5_metadata_apr2_1230_to_1300.jsonl  ← W2b eval window (midday diurnal)
```

Per-row format:
```json
{
  "timestamp": 0,
  "token_hash": "abc123...",
  "input_length": 13200,
  "output_length": 77,
  "messages": [
    {"role": "system", "tokens": 482},
    {"role": "user", "tokens": 1200},
    {"role": "assistant", "tokens": 800}
  ],
  "hash_ids": [0, 1, 2, ...]
}
```

Produced by: `data_processing/export_metadata_trace.py`

#### Synthetic Traces (gibberish Unicode — runnable by proxy)

Derived from metadata traces. Request bodies contain `enc.decode(random_ids)`
that tokenize to exact token counts. Used for replay experiments.

```
/mooncake_traces/synthetic/
  glm5_synthetic_apr2_0030_to_0100.jsonl
  glm5_synthetic_apr2_0100_to_0130.jsonl
  glm5_synthetic_apr2_1230_to_1300.jsonl
```

Produced by: `data_processing/build_synthetic_trace.py`
**Note**: Current synthetic traces have broken request bodies (random hex,
not decoded token IDs). Need regeneration with fixed script.

#### Original Mooncake Traces (private — contain real token IDs)

Traces built from raw data with actual tokenized message content.
These are what the experiments in the paper ran on.

```
/mooncake_traces/abstract_night_traces/with_bodies/
  glm5_apr2_0030_to_0100.jsonl       ← W1 tuning (2,754 rows)
  glm5_apr2_0100_to_0130.jsonl       ← W2a eval temporal (3,262 rows)
  glm5_apr2_1230_to_1300.jsonl       ← W2b eval diurnal (4,323 rows)
  glm5_0030_to_0100.jsonl            ← Apr 1 night (reference)
```

Produced by: `data_processing/build_mooncake_trace.py`

---

### `GORGO-bench-results` (environment: `alessio-dev`)

Experiment results, workload runs, and proxy traces.

#### Policy Matrix Sweep Results

```
/policy_matrix_sweep/c64/
  glm5_c64_tuning_p95ttft_v6/       ← p95 tuning (load_weight=[0.1,10.0])
    glm5_c64_tuning_p95ttft_v6_tune/   W1 tuning results
    glm5_c64_tuning_p95ttft_v6_eval0/  W2a nighttime eval
    learned_weights.json                rtt=0.39, prefill=1.88, load=6.38

  glm5_c64_tuning_p95ttft_v6/       ← p95 eval1 (midday, standalone)
    glm5_c64_tuning_p95ttft_v6_eval1/  W2b midday diurnal eval

  glm5_c64_tuning_p95ttft_v8/       ← p95 with gorgo-autotune (7 policies)
    (tuning only — evals pending)

  glm5_c64_tuning_p50ttft_v1/       ← p50 tuning
    glm5_c64_tuning_p50ttft_v1_tune/   W1 tuning results
    glm5_c64_tuning_p50ttft_v1_eval0b/ W2a nighttime eval
    glm5_c64_tuning_p50ttft_v1_eval1/  W2b midday diurnal eval
    learned_weights.json                rtt=1.21, prefill=0.52, load=1.69

  glm5_c64_eval_p95ttft_diurnal_v2/  ← load_weight=0 ablation (E2E regression)
```

#### Organized Workload Runs (namespaced by experiment)

```
/workload_runs/
  glm5_c64_tuning_p95ttft_v6_tune/    ← v6 W1 per-policy stats
  glm5_c64_tuning_p95ttft_v6_eval0/   ← v6 W2a per-policy stats
  glm5_c64_tuning_p95ttft_v6_eval1/   ← v6 W2b per-policy stats
  glm5_c64_tuning_p50ttft_v1_tune/    ← p50 W1 per-policy stats
  glm5_c64_tuning_p95ttft_v8/         ← v8 (with autotune) W1 stats
```

#### Proxy Traces

```
/proxy_traces/
  glm5_c64_tuning_p95ttft_v6/         ← v6 per-policy request traces
    tune_gorgo-hillclimb-p95/
      requests.jsonl                    Per-request routing decisions
      metrics.jsonl                     RTT probes
      tune.jsonl                        ES convergence events
```

---

## Local Repo (`/Users/alessio/GORGO/`)

### Specs

```
specs/c64/
  tuning/
    policy_matrix_c64_tuning_p95ttft.json   ← 7 policies (incl. autotune)
    policy_matrix_c64_tuning_p50ttft.json   ← p50 objective
    policy_matrix_c64_tuning_p99ttft.json   ← p99 objective
  eval/
    policy_matrix_c64_eval_p95ttft.json     ← v6 learned weights
    policy_matrix_c64_eval_p50ttft.json     ← p50 learned weights
    policy_matrix_c64_eval_p99ttft.json     ← p99 (placeholder)
  manifests/
    manifest_glm5_apr2_0030_0100.json       ← W1 (nighttime tuning)
    manifest_glm5_apr2_0100_0130.json       ← W2a (nighttime eval)
    manifest_glm5_apr2_1230_1300.json       ← W2b (midday diurnal eval)
```

### Data Processing Scripts

```
data_processing/
  export_metadata_trace.py        ← Raw parquets → metadata traces (privacy-safe)
  build_synthetic_trace.py        ← Metadata → synthetic Mooncake traces
  build_mooncake_trace.py         ← Raw parquets → Mooncake traces (original)
  build_eval_dataset.py           ← Raw parquets → tokenized parquets
  build_prefix_trie.py            ← Tokenized parquets → reuse stats
  measure_system_prompt_reuse.py  ← System prompt sharing analysis
```

### Figure Generation Scripts

```
scripts/
  paper_style.py                  ← Canonical color palette
  plot_ttft_bars.py               ← TTFT horizontal bar charts
  plot_rtt_timeseries.py          ← RTT over time
  plot_paper_cache_and_concentration.py  ← Cache vs latency + concentration
  plot_tune_convergence.py        ← ES convergence 4-panel
  plot_dataset_comparison.py      ← Dataset characterization
```

### Experiment Runner

```
experiment_runner/
  policy_matrix_app.py            ← Main experiment controller
  sequencer.py                    ← Multi-phase orchestrator (tune → eval)
```

---

## Key Learned Weights

| Experiment | rtt_weight | prefill_weight | load_weight | Objective |
|---|---|---|---|---|
| v6 (paper results) | 0.392 | 1.880 | 6.382 | neg_p95_ttft |
| p50 v1 | 1.206 | 0.517 | 1.689 | neg_p50_ttft |
| v2 (load ablation) | 0.227 | 0.907 | 0.0 | neg_p95_ttft (unconstrained) |

## Privacy Pipeline

```
Raw parquets (private)
    → export_metadata_trace.py (tokenizes, hashes, discards content)
Metadata traces (shareable: token counts + hash_ids, no content)
    → build_decoded_trace.py (TODO: generates gibberish Unicode from random token IDs)
Decoded traces (runnable: Mooncake format with synthetic request bodies)
    → policy_matrix_app.py (experiments)
```
