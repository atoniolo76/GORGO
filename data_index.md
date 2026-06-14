# GORGO Data Index

Where everything lives across the Modal volumes and local repo.

## Experiment Catalog (↔ `paper.md`)

Maps each experiment section in `paper.md` to its run IDs, trace windows, storage location, and weights. `paper.md` is the source of truth for result numbers; this file is the source of truth for where the data lives. All windows are UTC on April 2nd unless noted.

| `paper.md` section | Internal run ID / dir | Trace type | Windows | Storage | Learned weights | Status |
|---|---|---|---|---|---|---|
| p95 TTFT Objective (v6) | `glm5_c64_tuning_p95ttft_v6` | real | W1 0030–0100, W2a 0100–0130, W2b 1230–1300 | volume `…/c64/glm5_c64_tuning_p95ttft_v6/` (+ `/workload_runs/`, `/proxy_traces/`) | rtt=0.392, prefill=1.880, load=6.382 | tune + eval0 + eval1 ✓ |
| p50 TTFT Objective (v1) | `glm5_c64_tuning_p50ttft_v1` | real | W1 0030–0100 (eval0b/eval1 dirs also exist) | volume `…/c64/glm5_c64_tuning_p50ttft_v1/` | rtt=1.206, prefill=0.517, load=1.689 | paper shows W1 only |
| p95 + gorgo-autotune (v8) | `glm5_c64_tuning_p95ttft_v8` | real | W1 0030–0100 | volume `…/c64/glm5_c64_tuning_p95ttft_v8/` | fixed 1/1/1; fits per-replica `prefill_rate` | tuning only, evals pending |
| Decoded Synthetic (decoded_v9) | `glm5_c64_{tuning,eval}_p95ttft_000_glm5_decoded_apr2_*` | decoded synthetic | W1 0030–0100, W2a 0100–0130, W2b 1230–1300 | local `results/decoded_v9/` (`w1_tuning.json`, `w2a_eval0.json`, `w2b_eval1.json`) | rtt=1.119, prefill=1.106, load=1.446 | tune + W2a + W2b ✓ |
| 2D GORGO (decoded_v4 tuning) | `glm5_c64_tuning_p95ttft_2d_v4` (`…_2d_000_glm5_decoded_apr2_0030_to_0100`) | decoded synthetic | W1 0030–0100 | volume `…/c64/glm5_c64_tuning_p95ttft_2d_v4/` + local `results/2d_v4/` | rtt=0.5, queue=0.1 (2D; ES did not improve) | tuning only |

Present in the repo but **not yet written up in `paper.md`**:
- `glm5_c64_tuning_p95ttft_2d_v8_tune` — 2D model, decoded **Apr 5 16:15–16:45** (`…_2d_000_glm5_decoded_apr5_1615_to_1645`). ES converged to `rtt=5.0, queue=0.01`. Local: `results/2d_v8_tune/`. Likely the "next run" referenced at the end of the decoded_v4 section.
- `glm5_c64_eval_p95ttft_diurnal_v2` — `load_weight=0` ablation (rtt=0.227, prefill=0.907, load=0); source of the catastrophic E2E regression (12.58s) cited in the v6 narrative.

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

| Experiment | rtt_weight | prefill_weight | load_weight | queue_weight | Objective | Traces |
|---|---|---|---|---|---|---|
| v6 (paper results) | 0.392 | 1.880 | 6.382 | — | neg_p95_ttft | real |
| p50 v1 | 1.206 | 0.517 | 1.689 | — | neg_p50_ttft | real |
| v8 (autotune) | 1.0 (fixed) | 1.0 (fixed) | 1.0 (fixed) | — | neg_p95_ttft; fits `prefill_rate` | real |
| v2 (load ablation) | 0.227 | 0.907 | 0.0 | — | neg_p95_ttft (unconstrained) | real |
| decoded_v9 | 1.119 | 1.106 | 1.446 | — | neg_p95_ttft | decoded synthetic |
| 2D decoded_v4 | 0.5 | 1.0 (anchor) | — | 0.1 | neg_p95_ttft (ES no improvement) | decoded synthetic |
| 2D v8_tune (not in paper) | 5.0 | 1.0 (anchor) | — | 0.01 | neg_p95_ttft | decoded synthetic |

## Decoded Trace Experiments (privacy-safe synthetic data)

### decoded_v9: p95 TTFT on synthetic traces (tiktoken metadata + Qwen decoded text)

> Catalog ↔ `paper.md` → **Experiment: Decoded Synthetic Traces (decoded_v9)**. Tables below mirror the paper; `paper.md` is authoritative for numbers.

Learned weights: `rtt_weight=1.119, prefill_weight=1.106, load_weight=1.446`

W1 Tuning:
| Policy | TTFT p50 | TTFT p95 | TTFT p99 | E2E p95 |
|---|---|---|---|---|
| **gorgo-autotune** | **121ms** | **651ms** | 1,434ms | 2.25s |
| **gorgo-hillclimb** | 135ms | 711ms | 1,486ms | 2.25s |
| least-request | 194ms | 828ms | 1,448ms | **1.92s** |
| prefix-cache | 245ms | 883ms | 1,534ms | 2.35s |
| random | 248ms | 927ms | 1,532ms | 2.11s |
| least-load | 281ms | 927ms | 1,651ms | 2.24s |
| session-affinity | 302ms | 1,137ms | 2,035ms | 2.89s |

W2a Eval (nighttime):
| Policy | TTFT p50 | TTFT p95 | TTFT p99 | E2E p95 |
|---|---|---|---|---|
| **gorgo-autotune** | **120ms** | 693ms | 1,456ms | 2.26s |
| **gorgo-static** | 160ms | **675ms** | 1,581ms | **2.13s** |
| prefix-cache | 210ms | 936ms | 1,629ms | 2.29s |
| least-request | 278ms | 877ms | 1,590ms | 1.99s |
| random | 281ms | 966ms | 1,769ms | 2.14s |
| least-load | 288ms | 1,036ms | 1,874ms | 2.39s |
| session-affinity | 304ms | 1,320ms | 2,197ms | 3.10s |

gorgo-static wins TTFT p95 (675ms, 23% ahead of least-request 877ms) and E2E p95 (2.13s).
Rankings match real trace results: GORGO dominates, session-affinity last.

W2b Eval (midday): complete — see `paper.md` (decoded_v9 → W2b). gorgo-static wins p50/p95 TTFT but the margin collapses to noise (1,124ms vs random 1,105ms) because the window is dominated by a single 18-token bot user (78% of traffic). Stored in `results/decoded_v9/w2b_eval1.json`.

Results saved locally: `results/decoded_v9/` (`w1_tuning.json`, `w2a_eval0.json`, `w2b_eval1.json`)

## Privacy Pipeline

```
Raw parquets (private)
    → export_metadata_trace.py (tiktoken tokenizer — matches original trace builder)
Metadata traces (shareable: per-message token counts + hash_ids + system_prompt_hash, no content)
    → build_decoded_trace.py (Qwen tokenizer — "word\nword" text matching proxy/engine)
Decoded traces (runnable: Mooncake format with synthetic request bodies)
    → policy_matrix_app.py (experiments)
```

Key design decisions:
- **Metadata uses tiktoken**: matches `build_mooncake_trace.py` so same requests are selected
- **Decoded uses Qwen word pool**: proxy + engine tokenize with Qwen, so words must be 1 Qwen-token each
- **Intra-user prefix reuse**: preserved via per-user `user_token_strings` (same text across turns)
- **Cross-user system prompt**: preserved via `system_prompt_hash` (same text for matching hashes)
- **Round-trip fidelity**: "word\nword\n..." pattern gives 100% token count accuracy under Qwen

## 2D GORGO Results

### `glm5_c64_tuning_p95ttft_2d_v4` (decoded traces, tuning only)

> Catalog ↔ `paper.md` → **Experiment: 2D GORGO Cost Model (decoded_v4 tuning)**.
> Window: W1 tuning, decoded **Apr 2 00:30–01:00 UTC** (`glm5_decoded_apr2_0030_to_0100`); run ID `glm5_c64_tuning_p95ttft_2d_000_glm5_decoded_apr2_0030_to_0100`; concurrency 64; n=2,095/policy.

Location on volume:

```
GORGO-bench-results:/policy_matrix_sweep/c64/glm5_c64_tuning_p95ttft_2d_v4/
```

Local copies:

```
results/2d_v4/w1_tuning.json
results/2d_v4/learned_weights.json
```

Active 2D cost:

```text
score(u) = rtt_weight * rtt_ms(u) + uncached_tokens(u) + queue_weight * queued_uncached_tokens(u)
```

Initial/active weights:

```json
{"rtt_weight": 0.5, "queue_weight": 0.1}
```

W1 tuning result: `gorgo-hillclimb-p95-2d` achieved 122ms TTFT p50 and 567ms TTFT p95, beating all baselines on TTFT.

### `glm5_c64_tuning_p95ttft_2d_v9` (decoded traces, raw-queued load term)

First run with the fixed 2D load term (raw `queued_tokens` instead of `queued_uncached_tokens`) and tightened search ranges (`rtt_weight [0.05,2.0]`, `queue_weight [0.05,0.5]`).

Location on volume:

```
GORGO-bench-results:/policy_matrix_sweep/c64/glm5_c64_tuning_p95ttft_2d_v9/
  glm5_c64_tuning_p95ttft_2d_v9_tune/   ← tune (apr5 16:15–16:45) per-policy stats
  learned_weights.json                  ← rtt=0.276, queue=0.500 (queue pinned at ceiling)
```

Local copies:

```
results/2d_v9_tune/   (per-policy stats + learned_weights.json)
```

Active 2D cost (raw queued load term):

```text
score(u) = rtt_weight * rtt_ms(u) + uncached_tokens(u) + queue_weight * queued_tokens(u)
```

Tune window result (apr5, n=7,195, in-sample): `gorgo-hillclimb-p95-2d` wins TTFT p50 (673ms) and E2E p95 (8.91s, 35% ahead of least-load), competitive on TTFT p95 (2,514ms). Dramatic turnaround from the prior `queued_uncached` model, which collapsed to worst-of-5 on this same window (10,055ms p95, 30.2s E2E).

**Status:** tune phase only. Eval windows (apr6 15:05–15:35, apr7 19:45–20:15) launched but cancelled mid-eval0 by user; resume off saved weights with `--skip-tuning`.

### `glm5_c64_tuning_p95ttft_2d_v8_tune` (decoded traces, NOT yet in paper)

> Not yet written up in `paper.md`. This is the 2D follow-up tuning run on a high-diversity daytime window.
> Window: decoded **Apr 5 16:15–16:45 UTC** (`glm5_decoded_apr5_1615_to_1645`); run ID `glm5_c64_tuning_p95ttft_2d_000_glm5_decoded_apr5_1615_to_1645`; concurrency 64; ~7,221 routed requests (gorgo trace).

Local copies:

```
results/2d_v8_tune/glm5_c64_tuning_p95ttft_2d_000_glm5_decoded_apr5_1615_to_1645_{gorgo-hillclimb-p95-2d,least-load,least-request,prefix-cache,simple-session-affinity}.json
results/2d_v8_tune/learned_weights.json
results/2d_v8_tune/proxy_trace_gorgo/{manifest,metrics,requests,tune}.jsonl
```

ES-converged weights (note: ES moved off the init this time):

```json
{"rtt_weight": 5.0, "queue_weight": 0.01}
```
