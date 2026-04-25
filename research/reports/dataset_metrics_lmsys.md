# Dataset metrics: lmsys

> lmsys is the multi-turn, real-text chat dataset (substrate: lmsys/lmsys-chat-1m (lmsys-chat-1m)). n_requests=16,628 across n_sessions=9,741 (turns_per_session p95=5); prompts are heavy-tailed natural language (p50=17, p99=592 tokens), and prefix reuse is dominated by intra-session conversation history rather than a constant template — intra-session first-block reuse rate is 0.18, oracle hit rate climbs from 0.05 at 64 blocks to 0.09 at 16384 blocks. Top language is English (74% share).

## Source

- **kind**: `lmsys`
- **trace source**: `lmsys:data/lmsys/lmsys-chat.jsonl`
- **HF dataset**: `lmsys/lmsys-chat-1m` (canonical, gated)
- **loader config**:

  ```json
  {
    "max_conversations": 10000,
    "language_filter": "all-languages",
    "min_turns": 1,
    "max_turns": 16,
    "seed": 0
  }
  ```
- **loader params**:

  ```json
  {
    "arrival_rate_qps": 4.0,
    "max_output_tokens": 256,
    "tokenizer": "tiktoken:cl100k_base",
    "seed": 0
  }
  ```

## Volume

- Requests: **16,628**
- Sessions: **9,741**
- Trace duration: **4144.7 s**
- Empirical QPS: **4.01**

## Prompt / output length

| metric | prompt_tokens | output_tokens_budget |
|---|---:|---:|
| n | 16,628 | 16,628 |
| mean | 63.7 | 256.0 |
| std | 126.4 | 0.0 |
| min | 1.0 | 256.0 |
| p50 | 17.0 | 256.0 |
| p90 | 192.0 | 256.0 |
| p95 | 330.0 | 256.0 |
| p99 | 592.0 | 256.0 |
| max | 2,164.0 | 256.0 |

![prompt length histogram](figures/lmsys_prompt_length_hist.png)

## Interarrival / burstiness

- Mean IAT: **0.2493 s** (std 0.2503)
- CV² of IAT: **1.008** (≈1.0 → Poisson-like)
- Fano factor (1s windows): **1.019**
- Fano factor (10s windows): **0.999**
- Gini on interarrival gaps: **0.501**

![interarrival ccdf](figures/lmsys_interarrival_ccdf.png)

![qps over time](figures/lmsys_qps_timeseries.png)

## Prefix structure

- Block size: **16 tokens**
- Blocks per request: mean **3.5**, p50 1, p95 20
- Unique blocks: **52,293** (of 58,548 lookups)
- Block-reuse ratio: **0.107** (1 − unique/lookups)
- Unique first-blocks: **7,362**
- Top-10 first-blocks share: **0.085**
- First-block Zipf fit: s=**0.19**, R²=0.512
- All-block Zipf fit: s=**0.16**, R²=0.538

## Oracle cache hit-rate curve

Single unified LRU over blocks. Upper bound on what a prefix-aware
policy can achieve at that capacity; real multi-pod policies pay
partition overhead and will do strictly worse.

| capacity (blocks) | capacity (tokens) | hit rate |
|---:|---:|---:|
| 64 | 1,024 | 0.046 |
| 256 | 4,096 | 0.062 |
| 1,024 | 16,384 | 0.071 |
| 4,096 | 65,536 | 0.081 |
| 16,384 | 262,144 | 0.093 |

![cache hit curve](figures/lmsys_cache_hit_curve.png)

## Session / turn structure

- Turns per session: mean **1.7**, p50 1, p95 5, max 8
- Turns-per-session Gini: **0.332**
- Intra-session first-block reuse rate: **0.185**
- Prompt-length growth across turns (OLS slope, tokens/turn): mean **-7.02** over 1,637 sessions with ≥3 turns

## Language mix

- Tagged requests: **16,628**, unique languages: **74**

| lang | count | share |
|---|---:|---:|
| English | 12,371 | 0.744 |
| Portuguese | 679 | 0.041 |
| unknown | 642 | 0.039 |
| Russian | 544 | 0.033 |
| Spanish | 461 | 0.028 |
| German | 337 | 0.020 |
| Chinese | 327 | 0.020 |
| Italian | 284 | 0.017 |
| French | 239 | 0.014 |
| Japanese | 80 | 0.005 |

## Text statistics

- Natural language: **True**
- Empty prompts: **0**
- Degenerate prompts (<1 block): **7,819**
- Token-id sample (500 reqs): id range [0,100166], unique ids 7,352, mean 9919

## Reproduction

```bash
# Fetch chat data into data/lmsys/ (gitignored).
# HF_TOKEN required (lmsys/lmsys-chat-1m is gated; no fallback).
python scripts/fetch_lmsys_data.py --max-conversations 10000
python scripts/dataset_metrics.py --dataset lmsys
```
