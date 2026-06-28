# Results

// These first two experiments hasve hte old cost model where there were three weights.

## Experiment: p95 TTFT Objective (v6)

Learned weights: `rtt_weight=0.392, prefill_weight=1.880, load_weight=6.382`
Hyperparameter ranges: `prefill_weight=[1e-5, 5.0], load_weight=[0.1, 10.0], rtt_weight=[1e-5, 50.0]`

### W1: Tuning Window (nighttime, 00:30–01:00 UTC, April 2nd)

> $n=2{,}095$ requests per policy; 100% success rate. gorgo-hillclimb actively explores via (1+1)-ES.

| Policy | TTFT p50 | TTFT p95 | TTFT p99 | E2E p50 | E2E p95 | E2E p99 | ITL avg | Decode tok/s | SR |
|--|--|--|--|-|-|-|-|---|--|
| **gorgo-hillclimb-p95** | **180ms** | **1,010ms** | 2,660ms | 1.42s | **2.14s** | **2.95s** | 10.1ms | 105 | 100% |
| session-affinity | 194ms | 1,185ms | 6,757ms | 1.48s | 2.98s | 3.47s | 10.1ms | 106 | 100% |
| least-request | 197ms | 1,179ms | 2,062ms | 1.24s | 2.24s | 3.07s | 8.1ms | 125 | 100% |
| least-load | 271ms | 2,158ms | 8,750ms | 1.40s | 3.30s | 2.92s | 9.1ms | 116 | 100% |
| prefix-cache | 283ms | 1,305ms | 2,210ms | 1.31s | 2.74s | 3.32s | 9.6ms | 110 | 100% |
| random | 292ms | 1,192ms | 1,996ms | 1.27s | 2.38s | 3.24s | 8.5ms | 113 | 100% |

> **gorgo advantage (p95 TTFT): 14.3%** — gorgo-hillclimb-p95 1,010ms vs least-request 1,179ms _(in-sample tuning window)_

gorgo-hillclimb wins TTFT p50 (180ms, 7.2% gap over session-affinity at 194ms), TTFT p95 (1,010ms, 14.3% gap over least-request at 1,179ms), E2E p50 (tied with least-request), and E2E p95 (2.14s, 4.5% gap over least-request at 2.24s). The ES converged to prefill_weight=1.88 (cache-first), load_weight=6.38 (strong load avoidance), and rtt_weight=0.39 (moderate RTT preference).

### W2 Eval0: Nighttime Temporal (01:00–01:30 UTC, April 2nd)

> $n=2{,}207$ requests per policy; 100% success rate. gorgo-static deploys frozen W1-learned weights.

| Policy | TTFT p50 | TTFT p95 | TTFT p99 | E2E p50 | E2E p95 | E2E p99 | ITL avg | Decode tok/s | SR |
|--|--|--|--|-|-|-|-|---|--|
| **gorgo-static-p95** | **186ms** | **896ms** | 1,844ms | **1.23s** | **1.99s** | **2.89s** | 8.5ms | 121 | 100% |
| prefix-cache | 190ms | 1,041ms | 2,011ms | 1.35s | 2.54s | 3.16s | 9.4ms | 113 | 100% |
| session-affinity | 201ms | 981ms | 1,785ms | 1.54s | 2.64s | 3.47s | 10.6ms | 102 | 100% |
| least-request | 260ms | 1,131ms | 1,758ms | 1.23s | 2.16s | 2.90s | 8.1ms | 127 | 100% |
| least-load | 276ms | 1,144ms | 1,808ms | 1.42s | 2.43s | 3.28s | 9.5ms | 112 | 100% |
| random | 292ms | 1,280ms | 2,481ms | 1.30s | 2.49s | 4.38s | 8.5ms | 123 | 100% |

> **gorgo advantage (p95 TTFT): 8.7%** — gorgo-static-p95 896ms vs session-affinity 981ms _(held-out eval)_

gorgo-static sweeps TTFT p50 (186ms, 2.1% gap), p95 (896ms, 8.7% gap over session-affinity at 981ms), E2E p50 (1.23s, tied with least-request), E2E p95 (1.99s, 7.9% gap over least-request at 2.16s), and E2E p99 (2.89s). The frozen W1 weights generalize to the held-out nighttime trace with improved margins — no exploration tax from ES proposals.

### W2 Eval1: Midday Diurnal (12:30–13:00 UTC, April 2nd)

> $n=3{,}071$ requests per policy; 100% success rate. Different time-of-day from W1 (nighttime). Heavier traffic: 47% more requests.

| Policy | TTFT p50 | TTFT p95 | TTFT p99 | E2E p50 | E2E p95 | E2E p99 | ITL avg | Decode tok/s | SR |
|--|--|--|--|-|-|-|-|---|--|
| **gorgo-static-p95** | **195ms** | **1,136ms** | **2,060ms** | **1.29s** | **2.46s** | **3.24s** | **8.8ms** | **119** | 100% |
| session-affinity | 234ms | 1,519ms | 2,539ms | 1.69s | 3.78s | 6.70s | 12.1ms | 96 | 100% |
| least-request | 294ms | 1,297ms | 2,064ms | 1.36s | 2.60s | 3.48s | 9.0ms | 116 | 100% |
| prefix-cache | 280ms | 1,558ms | 2,491ms | 1.55s | 3.59s | 7.14s | 11.1ms | 103 | 100% |
| random | 292ms | 1,518ms | 2,457ms | 1.37s | 3.01s | 4.53s | 9.3ms | 115 | 100% |
| least-load | 297ms | 2,030ms | 3,765ms | 1.73s | 4.00s | 5.99s | 11.7ms | 96 | 100% |

> **gorgo advantage (p95 TTFT): 12.4%** — gorgo-static-p95 1,136ms vs least-request 1,297ms _(held-out eval)_

gorgo-static sweeps all metrics: TTFT p50 (195ms, 16.7% gap over session-affinity), p95 (1,136ms, 12.4% gap over least-request at 1,297ms), p99 (2,060ms), E2E p50 (1.29s), E2E p95 (2.46s, 5.4% gap), E2E p99 (3.24s), ITL (8.8ms), and decode throughput (119 tok/s). The load_weight=6.38 prevents the single-replica concentration that caused the catastrophic E2E regression in earlier experiments (12.58s with load_weight=0).

### Diff-in-diff: gorgo advantage grows under stress

| | gorgo p95 | session-affinity p95 | gorgo advantage |
|---|---|---|---|
| Eval0 (nighttime) | 896ms | 981ms | 8.7% |
| Eval1 (midday) | 1,136ms | 1,519ms | **25.2%** |
| Degradation night→midday | +27% | +55% | gorgo resists 2× better |

---

## Experiment: p50 TTFT Objective (v1)

Learned weights: `rtt_weight=1.206, prefill_weight=0.517, load_weight=1.689`

### W1: Tuning Window (nighttime, 00:30–01:00 UTC, April 2nd)

> Same trace as p95 tuning. $n=2{,}095$ requests per policy; 100% success rate.

| Policy | TTFT p50 | TTFT p95 | TTFT p99 | E2E p50 | E2E p95 | E2E p99 |
|--|--|--|--|-|-|-|
| **gorgo-hillclimb-p50** | **163ms** | **943ms** | 2,185ms | 1.45s | 2.40s | 3.95s |
| session-affinity | 196ms | 988ms | 1,654ms | 1.48s | 2.61s | 3.47s |
| least-request | 207ms | 1,102ms | 1,934ms | 1.24s | 2.10s | 3.07s |
| least-load | 264ms | 1,079ms | 1,874ms | 1.40s | 2.38s | 2.92s |
| random | 274ms | 1,186ms | 2,021ms | 1.27s | 2.33s | 3.24s |
| prefix-cache | 287ms | 1,052ms | 1,773ms | 1.31s | 2.34s | 3.32s |

> **gorgo advantage (p95 TTFT): 4.6%** — gorgo-hillclimb-p50 943ms vs session-affinity 988ms _(in-sample tuning window; p50-objective run)_

gorgo-hillclimb-p50 wins TTFT p50 (163ms, 16.8% gap over session-affinity at 196ms) and TTFT p95 (943ms, 4.6% gap over session-affinity at 988ms).

### Weight comparison: p95 vs p50 objective

| Weight | p95 objective | p50 objective | Interpretation |
|---|---|---|---|
| `rtt_weight` | 0.392 | **1.206** (3.1×) | p50 prioritizes RTT — typical requests benefit most from proximity |
| `prefill_weight` | 1.880 | **0.517** (0.3×) | p50 cares less about cache — median requests aren't the long-tail uncached ones |
| `load_weight` | 6.382 | **1.689** (0.3×) | p50 needs less load balancing — median requests don't see queueing pressure |

The ES discovers fundamentally different operating points depending on which percentile it optimizes. p95 optimization drives the policy toward cache-first routing with aggressive load avoidance (because tail requests are the ones stuck behind queues with cold caches). p50 optimization drives toward RTT-first routing (because the typical request is short enough that network latency dominates over cache effects).

---

## Design Iteration: Physical Rate + Adjustable Weight (hillclimb, May 29)

> **Not in the final paper results.** Recorded here as a design iteration that was tried and deliberately removed. Source: `specs/c64/EXPERIMENT_INDEX.md` at commit `1a7a7a8`, experiments `glm5_c64_tuning_p95ttft_v1` and `glm5_c64_eval_p95ttft_temporal_v1`.

### What was tried

After the 3-weight model (v6), a further experiment split the cost function into two *families* of parameters running simultaneously on the same prefill term:

```text
score(u) = rtt_weight × rtt_ms(u)
         + prefill_weight × prefill_rate(u) × (uncached_tokens(u) + queued_tokens(u))
```

- **`prefill_rate`** (ms/token) — physical hardware constant, calibrated per-replica by `proxy/calibrate.py` or the `fit` auto-tuner; stored in `per_target` since different GPUs prefill at different speeds. Typical measured value: 0.086–0.106 ms/tok.
- **`rtt_weight`, `prefill_weight`** — dimensionless ES-tuned amplification weights, stored in `defaults`. At `1.0` each, the score is a physically grounded time estimate in ms. The ES searched both.

The hyperparameter store explicitly separated the two families:

```json
{
  "defaults":   {"rtt_weight": ..., "prefill_weight": ...},
  "per_target": {"<replica_url>": {"prefill_rate": ...}}
}
```

### Results (Apr 2, nighttime, c=64)

ES converged at step 21. Learned: `prefill_weight=1.65, rtt_weight=5.0`. Per-replica calibrated rates: Ashburn 0.106 ms/tok, Frankfurt 0.086 ms/tok.

| Policy | TTFT p50 | TTFT p95 | E2E p95 | Routing |
|---|---|---|---|---|
| gorgo-static (frozen) | **141ms** | **956ms** | 2.88s | Ashburn=**100%**, Frankfurt=0%, Seoul=0% |
| session-affinity | 214ms | 1,047ms | 2.71s | — |
| least-request | 258ms | 1,377ms | 2.56s | — |

### Why it was removed

The `prefill_weight × prefill_rate` product is **not identifiable** — only the product affects routing, so the ES has a flat direction in which it can increase `prefill_weight` and decrease `prefill_rate` (or vice versa) without changing the score. More importantly, the ES reached the same degenerate corner as the 3-weight model: `rtt_weight → 5.0` (its ceiling), `prefill_weight=1.65` — effectively routing 100% of traffic to the closest replica (Ashburn) and zero to the other two.

The fundamental problem is the same as in all single-objective (TTFT-only) runs: the optimizer maximizes RTT emphasis and zeroes out load balancing whenever load is unconstrained. The physical rate adds calibration complexity without fixing this.

Commit `1a7a7a8` (May 30) reverted to a pure weight model and removed the calibration infrastructure, with the explicit note:

> *"Single-objective optimization always degenerates — whether the cost function has 2 or 3 terms, the ES always maximizes RTT emphasis and zeroes out load balancing when optimizing TTFT alone."*

The Pareto analysis from this era (in `EXPERIMENT_INDEX.md`) shows that the best operating point found was **Config B** (`rtt_w=1084, load_w=0.009`) which achieved TTFT p95 775ms with only 14% E2E penalty vs least-request — comparable to Config C's TTFT p95 762ms but with 36% E2E penalty. The physical-rate model found only Config C-style degenerate points.

### Concentration audit (all ~100% runs, all metrics)

Server-side scan over the bench-results volume found four ~100%-concentration
runs in this Apr 2 family (3 real + 1 smoke test). The smoke test had one
policy only, so no next-best comparison. For the 3 real runs, the table below
reports gorgo's percent improvement vs next-best policy on each metric:

| Run (100% concentration) | Policy | TTFT p50 | TTFT p95 | TTFT p99 | E2E p50 | E2E p95 | E2E p99 | ITL avg |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Apr 2 00:30-01:00 eval (5 baselines) | `gorgo-static` | **+36.7%** | +16.9% | +3.7% | -24.3% | -35.9% | -37.9% | -47.3% |
| Apr 2 00:30-01:00 eval (3 baselines) | `gorgo-static` | +34.0% | +31.5% | +33.0% | n/a | -10.9% | -63.6% | n/a |
| Apr 2 01:00-01:30 eval | `gorgo-static-p95` | +34.1% | +8.7% | -2.7% | n/a | -14.4% | -44.5% | -35.0% |
| Apr 2 12:30-13:00 eval | `gorgo-static-p95` | +31.6% | +17.0% | -12.4% | -50.6% | -380.3% | -388.0% | -150.8% |

The standout p50 result is Apr 2 00:30-01:00 (5-baseline matrix): TTFT p50
**125ms** (+36.7% vs next-best). It is also the clearest reward-hacking pattern:
large TTFT gains paired with broad regressions on E2E and ITL under single-replica concentration.

---

## Experiment: p95 TTFT with gorgo-autotune (v8)

Same setup as v6 but with gorgo-autotune added as a 7th policy. gorgo-autotune uses `mode: "fit"` — it keeps weights at 1.0 and instead fits a physical `prefill_rate` per replica from observed TTFT residuals every 16 samples.

### W1: Tuning Window (nighttime, 00:30–01:00 UTC, April 2nd)

> $n=2{,}095$ requests per policy; 100% success rate. 7 policies including gorgo-autotune.

| Policy | TTFT p50 | TTFT p95 | TTFT p99 | E2E p95 |
|--|--|--|--|--|
| **gorgo-autotune** | **123ms** | **776ms** | 1,800ms | 2.25s |
| gorgo-hillclimb-p95 | 145ms | 833ms | 1,722ms | 2.84s |
| session-affinity | 197ms | 970ms | 1,748ms | 2.66s |
| least-request | 210ms | 1,084ms | 2,055ms | 2.19s |
| least-load | 244ms | 1,194ms | 2,162ms | 2.40s |
| random | 285ms | 1,159ms | 1,972ms | 2.32s |
| prefix-cache | 287ms | 1,164ms | 2,268ms | 2.66s |

> **gorgo advantage (p95 TTFT): 20.0%** — gorgo-autotune 776ms vs session-affinity 970ms _(in-sample tuning window)_

gorgo-autotune beats gorgo-hillclimb on TTFT p50 (123ms vs 145ms, 15% better) and p95 (776ms vs 833ms, 7% better), with competitive E2E (2.25s vs 2.84s).

### gorgo-autotune: fitted parameters

gorgo-autotune keeps `rtt_weight=1.0, prefill_weight=1.0, load_weight=1.0` fixed and fits `prefill_rate` per replica:

| Replica | Fitted `prefill_rate` (ms/tok) | Notes |
|---|---|---|
| Replica 1 (Ashburn) | 0.107 | Sane — typical L40S idle prefill rate |
| Replica 2 (Seoul) | 77.17 | **Blown up** — rate inversion when uncached tokens ≈ 0 |
| Replica 3 (Frankfurt) | 0.134 | Sane |

The inflated rate on replica 2 (Seoul) effectively blacklists it — the cost model gives it an astronomical score. This accidentally produces good routing: traffic avoids the farthest replica entirely. But this is fragile — if the rate blows up on the *closest* replica, autotune would route everything far away and tank TTFT. The ES avoids this instability because it never inverts the TTFT equation.

### Note: why autotune looks good here

gorgo-autotune's strong W1 result is likely an artifact of which replica's rate blew up. Seoul (highest RTT, farthest) getting blacklisted is the best-case accident — it removes the worst replica from the pool. The eval windows will test whether this holds on different traffic.

---

## Experiment: Decoded Synthetic Traces (decoded_v9)

Privacy-safe synthetic traces derived from production metadata. Uses tiktoken for metadata extraction (matching original trace filtering) and Qwen tokenizer for decoded text generation (matching proxy/engine). Intra-user prefix reuse (52.6%) and cross-user system prompt reuse preserved.

### Per-turn / per-conversation structure of the v9 windows

The decoded_v9 evaluation windows are heavily multi-turn — the property that makes intra-user prefix reuse the dominant cache signal. Per-conversation turn statistics (`data_processing/window_turn_stats.py`, source `results/trace_summaries/glm5_window_stats.csv`):

| Window | conversations | avg turns/conv | median turns | p90 turns | max turns | multi-turn conv % | avg prior turns/request | avg conv length (tok) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| apr5 16:15–16:45 (tuning) | 1,310 | 30.3 | 10 | 91 | 279 | 77.6% | 18.8 | 19,676 |
| apr6 15:05–15:35 (eval) | 1,501 | 27.7 | 8 | 93 | 271 | 73.5% | 16.3 | 20,016 |
| apr7 19:45–20:15 (eval) | 1,155 | 21.5 | 5 | 63 | 320 | 66.6% | 20.4 | 12,408 |

Sessions average **~22–30 turns** (median 5–10) with long tails (max 270–320 turns), and 67–78% of conversations are multi-turn. Each request therefore carries ~16–20 prior turns of reusable context on average — directly motivating the cache-aware prefill term `T_prefill(x_r \ c_i)` in the GORGO cost model.

**Note on token-count basis:** the `avg_tokens` reported in `glm5_window_stats.csv` (apr5 ~20.5k, apr6 ~23.6k, apr7 ~18.2k) characterizes the window's *metadata* (full conversation prompts). The decoded replay files actually benchmarked (`mooncake_traces/decoded/glm5_decoded_apr{5,6,7}_*.jsonl`) average **~7,019 input tokens/request** over **24,069 requests / 1,219 users**, with **~82% block-level prefix reuse** (256-token `hash_ids` blocks: apr5 78.8%, apr6 77.6%, apr7 87.9% global). Block-level reuse is the signal the proxy/engine cache keys on and is not directly comparable to the token-level radix-trie reuse reported for the public datasets in Table 1.

Learned weights: `rtt_weight=1.119, prefill_weight=1.106, load_weight=1.446`

### W1: Tuning Window (nighttime, 00:30–01:00 UTC, April 2nd)

| Policy | TTFT p50 | TTFT p95 | TTFT p99 | E2E p95 |
|---|---|---|---|---|
| **gorgo-autotune** | **121ms** | **651ms** | 1,434ms | 2.25s |
| **gorgo-hillclimb** | 135ms | 711ms | 1,486ms | 2.25s |
| least-request | 194ms | 828ms | 1,448ms | **1.92s** |
| prefix-cache | 245ms | 883ms | 1,534ms | 2.35s |
| random | 248ms | 927ms | 1,532ms | 2.11s |
| least-load | 281ms | 927ms | 1,651ms | 2.24s |
| session-affinity | 302ms | 1,137ms | 2,035ms | 2.89s |

> **gorgo advantage (p95 TTFT): 21.4%** — gorgo-autotune 651ms vs least-request 828ms _(in-sample tuning window)_

### W2a: Nighttime Eval (01:00–01:30 UTC, April 2nd)

| Policy | TTFT p50 | TTFT p95 | TTFT p99 | E2E p95 |
|---|---|---|---|---|
| **gorgo-autotune** | **120ms** | 693ms | 1,456ms | 2.26s |
| **gorgo-static** | 160ms | **675ms** | 1,581ms | **2.13s** |
| prefix-cache | 210ms | 936ms | 1,629ms | 2.29s |
| least-request | 278ms | 877ms | 1,590ms | 1.99s |
| random | 281ms | 966ms | 1,769ms | 2.14s |
| least-load | 288ms | 1,036ms | 1,874ms | 2.39s |
| session-affinity | 304ms | 1,320ms | 2,197ms | 3.10s |

> **gorgo advantage (p95 TTFT): 23.0%** — gorgo-static 675ms vs least-request 877ms _(held-out eval)_

gorgo-static wins TTFT p95 (675ms, 23% ahead of least-request) and E2E p95 (2.13s). Rankings match real trace results.

### W2b: Midday Eval (12:30–13:00 UTC, April 2nd)

| Policy | TTFT p50 | TTFT p95 | TTFT p99 | E2E p95 |
|---|---|---|---|---|
| **gorgo-autotune** | **210ms** | 1,135ms | 2,813ms | 3.06s |
| **gorgo-static** | **210ms** | **1,124ms** | **2,117ms** | 3.07s |
| random | 287ms | 1,105ms | 2,215ms | **2.65s** |
| least-load | 291ms | 1,762ms | 3,577ms | 3.78s |
| least-request | 295ms | 1,252ms | 2,497ms | 2.64s |
| prefix-cache | 296ms | 1,431ms | 2,489ms | 3.24s |
| session-affinity | 364ms | 2,355ms | 4,844ms | 4.41s |

> **gorgo advantage (p95 TTFT): −1.7% (gorgo loses)** — gorgo-static 1,124ms vs random 1,105ms _(held-out eval; degenerate single-bot workload, see note below)_

### Note: W2b midday window composition issue

The W2b midday window has a fundamentally different workload composition that limits GORGO's advantage:

| Property | W1 (tuning) | W2b (midday) |
|---|---|---|
| Requests | 2,757 | 3,076 |
| Users | 180 | 121 |
| Avg input tokens | ~3,400 | **1,057** |
| Median input tokens | — | **18** |
| Multi-turn (>2 msgs) | — | **4.1%** |
| Single-turn | — | **95.9%** |
| Top user | — | **2,393 reqs at 18 tokens (78% of traffic)** |

The midday window is dominated by a single user sending 2,393 tiny 18-token requests — effectively a health-check or monitoring bot consuming 78% of all traffic. With 18-token prompts, there is no meaningful prefix to cache, so GORGO's cache-aware routing provides no benefit over random. gorgo-static still wins p50 and p95 TTFT but the margin is within noise of random on p95 (1,124ms vs 1,105ms).

The original real traces used `token_hash_filter_top20` filtering in `build_mooncake_trace.py` which selected multi-turn heavy users and excluded this type of lightweight traffic. The decoded metadata traces do not apply this filter, resulting in a workload regime closer to WildChat/LMSYS (short, low-reuse) than the production long-context setting GORGO targets.

Future work: apply equivalent user/length filtering to the metadata traces before generating decoded traces, or select different evaluation windows with more representative multi-turn traffic.

---

## HERE"S START OF IMPORTANT STUFF Planned Experiment: 2D GORGO Cost Model

The next experiment removes the redundant `prefill_weight` parameter and fixes own-prefill cost as the unit of the score:

```text
score(u) = rtt_weight * rtt_ms(u)
         + uncached_tokens(u)
         + queue_weight * queued_uncached_tokens(u)
```

This keeps prefill consequential — `uncached_tokens` remains the anchor term — but removes the arbitrary global scale factor. In the 3-weight model,

```text
score = a * rtt + b * uncached + c * queued
```

only the ratios `a / b` and `c / b` affect routing. Setting `b = 1` makes the search identifiable and reduces one flat direction in the ES landscape.

The load term also becomes cache-aware: instead of using raw `queued_tokens`, the 2D model uses `queued_uncached_tokens`, the number of cache-miss tokens already dispatched to a replica. This avoids over-penalizing a cache-warm replica whose queued requests mostly hit KV cache.

Initial values:

```json
{
  "rtt_weight": 0.5,
  "queue_weight": 0.1
}
```

Search ranges:

```json
{
  "rtt_weight": [0.05, 5.0],
  "queue_weight": [0.01, 2.0]
}
```

Run plan:
- Tune on decoded Apr 2 00:30–01:00 (W1)
- Eval on decoded Apr 6 15:05–15:35 (high-diversity window)
- Eval on decoded Apr 7 19:45–20:15 (high-diversity window)

---

## Experiment: 2D GORGO Cost Model (decoded_v4 tuning)

> **Window:** W1 tuning — decoded nighttime, 00:30–01:00 UTC, April 2nd (`glm5_decoded_apr2_0030_to_0100`). $n=2{,}095$ requests per policy; concurrency 64; 100% success rate.

Cost model:

```text
score(u) = rtt_weight * rtt_ms(u)
         + uncached_tokens(u)
         + queue_weight * queued_uncached_tokens(u)
```

Initial weights:

```json
{"rtt_weight": 0.5, "queue_weight": 0.1}
```

The ES did not improve on the initial point in this run, but the initial physics-informed policy performed very strongly on the decoded W1 tuning window:

| Policy | TTFT p50 | TTFT p95 | TTFT p99 | E2E p95 |
|---|---:|---:|---:|---:|
| **gorgo-hillclimb-p95-2d** | **122ms** | **567ms** | **1,335ms** | 2.40s |
| least-request | 262ms | 873ms | 2,278ms | **1.97s** |
| least-load | 206ms | 954ms | 1,806ms | 2.34s |
| prefix-cache | 288ms | 962ms | 1,745ms | 2.52s |
| simple-session-affinity | 301ms | 1,405ms | 3,813ms | 3.05s |

> **gorgo advantage (p95 TTFT): 35.0%** — gorgo-hillclimb-p95-2d 567ms vs least-request 873ms _(in-sample tuning window; largest margin in paper, but ES did not improve on the physics-informed init)_

The result suggests that anchoring the cost function on own uncached prefill work and using a lightly discounted queued-uncached term is a strong prior. Next run will use the more defensible initialization `rtt_weight=1.0, queue_weight=0.1` and include `gorgo-autotune` in both eval windows.

---

## Experiment: 2D GORGO Cost Model — raw-queued load term (decoded_v9 tuning)

> **Window:** decoded high-diversity afternoon, 16:15–16:45 UTC, April 5th (`glm5_decoded_apr5_1615_to_1645`). $n=7{,}195$ requests per policy; concurrency 64; 100% success rate. ~3.4× heavier than the W1 nighttime window.

This run fixes the load term in the 2D cost model. Earlier runs scored load with `queued_uncached_tokens` (cache-miss queued tokens only), which hid the true load on a replica whose queue was full of cache-hit requests. The load term now uses the raw `queued_tokens` counter:

```text
score(u) = rtt_weight * rtt_ms(u)
         + uncached_tokens(u)
         + queue_weight * queued_tokens(u)
```

The own-prefill term stays cache-aware (counts only the current request's uncached tokens); only the load term changed. Search ranges were also tightened to keep the `rtt_weight / queue_weight` ratio bounded (the ratio that sets the load-diversion threshold), capping `rtt_weight` and flooring `queue_weight`:

```json
{ "rtt_weight": [0.05, 2.0], "queue_weight": [0.05, 0.5] }
```

Learned weights (ES converged): `rtt_weight=0.276, queue_weight=0.500`. The ES drove `queue_weight` to its ceiling — the opposite of the prior run on this window, which slammed it to the floor (`queue_weight=0.01`) and collapsed (see below).

![2D GORGO hillclimb convergence on the decoded_v9 apr5 tuning window: best-incumbent hyperparameter trajectories (rtt_weight settles at 0.28, queue_weight rails at its 0.5 ceiling), Rechenberg sigma rise/decay, objective (neg p95 TTFT) climbing from −5.9 to −1.2, and the 1/5 success rate falling to zero at convergence.](figures/tune_convergence_2d_v9.png)

| Policy | TTFT p50 | TTFT p95 | TTFT p99 | E2E p95 |
|---|---:|---:|---:|---:|
| **gorgo-hillclimb-p95-2d** | **673ms** | 2,514ms | 4,473ms | **8.91s** |
| simple-session-affinity | 712ms | **2,428ms** | 5,190ms | 18.27s |
| least-load | 763ms | 2,447ms | 4,349ms | 13.69s |
| least-request | 968ms | 3,970ms | 7,012ms | 16.62s |
| prefix-cache | 1,477ms | 6,784ms | 9,613ms | 22.63s |

> **gorgo advantage (E2E p95): 34.9%** — gorgo-hillclimb-p95-2d 8.91s vs least-load 13.69s. gorgo also wins TTFT p50 (673ms, 5.5% over session-affinity) and is within ~3.5% of the best TTFT p95. _(in-sample tuning window)_

**Turnaround vs the broken load term.** On this exact window, the previous cost model (`queued_uncached` load term, ranges allowing `rtt_weight=5.0` / `queue_weight=0.01`) tuned itself into a degenerate RTT-greedy corner and collapsed:

| Metric (apr5 window) | prior (`queued_uncached`, `rtt=5.0/queue=0.01`) | raw-queued (`rtt=0.28/queue=0.50`) |
|---|---:|---:|
| gorgo TTFT p50 | 1,815ms | **673ms** |
| gorgo TTFT p95 | 10,055ms | **2,514ms** |
| gorgo E2E p95 | 30.22s | **8.91s** |
| gorgo rank (of 5) | worst | best (p50, E2E) |

Making queued cache-hit work visible to the load term — plus bounding the `rtt_weight/queue_weight` ratio — removes the load-concentration feedback loop: instead of piling traffic onto the nearest replica until its queue explodes, the policy now spreads load and wins E2E p95 decisively while staying competitive on TTFT.

### Why the cache-aware load term worked on Apr 2 but collapsed here

The earlier `queued_uncached` load term scored a replica's load as its *cache-miss* queued tokens only — it treats a queue full of cache-hit work as "unloaded." Whether that blind spot is harmless or catastrophic is decided entirely by the window's data composition (`results/trace_summaries/glm5_window_stats.csv`):

| Window | rps | n_users | top-user share | median tok | multi-turn % | diversity |
|---|---:|---:|---:|---:|---:|---:|
| apr2 00:30 (W1) | 2.1 | 249 | 45.9% | 13 | 46.1 | 1,022 |
| apr2 01:00 (W2a) | 2.3 | 256 | 43.4% | 13 | 47.4 | 1,115 |
| apr2 12:30 (W2b) | 3.5 | 307 | 38.2% | 34 | 52.6 | 1,747 |
| apr5 16:15 (this window) | 5.9 | 772 | 21.4% | 9,554 | 73.9 | 4,497 |
| apr6 15:05 | 6.5 | 814 | 20.4% | 9,807 | 74.1 | 4,897 |
| apr7 19:45 | 7.0 | 727 | 15.7% | 4,481 | 73.7 | 5,100 |

**On Apr 2 the blind spot is harmless.** The `median_tokens=13` next to `avg≈19k` shows a bimodal, whale-dominated regime: one user carries ~40–46% of traffic with giant multi-turn (highly prefix-reusable) contexts, while most other requests are tiny continuations. Concentrating the whale on one warm replica produces genuinely near-zero uncached work — so the load term reads ~0 *and the real cost is ~0*. At ~2 rps the fleet never saturates, so concentration is the correct move. All three Apr 2 windows share this shape, which is why frozen/held-out weights transferred: the windows are the same easy regime, not evidence of a robust cost model.

**On Apr 5 the same term is a blindfold.** Diversity triples (772 users, no whale at 21%), the median request is *real* 9.5k-token context rather than a 13-token continuation, and throughput doubles to ~6 rps. Now even cache-*hit* requests are large — they still consume KV memory, batch slots, and decode compute — so a queue "full of cache hits" is genuinely loaded, exactly the case `queued_uncached` reports as ~0. At ~6 rps the fleet saturates, the term says "send more," and the load-concentration loop runs to collapse. In short: `queued_uncached` only works when cache-hit queued work is genuinely cheap, which needs low load and short continuations — Apr 2 has both, Apr 5/6/7 have neither.

### How the init and ranges pinned the policy into the greedy corner

Routing is governed by the **`rtt_weight / queue_weight` ratio** (the load-diversion threshold). The collapsed run allowed `rtt_weight ∈ [0.05, 5.0]`, `queue_weight ∈ [0.01, 2.0]`, starting near `rtt=0.5, queue=0.1`:

- **Unbounded headroom, no floor.** Allowed max ratio `5.0/0.01 = 500` vs a starting ratio of `5` — ~100× room to amplify RTT and drive the load term toward off, with nothing flooring it away from zero.
- **The objective rewards walking that way.** The ES minimizes p95 *TTFT* (first-token) over a short rolling window. Concentrating onto the warm/near replica lowers TTFT *before* the queue backlog manifests (queue delay lags; a request can get its first token fast while the replica is oversubscribed for decode). Each step `rtt↑ / queue↓` looks locally better, and Rechenberg's 1/5-rule then shrinks σ and **locks it at the boundary** (`rtt=5.0, queue=0.01`).
- **`queued_uncached` removes the only brake.** A raw-load term would make concentration visibly raise the score and push back; the cache-aware term can't see the cache-hit queue it is creating, so the march to the corner is unopposed.

The fix attacks both: raw `queued_tokens` restores the feedback, and the bounded ranges (`rtt ∈ [0.05, 2.0]`, `queue ∈ [0.05, 0.5]`) cap the ratio at `40` (vs 500) so the load term can never be removed. The tell that the corner was an artifact rather than an optimum: on this same window the constrained ES then drove `queue_weight` to its **ceiling (0.5)** — it wanted *maximum* load avoidance, the exact opposite of the `queue=0.01` the unconstrained search "found."

> **Caveat:** this is the in-sample tuning window. Out-of-sample eval windows (Apr 6 15:05–15:35, Apr 7 19:45–20:15) were launched but cancelled mid-eval0; evals to be re-run off the saved weights with `--skip-tuning`. `queue_weight` pinning at the ceiling (0.5) suggests the next run should raise the `queue_weight` cap so the optimum sits inside the range.

---

## Experiment: Calibrated 2D GORGO — calibrate → tune → eval (v14)

> **Negative/diagnostic result.** First full run of the ms-normalized 2D cost model with physically calibrated rates. gorgo trails `simple-session-affinity` on both windows. Recorded honestly for the paper's analysis section.

The cost model is the ms-normalized 2D form, with `prefill_rate`/`queue_rate` now **calibrated physical constants** (not ES-tuned) and only the two dimensionless weights searched:

```text
score(u) = rtt_weight * rtt_ms(u)
         + prefill_rate * uncached_tokens(u)
         + queue_rate * queue_weight * queued_tokens(u)
```

**Phase 0 — Calibration** (decoded Apr 5 16:15–16:45, $n=7{,}195$). The proxy fits the physical rates online by least squares over each request, `ttft_ms ≈ per-replica intercept + prefill_rate·uncached + queue_rate·queued_tokens` (no engine `meta_info`; proxy-measured TTFT + real-time queued-token counter). Pooled result:

```text
prefill_rate = 0.057 ms / uncached-token
queue_rate   = 0.008 ms / queued-token
```

(per-replica fixed-effect intercepts 282–738 ms absorb RTT + fixed overhead). These are held fixed through tuning and eval.

**Phase 1 — Tuning** (decoded Apr 6 15:05–15:35, $n=7{,}630$; `least-load` errored out). ES searched `rtt_weight ∈ [0.1, …]`, `queue_weight`. Converged to `rtt_weight=0.1` (**floor-pinned**), `queue_weight=1.318`.

| Policy | TTFT p50 | TTFT p95 | TTFT p99 |
|---|---:|---:|---:|
| **simple-session-affinity** | **736ms** | **2,394ms** | 3,858ms |
| gorgo-hillclimb-p95-2d | 766ms | 2,895ms | 5,447ms |
| least-request | 1,365ms | 6,170ms | 9,465ms |
| prefix-cache | 1,781ms | 7,035ms | 9,547ms |

> **gorgo does not win:** session-affinity leads TTFT p95 (2,394ms vs gorgo 2,895ms, **−17%** for gorgo). gorgo is 2nd.

**Phase 2 — Eval** (decoded Apr 7 19:45–20:15, $n≈8{,}650$/policy). Deploys frozen weights (`rtt_weight=0.1, queue_weight=1.318`). The eval stalled draining a handful of dead in-flight requests and was **salvaged via `POST /workload/stop`** (partial-trace finalize; ~99.8% of requests completed, all 5 policies recovered):

| Policy | TTFT p50 | TTFT p95 | TTFT p99 |
|---|---:|---:|---:|
| **simple-session-affinity** | **717ms** | **2,248ms** | 4,338ms |
| gorgo-static-p95-2d | 1,219ms | 4,563ms | 6,846ms |
| least-load | 1,259ms | 5,401ms | 9,418ms |
| least-request | 1,474ms | 5,477ms | 7,088ms |
| prefix-cache | 1,602ms | 7,644ms | 18,803ms |

> **gorgo loses to session-affinity (p95 TTFT): −103%** — gorgo-static-p95-2d 4,563ms vs session-affinity 2,248ms _(held-out eval)_. gorgo is 2nd of 5, still well ahead of the load-only baselines.

---

## Experiment: Calibrated 2D GORGO — re-tune with raised rtt ceiling (v2)

> **Negative/diagnostic result.** Re-ran tuning+eval with a wider `rtt_weight` range and reasoned starters, reusing the v14 calibrated rates (`--skip-calib`). The ES drove `rtt_weight` to the **opposite** boundary (1.5 ceiling) vs v14 (0.1 floor), and the resulting weights generalized **worse**. This run exposed the core problem: **the per-window ES objective is dominated by load noise.**

Calibrated rates reused from v14 (`prefill_rate=0.057`, `queue_rate=0.008`, fixed). Search ranges `rtt_weight ∈ [0.01, 1.5]` (start 0.2), `queue_weight ∈ [0.25, 4.0]` (start 1.0).

**Phase 1 — Tuning** (decoded Apr 6 15:05–15:35, $n=7{,}630$, all 5 policies). ES converged to `rtt_weight=1.5` (**ceiling-pinned**), `queue_weight=0.647`.

| Policy | TTFT p50 | TTFT p95 | TTFT p99 |
|---|---:|---:|---:|
| **simple-session-affinity** | **751ms** | **2,470ms** | 4,559ms |
| least-load | 960ms | 3,367ms | 5,335ms |
| gorgo-hillclimb-p95-2d | 929ms | 3,397ms | 5,457ms |
| least-request | 1,404ms | 5,879ms | 8,060ms |
| prefix-cache | 3,204ms | 10,018ms | 12,658ms |

> gorgo (3,397ms p95) ≈ ties least-load, behind session-affinity (2,470ms). Note gorgo's tuning-window p95 is an average over the ES search (weights changing), not a clean run at the final weights.

**Phase 2 — Eval** (decoded Apr 7 19:45–20:15, $n=8{,}663$/policy). Deploys frozen `rtt_weight=1.5, queue_weight=0.647`. The eval hung on the drain step again; **3 of 5 policies salvaged** (`least-load`, `prefix-cache` were lost when the app terminated before finalizing):

| Policy | TTFT p50 | TTFT p95 | TTFT p99 |
|---|---:|---:|---:|
| **simple-session-affinity** | **744ms** | **2,254ms** | 4,059ms |
| least-request | 2,796ms | 6,922ms | 8,883ms |
| gorgo-static-p95-2d | 2,776ms | 8,037ms | 9,972ms |

> **gorgo is worst of the three (p95 TTFT 8,037ms)** — worse than least-request (6,922ms) and ~3.6× session-affinity (2,254ms). It is also **much worse than v14's gorgo on the same window** (4,563ms with `rtt=0.1`), i.e. the tuned `rtt=1.5` generalized badly.

### Diagnosis: why tuning is unstable and gorgo trails session-affinity

**1. The per-window ES objective is dominated by noise (root cause of the boundary pins).** The (1+1)-ES scores each weight candidate on a single rolling 128-request window of live traffic. From the v2 tuning trajectory, re-evaluating the *identical* weights (`rtt=1.5, queue_weight=0.647`) across consecutive windows produced `neg_p95_ttft` scores ranging from **−1.74 to −5.56** (p95 TTFT 1.7s–5.6s). The window-to-window load variance (±50%+) swamps the weight effect, so the ES locks onto whichever candidate lands in a lucky low-noise window, then Rechenberg's 1/5 σ-decay traps it there. This is why `rtt_weight` rails to a boundary (0.1 floor in v14, 1.5 ceiling in v2) — the pins are noise artifacts, not optima, and they generalize poorly out-of-sample.

**2. Ruled out (verified in code, not assumed):**
- *Stale metrics are not the cause.* Replica `/metrics` are scraped every 30s (median routing-decision age 15s), but `route_gorgo_2d` reads the load term from the **real-time** `endpoints_queued_tokens` counter and the cache term from the **real-time** radix trie; only RTT comes from the stale scrape, and RTT is stable.
- *`rtt=1.5` is not pure nearest-replica routing.* Measured chosen-target cost terms at `rtt=1.5`: `rtt≈183`, `prefill≈0–104`, `queue≈353` (p50 ms) — comparable scale; the queue term still participates.

**3. gorgo trails session-affinity even with reasonable weights.** Even v14's `rtt=0.1` (4,563ms) loses to session-affinity (2,248ms) on the eval window. Aggregate cache-hit fraction is comparable (gorgo 0.77 vs session-affinity 0.82), and gorgo is worse across the whole distribution (eval p50 too), not just the tail — pointing to session cache-*coherence* (deterministic session→replica pinning vs gorgo's per-request re-routing) and load handling under the high-load evening window, rather than gross cache misses. Direct confirmation needs a session key surfaced in the trace.

**Next steps:** (a) make the tuning objective robust — evaluate each ES candidate over far more data (≫128 samples, or a fixed wall-clock dwell, or whole-window p95) so the signal exceeds the load noise; (b) re-eval gorgo with robustly-tuned weights (expected near the low-`rtt` region the early ES steps favored) before concluding gorgo < session-affinity; (c) add the workload drain timeout so evals self-finalize instead of needing manual `/workload/stop`.

---

## Experiment: Held-out eval (apr6) — full 5-policy comparison @ time_scale=2.0

> **Held-out evaluation.** Frozen gorgo weights from the decoded_v9 apr5 16:15–16:45 hillclimb tuning (`rtt=0.276, queue=0.5`; non-physical-rate 2D, `score = rtt*rtt_ms + uncached + queue*queued`), deployed on the **held-out** apr6 15:05–15:35 high-diversity decoded window at `time_scale=2.0`. $n=7{,}630$/policy, 100% success, all policies within capacity (scheduling-slip p95 ≤ 8 ms — clean comparison).

| Policy | TTFT p50 | TTFT p95 | TTFT p99 | E2E p50 | E2E p95 | ITL avg | decode tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| **gorgo-static-p95-2d** | **491ms** | **1,584ms** | **2,222ms** | **1,797ms** | **3,285ms** | **10.8** | **105** |
| simple-session-affinity | 627ms | 1,875ms | 3,056ms | 2,008ms | 4,747ms | 13.0 | 94 |
| least-request | 634ms | 1,852ms | 2,484ms | 2,078ms | 3,991ms | 13.0 | 95 |
| least-load | 616ms | 1,818ms | 2,885ms | 2,166ms | 4,059ms | 12.9 | 92 |
| prefix-cache | 574ms | 1,798ms | 2,750ms | 2,068ms | 4,947ms | 14.4 | 91 |

> **gorgo sweeps every metric** on the held-out window: TTFT p95 **+15.5% vs session-affinity** (1,584 vs 1,875ms), E2E p95 **+30.8%** (3,285 vs 4,747ms), plus best p50/p99, ITL, and decode throughput. This is the cleanest headline result: held-out window, frozen weights, validated within-capacity load, beating all four baselines including session-affinity.

This is a **cleaner win than apr7 @ ts2** (where gorgo only ties session-affinity on TTFT), most likely because apr6 has the **longer median context** (median ~9.8k input tokens vs apr7's ~4.5k) → more prefill recoverable via cache-aware routing, so gorgo's lever is larger. The advantage tracks prompt length / recoverable prefill, consistent with the cost model.

**Source & method:** per-policy stats from `GORGO-bench-results` volume, `workload_runs/glm5_c64_eval_ts2_apr6_v2/glm5_c64_eval_ts2_apr6_000_glm5_decoded_apr6_1505_to_1535_<policy>.json` (the `stats` block: `ttft_seconds`, `request_e2e_seconds`, `itl_ms`, `decode_tokens_per_second`). Slip computed from per-request `sent_delay_ms − scheduled_delay_ms`. Run executed with the bounded drain-timeout fix (`WORKLOAD_DRAIN_TIMEOUT_SECONDS`), so all 5 policies self-finalized; the aggregated `sweep_matrix.json` was not written, stats taken from the per-policy `workload_runs` JSON.

---

## Experiment: Held-out eval (apr7) — full 5-policy comparison @ time_scale=2.0

> **Held-out evaluation, heavier window.** Same frozen weights (`rtt=0.276, queue=0.5`) deployed on the **held-out** apr7 19:45–20:15 decoded window at `time_scale=2.0`. $n=8{,}663$/policy, 100% success. Unlike apr6, this window sits **past the saturation knee for the concentrating/cache-blind baselines** at ts2 (see slip column), so it is the complementary "under stress" result.

| Policy | TTFT p50 | TTFT p95 | TTFT p99 | E2E p50 | E2E p95 | slip p95 |
|---|---:|---:|---:|---:|---:|---:|
| **gorgo-static-p95-2d** | 586ms | 1,891ms | 3,431ms | **2,383ms** | **7,102ms** | 0.04s |
| simple-session-affinity | **512ms** | **1,684ms** | **2,455ms** | 2,195ms | 14,664ms | 0.01s |
| least-request | 915ms | 4,806ms | 7,114ms | 3,769ms | 17,271ms | 5.6s |
| least-load | 633ms | 2,158ms | 3,753ms | 2,692ms | 10,391ms | 0.02s |
| prefix-cache | 1,474ms | 8,243ms | 12,167ms | 7,673ms | 22,141ms | 181s |

> **session-affinity wins TTFT p95 (−12.3% for gorgo) but gorgo wins E2E p95 by +51.6%** (7,102 vs 14,664ms) and has the **best E2E of all five**. This is the "E2E is the honest objective on saturating windows" result made concrete: TTFT actively *misleads* (continuous batching shields session-affinity's single-replica concentration on first-token), while E2E exposes that the concentrating replica is saturated.

**Per-policy saturation at this load** (the key story): only **gorgo, session-affinity, and least-load keep the client on schedule** (slip ≤ 0.04s), but by E2E only **gorgo is not melting** (7.1s vs 10–22s). The cache-greedy/cache-blind baselines fall over: `least-request` is client-over-capacity (slip 5.6s) and `prefix-cache` catastrophically so (**slip 181s**, E2E 22.1s) — its longest-prefix-match concentration saturates one replica so hard the open-loop generator backs up ~3 minutes. **There is no single `time_scale` at which all five policies are simultaneously within capacity at meaningful load**, because effective capacity is policy-dependent: balanced routing (gorgo, least-load) tolerates a higher offered rate before any replica saturates than concentrating routing (session-affinity, prefix-cache). "Saturated at the same offered load" is itself the result.

Paired with apr6, the two held-out windows give a complete picture: **apr6 (lighter) — gorgo sweeps every metric; apr7 (heavier) — gorgo wins E2E by 51.6% and is the only non-saturating policy** (session-affinity edges TTFT only because continuous batching hides its E2E collapse).

**Source & method:** per-policy `stats` from `GORGO-bench-results` volume, `workload_runs/glm5_c64_eval_ts2_apr7_v2/glm5_c64_eval_ts2_apr7_000_glm5_decoded_apr7_1945_to_2015_<policy>.json`; slip = per-request `sent_delay_ms − scheduled_delay_ms` (p95). Self-finalized via the drain-timeout fix. **Caveat:** `least-request` (slip 5.6s) and `prefix-cache` (slip 181s) are client-over-capacity here, so their figures are saturation-dominated; a strictly within-capacity 5-policy apr7 row requires `time_scale=3.0` (the concentrating policies' single-replica ceiling sits near the ts3 offered rate).

---

## Experiment: Held-out eval (apr7) — full 5-policy comparison @ time_scale=3.0 (within capacity)

> **Held-out evaluation, within capacity for all five.** Same frozen weights (`rtt=0.276, queue=0.5`) on the **held-out** apr7 19:45–20:15 decoded window at `time_scale=3.0` — the within-capacity row the ts2 section flagged as missing. $n=8{,}663$/policy (≤61 fails each), offered ≈11,329 input tok/s, scheduling-slip p95 ≤ 0.01s for **all** policies, so this is a clean routing-quality comparison with no saturation confound.

| Policy | TTFT p50 | TTFT p95 | TTFT p99 | E2E p50 | E2E p95 | slip p95 | conc% |
|---|---:|---:|---:|---:|---:|---:|---:|
| **gorgo-static-p95-2d** | **386ms** | **1,377ms** | **1,973ms** | **1,725ms** | **3,337ms** | 0.01s | 43.5 |
| simple-session-affinity | 439ms | 1,495ms | 2,313ms | 1,727ms | 3,901ms | 0.01s | 49.8 |
| least-load | 480ms | 1,637ms | 2,294ms | 1,931ms | 4,038ms | 0.00s | 35.9 |
| least-request | 509ms | 1,724ms | 2,485ms | 1,960ms | 4,402ms | 0.01s | 35.5 |
| prefix-cache | 516ms | 1,830ms | 2,801ms | 2,180ms | 11,980ms | 0.01s | 36.8 |

> **gorgo sweeps every metric** once apr7 is within capacity: TTFT p50 **+12.1% vs session-affinity** (386 vs 439ms), p95 **+7.9%** (1,377 vs 1,495ms), p99 **+14.7%** (1,973 vs 2,313ms), and E2E p95 **+14.5%** (3,337 vs 3,901ms; E2E p50 a tie). This is the decisive complement to the ts2 row: the **same window** where session-affinity *beat* gorgo on TTFT under stress flips to a gorgo sweep once the fleet has slack — confirming the ts2 TTFT loss was a saturation artifact (continuous batching shielding session-affinity's concentration), not a routing deficit.

gorgo achieves this **without the reward-hack signature**: its single-replica concentration is 43.5% — below session-affinity's 49.8% and above the load-balancers' ~36% — i.e. it spreads more than the cache-greedy policy yet exploits cache more than the cache-blind ones. Note `prefix-cache` still posts a 12.0s E2E p95 despite ≤0.01s slip: its longest-prefix-match concentration keeps one replica's decode batch saturated even when the client never falls behind, the in-capacity echo of its ts2 collapse.

Paired with apr6 ts2 and apr7 ts2, the three held-out points now read cleanly: **apr6 (light) — gorgo sweeps; apr7 ts3 (within capacity) — gorgo sweeps; apr7 ts2 (over capacity for the concentrating baselines) — gorgo wins E2E, loses TTFT to session-affinity only because batching hides its saturation.** The advantage is monotone in available capacity.

**Source & method:** recomputed from per-request traces on `GORGO-bench-results`, `proxy_traces/glm5_c64_eval_ts3_apr7/glm5_c64_eval_ts3_apr7_000_glm5_decoded_apr7_1945_to_2015_<policy>/requests.jsonl` (same `ttft_ns` = dispatch→first-token, `total_ns` = E2E methodology as the load-sweep/saturation tables; `error`-free rows only). offered tok/s = Σ`prompt_tokens` / replay span (`monotonic_s`); slip from `scheduling_slip_ms`; conc% = max single-replica share of `target_replica_key`. Run `glm5_c64_eval_ts3_apr7` (app `ap-m3DJJdJG3vmPNU6kod0rrw`), self-finalized via the drain-timeout fix.

---

## Experiment: Load Sweep (apr7) — saturation is what hurts gorgo's TTFT, not routing quality

> **Controlled experiment.** Replay the same apr7 19:45–20:15 high-diversity window at three arrival rates via `time_scale` (stretches inter-arrival times; identical requests/order/prompts/cache structure, only the rate varies). Fixed non-physical-rate 2D weights from the decoded_v9 tuning (`rtt_weight=0.276, queue_weight=0.5`; `prefill_rate=queue_rate=1.0` collapse the score to `rtt*rtt_ms + uncached + queue_weight*queued`), so load is the only variable. Trimmed to `gorgo-static-p95-2d` vs `simple-session-affinity` vs `least-request`.

| Load (`time_scale`) | arrival rate | regime | gorgo TTFT p95 | SSA TTFT p95 | gorgo E2E p95 | SSA E2E p95 |
|---|---|---|---:|---:|---:|---:|
| Full (`1.0`) | ~4.8 rps | **over capacity**† | 8,260ms† | 1,835ms† | 15.8s† | 17.9s† |
| Half (`2.0`) | ~2.4 rps | within capacity | 1,822ms | 1,707ms | **5,648ms** | 15,504ms |
| Third (`3.0`) | ~1.6 rps | within capacity | **1,378ms** | 1,556ms | **3,248ms** | 4,041ms |

> †**At full load the 3-replica fleet is over capacity for _every_ policy**, so ts1 is not a clean routing comparison. The open-loop replay cannot dispatch on schedule — it backs up by minutes waiting for concurrency slots, recorded as client-side scheduling-slip (scheduled-arrival→dispatch), separate from `ttft_ns` (dispatch→first-token). Slip p95 is **243s (SSA), 448s (gorgo), 491s (least-request)**. The TTFT/E2E figures for ts1 are `ttft_ns`/`total_ns` and therefore *exclude* that multi-minute arrival backlog; true user-perceived latency at full load is slip-dominated and unusable for all three. **ts1 marks the saturation ceiling, not a routing result.**

The valid routing comparison is **ts2 and ts3**, where scheduling slip is negligible (p95 ≤ 80ms for SSA and gorgo; least-request is still marginally over capacity at ts2, slip p95 7.2s). Across that range gorgo's standing improves **monotonically** as load backs off from the ceiling: at half load it is within 7% of session-affinity on TTFT p95 while winning E2E ~2.7× (5.6s vs 15.5s); at third load it **wins every metric** — TTFT p95 1,378ms vs 1,556ms (+11.4%), TTFT p99 2,124ms vs 4,589ms (+53.7%), and E2E p95 3,248ms vs 4,041ms. Two regimes, two winners: **at the capacity ceiling**, cache-greedy session-affinity is *less bad* — with zero spare capacity, minimizing total prefill work (high cache hits) beats spreading load, and gorgo's off-cache diversions only add work (gorgo is worst at ts1 on both `ttft_ns` and slip). **Below the ceiling**, load-aware routing wins outright, because now there is slack to exploit and avoiding queue buildup pays off. The saturation hypothesis is confirmed: gorgo's advantage scales inversely with saturation, and SSA's concentration (49.8% on one replica) eventually hurts even its own TTFT tail (p99 4,589ms at ts3) once the fleet is out of the saturated regime.

### ts3 full table (third load, `time_scale=3.0`)

| Policy | TTFT p50 | TTFT p95 | TTFT p99 | E2E p50 | E2E p95 |
|---|---:|---:|---:|---:|---:|
| **gorgo-static-p95-2d** | **392ms** | **1,378ms** | **2,124ms** | **1,665ms** | **3,248ms** |
| simple-session-affinity | 455ms | 1,556ms | 4,589ms | 1,757ms | 4,041ms |
| least-request | 546ms | 1,805ms | 2,623ms | 2,100ms | 4,917ms |

> $n=8{,}663$/policy, 100% success. gorgo sweeps all six metrics; the largest margin is TTFT p99 (+53.7% vs SSA), where SSA's single-replica concentration damages its tail.

### Why gorgo trades TTFT for E2E (request-level decomposition, half-load window)

Per-request signals at dispatch for the two extremes (ts2, $n=8{,}663$/policy):

| Signal | simple-session-affinity | gorgo-static-2d |
|---|---:|---:|
| TTFT p50 / p95 | 515 / **1,707ms** | 537 / 1,822ms |
| E2E p95 | 15,504ms | **5,648ms** |
| cache-hit rate (all) | **0.875** | 0.815 |
| avg uncached tokens | **1,033** | 1,478 |
| off-cache routing % (all / tail5%) | 2.3 / 4.6 | **11.1 / 16.4** |
| chosen replica's queued tokens | **62,037** | **22,961** |
| (fleet min queued available) | 9,842 | 21,926 |
| chosen scrape-latency (all / tail5%) | 286 / 376ms | 263 / **441ms** |

The **chosen-replica queue depth** is the smoking gun: session-affinity routes into replicas carrying **62k queued tokens** — 6× the least-loaded replica available (9.8k) — yet still posts TTFT p95 1.7s while its E2E p95 explodes to 15.5s. This is the **continuous-batching signature**: a saturated replica still admits a new request's prefill and emits a first token on time (TTFT is largely load-insensitive), but the decode backlog dilutes the running batch so every decode step crawls (E2E is load-sensitive). gorgo holds its chosen queue at ~23k (≈ the fleet minimum — it spreads load and keeps all three replicas balanced), buying the ~2.7× E2E win.

gorgo's slightly-worse TTFT is mostly **lost cache locality**, not RTT: it routes off the most-cached replica 11% of the time (16% on its tail) vs SSA's 2–5%, dropping its cache-hit rate 0.875→0.815 (~445 more uncached tokens/request to prefill), which shifts the whole TTFT distribution right (p50 +22ms, p95 +115ms). RTT is a secondary, tail-only effect — gorgo's top-5% TTFT requests show higher chosen scrape-latency (441 vs 376ms), the "avoid the loaded near/warm replica → land farther" cost, but across the body gorgo's chosen latency is actually *lower* (263 vs 286) because SSA's saturated replica inflates its own scrape latency via handler contention.

**Takeaway.** Prefill cache locality is genuinely the dominant TTFT lever (routing to the cache-warm replica minimizes uncached prefill), but in the **aggregated** prefill+decode regime the *same* replica then decodes the request, so cache-greedy concentration silently destroys E2E. TTFT and E2E are coupled through the shared replica, and TTFT-only optimization reward-hacks toward concentration precisely because TTFT is the axis that does not price the decode backlog. **E2E is the honest objective on saturating windows.**

### Implication: prefill/decode disaggregation should largely dissolve the tradeoff

The TTFT-vs-E2E tension above is an artifact of co-locating prefill and decode on one replica. Under PD-disaggregation \[DistServe, Splitwise, Mooncake], a request prefills on a prefill-pool worker (producing KV) and decodes on a separate decode-pool worker. The decode-batch dilution that destroys E2E is then handled by the decode scheduler, **decoupled from the prefill routing decision**. The prefill router GORGO implements would then score only the TTFT-relevant terms — RTT, uncached prefill (cache locality on the prefill pool), and **prefill-pool** queueing — while a separate, simpler decode load-balancer handles decode placement.

Crucially this should also **defuse the reward hack**: concentrating prefill onto a cache-warm worker no longer blows up decode E2E (decode is elsewhere), so the degenerate "concentrate to win TTFT" solution stops being globally harmful. The residual tradeoff moves *inside* TTFT — sending every request to one cache-warm prefill worker eventually causes **prefill-queue contention** at high saturation, which TTFT itself sees and prices. The objective becomes self-correcting: the load term that must be hand-constrained in the aggregated regime is one the prefill-pool queue signal now supplies directly. We do not evaluate a PD-disaggregated fleet here; this is a hypothesis for future work, consistent with the saturation mechanism the load sweep isolates.

---

## Saturation diagnostic: how to tell a run is over capacity (apr7 load sweep)

The apr7 `time_scale` sweep is also a clean reference for *diagnosing saturation*, because it places the same window/policies at three offered loads. The table below contrasts the four signals; the lesson is that **only E2E p95 and scheduling-slip discriminate saturation reliably** — TTFT and throughput mislead.

| run / policy | offered in tok/s | slip p95 | TTFT p95 | E2E p95 | saturated? |
|---|--:|--:|--:|--:|:--:|
| ts1 gorgo | 34,064 | 448s | 8,260 | 15,759 | yes |
| ts1 session-affinity | 34,035 | 243s | 1,835 | 17,940 | yes |
| ts1 least-request | 34,030 | 491s | 6,138 | 18,615 | yes |
| ts2 gorgo | 17,003 | 0.08s | 1,822 | 5,648 | no |
| ts2 session-affinity | 16,999 | 0.02s | 1,707 | 15,504 | yes |
| ts2 least-request | 16,999 | 7.2s | 4,361 | 16,847 | yes |
| ts3 gorgo | 11,327 | 0.01s | 1,378 | 3,248 | no |
| ts3 session-affinity | 11,326 | 0.01s | 1,556 | 4,041 | no |
| ts3 least-request | 11,326 | 0.01s | 1,805 | 4,917 | no |

(TTFT/E2E columns in ms.) **How to read it:**

- **E2E p95 is the primary saturation signal** (server-side). A saturated replica's decode batch dilutes, so E2E inflates to ~15–18s vs an unsaturated floor of ~3–5s. By this measure: ts1 saturated for all; **ts2 saturated for session-affinity and least-request but *not* gorgo** (gorgo's load-spreading de-saturates the fleet at a load where the concentrating/cache-blind baselines still melt); ts3 unsaturated for all.
- **Scheduling slip is the client-side signal** — `slip = sent_delay_ms − scheduled_delay_ms`, i.e. how late the open-loop generator fired a request vs its scheduled arrival (it can only fall behind). Slip blows up when the in-flight pipeline (64 workers + 128-slot queue) can't drain fast enough. ts1: minutes for all (offered ≈ drain ceiling). ts2: gorgo/SSA ~ms but least-request 7.2s (its slow requests exhaust the worker pool). ts3: ms for all.
- **TTFT p95 is misleading — do not use alone.** ts1 session-affinity shows TTFT p95 **1,835ms** (looks healthy) while its E2E is **17,940ms** (fully saturated): continuous batching admits the first token on time even on a melting replica, so a concentrating policy looks fine on TTFT while E2E is on fire.
- **Offered tok/s is not a saturation signal here** — it just tracks the offered rate (34k : 17k : 11k ≈ 1 : ½ : ⅓ with `time_scale`); wall-clock ran 1.00× the scheduled span in all cases, so aggregate token throughput "kept up" regardless of per-replica saturation.

Note the two signals measure *different axes*: **slip = is the client keeping up; E2E = is a replica saturated.** ts2 session-affinity has slip ≈ 0.02s (client fine) yet E2E 15.5s (replica saturated) — at that offered rate 64 workers absorb the slow requests without the queue filling, so the client never falls behind even though the replica is melting. Use both.

### Data sources and method (for reproducibility)

- **Source:** per-request traces on the Modal volume `GORGO-bench-results`, path `proxy_traces/glm5_c64_loadsweep_apr7_ts{1,2,3}_v2/glm5_c64_loadsweep_apr7_ts{N}_000_glm5_decoded_apr7_1945_to_2015_<policy>/requests.jsonl` (one JSON object per request, `kind=="request"`). Window: decoded apr7 19:45–20:15; fleet: 3 replicas (ap-seoul-1, eu-frankfurt-1, us-ashburn-1), 2×L40S each; concurrency 64; max_tokens 128; gorgo weights `rtt=0.276, queue=0.5` (decoded_v9). The aggregated `sweep_matrix.json` did not harvest for these v2 runs, so all figures were recomputed from `requests.jsonl` (status==200 only).
- **offered in tok/s** = Σ`request_tokens` ÷ wall-span, where wall-span = max(`monotonic_s` + `total_ns`/1e9) − min(`monotonic_s`).
- **slip p95** = p95 of `scheduling_slip_ms` (= `sent_delay_ms` − `scheduled_delay_ms`; both recorded per request).
- **TTFT p95** = p95 of `ttft_ns`/1e6; **E2E p95** = p95 of `total_ns`/1e6.
- **saturated?** = heuristic on E2E p95 (≳ ~3× the unsaturated decode floor of ~3–5s ⇒ saturated), corroborated by slip.
- Percentiles use linear interpolation between order statistics. The `ts1` figures are the over-capacity regime and exclude client-side slip from TTFT/E2E (those are dispatch→first/last-token); true user-perceived latency at ts1 is slip-dominated (minutes).
