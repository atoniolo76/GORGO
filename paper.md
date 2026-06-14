# Results

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

## Planned Experiment: 2D GORGO Cost Model

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

> **Caveat:** this is the in-sample tuning window. Out-of-sample eval windows (Apr 6 15:05–15:35, Apr 7 19:45–20:15) were launched but cancelled mid-eval0; evals to be re-run off the saved weights with `--skip-tuning`. `queue_weight` pinning at the ceiling (0.5) suggests the next run should raise the `queue_weight` cap so the optimum sits inside the range.
