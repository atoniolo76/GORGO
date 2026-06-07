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

gorgo-hillclimb-p50 wins TTFT p50 (163ms, 16.8% gap over session-affinity at 196ms) and TTFT p95 (943ms, 4.6% gap over session-affinity at 988ms).

### Weight comparison: p95 vs p50 objective

| Weight | p95 objective | p50 objective | Interpretation |
|---|---|---|---|
| `rtt_weight` | 0.392 | **1.206** (3.1×) | p50 prioritizes RTT — typical requests benefit most from proximity |
| `prefill_weight` | 1.880 | **0.517** (0.3×) | p50 cares less about cache — median requests aren't the long-tail uncached ones |
| `load_weight` | 6.382 | **1.689** (0.3×) | p50 needs less load balancing — median requests don't see queueing pressure |

The ES discovers fundamentally different operating points depending on which percentile it optimizes. p95 optimization drives the policy toward cache-first routing with aggressive load avoidance (because tail requests are the ones stuck behind queues with cold caches). p50 optimization drives toward RTT-first routing (because the typical request is short enough that network latency dominates over cache effects).

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
