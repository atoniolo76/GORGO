# Experiment Index

Quick-reference for all c=64 GORGO routing experiments. Ordered
chronologically. Each entry links to Modal volume data and local results.

Structured data: [`experiment_index.json`](experiment_index.json)

---

## Pareto frontier summary

Three Pareto-optimal operating points (TTFT p95 vs E2E p95):

| Config | TTFT p95 | E2E p95 | Tradeoff |
|--------|----------|---------|----------|
| least-request | 1,058ms | 2,137ms | Best E2E, worst TTFT |
| **Config B** (rtt=1084, load=0.009) | **775ms** | **2,441ms** | **Knee of frontier: 27% better TTFT, 14% E2E cost** |
| Config C (rtt=4132, load=0.004) | 762ms | 2,905ms | Best TTFT, 36% E2E cost |

Config B is the high-value operating point for the paper.

---

## Pre-RTT-weight era (May 5–6, c=32)

Cost function: `score = rtt_seconds + t_prefill × uncached + queued_w × (queued + num_used)`

No explicit RTT weight — RTT contributed ~86ms to a score dominated by
prefill (thousands of score units). Optimizer always zeroed queue term.

### abstract_night_glm5_w1_v1 — Tuning

| | |
|---|---|
| Date | 2026-05-05 |
| Trace | `glm5_0030_to_0100` (Apr 1 night, tuning window) |
| Concurrency | 32 |
| Learned | `t_prefill=0.983, queued_w=0.0004` |
| Volume | `GORGO-bench-results:/policy_matrix_sweep/abstract_night/glm5_w1_v1/` |
| Local | `results/policy_matrix_sweep/abstract_night/glm5_w1_v1/` |

| Policy | TTFT p50 | TTFT p95 | TTFT p99 | E2E p50 | E2E p95 | E2E p99 |
|--------|----------|----------|----------|---------|---------|---------|
| gorgo-hillclimb | 288ms | 1,466ms | 2,307ms | 1,876ms | 6,087ms | 34,043ms |

### abstract_night_glm5_apr2_v1 — Eval (held-out)

| | |
|---|---|
| Date | 2026-05-05 |
| Trace | `glm5_apr2_0030_to_0100` (Apr 2 night, eval window) |
| Concurrency | 32 |
| Volume | `GORGO-bench-results:/policy_matrix_sweep/abstract_night/glm5_apr2_v1/` |
| Local | `results/policy_matrix_sweep/abstract_night/glm5_apr2_v1/` |

| Policy | TTFT p50 | TTFT p95 | TTFT p99 | E2E p50 | E2E p95 | E2E p99 |
|--------|----------|----------|----------|---------|---------|---------|
| gorgo-hillclimb | 283ms | 1,416ms | 2,118ms | 1,554ms | 3,370ms | 13,711ms |
| gorgo-static | 329ms | 1,387ms | 1,947ms | 1,603ms | 3,273ms | 8,945ms |

### abstract_night_glm5_stress_v1 — Eval (stress)

| | |
|---|---|
| Date | 2026-05-06 |
| Trace | `glm5_apr2_0030_to_0100` (Apr 2 night) |
| Concurrency | 32 |
| Learned | `t_prefill=0.043, queued_w=0.00006` |
| Volume | `GORGO-bench-results:/policy_matrix_sweep/abstract_night/glm5_stress_v1/` |
| Local | `results/policy_matrix_sweep/abstract_night/glm5_stress_v1/` |

| Policy | TTFT p50 | TTFT p95 | TTFT p99 | E2E p50 | E2E p95 | E2E p99 |
|--------|----------|----------|----------|---------|---------|---------|
| gorgo-hillclimb | 281ms | 1,331ms | 1,832ms | 1,591ms | 3,287ms | 11,156ms |

---

## RTT-weight era (May 26–28, c=64)

Cost function: `score = rtt_w × rtt_seconds + prefill_w × uncached + load_w × (queued + num_used)`

Adding explicit RTT weight was the single biggest improvement: TTFT p50
dropped from ~285ms to ~125ms. The optimizer's behavior: maximize
`rtt_weight`, minimize `load_weight`.

### glm5_c64_eval_v1 — Config B (balanced) ★

| | |
|---|---|
| Date | 2026-05-26 |
| Trace | `glm5_apr2_0030_to_0100` (Apr 2 night) |
| Concurrency | 64 |
| Learned | `prefill_w=0.038, load_w=0.009, rtt_w=1084` |
| Volume | `GORGO-bench-results:/policy_matrix_sweep/c64/glm5_c64_eval_v1/` |
| **Significance** | **Pareto-optimal knee. Best combined TTFT+E2E.** |
| **Origin** | **Final converged output of v1 ES tuner** (started at rtt_w=2000, prefill_w=0.2, load_w=0.01; ES reduced rtt_w and kept load_w, finding a balanced equilibrium rather than the RTT-extreme trajectory v8 later found) |

| Policy | TTFT p50 | TTFT p95 | TTFT p99 | E2E p50 | E2E p95 | E2E p99 |
|--------|----------|----------|----------|---------|---------|---------|
| least-request | 321ms | 1,131ms | 1,573ms | 1,000ms | 2,202ms | 6,697ms |
| session-affinity | 376ms | 1,257ms | 1,873ms | 1,241ms | 2,680ms | 10,822ms |
| **gorgo-static** | **212ms** | **775ms** | **1,055ms** | **1,205ms** | **2,441ms** | 10,953ms |

Score decomposition at typical request (RTT=0.086s, 5k uncached, 20k queued):
- RTT: 1084 × 0.086 = **93**
- Prefill: 0.038 × 5000 = **190**
- Load: 0.009 × 20000 = **180**
- All three terms comparable → balanced routing.

### glm5_c64_eval_v8 — Config C (RTT-dominated)

| | |
|---|---|
| Date | 2026-05-28 |
| Trace | `glm5_apr2_0030_to_0100` (Apr 2 night) |
| Concurrency | 64 |
| Learned | `prefill_w=0.060, load_w=0.004, rtt_w=4132` |
| Volume | `GORGO-bench-results:/policy_matrix_sweep/c64/glm5_c64_eval_v8/` |
| Local | `results/glm5_c64_eval_v8/` |
| **Significance** | Best TTFT ever, but E2E regression vs baselines. |

| Policy | TTFT p50 | TTFT p95 | TTFT p99 | E2E p50 | E2E p95 | E2E p99 |
|--------|----------|----------|----------|---------|---------|---------|
| random | 284ms | 1,165ms | 2,047ms | 1,297ms | 2,379ms | 3,208ms |
| least-request | 209ms | 1,058ms | 1,758ms | 1,224ms | 2,137ms | 2,876ms |
| prefix-cache | 300ms | 1,026ms | 1,450ms | 1,466ms | 2,454ms | 2,964ms |
| session-affinity | 197ms | 917ms | 1,554ms | 1,500ms | 2,552ms | 3,170ms |
| **gorgo-static** | **125ms** | **762ms** | **1,397ms** | 1,521ms | 2,905ms | 3,967ms |

---

## New cost model era (May 29, c=64)

Cost function: `score = rtt_w × rtt_ms + prefill_w × (prefill_rate × uncached + queue_rate × queued)`

Removed `load_weight` as separate term. Added calibration of
`prefill_rate`. Confirmed the same single-replica-concentration
behavior. 100% of traffic to closest replica.

### glm5_c64_tuning_p95ttft_v1 — Config D tuning

| | |
|---|---|
| Date | 2026-05-29 |
| Trace | `glm5_apr2_0030_to_0100` (Apr 2 night) |
| Concurrency | 64 |
| Learned | `prefill_w=1.65, rtt_w=5.0` (converged step 21) |
| Volume | `GORGO-bench-results:/policy_matrix_sweep/c64/glm5_c64_tuning_p95ttft_v1/` |
| Tune trace | `proxy_traces/glm5_c64_tuning_p95ttft_000_..._gorgo-hillclimb-p95/tune.jsonl` |
| Calibration | Integrated: prefill_rate=0.086–0.106 ms/tok per-replica |

### glm5_c64_eval_p95ttft_temporal_v1 — Config D eval (held-out)

| | |
|---|---|
| Date | 2026-05-29 |
| Trace | `glm5_apr2_0100_to_0130` (Apr 2 night, temporal generalization) |
| Concurrency | 64 |
| Frozen | `prefill_w=1.65, rtt_w=5.0` |
| Volume | `GORGO-bench-results:/policy_matrix_sweep/c64/glm5_c64_eval_p95ttft_temporal_v1/` |
| **Significance** | Confirmed 100% traffic concentration on closest replica. |

| Policy | TTFT p50 | TTFT p95 | TTFT p99 | E2E p50 | E2E p95 | E2E p99 |
|--------|----------|----------|----------|---------|---------|---------|
| random | 285ms | 1,304ms | 2,061ms | 1,306ms | 2,517ms | 3,428ms |
| least-request | 258ms | 1,377ms | 3,608ms | 1,293ms | 2,561ms | 5,687ms |
| least-load | 280ms | 1,296ms | 1,850ms | 1,402ms | 2,627ms | 3,384ms |
| prefix-cache | 284ms | 1,309ms | 2,078ms | 1,390ms | 2,545ms | 3,772ms |
| session-affinity | 214ms | 1,047ms | 1,691ms | 1,534ms | 2,705ms | 3,360ms |
| **gorgo-static-p95** | **141ms** | **956ms** | **1,736ms** | 1,497ms | 2,879ms | 4,855ms |

Per-replica routing: Ashburn=100%, Frankfurt=0%, Seoul=0%.

---

## Key findings for the paper

1. **RTT weight is the single largest improvement** — adding explicit RTT
   weighting dropped TTFT p50 from ~285ms to ~125ms (56% reduction).

2. **Single-objective optimization always degenerates** — whether the cost
   function has 2 or 3 terms, the ES always maximizes RTT emphasis and
   zeroes out load balancing when optimizing TTFT alone.

3. **The Pareto frontier has curvature** — Config B (rtt=1084, load=0.009)
   achieves 98% of Config C's TTFT improvement (775ms vs 762ms) while
   paying only 14% E2E cost vs least-request (vs 36% for Config C).

4. **The load term matters for E2E** — but only when it produces score
   contributions comparable to the RTT term. At Config B, all three terms
   contribute ~100–190 score units. At Config C, RTT alone contributes
   ~355, overwhelming the load term's ~80.

5. **Next step: constrained or multi-objective tuning** — the cost function
   structure is capable of Pareto-optimal routing. The tuner needs to stop
   at the frontier knee rather than sliding to the TTFT endpoint.

---

## Volume paths reference

All on `GORGO-bench-results` (Modal volume, `alessio-dev` environment):

```
/policy_matrix_sweep/
  abstract_night/
    glm5_w1_v1/          ← pre-RTT tuning (May 5)
    glm5_w2_v1/          ← pre-RTT eval temporal (May 5)
    glm5_apr2_v1/        ← pre-RTT eval held-out (May 5)
    glm5_stress_v1/      ← pre-RTT stress (May 6)
    glm5_midday_stress_v1/ ← pre-RTT midday (May 6)
    glm5_w1_stress_v1/   ← pre-RTT stress w/ W1 params (May 7)
  c64/
    glm5_c64_tuning_v1/ through v8/  ← RTT-weight tuning iterations (May 26-28)
    glm5_c64_eval_v1/    ← CONFIG B eval (May 26) ★
    glm5_c64_eval_v8/    ← CONFIG C eval (May 28)
    glm5_c64_tuning_p95ttft_v1/  ← new model tuning (May 29)
    glm5_c64_eval_p95ttft_temporal_v1/  ← CONFIG D eval (May 29)
```
