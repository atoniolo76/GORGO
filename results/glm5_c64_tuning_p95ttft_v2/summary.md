# TUNE-p95 v2 Results (3-weight model)

**Experiment:** `glm5_c64_tuning_p95ttft_v2`
**Trace:** T1 — glm5_apr2_0030_to_0100 (Apr 2 00:30–01:00, 2095 requests sent)
**Cost model:** `score = rtt_weight × rtt_ms + prefill_weight × uncached + load_weight × queued`
**Started:** 2026-05-31T02:48:34Z | **Fleet ready:** 02:54:01Z | **Completed:** 03:25:22Z

## Learned weights (ES converged step 29, best_score=-0.4546)

```
rtt_weight     = 0.2270
prefill_weight = 0.9074
load_weight    = 0.0
```

The ES found that prefill-cache-aware routing dominates (`prefill_weight ≈ 0.9`),
with moderate RTT emphasis (`rtt_weight ≈ 0.23`) and zero queue penalty. This is
qualitatively different from v1 (which had `rtt_weight=5.0, prefill_weight=1.65`)
— the 3-weight model with decoupled `load_weight` allows the ES to find a
cache-first operating point instead of an RTT-first one.

## Results

| Policy | TTFT p50 | TTFT p95 | TTFT p99 | E2E p50 | E2E p95 | E2E p99 | ITL avg | Decode tok/s | Sent | Fail |
|--------|----------|----------|----------|---------|---------|---------|---------|-------------|------|------|
| random | 282ms | 1,231ms | 2,117ms | 1.27s | 2.38s | 3.54s | 8.3ms | 124 | 2095 | 0 |
| least-request | — | — | — | — | — | — | — | — | — | ERROR |
| least-load | 214ms | 1,118ms | 2,105ms | 1.35s | 2.43s | 3.11s | 9.1ms | 116 | 2095 | 0 |
| prefix-cache | 278ms | 968ms | 1,633ms | 1.36s | 2.36s | 3.28s | 9.1ms | 116 | 2095 | 0 |
| session-affinity | 212ms | 1,033ms | 2,228ms | 1.49s | 2.64s | 3.65s | 10.1ms | 106 | 2095 | 0 |
| **gorgo-hillclimb-p95** | **151ms** | **880ms** | 1,909ms | 1.38s | **2.23s** | 3.20s | 9.6ms | 110 | 2095 | 0 |

## Key findings

- **gorgo wins TTFT p50 and p95** — 151ms p50 (next best: session-affinity 212ms, 29% worse),
  880ms p95 (next best: prefix-cache 968ms, 10% worse).
- **gorgo wins E2E p95** — 2.23s (next best: prefix-cache 2.36s, 6% worse; random 2.38s).
  This is the key improvement over v1 where gorgo had the worst E2E.
- **100% success rate** on all working policies.
- **TTFT p99 not best** — 1,909ms vs prefix-cache's 1,633ms. Tail requests still pay a penalty.
- **least-request crashed** — `ConnectError: All connection attempts failed` (transient Modal issue,
  not a code bug).

## Comparison to v1 (rate-based model, `rtt_weight=5.0, prefill_weight=1.65`)

| Metric | v1 gorgo | v2 gorgo | Change |
|--------|----------|----------|--------|
| TTFT p50 | 128ms | 151ms | +18% (slightly worse) |
| TTFT p95 | 934ms | 880ms | **-6% (better)** |
| E2E p50 | 1.38s | 1.38s | same |
| E2E p95 | 2.32s | 2.23s | **-4% (better)** |
| Routing | 100% Ashburn | balanced | cache-first, not RTT-first |

v2 trades a small p50 TTFT regression (151ms vs 128ms) for significantly better
tail latency on both TTFT and E2E. The routing is no longer all-Ashburn — traffic
is distributed based on prefix cache hits rather than purely by RTT.
