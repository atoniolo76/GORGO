# Reward Hacking in Single-Objective TTFT Routing

> Cross-cutting finding consolidated from three independent GORGO runs. Companion to
> `paper.md` (result numbers) and `formatting_instructions.tex` Appendix C
> (load-weight ablation). This document is the source of truth for the
> reward-hacking narrative.

## The phenomenon

When the GORGO weights are tuned to minimize a **TTFT-only** objective with an
**unconstrained** search range, the evolutionary search does not learn "good
routing" — it learns to **game the metric**. It discovers it can drive measured
p95 TTFT down by routing ~100% of traffic to the single nearest / fastest
replica, and zeroing out (or never using) the load term.

This is reward hacking in the precise sense: the optimizer exploits a gap
between the **measured reward** (negative p95 TTFT) and the **true objective**
(good end-to-end serving with balanced load). The gap exists because **TTFT does
not price queueing or decode contention until the replica saturates** — at low
and moderate load a single overloaded replica still returns its first token
quickly, so the proxy reward looks excellent while end-to-end latency and load
distribution silently collapse.

## Mechanism

```text
score(u) = rtt_weight · rtt_ms(u) + prefill_term(u) + load_weight · queued(u)
```

- The ES is free to push `rtt_weight` up (toward its ceiling) and/or `load_weight`
  toward 0.
- Both moves make the score dominated by "distance to the closest replica."
- Result: the policy collapses to **nearest-replica routing** → one replica gets
  ~100% of traffic.
- On light traffic this *wins* TTFT (warm cache, low RTT, queue not yet visible).
- On heavier traffic the chosen replica **saturates**: decode throughput craters,
  the queue explodes, and E2E (and eventually TTFT itself) blow up.

## Evidence: three independent instances

The same degeneration appears across **three different cost-model variants**,
which makes it a robust failure mode of single-objective TTFT routing rather than
a quirk of one configuration.

| Instance | Run | Degenerate weights | TTFT (the "win") | The hack | True-objective cost |
|---|---|---|---|---|---|
| **Load-weight = 0** | `glm5_c64_eval_p95ttft_diurnal_v2` | rtt=0.227, prefill=0.907, **load=0** | p50 190ms / **p95 1,101ms** (best) | 100% to one replica | **E2E p95 12.58s** (4.8× least-request); decode 70 tok/s |
| **Physical-rate iteration (May 29)** | `glm5_c64_tuning_p95ttft_v1` (removed) | **rtt=5.0 (ceiling)**, prefill=1.65 | **p50 141ms / p95 956ms** (best of all) | Ashburn **100%**, Frankfurt 0%, Seoul 0% | E2E 2.88s (worse than session-affinity 2.71s & least-request 2.56s) |
| **2D `queued_uncached` (apr5)** | prior to `glm5_c64_tuning_p95ttft_2d_v9` | **rtt=5.0**, queue=0.01 | — | RTT-greedy corner | **TTFT p95 10.06s, E2E 30.22s** — worst of 5 (saturated) |

## Concentration Audit: all ~100% runs (Apr 2 trace family)

Server-side sweep-matrix scan on the bench-results volume found four runs at
~100% single-replica routing concentration:

| Run | Policy | Requests | Concentration | Note |
|---|---|---:|---:|---|
| `glm5_c64_eval_000_glm5_apr2_0030_to_0100_gorgo-static` | `gorgo-static` | 2,096 | 100% | real eval run |
| `glm5_c64_eval_p95ttft_000_glm5_apr2_0100_to_0130_gorgo-static-p95` | `gorgo-static-p95` | 2,212 | 100% | real eval run |
| `glm5_c64_eval_p95ttft_000_glm5_apr2_1230_to_1300_gorgo-static-p95` | `gorgo-static-p95` | 3,077 | 100% | real eval run |
| `moon_smoke_001_00028_20260401T140000Z_token_hash_filter_top20_gorgo-tuned` | `gorgo-tuned` | 1,000 | 100% | smoke test (single policy; no next-best comparison) |

Improvement is computed as percent vs next-best policy on each metric
(`(next_best - gorgo)/next_best`, so positive is better for lower-is-better
latency metrics).

| Run (100% concentration) | Policy | TTFT p50 | TTFT p95 | TTFT p99 | E2E p50 | E2E p95 | E2E p99 | ITL avg |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Apr 2 00:30-01:00 eval (5 baselines) | `gorgo-static` | **+36.7%** | +16.9% | +3.7% | -24.3% | -35.9% | -37.9% | -47.3% |
| Apr 2 00:30-01:00 eval (3 baselines) | `gorgo-static` | +34.0% | +31.5% | +33.0% | n/a | -10.9% | -63.6% | n/a |
| Apr 2 01:00-01:30 eval | `gorgo-static-p95` | +34.1% | +8.7% | -2.7% | n/a | -14.4% | -44.5% | -35.0% |
| Apr 2 12:30-13:00 eval | `gorgo-static-p95` | +31.6% | +17.0% | -12.4% | -50.6% | -380.3% | -388.0% | -150.8% |

Standout p50 case (the one that looked "too good"): Apr 2 00:30-01:00 eval,
5-baseline matrix, where `gorgo-static` reaches TTFT p50 **125ms** (+36.7% vs
next-best) and TTFT p95 **762ms** (+16.9%), while losing badly on E2E and ITL.

### Instance 1 — the controlled ablation (clearest)

Midday diurnal trace, `gorgo-static`, unconstrained vs constrained load weight:

| Config | TTFT p50 | TTFT p95 | E2E p50 | E2E p95 | Decode tok/s | Concentration |
|---|---:|---:|---:|---:|---:|---:|
| `w_load = 0` (hacked) | **0.190s** | **1.101s** | 2.07s | 12.58s | 70 | **100%** |
| `w_load = 6.38` (fixed) | 0.195s | 1.136s | **1.29s** | **2.46s** | **119** | 60% |
| Δ | +2.6% | +3.2% | −37.7% | **−80.4%** | +70% | — |

Reading: constraining the load weight trades **3% TTFT p95** for **5× better E2E p95**
and balanced routing. The "hacked" policy only wins because the light nighttime
tuning window (~1.2 req/s) never saturates the single replica; on the heavier
midday window (~1.7 req/s) the hack self-destructs (E2E 12.58s).

### Instance 2 — independent confirmation (physical-rate model)

A different cost model (physical prefill rate + adjustable weights) reached the
*same* corner from a different direction: the ES drove `rtt_weight` to its 5.0
ceiling and routed 100% to the nearest replica (Ashburn). It posted the best TTFT
of any policy (p95 956ms) while losing on E2E — the hack is visible even when it
"wins."

### Instance 3 — recurs in the 2D model

Before the load term was fixed (raw `queued_tokens` + bounded
`rtt_weight/queue_weight` ratio), the 2D model on the apr5 window converged to
`rtt=5.0, queue=0.01` and collapsed to worst-of-5 (TTFT p95 10.06s, E2E 30.22s).
The fix (`rtt=0.276, queue=0.5`) recovered best-in-class E2E (8.91s).

## The fix (consistent across all three)

Constrain the search so the load signal cannot be removed:
- **Floor the load/queue weight** (e.g. `w_load ∈ [0.1, 10]` → 6.38; `queue_weight` floored).
- **Cap `rtt_weight`** so the RTT term cannot dominate the cache/load terms.
- Equivalently: bound the `rtt_weight / load_weight` ratio that sets the
  load-diversion threshold.

With the load term retained, the same ES finds a non-degenerate optimum that wins
**both** TTFT and E2E and keeps routing balanced.

## Why this is a strong result for the paper

1. **It motivates the core design choice** (the explicit, constrained load term) —
   it is the *reason* GORGO's cost model is shaped the way it is, not a footnote.
2. **It generalizes**: same failure across 3 cost-model variants ⇒ a claim about
   single-objective routing, not one config.
3. **It gives a clean home** for otherwise-negative runs (physical-rate iteration,
   2D degeneration) — they become evidence, not dead ends.
4. **It connects to the live work**: the same mechanism explains why high
   `rtt_weight` hurt on the apr7 eval and motivates the load sweep.

### Suggested incorporation
- **Promote** a 3–4 sentence version + the Instance-1 concentration table from
  Appendix C into the **main text** (end of §3 Cost Model, or a short §5
  subsection) as a motivating result.
- **Retitle** Appendix C: *"Reward Hacking: Single-Objective TTFT Concentrates
  Load,"* opening with the proxy-vs-true-objective gap.
- **Cite all three instances** as a recurring pattern; keep full tables/figures in
  the appendix.
