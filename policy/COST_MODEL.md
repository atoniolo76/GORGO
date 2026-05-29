# GORGO Cost Model: Design Rationale

Why the cost model is shaped this way. Companion to `POLICIES.md` (the *what*).
Numbers are from the GLM5 c=64 eval (`Qwen3.5-35B-A3B-FP8`, 2×L40S, 3 regions, open-loop c=64).

The signals come from the inference engine: SGLang's RadixAttention prefix
cache and Prometheus metrics [`sglang2024`], on top of paged KV-cache memory
management [`vllm2023`]. The two findings that shape the model below — that
queued prefill drains *slower* and *overlaps* with decode (§4), and that KV
occupancy is a nonlinear pressure signal (§3) — are direct consequences of
**continuous batching** [`orca2022`] and **chunked prefill** [`sarathi2024`].
See [References](#references).

## The cost function

```
score(u) = rtt_weight     × rtt_ms(u)
         + prefill_weight × ( prefill_rate(u) × uncached_tokens
                            + queue_rate(u)   × queued_tokens(u) )
```

Two cost sources: network RTT, and prefill work ahead of the request (its own
uncached tokens + the backlog queued on the replica). No contention term (§3).

Each term is `weight × rate × count`: **rate = physics, weight = preference.**
At `weight = 1.0` the score is a literal predicted delay; a learned `weight ≠ 1`
means "this term matters more/less than its physical magnitude" — not a fudge
factor. This only holds if `rate × count` is physically honest, which requires:

## 1. Units must agree before weighting

Every term is summed, so all must be in the same unit (ms). Two original
mismatches, both distorting what the tuner learned:

- **RTT in seconds vs. token counts.** RTT (0.02–0.6) was ~10,000× smaller than
  token terms (0–45k), so the tuner drove `rtt_weight` to 4132 to compensate.
  Result: gorgo-static routed **100% to the nearest region**, never considering
  load — worst E2E (p95 2.9s vs 2.1s) and decode (94 vs 127 tok/s). Fix: RTT →
  ms (`×1000`), cap `rtt_weight` to `[1e-5, 5]`.
- **Calibrator emitted seconds/token** while scoring expected ms/token (1000×
  off). Fix: `recommend_rates` converts at the one point rates become routing
  params.

Lesson: a unit mismatch never announces itself — it's absorbed into whatever
knob the tuner can move, yielding a degenerate policy that still "wins" the metric.

## 2. Calibrate idle; fit what you can't

`prefill_rate` is a hardware constant — the per-token prefill cost the engine
pays for uncached tokens — measured cleanly on an **idle** replica where
`TTFT = network + prefill`:

```
prefill_rate = (TTFT − RTT) / uncached_tokens     # ~0.06 ms/tok on L40S
```

`proxy/calibrate.py` does ~32 sequential cache-flushed probes, takes the median.
Idle calibration beats live discovery for experiments: no convergence window
(a cold default biases the first 5–10 min of an eval), reproducible, and it
turns the auto-tuner's job from *discovery* into *drift tracking*.

## 3. No contention term

Dropped `num_used_tokens` (KV occupancy) from the model:

- `corr(queued_tokens, num_used_tokens) = −0.035` — not a redundant load
  signal; it's a 30s-stale step function (50 distinct values / 2,095 requests).
- Non-predictive: TTFT slope ~0.0016 ms/tok; ITL R² = 0.002.
- The saturation signal it would guard (`sglang:utilization`) reported 0.000
  and the pool never neared capacity.

Conceptually it never belonged in a `rate × count` term: `num_used_tokens` is
SGLang's KV-pool occupancy [`sglang2024`, `vllm2023`] — a *stock* (how full
now), not a *flow* (drainable work). Multiplying a stock by a per-token rate
doesn't yield a wait time. Its real effect is nonlinear (batch contention +
saturation cliff), not a linear additive term. Load balancing instead falls out
of `queued_tokens` — a real-time counter that rises on dispatch, falls on
completion, and self-limits pile-on.

> If a future high-load regime nears KV saturation, re-add it as a nonlinear
> barrier `f(utilization)` that *multiplies* the rates — not a linear term — and
> only once `utilization` is actually populated.

## 4. Two prefill rates: the batching discount

`queued_tokens` is prefill work, but must not share `prefill_rate`. Regressing
TTFT against each count separately:

| Count | Slope (ms/tok) |
| --- | --- |
| own uncached tokens | 0.063 (full serial cost) |
| queued tokens | 0.010 (~6× cheaper) |

Under continuous batching [`orca2022`], queued work doesn't serialize ahead of
you: chunked prefill [`sarathi2024`] interleaves prefill chunks with ongoing
decode and batches concurrent prefills, so a queued token adds far less to your
first-token latency than one of your own tokens does. A single rate
overestimates queue cost ~6× and over-avoids mildly loaded replicas (20k queued
→ scored 1223 ms when true cost was ~223 ms). So:

- `prefill_rate` — own tokens; calibrated idle.
- `queue_rate` — queued tokens; **can't** be measured idle (no queue) and is fit
  from the residual:

```
queue_rate = (TTFT − RTT − prefill_rate × uncached) / queued_tokens   # ~0.01 ms/tok @ c=64
```

`_fit_queue_rate` does a robust ratio-of-medians, returning `None` (keep prior)
below 10 qualifying samples.

## 5. Tuning: measured vs. searched, decoupled

| Parameter | Role | Mechanism |
| --- | --- | --- |
| `prefill_rate` | physics | idle calibration, pinned |
| `queue_rate` | physics | residual fit, continuous, per-replica |
| `rtt_weight`, `prefill_weight` | preference | online-ES (2D), continuous, global |

The search is a Gaussian (1+1)-ES with Rechenberg's 1/5 success rule
[`rechenberg1973`] — fast and surrogate-free in the 2D weight space.

**Hillclimb (ES) and rate-fitting don't interfere**, by construction:

- `prefill_rate` is pinned — a fixed reference, not a moving part.
- `queue_rate` is fit from a residual that is **orthogonal to the weights** (it
  uses the measured prefill rate and observed counts, not `rtt_weight` /
  `prefill_weight`). So ES weight moves don't change what `queue_rate` fits to.
- The ES therefore sees a **stable cost surface** and its score deltas are
  attributable to its own moves — the 1/5 step-size rule adapts correctly
  instead of chasing a moving target.

Rates are measured, weights are searched, and the two are decoupled so both can
run continuously.

## 6. Three-phase rollout

- **Phase 0 — calibrate `prefill_rate`** (idle, on a *separate* data window),
  POST per-replica, pin for the run.
- **Phase 1 — tune** at target concurrency: each hop fits `queue_rate` (residual)
  *and* ES-searches `(prefill_weight, rtt_weight)` against the objective.
  Extract converged weights.
- **Phase 2 — evaluate** on held-out traces: re-calibrate `prefill_rate` for the
  new fleet, freeze Phase 1 weights, fit `queue_rate` online, run baselines in
  parallel.

The ES never touches the rates; rates are measurements, not search targets.

## Principles

1. Reduce every term to one unit before weighting — mismatches hide in the tuner.
2. Separate physics (rates) from preference (weights); weights default to 1.0.
3. Only put drainable *work* in a `rate × count` term (queued yes, occupancy no).
4. One rate per distinct physical process (own-prefill ≠ queued-prefill, ~6×).
5. Calibrate idle what you can; fit from residuals what you can't.
6. Decouple search from fitting so the ES sees a stable surface.

## References

Keys match the paper's `references.bib` (`formatting_instructions.tex`).
`orca2022` and `sarathi2024` are additions specific to this design note — they
are the mechanism behind the queued-prefill discount (§4) and are worth adding
to the bib if cited in the paper.

- `sglang2024` — Zheng et al. *SGLang: Efficient Execution of Structured Language Model Programs.* NeurIPS 2024. (RadixAttention prefix cache; `num_used_tokens`/`utilization` metrics.)
- `vllm2023` — Kwon et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP 2023. (Paged KV-cache; occupancy as memory pressure.)
- `orca2022` — Yu et al. *Orca: A Distributed Serving System for Transformer-Based Generative Models.* OSDI 2022. (Iteration-level / continuous batching — why queued work overlaps decode.)
- `sarathi2024` — Agrawal et al. *Taming Throughput–Latency Tradeoff in LLM Inference with Sarathi-Serve.* OSDI 2024. (Chunked prefill / stall-free batching — prefill chunks interleave with decode, the ~6× queued-prefill discount.)
- `qin2024mooncake` — Qin et al. *Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving.* 2024. (KV-cache-centric serving; trace format.)
- `preble` — Srivatsa et al. *Preble: Efficient Distributed Prompt Scheduling for LLM Serving.* 2024. (Longest-prefix-match routing baseline.)
- `aibrix2024` — AIBrix Team. *AIBrix: Scalable, Cost-Effective LLM Inference Infrastructure.* 2024. (Production prefix-cache router baseline.)
- `distserve2024` — Zhong et al. *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving.* OSDI 2024. (PD disaggregation — would change the rate decomposition.)
- `splitwise2024` — Patel et al. *Splitwise: Efficient Generative LLM Inference Using Phase Splitting.* ISCA 2024. (PD disaggregation.)
- `rechenberg1973` — Rechenberg. *Evolutionsstrategie: Optimierung technischer Systeme nach Prinzipien der biologischen Evolution.* 1973. ((1+1)-ES, 1/5 success rule.)
