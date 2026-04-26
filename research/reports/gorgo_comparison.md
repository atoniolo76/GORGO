# Gorgo (main's deployed routing rule) vs rome's policy library

> Bead: go-4jm
>
> rome and main both call something "gorgo," but the two are different
> objects. On main, `utils/lb_aibrix.py:route_gorgo` is a single ~50-LOC
> scoring rule deployed as the production routing policy. On rome,
> "gorgo" names the whole simulator + 12-policy comparison harness.
> This report reconciles the two: we port main's `route_gorgo` into
> rome as `src/routing_harness/policies/gorgo.py` and benchmark it
> against the front-runners from §6 of `routing-comparison.md`
> (`prefix-cache-preble`, `pd-preble`, `least-kv-cache`) under
> identical simulated conditions.

## TL;DR

* Gorgo's scoring rule ports cleanly into rome — the only new state
  it requires is a per-pod `queued_prompt_tokens` counter (Σ
  prompt-token lengths of in-flight prefills), which we add to the
  engine alongside existing `active_prefill` / `pending_work_ms`
  bookkeeping.
* On the simulator's synthetic-potent workload (256 families × 1024-
  token shared heads, the §6 grid), gorgo is **substantially worse
  than the front-runners under load** at qps∈{4, 8} — by 10–14s of
  p95 — and **roughly tied or slightly better at qps∈{16, 32}**. The
  loss at low QPS is structural and traces to gorgo's lack of an
  exploit/explore gate (or any explicit hotspot mechanism): once one
  pod warms a prefix it monotonically wins all subsequent requests
  for that family because the queue term it relies on stays near
  zero between sequential arrivals at low QPS.
* On ShareGPT, the rankings invert above the saturation line but
  gorgo still loses meaningfully at low QPS (qps=4: 5,369ms vs
  ~1,605ms for the leaders).
* Hyperparameter sensitivity (3 × 3 grid on `t_prefill` ×
  `queued_tokens_weight` at qps=8, zipf=1.1) is very flat — p95
  varies by ~370ms across all 9 settings. The defaults are not
  pathological; the gap to prefix-cache-preble is structural, not a
  tuning issue.
* lmsys was in-scope per the bead but the dataset is not fetched on
  this worktree (see *Workload coverage* below); it is omitted from
  the comparison rather than reported with a fabricated stand-in.

## What gorgo is

The scoring rule, lifted from `origin/main:utils/lb_aibrix.py`:

```
score(u) = m.latency
         + max(0, request_tokens - cached_tokens) * t_prefill
         + (queued_tokens + m.num_used_tokens) * queued_tokens_weight
min wins
```

Three additive terms: a latency baseline, a soft prefix-affinity
penalty for the uncached tail, and a load penalty combining queued
prompt tokens and resident KV tokens. Two scalar hyperparameters
(`t_prefill`, `queued_tokens_weight`) are tuned online in production;
in this comparison we report the deployed defaults
(`t_prefill = 0.05`, `queued_tokens_weight = 0.001`) plus a
sensitivity grid.

### Signal mapping main → rome

| `route_gorgo` term         | rome simulator equivalent                           |
|----------------------------|------------------------------------------------------|
| `m.latency`                | `pod.ewma_latency_ms`                                |
| `request_tokens`           | `len(request.prompt_tokens)`                         |
| `cached_tokens`            | block-level prefix match × `block_size`              |
| `queued_tokens`            | **NEW** `pod.queued_prompt_tokens` (engine-tracked)  |
| `m.num_used_tokens`        | Σ `entry.token_count` over `KVCacheState.pods[pid]`  |

`queued_prompt_tokens` is the only new state the port required. We
chose to track the live signal in the engine (option a in the bead's
scope notes) rather than approximate it as
`active_prefill × mean_prompt_tokens` (option b) — the live signal is
what main reads from SGLang's `/metrics`, and the cost of carrying
one more counter is negligible. The engine increments on dispatch in
`_apply_side_effects` and decrements on retirement in `_retire_up_to`,
mirroring the existing `pending_work_ms` accounting (the symmetry is
verified by the engine accounting tests in
`tests/unit/test_engine_pending_work_accounting.py`, which still
pass).

## Workload coverage

| Workload     | Status   | Notes                                              |
|--------------|----------|-----------------------------------------------------|
| synthetic    | ✓        | `example_run.yaml` potent config, 4 pods × 4 GiB    |
| sharegpt     | ✓        | 1,000 conversations, mock tokenizer (no tiktoken)   |
| lmsys        | skipped  | `data/lmsys/lmsys-chat.jsonl` is not present on this worktree (gated download, requires `HF_TOKEN`). Re-running the sweep here once the dataset lands is a one-config-flip follow-up; no policy or simulator change needed. |
| code\_completion | not run | Out of scope per the bead (top-3 in §6 are prefix-cache-preble, pd-preble, least-kv-cache; lmsys was the explicit add). |

The synthetic and sharegpt sweeps cover the same 4-QPS × 3-zipf grid
the §6 colocated sweep uses (zipf has no effect on sharegpt's real
turn-mix, but the grid is preserved for symmetry); single seed per
the bead's compute-throttle directive.

## Synthetic-potent: median p95 (ms) over 3 zipf values, single seed

4 pods × 4 GiB KV; 256 prefix families × 1024-token shared heads;
2,000 requests/run. Numbers are medians across `zipf_s ∈
{0.7, 1.1, 1.5}`.

| Policy                 | qps=4   | qps=8   | qps=16  | qps=32  | hit_rate | skew  |
|------------------------|---------|---------|---------|---------|----------|-------|
| gorgo                  | 14,313  | 14,985  | 15,481  | 15,377  | 0.727    | 0.994 |
| least-kv-cache         | 940     | 12,505  | 15,337  | 15,257  | 0.729    | 0.107 |
| pd-preble              | 944     | 1,038   | 15,665  | 15,697  | 0.669    | 0.053 |
| prefix-cache-preble    | 938     | 1,007   | 15,041  | 15,041  | 0.749    | 0.121 |

**Reading the table.**

* At qps=4 / 8 (under-loaded), the front-runners hold p95 near 1s
  while gorgo collapses to ~14–15s — saturation. The skew column
  confirms why: gorgo's load distribution is `0.99` (one pod is
  doing essentially everything) versus 0.05–0.12 for the others.
  The mechanism is straightforward in the simulator: at low QPS
  every request retires before the next arrives, so
  `queued_prompt_tokens` drops back to zero between dispatches and
  the queue term contributes nothing. With latencies tied (all pods
  near `initial_warm_latency_ms`) and the prefix term favoring
  whichever pod warmed the family first, gorgo keeps re-electing the
  same pod and a positive feedback loop locks the family to that
  pod. There is no exploit/explore gate to break the cycle the way
  prefix-cache-preble's `missed < cached` test does.
* At qps=16 / 32 (saturated), all four policies cluster around
  ~15s p95 because the cluster is the bottleneck regardless of
  routing. Gorgo is comparable to or slightly better than pd-preble
  (gorgo wins by 184–544ms across the qps≥16 cells; see the
  leadership matrix below) and slightly worse than
  prefix-cache-preble (216–616ms). hit_rate is essentially tied
  with prefix-cache-preble (0.727 vs 0.749) — gorgo is *not* failing
  to capture prefix reuse, it is failing to balance load.

### Leadership matrix (synthetic)

Gorgo p95 minus baseline p95 in ms; negative = gorgo wins.

`gorgo - prefix-cache-preble`:

| zipf \ qps | qps=4   | qps=8   | qps=16 | qps=32 |
|------------|---------|---------|--------|--------|
| 0.7        | +14,832 | +14,860 | +216   | +232   |
| 1.1        | +13,374 | +13,978 | +440   | +336   |
| 1.5        | +10,650 | +11,254 | +616   | +112   |

`gorgo - pd-preble`:

| zipf \ qps | qps=4   | qps=8   | qps=16 | qps=32 |
|------------|---------|---------|--------|--------|
| 0.7        | +14,818 | +14,834 | -224   | -272   |
| 1.1        | +13,369 | +13,946 | -184   | -320   |
| 1.5        | +10,641 | +11,243 | -296   | -544   |

`gorgo - least-kv-cache`:

| zipf \ qps | qps=4   | qps=8   | qps=16 | qps=32 |
|------------|---------|---------|--------|--------|
| 0.7        | +14,826 | +14,842 | +104   | +104   |
| 1.1        | +13,373 | -208    | +144   | +120   |
| 1.5        | +10,633 | -280    | +24    | -248   |

## ShareGPT: median p95 (ms), single seed

1,000 conversations × 16 max turns; mock tokenizer
(`tokens_per_char=0.25`); same 4-pod / 4 GiB topology.

| Policy                 | qps=4   | qps=8   | qps=16  | qps=32  | hit_rate | skew  |
|------------------------|---------|---------|---------|---------|----------|-------|
| gorgo                  | 5,369   | 5,393   | 5,425   | 5,449   | 0.014    | 1.268 |
| least-kv-cache         | 2,617   | 4,897   | 5,393   | 5,449   | 0.017    | 0.381 |
| pd-preble              | 1,605   | 5,153   | 5,497   | 5,497   | 0.015    | 0.138 |
| prefix-cache-preble    | 1,604   | 5,209   | 5,497   | 5,497   | 0.017    | 0.129 |

**Reading the table.** Same story compressed: gorgo is decisively
worse at qps=4 (3.7s gap to leaders) and tied or slightly better at
saturation. The skew (1.27) again signals load concentration, not a
prefix-cache miss. ShareGPT's hit_rate floor is much lower than
synthetic's (~0.015 vs 0.73) because ShareGPT does not have the
synthetic config's 1024-token deterministic shared head — most reuse
opportunity is intra-session conversation history, which is small per
session. The relative rankings are unchanged from synthetic.

## Hyperparameter sensitivity (synthetic, qps=8, zipf=1.1)

Median p95 (ms) over a 3 × 3 grid:

| t_prefill \ qtw | qtw=0.0001 | qtw=0.001 | qtw=0.01 |
|-----------------|------------|-----------|----------|
| 0.01            | 15,137     | 15,105    | 14,921   |
| 0.05 (default)  | 15,017     | 14,985    | 15,289   |
| 0.2             | 15,105     | 15,121    | 14,985   |

**Reading the table.** The grid is essentially flat — p95 ranges
14,921–15,289ms, a 368ms spread across two-orders-of-magnitude
sweeps in both knobs. None of the corners closes the ~14s gap to
prefix-cache-preble at this QPS. Changing `t_prefill` alone (with
`queued_tokens_weight=0`) reduces gorgo to "latency + prefix
affinity," which is structurally similar to plain `prefix-cache`
(which the §6 sweep already showed under-performs preble); changing
`queued_tokens_weight` alone (with `t_prefill=0`) drops the prefix
bias and turns gorgo into a near-`least-kv-cache` variant. Neither
substitution is going to outperform the gated, hotspot-aware
front-runners on this workload.

This finding is consistent with main's online tuning: tuning
`t_prefill` and `queued_tokens_weight` is meaningful when the live
signals genuinely change between configurations (a busy production
cluster generates non-zero queued_tokens during normal operation).
In a deterministic sequential simulator, those signals are diluted
because there is no overlap between requests at low QPS — the
hyperparameters cannot rescue gorgo from the structural problem.

## Where gorgo wins, where it loses, where it matches

* **Wins** — qps=16 / 32 vs `pd-preble` on synthetic (184–544ms), and
  scattered cells against `least-kv-cache` at qps=8/zipf≥1.1 and
  qps=32/zipf=1.5 (200–280ms). At saturation, gorgo's composite
  signal is mildly better than pd-preble's role-aware rule on a
  colocated topology where the role split is moot.
* **Loses** — qps=4 / 8 across both workloads vs all three
  front-runners (10–14s p95 on synthetic, 3.7s on sharegpt). The
  loss is structural: no exploit/explore gate, no hotspot
  redirect, and the queue term that should deflect is empty at low
  QPS.
* **Matches** — hit_rate is competitive across the grid (0.727 vs
  prefix-cache-preble's 0.749 on synthetic). Gorgo captures prefix
  reuse correctly; the failure is on the load-balancing side.

## What this comparison can and cannot say

This is a simulator comparison. It uses rome's analytic cost model,
which has been calibrated against measured per-token coefficients on
A100 (`research/data/calibration/`, see `go-h3r` lineage report) but
is not the production execution engine main is deployed against.
Three caveats:

1. **The arrival model is sequential.** The simulator processes
   requests in arrival order without true overlapping execution.
   Production traffic interleaves: `queued_tokens` accumulates as
   new requests arrive while existing ones are still in prefill.
   That signal is what gorgo's `queued_tokens_weight` is calibrated
   for. The simulator's discrete arrival model dilutes this signal
   at low QPS, which is precisely the regime where gorgo
   under-performs in this comparison. This does not mean gorgo is
   bad in production; it means the simulator under-rewards gorgo's
   load-deflection mechanism at light load.
2. **Online tuning is not modeled.** Main tunes `t_prefill` and
   `queued_tokens_weight` against live latency observations. The
   sensitivity grid here is a static cut, not the trajectory main's
   tuner would follow. The flatness of that grid (§Hyperparameter
   sensitivity) is suggestive — there isn't an obvious better
   point — but it is not a substitute for the live tuner.
3. **Use main's deployment for ground truth.** The simulator is a
   ranking tool for policy *design*, useful for asking "would this
   structural change help?" Per-policy absolute numbers should not
   be quoted as performance predictions for production.

The takeaway from this comparison is design-level, not operational:
**gorgo's scoring rule lacks the explicit hotspot / exploit-explore
gating that the simulator's front-runners use, and it pays for that
absence at light load.** A constructive follow-up would be to add a
hotspot deflection rule to the gorgo policy (e.g., redirect to
lightest pod when the best-scored pod's `pending_work_ms` exceeds a
ratio of the lightest's, mirroring `prefix-cache-preble`'s
`th_bal=1.5` mechanism) and re-run the comparison. That change is
out of scope for this bead.

## Reproduction

```bash
# Sweep configs land in research/data/gorgo_comparison/<workload>/.
uv run python -m routing_harness.cli sweep \
    --config configs/gorgo_comparison_synthetic.yaml > /tmp/synth.json
uv run python -m routing_harness.cli sweep \
    --config configs/gorgo_comparison_sharegpt.yaml > /tmp/sharegpt.json
uv run python -m routing_harness.cli sweep \
    --config configs/gorgo_hyperparam_grid.yaml > /tmp/hp.json

# Aggregate into the tables in this report:
uv run python scripts/aggregate_gorgo_sweep.py
```

ShareGPT requires `data/sharegpt/sharegpt.jsonl` (10k conversations,
gitignored). To bring lmsys into this comparison, fetch
`data/lmsys/lmsys-chat.jsonl` (gated; `HF_TOKEN` required), add a
`gorgo_run_lmsys.yaml` mirroring the sharegpt config, and rerun.
