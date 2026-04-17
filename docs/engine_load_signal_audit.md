# Engine-side Load-Signal Audit (go-s13)

Author: gorgo/polecats/guzzle
Date: 2026-04-15
Scope: audit only, no fixes committed. Independent investigation of scout's
hypothesis that `PodRuntime.queued` is dead state. Files examined:
`scout/src/routing_harness/simulator/engine.py`,
`scout/src/routing_harness/core.py`,
`scout/src/routing_harness/cost_model.py`, and every policy under
`scout/src/routing_harness/policies/`.

## TL;DR

1. **Scout's hypothesis is confirmed, mechanically.** `PodRuntime.queued` has
   zero writers in production code; it stays 0 for the entire simulation. Seven
   policies read it.
2. **But that is not why `prefix-cache-preble` fails to deflect.** Under
   qps=4, zipf=1.5 on the example 4-pod topology, the load signal without
   `queued` already exceeds preble's 0.9 threshold in **26.5% of decide
   calls** (530/2000). The hotspot branch fires; deflection still doesn't
   happen because `match > 0` never holds on any alternative pod.
3. **The binding bug in preble is the `match > 0` gate** on deflection
   candidates, compounded by prefix-greedy routing having already mono-homed
   the hot prefix onto one pod. Zero of the 530 over-threshold calls had any
   alternative pod with prefix overlap; all 530 had an alternative with
   slack (load < 0.9).
4. **Two engine-side issues compound the problem** independently of
   `queued`: (i) no admission enforcement — `max_concurrent_prefill` is only a
   cost-model parameter, so `active_prefill` grows unbounded (observed 40+
   on a single pod); (ii) on role=BOTH pods, each in-flight request counts
   once in `active_prefill` and once in `active_decode`, so the load
   denominator is double-counted against a capacity that sums the two.

## 1. PodRuntime.queued end-to-end

`PodRuntime.queued` is defined in `core.py:88` with default `0`. Across
`scout/src/routing_harness/`:

```
$ rg '\bqueued\b' scout/src/routing_harness/
core.py:88:    queued: int = 0
policies/session_affinity.py:48:    key=lambda p: (p.active_prefill + p.active_decode + p.queued, p.spec.pod_id),
policies/vtc_basic.py:70:        return p.ewma_latency_ms * (p.active_prefill + p.active_decode + p.queued)
policies/least_busy_time.py:3 (docstring)
policies/least_busy_time.py:36: return p.ewma_latency_ms * (p.active_prefill + p.active_decode + p.queued)
policies/prefix_cache_preble.py:10 (docstring)
policies/prefix_cache_preble.py:71:    load = (p.active_prefill + p.active_decode + p.queued) / cap
policies/least_request.py:35: key=lambda p: (p.active_prefill + p.active_decode + p.queued, p.spec.pod_id)
policies/least_request.py:40-41 (rationale + score)
policies/pd.py:62: return p.ewma_latency_ms * (p.active_decode + p.queued)
policies/prefix_cache.py:64: key=lambda p: (p.active_prefill + p.queued, p.spec.pod_id)
```

Writers in production code: **none**. The only write is in the test fixture
`tests/unit/test_policies_individual.py:55` (`cluster.pods["p0"].queued = 100`),
which exists exactly to exercise the read path.

The simulator increments `active_prefill` / `active_decode` on decision
(`engine.py:253-254`) and decrements them in `_retire_up_to` (`engine.py:95-99`).
There is no admission-queue stage between arrival and dispatch in the
engine, so there is no natural writer for `queued`.

**Policies that read `queued`** (count matches scout's list): `session_affinity`,
`vtc_basic`, `least_busy_time`, `prefix_cache_preble`, `least_request`, `pd`,
`prefix_cache`. Seven policies treat a permanently-zero field as part of
their load signal.

The practical effect is that every `+ p.queued` term is a no-op: the policies
rank on `active_prefill + active_decode` alone. This is not wrong, just
misleading documentation — the signal does not include the queued dimension
those policies claim to use.

## 2. Empirical load-signal distribution (qps=4, zipf=1.5, prefix-cache-preble)

Throwaway instrumentation at `/tmp/go-s13-instrument.py` (not committed)
wraps `PreblePrefixCachePolicy.decide` and logs, per candidate pod per
decide call: `(pod_id, active_prefill, active_decode, queued, cap, load,
match)`. Topology, compute, and network params match
`scout/configs/example_run.yaml`. Workload: `SyntheticParams(n_requests=2000,
arrival_rate_qps=4.0, zipf_s=1.5, n_prefix_families=64,
shared_prefix_tokens=512, seed=0)`. Single seed.

Run output:

```
rows: 8000                         # 4 pods × 2000 decide calls
load min/mean/max: 0.0000 / 0.3473 / 2.2000
active_prefill+active_decode min/mean/max: 0 / 6.95 / 44
queued: min=0 max=0 (unique values: [0])
load > 0.9 observations: 1284 of 8000 (16.05%)
p50/p95/p99 latency_ms: 942.8 / 12168.7 / 13240.7
hotspot-avoid deflections: 0 of 2000

decide calls: 2000
  top_load > 0.9: 530 (26.50%)
  ...AND any alt has match>0: 0
  ...AND any alt has match>0 AND load<0.9: 0
  ...AND any alt has load<0.9 (any match): 530
```

Reading this:

- **`queued` is 0 for every one of the 8000 pod-observations.** Scout's
  hypothesis confirmed at runtime.
- **The signal already reaches preble's 0.9 threshold anyway.** In 26.5% of
  decide calls (530/2000), the top-scoring pod has `load > 0.9`. The hotspot
  branch in `prefix_cache_preble.py:76` *does* execute. Fixing `queued`
  would only make this fire *more* often, not change the outcome.
- **Zero of those 530 have an alternative pod with `match > 0`.** The
  deflection loop at `prefix_cache_preble.py:77-84` requires
  `match > 0 AND load < self.hotspot_threshold` on the alternative. The
  `match > 0` predicate never holds: under zipf-1.5 skew, prefix-greedy
  scoring (`alpha * match - beta * load`) routes the hot family to one pod
  consistently, so cached prefix blocks for hot families exist on exactly
  one pod. Alternatives have `match = 0`.
- **All 530 do have an alternative with slack.** Load heterogeneity is real;
  it's just that preble refuses to use it because the slack pods don't hold
  the prefix.

**Does the load signal ever reach 0.9?** Yes, frequently (26.5% of calls).
**Does preble still fail to deflect?** Yes, at 100% rate — because the
deflection criterion requires a property (prefix match on the alternative)
that the upstream scoring has systematically eliminated.

This is the answer to the bead's conditional: "If yes but preble still does
not deflect, the bug is elsewhere." The bug is *elsewhere*: it is the
`match > 0` gate on the deflection candidate, interacting with the
prefix-greedy primary score.

## 3. Request-lifecycle retirement and load-signal accuracy

The engine's request-lifecycle model (from `engine.py:104-228`):

1. Advance `now = max(now, req.arrival_ts)`.
2. `_retire_up_to(now)` — pop every pending completion whose scheduled
   retirement ≤ `now`; decrement `active_prefill` / `active_decode`.
3. `policy.decide(...)` — policy sees post-retirement counters.
4. `cost_model.estimate(...)` — computes `queueing_ms` via M/M/1 on
   `active_prefill / max_concurrent_prefill`.
5. `_apply_side_effects(...)` — increment `active_prefill`, `active_decode`;
   schedule retirement at `now + cost.total_ms / 1000`.

This is *not* the "instantaneous completion" mistake the bead asks about.
The `active_*` counters hold requests in flight for their full end-to-end
latency (`cost.total_ms` includes `queueing_ms`), and `_retire_up_to` only
drains entries whose scheduled completion is past. Hence `active_prefill`
on the hot pod grows to 40+ under load; it does not collapse to zero
between arrivals.

But two real problems lurk here, both independent of `queued`:

**(a) Admission is unbounded.** `max_concurrent_prefill` is not enforced
as an admission limit — it is only the denominator of the M/M/1 utilization
in `cost_model._mm1_wait_ms`, clamped at ρ=0.99 (`cost_model.py:124`). So
`active_prefill` can exceed capacity arbitrarily, which is what we observe
(44 > 4). The policy's load signal is still monotone in true pressure, but
the *meaning* of `load > 1.0` is "this pod is over-subscribed" with no
mechanism to back-pressure. Any load-aware policy that uses
`(active / cap)` is interpreting an unbounded numerator against a fixed
denominator.

**(b) Role=BOTH double-counts.** On a pod with `role=BOTH`, a single
in-flight request sits in both `active_prefill` and `active_decode` for
its full lifetime (`engine.py:253-254`: `pod.active_prefill += 1;
decode_pod.active_decode += 1`, where `decode_pod == pod` when
colocated). Preble's denominator sums the two capacities
(`max_concurrent_prefill + max_concurrent_decode`, 4 + 16 = 20 in the
example), but the numerator double-counts every colocated request. With
22 real requests on a pod, `active_prefill + active_decode = 44` and
`load = 44/20 = 2.2`. Ranking across pods is preserved (all pods
double-count symmetrically), but the 0.9 threshold was presumably
calibrated against a non-double-counted signal and is effectively
~0.45 in practice.

**Feedback interaction.** EWMA of observed latency on the hot pod
(`_apply_side_effects`, `engine.py:245`) rises as load rises, which inflates
the cost model's predicted `queueing_ms` for subsequent requests, which
inflates their `cost.total_ms`, which pushes their scheduled retirement
further into the future, keeping `active_*` higher. The system is
self-reinforcing on hot pods. This is probably intended behavior for
realism, but it means the load signal has memory beyond the current set
of in-flight requests.

## 4. Candidate fixes, ranked by invasiveness

### (a) Write `PodRuntime.queued` on dispatch / completion — minimal

Surface change: in `engine.py:253-254`, increment `pod.queued += 1` on
dispatch; in `engine.py:95-99`, decrement on retirement. (Or redefine
`queued` = `active_prefill + active_decode` and drop the `+ queued` terms
everywhere, which is semantically the same.)

**Effect on preble in the measured regime**: none useful. The 26.5% of
calls where `top_load > 0.9` would become more frequent (because `queued`
would just add itself to `active_*` once more, roughly doubling the
numerator), but the deflection is blocked by the `match > 0` gate, not by
the threshold being too low.

**Why it's still worth doing**: (i) the field is documented as semantically
meaningful; the gap between docstring and behavior is a correctness bug
in the data model, not just an observability one; (ii) it removes a trap
for future policy authors who believe the readme.

**Invasiveness**: ~6 lines. No schema change. Zero effect on metrics in
regimes where the match-gate dominates.

### (b) Expose a separate `effective_queueing_ms` (or M/M/1 wait) on `ClusterState` — moderate

Today, `cost_model._mm1_wait_ms` is computed at request time and thrown
away into `CostBreakdown.queueing_ms` — policies cannot see it before
deciding. Exposing a cached per-pod `ewma_queueing_ms` (updated each
completion) or a live `projected_wait_ms(pod, service_ms)` on
`ClusterState` would let load-aware policies score on *expected latency
contribution*, not just *counter sum*.

**Effect on preble**: still blocked by the `match > 0` gate in the
measured regime, but preble's `load` formula would become a genuine
latency proxy (not a double-counted capacity ratio), and the
hotspot-threshold could be calibrated meaningfully. More importantly,
`least_busy_time` and `pd` (which use `ewma_latency_ms * active`) would
get a more honest estimate.

**Invasiveness**: new method on `ClusterState` + one new EWMA in
`_apply_side_effects` + plumbing in cost model. Touches roughly 4 files;
preserves policy contract (read-only `ClusterState`).

### (c) Surface the M/M/1 wait estimate as a first-class field on `PodRuntime` — most invasive

A per-pod `projected_queueing_ms` field, maintained by the engine on every
dispatch/retirement, indexed by a representative service time. Lets
policies score directly on predicted latency. Effectively moves part of
`cost_model.estimate` into the engine's per-pod state so policies can
read it without invoking the cost model.

**Effect on preble**: changes the composite score from
`alpha * match - beta * load_ratio` to
`alpha * match - beta * projected_queueing_ms`, which is a meaningful
latency trade-off. Still blocked by `match > 0`, but would make
`prefix_cache_preble` behavior in non-mono-homed regimes more sensible.

**Invasiveness**: schema change on `PodRuntime`; touches engine, cost
model, and every policy that reads `queued` today. ~10 files.

### Orthogonal: fix the `match > 0` gate in preble itself

Not part of the engine-side audit scope, but noted for completeness.
Under the measured regime, fixes (a)-(c) all leave preble's deflection
blocked. The direct fix is in `prefix_cache_preble.py:77-84`: relax
`match > 0` on the deflection candidate, or keep it but add a second
deflection tier (`match == 0` alternative when `top_load` exceeds a
harder threshold like 1.5 — i.e. when the hot pod is genuinely saturated
and losing the prefix affinity is a net win). This is an algorithmic
choice, not an engine change.

### Orthogonal: role=BOTH double-counting

In every `queued`-reading policy, swap
`(p.active_prefill + p.active_decode + p.queued) / cap` for something
that does not double-count colocated requests. Either count distinct
in-flight requests, or split `cap` so prefill and decode load are
evaluated against their own capacities and combined. Small policy-side
change; no engine change required.

### Orthogonal: admission enforcement

If the simulator is meant to model real behavior under saturation, the
engine should enforce `max_concurrent_prefill` by deferring dispatch
rather than letting `active_prefill` run unbounded. This is a bigger
change (needs an actual admission queue, and the `queued` field would
finally be the natural place for it to live — which closes the loop
with fix (a)).

## 5. Cross-policy benefit of each fix

| Fix | preble | prefix_cache | least_request | session_affinity | least_busy_time | vtc_basic | pd |
|---|---|---|---|---|---|---|---|
| (a) write `queued` | negligible (blocked by match-gate) | mild — tie-break disambiguation on hot pod | mild — same | mild | moderate — `ewma_latency * (active + queued)` becomes sensitive to admission pressure if (a) is paired with an admission queue | moderate (same reason) | moderate |
| (b) expose queueing estimate | moderate — hotspot threshold becomes latency-calibrated | low | low | low | **high** — its core signal is busy-time, which is `latency * active`; a projected queueing term makes the signal predictive instead of retrospective | **high** — fairness-weighted cost is latency-shaped | **high** — splits load across prefill/decode legs |
| (c) first-class `projected_queueing_ms` on PodRuntime | high *in non-mono-homed regimes* — load signal is latency instead of counter-ratio | low | moderate | low | high | high | high |

Notes:

- **`prefix_cache`** does not show `active_decode` in its score (only
  `active_prefill + queued`, `policies/prefix_cache.py:64`). It already
  avoids the role=BOTH double-count. Fix (a) would give it the
  admission-pressure sensitivity the others get from `queued`.
- **`session_affinity`** uses `queued` as a load tiebreaker among
  prefill-capable pods that don't already own the session. Fix (a) makes
  the tiebreaker meaningful.
- **`pd`** uses `(active_decode + queued)` on decode pods. Under
  PD-disaggregation, decode pods are not role=BOTH, so no double-count,
  but `queued` is still dead and fix (a) would make decode pressure
  visible. `pd` is probably the cleanest case where (b)/(c) give the
  biggest win: predicting queueing on the decode leg separately is the
  whole point of PD.
- **`least_busy_time` and `vtc_basic`** multiply `ewma_latency_ms * (active
  + queued)`. Without admission enforcement, `active` grows without
  bound and `ewma_latency` rises with it, so the product is already
  hot-pod-avoidant in practice. Fixes (b)/(c) make the signal *predictive*
  (what latency will this request see?) rather than *retrospective*
  (how bad has this pod been?), which matters when a pod is transitioning
  between hot and cold.

## Appendix A — instrumentation provenance

Throwaway script: `/tmp/go-s13-instrument.py` (not committed). It imports
`scout/src/routing_harness` via `PYTHONPATH=src`, instantiates the
example topology + cost/network/scheduler params from
`configs/example_run.yaml`, generates a 2000-request synthetic trace at
qps=4 zipf=1.5 seed=0, wraps `PreblePrefixCachePolicy.decide` to log
per-pod observations to `/tmp/go-s13-decide-log.csv`, and prints summary
stats. Summary reproduces deterministically given the same scout commit;
no engine-side files were modified or committed.

## Appendix B — caveats

- **Single seed, single workload point.** The bead asked for one synthetic
  sweep point; this audit delivers one (seed=0). Across seeds the
  26.5%/0/530 figures will vary in magnitude, but the structural finding
  (0 alternatives with match>0 under mono-homing) is invariant given the
  prefix-greedy primary score.
- **Synthetic shared-head workload.** Real workloads (lmsys, sharegpt)
  have multi-turn session structure that can distribute hot prefixes
  across multiple pods via `session_affinity`-style decisions. The
  mono-homing observation is specific to stateless prefix-greedy
  scoring on a single-turn synthetic trace; preble's behavior under
  session-structured workloads warrants a separate run.
- **Did not test whether fixing `queued` changes any *other* policy's
  behavior** beyond preble. The cross-policy ranking in §5 is from reading
  the scoring code, not from empirical runs.
