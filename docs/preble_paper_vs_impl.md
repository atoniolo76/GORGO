# Preble paper vs. our implementation

> **Bead:** go-ggf (independent investigation A)
> **Scope:** design doc only. No code changes. Scout approves implementation
> separately. Counterpart investigation (polecat B) did not inform this write-up.

## 0. TL;DR

Preble's mechanism that mitigates hotspotting is **time-domain load accounting
plus relative-imbalance rebalancing**. Our `prefix-cache-preble` reduces that to
**slot-count occupancy with an absolute threshold**. The reduction kills the
signal in exactly the regimes where the paper says the mitigation should fire,
which matches scout's observation that `prefix-cache-preble` and `prefix-cache`
produce millisecond-identical latency across all 12 (qps × zipf) regimes.

**The single highest-leverage fix** is to replace the load signal with an
estimated-wait-time accumulator over a window `H`, and replace the absolute
hotspot threshold with a max/min ratio. The dead `queued` counter is a symptom,
not the root cause: even if we set `queued` correctly, the underlying signal
(instantaneous in-flight count vs. slot budget) does not capture the buildup
that Preble's rebalancer is designed to detect.

## 1. What Preble does (paper)

**Preble**, Zhong et al., *"Efficient Distributed Prompt Scheduling for LLM
Serving,"* ICLR 2025 / arXiv:2407.00023. Two-tier scheduler: a **global**
router assigns requests to GPUs, a **local** per-GPU scheduler orders the
batch. Only the global router is comparable to what our harness models.

### 1.1 Global scoring (the "E2" algorithm)

Preble scores a candidate GPU `i` with a **total predicted cost** composed of
three time-domain terms, all in units of GPU-compute milliseconds:

```
Cost_i = L_i + M_i + P_i
```

- **`L_i` — historical load over window `H`.** For every request `r` served
  by GPU `i` in the last `H` seconds:
  `L_i = Σ_r (PT_r + DT_r)`, where `PT_r` is the regression-predicted prefill
  time on the uncached tail and `DT_r` is the regression-predicted decode
  time. The key property: `L_i` is a **time estimate** for work still in
  flight on GPU `i`, not a count of outstanding requests.
- **`M_i` — eviction penalty.** If assigning this request to GPU `i` would
  evict cached prefix nodes, `M_i = Σ_j (PT_j · N_j)` where `PT_j` is the
  prefill cost to recompute evicted node `j` and `N_j` is its hit ratio
  within `H`. I.e. evicting a popular prefix is charged back as future
  recomputation cost.
- **`P_i` — new-request prefill cost.** Expected prefill time for the
  uncached portion of the incoming request.

On top of the cost, Preble layers an **exploit/explore** gate:

```
if missed_len < cached_len:   # "recomputation saved" > "new work"
    assign to GPU with longest matched prefix   (exploit)
else:
    assign to the GPU that minimizes Cost_i     (explore / balance)
```

This is *not* a linear combination. It is a conditional: only when cache
reuse dominates does the policy bind to the prefix-owner; otherwise it falls
back to pure load-minimization.

### 1.2 Load rebalancing (hotspot mitigation)

Two mechanisms, both driven by time-domain signals:

1. **Load shifting** fires when
   `max_GPU_load > Th_bal · min_GPU_load`
   — a **ratio** between the heaviest and lightest GPU, not an absolute
   threshold on any single GPU. When it fires, future *exploitative*
   requests for hot prefixes are redirected from heavy to light GPUs. The
   redirection is sticky-for-the-request, not a migration: in-flight work
   stays where it is.
2. **Prefix auto-scaling** replicates a hot prefix onto an additional GPU
   when the average queueing time on the owner doubles within window `H`.
   The replication makes the new GPU a legitimate "exploit" target for
   subsequent requests; the rebalancer then redistributes across both.

The trigger for both is **observed wall-clock queueing behavior** over a
window, not the instantaneous state of any counter.

### 1.3 Data structure

Global **radix tree** over token sequences. Each node stores token count,
the **set of GPUs that cache this node**, and the per-GPU request count
touching this node within window `H`. The per-node GPU-set is how the
rebalancer expresses "replicate this prefix"; the per-GPU request count is
what `N_j` in the eviction term reads.

### 1.4 Scheduling granularity

- Global (router): **request-level**. One decision per arrival. No chunking.
- Local (per-GPU): **iteration-level** continuous batching with a
  priority-group fair scheduler for queue ordering inside the pod.

### 1.5 Migration requirement

**No cross-GPU migration of in-flight requests.** Rebalancing is dispatch-time
redirection plus prefix replication. A heavy GPU stays heavy for the work it
already accepted.

## 2. What we do (impl)

`src/routing_harness/policies/prefix_cache_preble.py`.

### 2.1 Scoring

Lines 39–90. For each prefill-capable pod `p`:

```python
# prefix_cache_preble.py:64–73
match = 0
for h in hashes:
    if kv_cache.has(p.spec.pod_id, h):
        match += 1
    else:
        break
cap = max(1, p.spec.max_concurrent_prefill + p.spec.max_concurrent_decode)
load = (p.active_prefill + p.active_decode + p.queued) / cap
score = self.alpha * match - self.beta * load
```

Defaults: `alpha=1.0`, `beta=0.5`, `block_size=16`, `hotspot_threshold=0.9`
(lines 42–45). Pods are sorted by `(-score, pod_id)`; the top pod wins
unless hotspot-avoidance fires.

### 2.2 Hotspot avoidance

Lines 76–84:

```python
if top_load > self.hotspot_threshold:
    for _, match, load, p in scored[1:]:
        if match > 0 and load < self.hotspot_threshold:
            return Decision(..., rationale=f"hotspot-avoid top={top_pod...}")
```

Trigger is an **absolute** threshold on the top pod's `load` value. Deflection
target is the next-highest-scored pod with **any** match and load below the
same absolute threshold.

### 2.3 The load signal is broken by construction

Two independent problems:

**Problem A — `queued` is dead state.** `PodRuntime.queued` is declared in
`core.py:88` with default `0` and **no simulator code ever assigns to it**
(confirmed by grep across `src/routing_harness/simulator/`). The numerator
`active_prefill + active_decode + queued` is therefore identical to
`active_prefill + active_decode`. Six other policies read `queued` too
(`least_request.py:35`, `least_busy_time.py:36`, `session_affinity.py:48`,
`per_tenant_load_balance.py:70`, `pd.py:62`, `prefix_cache.py:64`), so the dead-state bug
is load-bearing beyond Preble — but only Preble uses it for a *threshold*
decision rather than an ordering.

**Problem B — even with `queued` wired up, the signal is wrong.** Our
denominator is `max_concurrent_prefill + max_concurrent_decode` (lines
70–71), which on the baseline topology is `4 + 16 = 20`. Hotspot fires at
`load > 0.9`, i.e. `(active_prefill + active_decode) > 18` on a single pod.
But the cost-model's queueing formula (`cost_model.py:_mm1_wait_ms`,
lines 127–139) uses a *different* denominator — **only
`max_concurrent_prefill = 4`** — so queueing latency already explodes at
`active_prefill ≈ 3`. The two normalizations disagree by 5×. The hotspot
rebalancer is watching for a condition that the cost model has already
declared unreachable.

Concretely at qps=4, zipf=1.5 (scout's "should trigger" regime), you need
~18 concurrent requests on one pod before Preble's branch fires. The
simulator retires requests at their *projected* completion (`engine.py:85–102`),
so `active_prefill + active_decode` tops out near the sum of per-request
durations divided by inter-arrival time — typically well below 18 even on
a hot pod.

### 2.4 No eviction term, no replication, no window

- `M_i` is not modeled. Nothing in `prefix_cache_preble.py` consults
  `KVCacheState.pods[...].entries` to estimate eviction cost or per-entry
  hit ratios. Evictions are LRU-by-byte (`kv_cache.py:47–64`) and the
  scorer is unaware of them.
- Prefix **auto-scaling / replication** is absent. The only path by which
  a prefix ends up on multiple pods is the engine's peer-pull logic
  (`engine.py:130–154`), which installs on the destination but is *not*
  triggered by hotspot detection — it fires whenever a strictly longer
  cross-pod prefix exists.
- There is no **window `H`**. The load signal is instantaneous. `ewma_latency_ms`
  exists on `PodRuntime` (`core.py:89`) and is updated in the engine
  (`engine.py:244–245`), but the Preble policy does not read it.

### 2.5 Data structure

Per-pod `OrderedDict`-backed LRU keyed by block hash
(`kv_cache.py:30–64`). **No radix tree**, no per-node GPU-set, no per-node
hit-count within a window. Block membership is checked point-wise
(`KVCacheState.has(pod_id, hash)`, `kv_cache.py:77–79`). The `owners_of`
helper (called from `engine.py:135`) returns the pod-set for a single
block, not a tree traversal.

## 3. Divergences (paper vs. impl)

| Dimension | Preble paper | Our impl | Severity |
|---|---|---|---|
| Load signal | Σ regression-predicted PT+DT over window `H`, per GPU | `(active_prefill + active_decode + queued) / capacity`, instantaneous | **Critical — kills the signal** |
| Hotspot trigger | `max_load > Th_bal · min_load` (relative) | `top_load > 0.9` (absolute) | **Critical — wrong regime** |
| Route rule | `exploit` iff `missed_len < cached_len`, else minimize cost (conditional) | `score = α·match − β·load` (linear combination) | **High — different shape of tradeoff** |
| Eviction cost `M_i` | Explicit term in cost | Not modeled | Medium |
| Prefix auto-scaling | Replicate hot prefix when queueing time doubles over `H` | Absent; replication is incidental via peer-pull | Medium |
| `queued` counter | (Preble maintains per-GPU queue with priority groups) | Declared, never set → dead state | High — but symptomatic of load-signal problem |
| Queueing-denominator consistency | N/A — Preble uses time, not slots | Policy uses `P+D` slots, cost model uses `P` slots only; ratio differs 5× | High — silent misalignment |
| Prefix data structure | Global radix tree, per-node GPU-set, per-node window count | Per-pod block-hash LRU, no tree | Medium (needed only for `M_i` / replication) |
| Migration | None (rebalancing is dispatch-time redirection) | None | Match |
| Scheduling granularity | Request-level at global, iteration-level at local | Request-level (engine is one-pass) | Match for global tier |

## 4. Why scout's sweep shows no difference

Two compounding reasons, both traceable to section 2.3:

1. **The hotspot branch almost never fires.** Under the baseline topology
   (`cap = 20`, `hotspot_threshold = 0.9`), a pod needs ~18 concurrent
   in-flight requests before the rebalancer activates. At qps=4 over 4 pods
   average utilization per pod is ~1 request in flight; at qps=32 it rises
   but the M/M/1 queueing penalty in `cost_model.py:171–175` (which uses
   `max_concurrent_prefill = 4` as the denominator) drives per-request
   latency up so sharply that `active_prefill` retires before accumulating
   toward 18.
2. **When it doesn't fire, the linear term is dominated by match.**
   `alpha = 1.0`, `beta = 0.5`. One block of prefix match contributes `+1`
   to the score; a fully loaded pod contributes `−0.5 · 1.0 = −0.5`.
   Typical requests have many matching blocks on the hot pod and zero or
   near-zero `load` on all pods (because the denominator is 20 not 4), so
   the cache-affinity term wins on every decision. The ordering reduces to
   pure prefix-match, which is exactly what `prefix_cache.py` does
   (modulo the no-match fallback rule, which is symmetric).

Result: `prefix-cache-preble` and `prefix-cache` produce the same dispatch
sequence, and therefore the same latency to the millisecond.

## 5. Recommended minimum fix

The deliverable is not faithful Preble. It is a Preble stand-in that **the
sweep can distinguish from plain prefix-cache in the regimes the paper says
it should**. Scope ordered by leverage:

### 5.1 Replace the load signal with an estimated-wait accumulator (highest leverage)

Change the per-pod load to a time-domain quantity, not a counter ratio. A
cheap, in-harness version that does not require new instrumentation:

```
load_ms_i = active_prefill_i · S_prefill + active_decode_i · S_decode
```

where `S_prefill` and `S_decode` are the cost-model's representative service
times for the workload (already computed as `avg_service_ms` in
`cost_model.py:167–170`; `S_decode` analogously from `ComputeParams`). This
is Preble's `L_i` with in-flight work stand-ins for the windowed sum, which
is the cheapest approximation that is still *time-domain*.

Rationale: the signal the paper relies on is "how long until this GPU drains"
— a quantity with units of milliseconds. Our current signal is "what
fraction of slot budget is occupied" — a dimensionless ratio whose relationship
to queue drain time is policy-by-policy.

**Implication for `queued`.** Leave it alone. With a time-domain load, the
dead counter is irrelevant to Preble. Fixing `queued` in the simulator is a
separate bead (it also affects `least_request`, `per_tenant_load_balance`, `pd`,
`least_busy_time`, `session_affinity`, `prefix_cache`; each of those has its
own remediation story). File separately if scout wants the counter lit up
system-wide; do not bundle here.

### 5.2 Replace the absolute threshold with a max/min ratio

```
if max_pod.load_ms > Th_bal · min_pod.load_ms:   # Th_bal ≈ 1.5–2.0
    redirect top-scoring request from max_pod to the best-match light pod
```

This is Preble's load-shifting trigger literally. It fires on *imbalance*,
not on saturation, and therefore is not silenced when the whole cluster is
underloaded. It also does not require a global target threshold, only a
local comparison — so the parameter does not need to be retuned per
topology.

### 5.3 Replace the linear score with the `exploit`/`explore` conditional

```
cached_tokens = block_size · best_match_on_best_pod
missed_tokens = len(prompt_tokens) − cached_tokens
if missed_tokens < cached_tokens:
    return best_match_pod                    # exploit
else:
    return argmin_i load_ms_i                # explore
```

This is algorithmically closer to the paper than the linear tradeoff. It
also gives the sweep a sharper hypothesis: the exploit regime should
dominate at high zipf_s (fat-head workloads), the explore regime at low
zipf_s (uniform workloads). The current linear score smears both into one
curve and obscures the crossover.

### 5.4 Defer (not minimum-fix scope)

The following are part of real Preble but can be excluded from a first
representative implementation without erasing the hotspot signal:

- Eviction term `M_i`: requires a radix tree and windowed hit counts. Skip.
- Prefix auto-scaling / replication: would require engine changes to
  warm-install on a second pod on demand. Skip.
- Priority-group local scheduler: our engine doesn't model per-pod queues
  as orderable sets. Skip.
- Real regression for `PT_r`, `DT_r`: the cost model's analytic `S_prefill`,
  `S_decode` are good enough for a sweep. Skip.

### 5.5 What to verify after the fix lands

In the scout sweep, the representative Preble should:

- At `qps=4, zipf_s=1.5`: diverge from `prefix-cache` in the direction of
  lower p95/p99, because the rebalancer now fires when one family's
  prefix-owner accumulates more load than the lightest pod.
- At `qps=32, zipf_s=0.7`: converge with `least-busy-time` / `least-request`
  on latency because cache affinity is weak and the explore branch
  dominates.
- At `qps=4, zipf_s=0.7`: remain close to `prefix-cache` because no pod
  accumulates a dominant share.

If any of those three do not hold after the fix, the remaining gap is not
in the load signal and the eviction term is the next suspect.

## 6. Open questions for scout

1. Is it acceptable to drop `queued` from the Preble policy entirely and
   file a separate bead for lighting up `queued` in the simulator? (Our
   recommendation: yes — the two are decoupled.)
2. Preferred parameter surface: expose `Th_bal` as a policy field, or
   hard-code to 1.5 with a FIXME? (Recommendation: expose, since it is the
   parameter most likely to be swept.)
3. The cost model's queueing denominator (`max_concurrent_prefill` only)
   and the policy's load denominator (`max_concurrent_prefill +
   max_concurrent_decode`) disagree by a factor of 5 on the baseline
   topology. If we migrate to a time-domain load signal the policy
   disagreement disappears, but the cost model still treats decode slots
   as free from a queueing-latency standpoint. Is that intentional, or a
   third bead to file? (Out of scope for this design doc.)
