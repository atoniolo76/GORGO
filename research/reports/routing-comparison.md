# LLM Inference Routing Strategies: A KV-Cache-Aware Comparison

> **Status**: Draft scaffold. Quantitative results are `{{placeholders}}`
> to be filled after running the harness on real hardware or against
> traces. Do not cite numerical claims from this document until the
> placeholders are replaced.

## 0. Abstract

LLM serving systems increasingly treat the *routing* layer — the policy
that decides which pod serves which request — as a first-class research
object. The design space has fragmented: cache-oblivious load
balancers, KV-cache-aware routers (SGLang router, Mooncake Conductor,
NVIDIA Dynamo, Ant AI Gateway, AIBrix), hotspot-aware variants
(Preble-style), prefill-decode disaggregation (PD), and fairness
mechanisms (VTC). Systems that look similar on paper differ in the
*second-order costs* they pay: KV transport bandwidth, scheduling
latency, queueing, head-of-line blocking, and hotspot formation.

We present a unified, dependency-light experiment harness with pluggable
policies, a configurable cost model, and deterministic workload
generation (including a lmsys-chat-1m adapter). Eleven policies are
evaluated against a common set of metrics. We quantify the central
tradeoff between **maximizing KV-cache reuse** and **minimizing end-to-
end latency**, and decompose latency into routing, queueing, compute,
network, and KV-transport components.

## 1. Introduction

### 1.1 Motivation

Prefix-aware routing (SGLang router, Mooncake Conductor) can
dramatically cut prefill cost by reusing KV state already cached on a
specific pod. But naive prefix-locality concentrates load on a few
"hot" pods, trading cache reuse for queueing delay. Preble-style
approaches argue for a joint objective; PD disaggregation argues for
decoupling the two phases; VTC argues that fairness constraints change
the Pareto frontier entirely. Each of these proposals has been
evaluated against its own baseline on its own workload; the field has
not produced an apples-to-apples comparison across the design space.

**Research question.** *Which* prefix-aware routing strategy wins, for
which workload regime and which cluster topology, and by how much — and
does any of them clearly improve on the routing layer GORGO already
ships (a random-choice proxy over replica URLs; see
`proxy/modal_proxy.py`)?

This phrasing is deliberate on three counts:

1. **Not "does prefix-aware routing win"** — that framing treats
   prefix-awareness as a single binary. SGLang-style longest-prefix,
   Mooncake-style KV-aware dispatch, Preble-style load-adjusted
   matching, and PD disaggregation are all "prefix-aware" but make
   fundamentally different tradeoffs; lumping them hides the result we
   actually care about.
2. **Situating a fragmented space.** The related-systems literature
   (Section 2) is a catalog of point designs, each with its own cost
   model, workload, and evaluation protocol. A common harness that
   holds those constant while varying only the policy is what lets us
   compare them head-to-head.
3. **Grounded in GORGO.** GORGO's production serving path is
   `engine/modal_sglang.py` (SGLang + Qwen3.5-35B-A3B-FP8, multi-
   replica on H100) fronted by the naive random-choice proxy in
   `proxy/modal_proxy.py`. Any policy that demonstrably beats the
   random baseline under GORGO's workload and topology is a candidate
   to replace that proxy; any that does not, isn't.

### 1.2 Contributions

- **A taxonomy** of routing strategies along five orthogonal axes
  (Section 3) that separates "reads cache state" (a selection signal)
  from "carries private state across requests" (a memory axis) — a
  distinction the existing literature collapses.
- **A harness** with 11 pluggable policies (SGLang-ish, Mooncake-ish,
  Preble-ish, PD, per-tenant load balance, session-affinity, five
  cache-oblivious baselines), a prefix-level KV model, and a
  deterministic discrete-event simulator (Section 4).
- **A comparison protocol** with explicit cost model, metrics, and
  reproducibility guarantees (Section 5). Policies are evaluated
  against the same workload/topology/cost assumptions; the random
  baseline (`policy=random`) is the direct stand-in for GORGO's
  current proxy.
- **A first characterization** of the reuse-vs-latency Pareto curve on
  synthetic workloads and lmsys-chat-1m turn replays (Sections 6–7,
  quantitative results to be filled after the sweep lands).
- **A GORGO-fit recommendation**: which policy (if any) beats random
  on GORGO's workload enough to justify the deployment cost of
  replacing the current proxy (Section 7.4, to be written after the
  sweep).

## 2. Related Systems (conceptual references, not reimplementations)

| System | Key idea | What we model |
|-----------------------|--------------------------------------------|----------------------|
| Mooncake Conductor | KV-cache-aware dispatch; hotspot detect. | `prefix-cache`, `prefix-cache-preble` |
| SGLang Router | Prefix-tree routing on a shared cache. | `prefix-cache` |
| NVIDIA Dynamo | PD disaggregation + scheduling. | `pd` policy |
| Ant Group AI Gateway | Heterogeneous dispatch under SLA. | Cost model hooks |
| AIBrix | Cluster-level LLM serving orchestration. | Topology abstraction |
| Preble | Prefix-locality + load balancing. | `prefix-cache-preble` |
| (none — VTC paper not implemented) | Sheng et al. OSDI'24 schedule admission order to bound max-min fairness. We do not implement this. | — |

We do **not** claim parity with any of these systems; policies are
baselines inspired by their published designs with documented
assumptions (see `src/routing_harness/policies/`).

## 3. Taxonomy

We locate each policy along five axes chosen to be orthogonal — that is,
a policy's value on one axis should not mechanically determine its value
on another. The first cut of this document conflated *reading*
KV-cache state (a selection signal) with *carrying* private state across
requests, which meant prefix-aware policies were doubly counted against
both "cache awareness" and "statefulness." It also listed "hotspot
mitigation" as a standalone axis even though every Preble-style
mitigation mechanism we care about is either a linear combination of
signals (belongs on the selection axis as `composite`) or a
post-dispatch movement (belongs on the migration axis). The revision
below retires those conflations.

1. **Selection criterion** — the primary signal the policy consults to
   score candidate pods.
   Values: `random` | `load` | `capacity` | `cache-affinity` |
   `identity` | `fairness-debt` | `composite`.
   `load` covers any backlog or utilization signal (queue depth,
   busy-time, EWMA latency, EWMA throughput). `capacity` covers
   free-resource signals such as free KV bytes. `cache-affinity` covers
   prefix-match length against the per-pod KV cache. `identity` covers
   session- or tenant-keyed stickiness. `fairness-debt` covers
   accumulated token counters. `composite` is a linear or phase-split
   combination of two or more of the above.

2. **Policy state scope** — memory the *policy itself* carries across
   `decide` calls, independent of cluster or KV state owned by the
   simulator.
   Values: `stateless` | `per-session` | `per-tenant`.
   A policy that inspects `KVCacheState.has(...)` or `Pod.active_*` to
   make a decision is still `stateless` on this axis; only private
   per-policy dataclass fields that accumulate observations count.

3. **Fairness model** — how multi-tenant or multi-session contention is
   mediated.
   Values: `best-effort` | `session-sticky` | `tenant-weighted`.
   Stickiness is distinct from tenant-weighting: it isolates a session
   onto one pod (useful for cache warm-up) but does not attempt to
   equalize throughput across tenants.

4. **Topology requirement** — cluster-role assumption.
   Values: `any` | `pd-aware` | `pd-required`.
   `pd-aware` policies exploit role-split pools when present and
   tolerate colocated topologies; `pd-required` policies would refuse
   to run on a fully colocated cluster.

5. **Migration / rebind** — whether the policy moves, abandons, or
   reassigns work after the initial dispatch decision.
   Values: `none` | `rebind-on-fail` | `cross-pod-pull`.
   This axis is included for completeness of the design space; most
   currently implemented policies are `none`.

**Table 1** — placement of all 11 policies:

| Policy                  | Selection       | State         | Fairness          | Topology   | Migration         |
|-------------------------|-----------------|---------------|-------------------|------------|-------------------|
| `random`                | `random`        | `stateless`   | `best-effort`     | `any`      | `none`            |
| `least-request`         | `load`          | `stateless`   | `best-effort`     | `any`      | `none`            |
| `least-busy-time`       | `load`          | `stateless`   | `best-effort`     | `any`      | `none`            |
| `least-latency`         | `load`          | `stateless`   | `best-effort`     | `any`      | `none`            |
| `throughput`            | `load`          | `stateless`   | `best-effort`     | `any`      | `none`            |
| `least-kv-cache`        | `capacity`      | `stateless`   | `best-effort`     | `any`      | `none`            |
| `prefix-cache`          | `cache-affinity`| `stateless`   | `best-effort`     | `any`      | `none`            |
| `prefix-cache-preble`   | `composite`     | `stateless`   | `best-effort`     | `any`      | `none`            |
| `pd`                    | `composite`     | `stateless`   | `best-effort`     | `pd-aware` | `none`            |
| `session-affinity`      | `identity`      | `per-session` | `session-sticky`  | `any`      | `rebind-on-fail`  |
| `per-tenant-load-balance` | `fairness-debt` | `per-tenant` | `tenant-weighted` | `any`      | `none`            |

Orthogonality notes:

- **Selection criterion vs. state scope.** `prefix-cache` is
  (`cache-affinity`, `stateless`) because it reads `KVCacheState` but
  stores nothing privately; a hypothetical per-session-learned
  cache-affinity policy would share axis 1 and move to `per-session`
  on axis 2. The two axes vary independently.
- **State scope vs. fairness model.** Fairness-weighted policies do
  need state, but the converse is not true: `session-affinity` carries
  `per-session` bindings yet its fairness model is `best-effort`
  (stickiness is for cache warm-up, not balancing).
- **Topology and migration** are independent of the other three. Any
  selection criterion could in principle be paired with any topology
  requirement, and migration is a post-dispatch behavior.
- **Composite at selection, one axis down.** The old taxonomy's
  "hotspot mitigation" axis duplicated signal-fusion: a Preble-style
  linear combination is captured as `composite` on axis 1, and the
  threshold-based deflection is a property of that composite, not a
  separate axis. Migration-based hotspot mitigation (e.g. some Mooncake
  variants) is captured on axis 5, not bolted onto axis 1.

## 4. Experimental Harness

### 4.1 Architecture

See `docs/harness_overview.md` for the full architectural rationale.
Summary:

- **Policies** register via `@register_policy(...)` and satisfy the
  `RoutingPolicy` protocol.
- **ClusterState** exposes read-only per-pod runtime state; simulator
  owns mutation. Contract tests enforce non-mutation by policies.
- **KVCacheState** is a per-pod LRU over deterministic block-level
  prefix hashes (16-token blocks by default) with byte-budget
  eviction.
- **CostModel** (`AnalyticCostModel`) decomposes latency into
  `routing + queueing + compute_prefill + compute_decode + network +
  kv_transport` with closed-form, documented coefficients. An
  `InstrumentedCostModel` scaffolding exists for future replacement
  with measured values.
- **Simulator** iterates in arrival order, charges cost, updates KV
  state, and records per-request breakdowns.

### 4.2 Workload

Two adapters:

- **Synthetic** (`workload.synthetic`): Poisson arrivals, Zipf-
  distributed prefix families over `n_prefix_families`, session-id
  sampling. Parameterized by `n_requests`, `arrival_rate_qps`,
  `zipf_s`, prompt-length range, session count, and seed.
- **lmsys-chat-1m** (`workload.lmsys`): JSONL loader plus a
  deterministic char→token mapping that preserves prefix overlap in
  the source text (a real tokenizer can be swapped in later). Download
  is stubbed; users supply a local path.

### 4.3 Topology and cost parameters

All experiment-defining fields are explicit in config (no silent
defaults). Base topologies:

- **Colocated**: 4 pods × 1 GPU, `role=both`, 4 GiB KV each.
- **PD-disaggregated**: 2 prefill + 2 decode pods, 8 / 4 GiB KV.

See `configs/example_run.yaml`, `configs/example_pd_run.yaml`.

### 4.4 Sweep design

Cartesian over:

- `policy.policy_id` × 11
- `workload.params.arrival_rate_qps` ∈ {4, 8, 16, 32}
- `workload.params.zipf_s` ∈ {0.7, 1.1, 1.5}
- `seed` ∈ {0, 1, 2}

Total: 11 × 4 × 3 × 3 = **396 runs** per topology.

## 5. Metrics

- **Latency**: p50, p95, p99, mean; plus per-component decomposition.
- **KV reuse**: hit rate (any-cached), reuse-captured vs reuse-
  available blocks, capture rate = captured / available.
- **Network**: total KV transport bytes, per-request KV transport ms.
- **Throughput/goodput**: req/s and tokens/s, tail-collapse onset.
- **Fairness / load skew**: per-pod busy-ms distribution, skew =
  (max − min) / mean.
- **Scheduling overhead**: modeled `routing_ms` per request;
  instrumentation hook available for measured overrides.
- **Hotspot**: prefix popularity distribution and, when hotspot ground
  truth is defined, precision/recall of the policy's hotspot-avoidance
  decisions (Preble-style only).

## 6. Results (sweep v4)

Sweep v4: 11 policies × 4 QPS × 3 Zipf × 3 seeds = 396 runs.
Workload: 256 prefix families, 1024-token shared heads, 2000
requests/run. 4-pod colocated topology, 4 GiB KV cache/pod (2048
blocks), so working set is 8× per-pod capacity.

### 6.1 Headline table (median p95 across seeds and Zipf values)

Refreshed 2026-04-25 against the current cost model (M/M/1 queueing
go-4lp, KV-pull/prefill-overlap go-npl, fluid fair-share go-uy0,
non-consecutive block residency go-uce).

| Policy                  | qps=4  | qps=8  | qps=16  | qps=32  | hit_rate | skew  |
|-------------------------|--------|--------|---------|---------|----------|-------|
| random                  |  1,001 | 14,793 |  14,969 |  14,969 |    0.730 | 0.052 |
| least-request           |    938 |  1,012 |  14,905 |  14,977 |    0.731 | 0.186 |
| prefix-cache            |    950 | 13,809 |  14,777 |  14,761 |    0.761 | 0.472 |
| **prefix-cache-preble** | **938** | **1,007** |  14,761 |  14,785 |    0.746 | **0.117** |
| least-busy-time         |    940 |  1,009 |  14,873 |  14,969 |    0.729 | 0.125 |

(Full grid for all 11 policies: see `scripts/aggregate_v4_sweep.py`
output. `pd` colocated-fallback gives 942 / 1,033 / 15,369 / 15,369
ms, matching the §6.4 colocated row.)

### 6.2 Preble vs prefix-cache: p95 margin

| QPS | Margin (ms)  | Interpretation                                 |
|-----|--------------|------------------------------------------------|
|   4 |        −12   | Noise — both ≈ 940 ms, prefix-cache no longer   |
|     |              | mono-homes badly at this load                   |
|   8 | **−12,802**  | Strongest win — Preble's load-aware gate avoids |
|     |              | the prefix-cache mono-homing collapse           |
|  16 |        −16   | Tied, queueing saturates                       |
|  32 |        +24   | Tied, queueing saturates                       |

### 6.3 Hotspot mitigation

Preble's overall skew (0.12) is roughly 4× lower than prefix-cache's
(0.47). The gap is largest at qps=4 (0.12 vs 0.69) where prefix-cache's
Zipf-driven mono-homing is most visible; both compress under saturation
(qps≥16) as every pod becomes a hot pod. Random achieves the lowest
skew (0.05) but pays for it with the lowest hit_rate among non-PD
policies and the worst qps=8 p95. The relative-imbalance rebalancer
(th_bal=1.5) fires when the exploit target is 1.5× the lightest pod's
load, distributing hot families without abandoning prefix affinity
entirely.

| Policy              | overall skew | hit_rate |
|---------------------|--------------|----------|
| random              |        0.052 |    0.730 |
| least-request       |        0.186 |    0.731 |
| prefix-cache        |        0.472 |    0.761 |
| prefix-cache-preble |        0.117 |    0.746 |
| least-busy-time     |        0.125 |    0.729 |

### 6.4 PD gains

Dedicated PD-topology sweep (`configs/example_pd_potent_sweep.yaml`)
paired with a matched colocated re-run
(`configs/example_colocated_potent_sweep.yaml`) over the potent
synthetic workload (256 families × 1024-token shared heads, 2000
requests/run). Both topologies hold 4 GPUs total: the colocated
cluster has 4 × 1-GPU pods at `role=both` (4 × 4 GiB KV); the PD
cluster has 2 × 1-GPU prefill pods (8 GiB KV each, for staging) and
2 × 1-GPU decode pods (4 GiB each). Aggregate prefill slot count
matches at 16 (4 pods × 4 colocated, 2 pods × 8 prefill on PD).
Five-policy slate × 4 QPS × 3 Zipf × 3 seeds = 180 runs/topology;
medians fold across Zipf and seed.

**Note on comparability with §6.1.** As of the 2026-04-25 refresh
(go-rr0), §6.1 was re-run against the current cost model — colocated
`prefix-cache-preble` p95 = 938 / 1,007 ms at qps=4 / 8 in §6.1
matches the `colocated` row of this table to the millisecond, so the
two sections can now be cross-referenced directly.

**Table — median p95 (ms) by topology × policy × qps:**

| Topology   | Policy                  | qps=4 | qps=8  | qps=16 | qps=32 | hit_rate | skew |
|------------|-------------------------|-------|--------|--------|--------|----------|------|
| PD         | `pd`                    |   949 |  8,225 | 14,777 | 14,785 |   0.763  | 0.23 |
| PD         | `pd-preble`             |   953 |  1,046 | 14,761 | 14,777 |   0.749  | 0.01 |
| PD         | `prefix-cache`          |   949 |  8,225 | 14,777 | 14,777 |   0.763  | 0.23 |
| PD         | `prefix-cache-preble`   |   953 |  1,041 | 14,761 | 14,761 |   0.753  | 0.01 |
| PD         | `random`                |   970 | 14,241 | 14,793 | 14,817 |   0.746  | 0.00 |
| colocated  | `pd` (fallback)         |   942 |  1,033 | 15,369 | 15,369 |   0.694  | 0.56 |
| colocated  | `pd-preble` (fallback)  |   944 |  1,038 | 15,553 | 15,401 |   0.678  | 0.14 |
| colocated  | `prefix-cache`          |   950 | 13,809 | 14,777 | 14,761 |   0.762  | 0.69 |
| colocated  | `prefix-cache-preble`   |   938 |  1,007 | 14,761 | 14,785 |   0.736  | 0.12 |
| colocated  | `random`                | 1,001 | 14,793 | 14,969 | 14,969 |   0.730  | 0.05 |

**Three-way head-to-head — colocated baseline vs PD-plain vs PD+Preble:**

| QPS | colocated `prefix-cache-preble` | PD `pd` | PD `pd-preble` | best PD vs colocated |
|-----|---------------------------------|---------|----------------|----------------------|
|   4 |   938                           |   949   |   953          | colocated by 11 ms (noise) |
|   8 | 1,007                           | 8,225   | 1,046          | colocated by 39 ms (Preble required for PD parity) |
|  16 | 14,761                          | 14,777  | 14,761         | tied (saturation) |
|  32 | 14,785                          | 14,785  | 14,777         | tied (saturation) |

Three observations:

1. **At matched 4-GPU count, PD topology does not win.** Across the
   QPS range — including at low load where PD's phase isolation should
   help most — the best PD configuration matches colocated to within
   single-digit-percent p95. The under-load regime (qps=4) is a tie:
   prefill rarely queues on either topology, so the dedicated prefill
   pool gains nothing. The saturated regime (qps≥16) is also a tie
   because the prefill bottleneck is the same in both topologies (16
   prefill slots aggregate). The transitional regime (qps=8) is where
   `pd-preble` earns its keep — without the Preble gate on the
   prefill pool, plain `pd` drops 8× to 8,225 ms.

2. **Plain `pd` cache-locks on the prefill pool exactly as
   prefix-cache mono-homes on colocated.** At qps=8 the three seeds
   span (1,047, 9,297, 12,609) ms p95 for `pd` — the same cache-lock
   pathology F25 fixed for the colocated cache-affinity policies,
   but the failure mode lives one level deeper. Once the first
   identical-prompt request lands on `pf0`, prefix-match wins every
   tie-break against `pf1`, and the warmer pod accumulates queue
   depth without the load-aware deflection that `pd-preble` adds.
   `pd-preble` flattens these three seeds to (1,041, 1,046, 1,060) ms
   — same range as colocated `prefix-cache-preble`. Skew confirms the
   mechanism: 0.23 (pd) → 0.01 (pd-preble).

3. **PD lifts hit_rate slightly but does not convert it into latency.**
   Pooling cache across two prefill pods (vs four colocated pods)
   raises the median hit_rate from 0.736 (colocated
   `prefix-cache-preble`) to 0.749–0.763 (PD policies). The win is
   real but the latency does not move with it: at qps=8, PD
   `prefix-cache-preble` p95 = 1,041 ms vs colocated 1,007 ms despite
   the higher hit_rate. The reuse-vs-latency frontier is dominated by
   queueing variance at this regime, not by cache savings.

**Recommendation.** PD topology, on this workload at this scale, is
not justified by routing-layer wins alone — the colocated baseline
plus a Preble gate captures the same operating point with fewer pod
roles. PD's plausible advantages (independently scaled prefill /
decode pools, batched decode pipelines per §9.1 `decode_batch_k`,
asymmetric accelerators per phase) live outside the matched-GPU,
analytic-cost-model regime this sweep evaluates. The recommendation
to the GORGO-fit conclusion in §7.4 is unchanged: `prefix-cache-preble`
on the existing colocated topology remains the strongest candidate.

### 6.5 Fairness under contention

> `{{figure_vtc_fairness}}` — per-session analysis deferred to
> calibration phase 2.

## 7. Analysis (to be written)

### 7.1 The reuse-vs-latency frontier

Confirmed (2026-04-25 refresh against §9.1-fixed cost model):

- **Prefix matching alone does not monotonically improve p95.** Plain
  `prefix-cache` has the highest hit_rate (0.761) but blows up in the
  transitional regime — qps=8 p95 = 13,809 ms vs prefix-cache-preble's
  1,007 ms. At qps=8 the cluster has enough utilization that Zipf-
  driven mono-homing on a hot pod creates queueing pressure that
  exceeds the cache-reuse savings; at qps=4 the cluster is under-
  loaded enough that mono-homing has no observable cost (prefix-cache
  ties the load-balancers within ~12 ms).
- **Load-awareness flattens the frontier where it matters.**
  `prefix-cache-preble` retains most of the hit-rate gain (0.746 vs
  0.761) while cutting qps=8 p95 by 12.8 s — a ~13× reduction over
  plain prefix-cache. The exploit/explore gate preserves cache affinity
  when reuse dominates and falls back to load-balancing when it
  doesn't. Overall skew drops from 0.47 to 0.12 (~4× reduction).
- **The tradeoff favors Preble in the transitional regime, not low
  load.** Preble sacrifices 1.5 pp of hit_rate for a ~13× p95 reduction
  at qps=8. At qps=4 the gain is noise (12 ms) because no policy is
  yet stressed. At qps≥16 the distinction vanishes — queueing
  saturates regardless of policy and all policies converge to
  ~14.7-15.0 s p95.

### 7.2 Second-order costs

- **KV transport**: when (if ever) does pulling from a peer beat
  cold-prefilling locally? The cost model predicts a crossover point;
  we measure whether the simulator recovers it.
- **Scheduling overhead**: at what cluster size does `routing_ms`
  dominate routing decisions? (Linear in `len(cluster)` in our model;
  dominated by `per_pod_consideration_us`.)
- **Queueing**: head-of-line blocking as `arrival_rate_qps` approaches
  aggregate prefill capacity.

### 7.3 PD versus colocated

At what ratio of prefill:decode cost does disaggregation help?

### 7.4 GORGO-fit recommendation

GORGO's production routing is random (`proxy/modal_proxy.py`:
`random.choice(replica_urls)` over replicas discovered through the
`GORGO-replicas` Modal Dict). That corresponds directly to the
`random` policy in our harness. This section answers, for the workload
mix we actually serve:

- **Does any policy beat random** by enough margin (on p99 latency or
  prefill hit rate) to justify the operational cost of a real proxy?
- **Which one**, and under what assumptions about the workload's
  prefix-reuse mass? (The answer almost certainly depends on how much
  system-prompt and multi-turn reuse actually shows up in production
  traffic; until calibration (§2.2 below) is done against live
  SGLang metrics, the recommendation is conditional on the synthetic
  reuse knob, not measured.)
- **What would have to be true** — in workload shape, cluster
  topology, or latency budget — for the recommendation to flip?

**Preliminary answer (pre-calibration):** Under the synthetic workload,
`prefix-cache-preble` is the strongest candidate to replace `random`:
at qps=8 it cuts p95 from 14.8 s to 1.0 s (~14× improvement over
`random`, ~13× over plain `prefix-cache`) while preserving 0.75
hit_rate. `least-request` and `least-busy-time` match `random` on
hit_rate with similar qps=8 p95 (~1.0 s) and are zero-risk drop-ins
that capture the load-balancing win without the cache-aware logic.
The choice between them depends on how much prefix reuse actually
exists in production traffic — if GORGO's system-prompt and multi-turn
reuse is substantial, Preble's extra 1.5 pp of hit_rate compounds;
if traffic is mostly unique prompts, `least-request` is simpler and
equally effective. Calibration phase 2 (§9) will resolve this.

The answer feeds directly into a follow-up proposal to replace
`random.choice` with the winning policy behind an SGLang-compatible
proxy.

## 8. Reproducibility

- All configs are full snapshots and are saved with each run under
  `results/<run_id>/config.json`.
- All RNG use goes through seeded instances; no global seeds.
- Synthetic workload generation is deterministic per seed;
  lmsys replay is deterministic per (path, seed) after a user-
  supplied local dataset dump.
- `run_id` = blake2b(snapshot) for content-addressed runs.
- No network calls at run time once the lmsys local path is provided.

## 9. Gaps to be filled by running the harness

1. **Absolute numbers.** Every quantitative claim in Sections 6–7 is
   conjectural until the sweep has been executed.
2. **Cost model calibration.** Phase 1 design + Phase 2 execution
   pipeline landed (bead go-8cm). `scripts/calibrate.py` runs a vLLM
   microbenchmark sweep on Modal A100-80GB (three sub-sweeps per
   `docs/calibration_plan.md` §3: prefill / decode@batch=1 /
   decode-batch) and fits the five `ComputeParams` coefficients
   locally, emitting `configs/calibrated_a100.yaml` +
   `research/data/calibration/<ts>/fit_summary.json`; unit tests in
   `tests/unit/test_calibrated_coefficients.py` enforce the §6
   acceptance gates (R² thresholds, residual-SE bounds,
   `k ∈ [0.1, 2.0]`, monotonic amortization) and skip cleanly when
   the calibrated config is absent. **Rerun-blocked on HF access**:
   the HF account behind the `hf_token_rome` Modal secret is not
   authorized for `meta-llama/Meta-Llama-3-8B-Instruct` (gated repo,
   403 Forbidden on model config fetch, 2026-04-25). Until the rerun
   lands (tracked as go-26c), the `AnalyticCostModel` coefficients
   shipped in `configs/example_run.yaml` remain illustrative and the
   quantitative claims in §6–§7 are conditional on the
   pre-calibration relative ordering being preserved (which §9.1
   argues for, on the axes where the approximations are symmetric).
   The `InstrumentedCostModel` scaffolding supports overriding
   analytic values with observations per-pod once a calibrated
   config ships.
3. **Tokenizer fidelity.** The lmsys adapter ships two tokenizers: a
   content-hashed block-structured *mock* (default, zero deps) and a
   real `tiktoken:<encoding>` path behind the `tokenizers` optional
   extra. Real tokenization changes block alignment and can materially
   shift capture rates; the real path should be used for any published
   absolute number on lmsys (see §9.1 for the bias direction).
4. **Real KV transport.** We model transport as
   `rtt + bytes/bandwidth`. Real NCCL / RDMA transfers include
   setup, backpressure, and bandwidth sharing effects not captured
   here.
5. **Scheduling overhead.** Modeled as a constant + per-pod
   consideration term. Real routers have additional costs
   (consistent-hash lookups, prefix-tree maintenance, cluster-state
   sync) that we do not yet account for.
6. **Preemption and migration.** Our simulator does not model
   preemption; migration metrics are edge-case only. Policies that
   rely on migration (some Mooncake variants) are evaluated as if
   migration is free when chosen.
7. **Hotspot ground truth.** Precision/recall of hotspot detection
   requires a ground-truth labeler, not yet defined. Today we only
   report load skew as a proxy.
8. **Dataset coverage.** lmsys-chat-1m is English-heavy and
   conversation-style; ShareGPT, summarization, and code-completion
   workloads will shift prefix-reuse distributions materially and are
   out of scope here.

### 9.1 Known simulation biases (surfaced in peer review v1)

Each item below is acknowledged as a model-fidelity limitation, with
the *direction* of the resulting bias called out so readers know which
policies are favored or penalized by the approximation. The harness is
correct enough to support relative comparisons on the axes these biases
treat symmetrically; absolute numbers are not claimed.

- **Queueing formula is now M/M/1 (go-4lp).** Previously a linear
  occupancy × service formula; reviewer C estimated it under-reported
  queueing latency by ~8× at high load. Replaced with a single-server
  M/M/1 waiting-time approximation `W_q = ρ/(1-ρ) · S`, where ρ is
  slot occupancy (clamped at 0.99 for numerical stability at
  saturation) and S is the request's own prefill service time as a
  proxy for average service. *Residual bias:* service-time proxy is
  per-request, not a workload-wide average, so highly variable
  prompt lengths bias the wait estimate toward the incoming request
  rather than toward true queue composition. Relative ordering across
  policies is preserved.
- **Decode throughput is constant; no batch-size dependence.**
  *Status:* addressed in cost_model.py via `ComputeParams.decode_batch_k`
  (go-24m). At `k=0` (default, preserved for back-compat run_ids) the
  constant-decode model still applies. At `k>0` the effective per-token
  decode cost is `decode_ms_per_token / (1 + k · log(1 + max(0, batch -
  1)))`, where `batch` is the decode pod's concurrent decode count
  inclusive of the request. Sublinear by construction; batch=1
  reproduces the baseline. *Residual bias at `k=0`:* over-states decode
  latency at high concurrency, biases *against* policies that batch
  well (e.g. PD with batched decode pods). *Next step:* calibrate `k`
  against a measured vLLM / TRT-LLM decode-latency curve.
- **Concurrent KV pulls now share fabric bandwidth (fluid fair-share).**
  The engine tracks in-flight transfers on the inter-pod fabric and
  charges each new transfer against the sum of overlapping bytes, so
  `kv_transport_ms ≈ inter_pod_rtt + Σbytes_in_flight / bandwidth`.
  Prior model treated every transfer as if it owned the full fabric
  (bead go-uy0). *Direction:* tightens the previous under-estimate and
  now biases *against* pull-heavy policies when the fabric saturates;
  when only one transfer is in flight the formula reduces to the
  uncontended `rtt + bytes/B`, so low-concurrency runs are unchanged.
- **KV pull now overlaps with prefill compute (go-npl).** *Status:*
  addressed. `CostBreakdown.total_ms` composes the prefill phase as
  `max(compute_prefill_ms, kv_transport_ms)` rather than their sum —
  the pull is modeled as async-initiated at dispatch (RDMA / NCCL
  style) and therefore runs in parallel with prefill on the uncached
  tail. The phase bottleneck is the slower of the two. The raw wire
  time is still reported in the `kv_transport_ms` field and the
  fabric-contention heap in the engine still charges the full
  transport duration against the inter-pod bandwidth (concurrency
  tracking is unchanged; only the charge-timing is). *Residual bias:*
  this is an upper bound on overlap benefit — real implementations
  hit setup, synchronization, and dependency-chain stalls that
  prevent perfect parallelism. The symmetric counterpart is
  non-consecutive block residency (below), which biases in the
  opposite direction (over-estimates usable cross-pod reuse). Option
  (a) from the bead, chosen over the chunked-pipeline (b) for
  simplicity; relative ordering across policies is preserved because
  every policy sees the same rule.
- **lmsys mock tokenizer (0.25 tokens/char vs. ~0.75 real English).**
  *Direction:* under-estimates prompt length, which under-estimates
  available reuse and biases *against* prefix-aware policies on lmsys.
  *Status:* addressable — set `tokenizer: "tiktoken:cl100k_base"` in
  the lmsys workload params and install the `tokenizers` optional
  extra (`pip install 'GORGO[tokenizers]'`). Bias persists only for
  runs that keep the default mock.
- ~~**Non-consecutive block residency.** `owners_of` only checks
  per-block presence, not whether a pod owns a *consecutive* prefix.
  *Direction:* over-estimates usable cross-pod reuse and biases
  *toward* pull-heavy policies.~~ *Addressed (go-uce).* `owners_of`
  now accepts the request's ordered hash list and filters to pods
  whose cache contains every predecessor up to the queried block, so
  a scattered resident (blocks 0 and 2 but not 1) no longer qualifies
  as a reuse source for block 2. Engine pull-decision callsite passes
  `hashes` as context; the legacy per-block form is retained for
  single-block introspection.
- **`active_prefill` / `active_decode` retirement approximation.** The
  simulator now increments on dispatch and retires on projected
  completion via a heap, giving a non-monotonic load signal; it is
  still not a true event-driven scheduler. *Direction:* acceptable for
  p99 comparison at moderate concurrency; absolute throughput numbers
  should not be read off.

## 10. How to fill this report

Once runs are permitted:

```bash
routing-harness sweep --config configs/example_sweep.yaml
# results/ populated; index.json updated
# grd fill research/grd.yaml
```

The `grd fill` step (or a follow-up aggregation script) reads
`results/index.json` + `records.csv`s, emits tables into
`research/tables/`, figures into `research/figures/`, and substitutes
placeholders in this document.

## 11. Acknowledgements & references

- Mooncake KV-cache routing issue thread:
  <https://github.com/kvcache-ai/Mooncake/issues/977>
- Paper: <https://arxiv.org/abs/2603.20397>
- Paper: <https://arxiv.org/pdf/2303.06865>
- Internal notes (Google doc referenced by the task).

---

**Harness version**: `routing_harness` v0.1.0.
**Report generated from**: `research/grd.yaml`.
