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

Prefix-aware routing (e.g. SGLang, Mooncake Conductor) can dramatically
cut prefill cost by reusing KV state already cached on a specific pod.
But naive prefix-locality concentrates load on a few "hot" pods,
trading cache reuse for queueing delay. Preble-style approaches argue
for a joint objective; PD disaggregation argues for decoupling the two
phases; VTC argues that fairness constraints change the Pareto
frontier entirely.

Existing evaluations are pairwise and report-specific. We need a
harness that holds workload, topology, and cost assumptions constant
while varying only the policy.

### 1.2 Contributions

- **A taxonomy** of routing strategies along five axes (Section 3).
- **A harness** with 11 pluggable policies, a prefix-level KV model,
  and a deterministic discrete-event simulator (Section 4).
- **A comparison protocol** with explicit cost model, metrics, and
  reproducibility guarantees (Section 5).
- **A first characterization** of the reuse-vs-latency Pareto curve
  on synthetic workloads and lmsys-chat-1m turn replays
  (Sections 6–7, quantitative results to be filled).

## 2. Related Systems (conceptual references, not reimplementations)

| System | Key idea | What we model |
|-----------------------|--------------------------------------------|----------------------|
| Mooncake Conductor | KV-cache-aware dispatch; hotspot detect. | `prefix-cache`, `prefix-cache-preble` |
| SGLang Router | Prefix-tree routing on a shared cache. | `prefix-cache` |
| NVIDIA Dynamo | PD disaggregation + scheduling. | `pd` policy |
| Ant Group AI Gateway | Heterogeneous dispatch under SLA. | Cost model hooks |
| AIBrix | Cluster-level LLM serving orchestration. | Topology abstraction |
| Preble | Prefix-locality + load balancing. | `prefix-cache-preble` |
| VTC | Virtual token counter for fairness. | `vtc-basic` |

We do **not** claim parity with any of these systems; policies are
baselines inspired by their published designs with documented
assumptions (see `src/routing_harness/policies/`).

## 3. Taxonomy

Five orthogonal axes. Any system can be located along each:

1. **Cache awareness**: none → capacity-aware → prefix-aware →
   prefix+load-aware.
2. **Phase separation**: colocated → disaggregated (PD) → hierarchical.
3. **Fairness**: best-effort → tenant-fair → per-session-fair (VTC).
4. **Statefulness**: stateless hash → EWMA → full session affinity.
5. **Hotspot mitigation**: none → threshold-based → score-penalized
   (Preble-style) → migration-based.

Each implemented policy is placed in this space in Table 1.

> `{{table_policy_taxonomy}}` — placement of all 11 policies across
> the five axes.

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

## 6. Expected Results (placeholders)

### 6.1 Headline table

> `{{table_headline}}` — per-policy p50/p95/p99 (ms), capture_rate,
> load skew, migrations, mean KV transport (KiB/request), averaged over
> seeds and a fixed (qps, zipf_s) point.

### 6.2 Reuse vs latency Pareto

> `{{figure_reuse_vs_latency}}` — scatter of capture_rate (x) against
> p95 latency (y), one point per (policy, qps, zipf_s). Pareto front
> highlighted; we expect `prefix-cache-preble` and `pd` to dominate
> the corner, `random` to sit on the inefficient side, and
> `prefix-cache` to be fastest at low load but degrade under high
> Zipf s due to hotspot concentration.

### 6.3 Hotspot mitigation

> `{{figure_hotspot_mitigation}}` — skew vs `zipf_s` for
> `prefix-cache` vs `prefix-cache-preble`. We expect the gap to widen
> as `zipf_s` increases.

### 6.4 PD gains

> `{{figure_pd_vs_colocated}}` — p99 latency for `pd` on PD topology
> vs `prefix-cache` on colocated topology at matched total GPU count.
> We expect PD to win on decode-heavy workloads and lose when prefill
> dominates (small outputs).

### 6.5 Fairness under contention

> `{{figure_vtc_fairness}}` — per-session latency CDF for `vtc-basic`
> vs `least-busy-time` when one session dominates token volume.

## 7. Analysis (to be written)

### 7.1 The reuse-vs-latency frontier

Filling out:

- Does prefix matching *alone* monotonically improve p95? (Hypothesis:
  no; hotspotting degrades p95 above some Zipf threshold.)
- Does adding load-awareness flatten the frontier (Preble), and at
  what cost to capture rate?

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
2. **Cost model calibration.** `AnalyticCostModel` coefficients are
   illustrative. Real prefill/decode ms/token must be measured on a
   given model × GPU before any quantitative claim is made. The
   `InstrumentedCostModel` scaffolding supports overriding analytic
   values with observations.
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

- **Queueing formula is not M/M/1.** A reviewer estimated the current
  linear formula under-estimates queueing latency by ~8× at high load.
  *Direction:* under-reports absolute p99; relative ordering at tails
  is preserved because the term scales with `active_prefill` for every
  policy.
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
- **KV pull is still synchronous — no RDMA pipelining.** *Direction:*
  over-penalizes small cross-pod pulls even with contention tracking,
  because setup cost is paid in full per transfer.
- **lmsys mock tokenizer (0.25 tokens/char vs. ~0.75 real English).**
  *Direction:* under-estimates prompt length, which under-estimates
  available reuse and biases *against* prefix-aware policies on lmsys.
  *Status:* addressable — set `tokenizer: "tiktoken:cl100k_base"` in
  the lmsys workload params and install the `tokenizers` optional
  extra (`pip install 'GORGO[tokenizers]'`). Bias persists only for
  runs that keep the default mock.
- **Non-consecutive block residency.** `owners_of` only checks
  per-block presence, not whether a pod owns a *consecutive* prefix.
  *Direction:* over-estimates usable cross-pod reuse and biases
  *toward* pull-heavy policies.
- **`active_prefill` / `active_decode` retirement approximation.** The
  simulator now increments on dispatch and retires on projected
  completion via a heap, giving a non-monotonic load signal; it is
  still not a true event-driven scheduler. *Direction:* acceptable for
  p99 comparison at moderate concurrency; absolute throughput numbers
  should not be read off.
- **Taxonomy axis overlap.** Statefulness and cache-awareness in §3
  are partially collinear (stateful policies are usually
  cache-aware). Revising the taxonomy into orthogonal axes is deferred
  to a follow-up.

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
