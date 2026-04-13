# Hotspot Ground-Truth Labeling — Design Note (proposal, unapproved)

> Status: **proposal, awaiting scout approval**. Scope: unblock the §6.3
> placeholder in `research/reports/routing-comparison.md` and gap #7 in §9
> ("Hotspot ground truth: precision/recall of hotspot detection requires a
> ground-truth labeler, not yet defined"). No labeler code is landed yet.
> Filed against `go-n2h`; discovered-from `go-y55`.

## What we need

A binary, per-request label `is_hot(req) ∈ {0, 1}` such that we can compute
precision and recall of `prefix-cache-preble`'s hotspot-avoidance decisions
(the `top_load > hotspot_threshold` branch in `prefix_cache_preble.py:76`),
and do so in a way that is **not tautological with the policy's own inputs**.

The object being classified is the *request-to-pod decision*, not the pod.
A decision is a true positive when (a) the request targets a hot prefix
AND (b) the policy avoids a pod that is concurrently over-loaded.

## Candidates considered

| # | Definition | Verdict |
|---|------------|---------|
| 1 | **Pod-level, threshold on busy-ms** (pod is hot at `t` if `busy_ms(t, t+w) > p90(cluster)`) | Reject — tautological. The policy's own input signal is `(active+queued)/capacity`, a monotone function of the same busy-ms window. Precision of "avoids high-busy-ms pods" against "high-busy-ms pods" is circular, and the load-skew metric already reports this. Also unstable at small `len(cluster)` (≤4 pods → p90 is degenerate). |
| 2 | **Prefix-level, threshold on arrival rate** (prefix is hot if its observed rate exceeds `zipf_tail_threshold = c × total_qps / n_prefix_families`) | Useful but secondary. Arrival rate is observable in both synthetic and lmsys traces, so this is the only candidate that works on replayed data. Threshold `c` is arbitrary; results vary with `zipf_s`. Label is noisy at the head/tail boundary. |
| 3 | **Synthetic with injected hotspots** (workload generator marks a fixed subset of prefix families as "hot", inflates their arrival rate deterministically, emits the label on each `Request`) | **Proposed primary.** Ground truth is exact (not inferred from a threshold), seed-reproducible, and independent of any policy's input signal. Cost: new workload flag + a `Request.metadata["hot"]` column. |
| 4 | Human annotation / offline clustering | Rejected — overkill for a simulator. |

## Proposal

1. **Primary ground truth: option 3 (injected synthetic).** Add a
   `hotspot_families: tuple[int, ...]` and `hotspot_rate_multiplier: float`
   to `SyntheticParams`. When a request is drawn from a family in
   `hotspot_families`, `metadata["hot"] = True`. The Zipf draw continues
   as today; the multiplier is applied by re-sampling the family with
   probability proportional to the multiplier (a mixture, not a
   post-hoc reweight — keeps arrivals Poisson per family).
2. **Secondary ground truth for lmsys replay: option 2 (rate threshold).**
   Compute per-prefix arrival rate over a sliding window of the replayed
   trace; mark the top-k families (default k chosen so cumulative mass ≈
   configurable `hot_mass_frac`, default 0.3) as hot. This is inferred,
   not injected, and its weaknesses are disclosed when it is used.
3. **Decision label.** A policy decision for request `r` is counted as a
   *hotspot-avoidance action* when its rationale string is
   `hotspot-avoid …` (already emitted by `prefix_cache_preble.py:82`).
   - TP: `is_hot(r) ∧ avoidance_action`
   - FP: `¬is_hot(r) ∧ avoidance_action`
   - FN: `is_hot(r) ∧ ¬avoidance_action` **and** the top-scoring pod was
     above the load threshold (i.e., there was a hotspot to avoid).
   - TN: everything else.
4. **Metrics reported in §6.3.** `precision`, `recall`, and
   `decisions_per_1k_requests` for `prefix-cache-preble` across the sweep.
   Only reported on workloads where a ground-truth mode is active;
   absent otherwise (instead of a misleading zero).

## Tradeoffs called out

- **False-positive risk.** Option 3 is exact by construction; option 2
  has boundary noise (families just below the threshold are labeled
  cold but behave hot). We disclose the threshold `c` and `hot_mass_frac`
  next to every P/R figure.
- **Observability / transferability.** Option 3 does not exist on
  replayed lmsys data. We accept this: synthetic figures are the
  primary claim; lmsys P/R is reported separately, with option-2
  caveats, as a sanity check.
- **Stability under workload shift.** A fixed `hotspot_rate_multiplier`
  produces labels that are invariant under scaling `arrival_rate_qps`
  and `zipf_s`. Option 1 would flip labels on every QPS change; option
  2 flips labels on `zipf_s` change (both avoided).
- **Coupling to policy internals.** The decision label (#3) reads
  `rationale`, a string. That is load-bearing and should be pinned by
  test. Alternative: add an explicit `Decision.tag: str | None` enum.
  We propose adding the enum in the same landing as the labeler.
- **Small-cluster artefacts.** All options degrade at `n_pods ≤ 4`;
  option 3 at least keeps ground truth sharp. We cap the reported sweep
  at `n_pods ≥ 4` and note the floor.

## Open questions for scout

1. Approve **option 3 primary + option 2 fallback**, or prefer a single
   definition? (Picking one simplifies the report; two widens coverage.)
2. Accept **rationale-string decision tagging** as the short-term label
   source, with the `Decision.tag` enum as a follow-up?
3. Default values: `hotspot_families = (0, 1, 2)`,
   `hotspot_rate_multiplier = 5.0`, `hot_mass_frac = 0.3` — OK to land,
   or specify different priors?

## Next step (gated on approval)

After scout approves: implement the labeler in
`src/routing_harness/workload/synthetic.py` + a `hotspot.py` evaluator
module, wire into `MetricsCollector.summary()`, add unit tests, run the
sweep for `prefix-cache-preble`, and fill the §6.3 placeholder. Leave
§9 gap #7 open until the figure lands.
