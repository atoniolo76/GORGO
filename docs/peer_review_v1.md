# Peer Review v1 — Findings and Response

Three independent critics reviewed the first cut of the harness:

- **Critic A** (ML-systems research peer review, publication-readiness).
- **Critic B** (production-grade Python code review, bug hunt).
- **Critic C** (simulation-fidelity deep dive on cost + KV models).

A fourth critic (Codex, `codex review`) was dispatched but failed in its
sandbox (`bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`)
before producing findings. We proceed on the three that landed.

Their findings overlapped to a striking degree on the correctness
issues; the severity scale below collapses all three.

## Must-fix (correctness — break the simulation's claims)

1. **`per-tenant-load-balance` (formerly `vtc-basic`) does not enforce fairness.**
   Picks pod by least-busy-time; counter is written to `Decision.score` but never
   consulted, and `observe_completion` is never called by the engine.
   It is a silent alias for `least-busy-time`. *Action:* wire an
   optional `on_complete` hook in the engine; update the policy to use the
   counter as a pod-selection priority; add a test.
2. **`pd` does not model prefill-decode disaggregation.** The engine
   routes only to `decision.prefill_pod_id`; the prefill→decode KV
   handoff is never costed. *Action:* charge
   `kv_bytes_per_token * prompt_tokens / bandwidth + inter_pod_rtt`
   when `prefill_pod_id != decode_pod_id`; install prefix blocks on
   *both* pods post-decode.
3. **Cross-pod KV pull corrupts cache state.** The engine increments
   `captured` for pulled blocks but never calls `kv_cache.install` on
   the destination pod. Subsequent requests believe prefixes are
   cached that were never written. Capture-rate metric is overstated.
   *Action:* `install` the pulled blocks on the destination pod with
   the pulled byte count.
4. **`active_prefill` / `active_decode` never increment during a run.**
   All policies that use load as a signal (Preble, LBT, per-tenant-load-balance) route
   against a frozen snapshot. Comparative advantage of load-aware
   policies is structurally suppressed. *Action:* increment on
   `decide`, decrement on a modeled completion; since the engine is
   one-pass without true time advance, use an arrival-count
   approximation documented as such.
5. **`session_affinity._bindings` leaks into the config snapshot.**
   `asdict()` serializes the runtime binding map into `config.json`,
   making `run_id` change between otherwise-identical runs and
   breaking the content-addressed reproducibility claim. *Action:*
   exclude private `_*` fields from the snapshot.
6. **`prefix-cache-preble` is Preble-*inspired*, not Preble.** The
   formula is a simple linear combination with magic-number
   coefficients. The actual Preble paper optimizes over a reuse graph.
   *Action:* rename in docs (not the policy id, which would break
   config stability) and strengthen the docstring to cite the
   deviation.

## Should-fix (structural + hygiene)

7. **`modal` is a mandatory dep but unused by the harness.** Install
   pulls a large unrelated SDK. *Action:* move to
   `[project.optional-dependencies]` so the harness installs lean.
8. **`get_policy` fails out-of-box** if the caller doesn't also import
   `routing_harness.policies`. *Action:* auto-register at package
   import by importing `policies` from `routing_harness/__init__.py`.
9. **`merge_sorted` is dead code** with a broken dead branch
   (`trace.py:55`). *Action:* delete; nobody calls it.
10. **`_build_dc` is dead code** (`config/schema.py:144`); the
    `ConfigError` guarantee is satisfied *accidentally* by
    `TypeError` from `**kwargs`. *Action:* delete it and replace with
    explicit required-key checks that actually raise `ConfigError`.
11. **e2e test uses `pfx_cap >= rnd_cap`** — passes even if both are
    zero. *Action:* also assert `pfx_cap > 0`.
12. **Contract test does not check `kv_cache` mutation** — policies
    could mutate KV state and tests would not notice. *Action:*
    snapshot + compare cache entries too.
13. **`capture_rate` denominator is aggregate-weighted.** One request
    with 1000 available blocks dominates 999 requests with 1 each.
    *Action:* report both macro (mean over requests) and micro
    (aggregate) in the metrics summary.
14. **TTFT is missing** from required metrics. *Action:* add
    `ttft_ms = routing + queueing + compute_prefill + kv_transport`
    and report percentiles.
15. **Sweep config uses colocated base, so `pd` runs its fallback
    path** (degrades to prefix-cache when no prefill/decode role
    split). *Action:* add a PD-specific sweep config that uses the
    disaggregated topology for the `pd` policy.
16. **Typo `unchached` in cost_model.py.** *Action:* fix.

## Documented as gaps (out-of-scope; model-fidelity, not correctness)

Moved into `research/reports/routing-comparison.md` §9 explicitly,
each with a direction-of-error annotation so readers know the bias:

- ~~**Queueing formula is not M/M/1**~~ (Critic C: under-estimates by
  ~8× at high load). *Addressed in go-4lp.* `AnalyticCostModel` now
  uses `W_q = ρ/(1-ρ) · S` with ρ clamped at 0.99 and S = this
  request's uncached prefill cost. Residual: service-time proxy is
  per-request rather than a workload-wide average.
- **Decode throughput is constant; batch-size dependency absent.**
  *Status (go-24m):* now configurable via
  `ComputeParams.decode_batch_k`. Default `k=0` preserves the constant
  behavior (and pinned run_ids); `k>0` amortizes per-token decode
  sublinearly with the decode pod's concurrent batch. Remaining gap is
  calibration of `k` against measured serving data — see §9.1 of
  research/reports/routing-comparison.md.
- **KV pull is synchronous, no RDMA pipelining.** *Direction:*
  over-penalizes small cross-pod pulls → biases *against* policies
  that exploit fine-grained prefix sharing.
- **`lmsys` mock tokenizer: 0.25 tokens/char (actual English ~0.75).**
  *Direction:* under-estimates prompt length → under-estimates
  available reuse → biases *against* prefix-aware policies on lmsys.
  *Status:* addressed in go-pf8. `TraceParams.tokenizer` now accepts
  `"tiktoken:<encoding>"` (e.g. `"tiktoken:cl100k_base"`); install the
  `tokenizers` optional extra to enable. Default remains `"mock"` so
  the base install works without external deps, and an unmet
  `tiktoken:*` request raises a loud `RuntimeError` rather than
  silently regressing to the biased mock.
- **Non-consecutive block residency** breaks the strict "prefix"
  assumption in `owners_of`. *Direction:* over-estimates usable
  cross-pod reuse → biases *toward* pull-heavy policies.
- **`active_prefill` approximation** (increment-only without
  decrement): load signal is monotonic rather than oscillating.
  *Direction:* over-estimates saturation under sustained load;
  acceptable for p99 comparison, not for absolute throughput.
- **Taxonomy axis overlap** (statefulness vs cache awareness).
  *Action (deferred):* revise taxonomy in a follow-up; current
  5-axis framing is retained with a footnote explaining the
  overlap.

## New beads filed (out-of-scope for this iteration)

- PD sweep topology + PD ablation experiment design.
- ~~M/M/1 queueing model implementation + calibration.~~
  Addressed in go-4lp via `_mm1_wait_ms` with ρ clamped at 0.99;
  calibration vs. measured serving data remains.
- ~~Real-tokenizer (tiktoken or model-native) lmsys adapter.~~
  Addressed in go-pf8 via optional `tokenizers` extra and
  `TraceParams.tokenizer` flag.
- ~~Batch-size-dependent decode cost + continuous-batching model.~~
  Addressed in go-24m — `ComputeParams.decode_batch_k` with logarithmic
  amortization; default 0.0 preserves legacy run_ids.
- Network-contention model for concurrent KV pulls.
