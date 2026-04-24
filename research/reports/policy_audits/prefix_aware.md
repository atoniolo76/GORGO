# Policy audit: prefix-aware group

> **Bead:** go-5m8 (child of epic go-czi).
> **Policies:** `prefix-cache` (`src/routing_harness/policies/prefix_cache.py`),
> `prefix-cache-preble` (`src/routing_harness/policies/prefix_cache_preble.py`).
> **Baseline commit:** `b29e1fc` (includes Preble MVP 98465a6, throughput+sweep 4540cca,
> pending_work_ms load signal 50f5913, blake2b incremental fc039e4).

## 0. TL;DR

Both policies are functionally correct under sequential dispatch. The Preble
implementation now matches the paper on its three minimum-viable mechanisms
(exploit/explore gate, time-domain load signal, relative-imbalance hotspot
threshold). The prior "mechanism reduces to slot-count" criticism in
`docs/preble_paper_vs_impl.md` no longer applies: `PodRuntime.pending_work_ms`
is maintained by the engine as `L_i`, and the hotspot check uses `th_bal *
min_load` rather than an absolute threshold.

Remaining divergences from the paper (`M_i`, prefix auto-scaling, radix tree,
priority-group local scheduler, window `H`) are deferred in scope and declared
in the policy docstring. Four lower-severity findings (two Preble-local, one
cross-policy, one hash-utility) are documented below with discovered-from
beads filed. None are release-blocking for the sweep regime in
`research/reports/routing-comparison.md` §6.

## 1. prefix_cache.py (basic longest-prefix-match)

### 1.1 What it does

For each prefill-capable pod, walks the request's block-hash sequence in order
and counts the longest *consecutive* prefix that pod holds. Selects the pod
with the longest match; ties break on lexicographically smallest `pod_id`. If
no pod has any match, falls back to pick the pod minimizing
`(active_prefill + queued, pod_id)` — effectively least-requests.

### 1.2 Correctness

| Check | Status | Evidence |
|---|---|---|
| Block-by-block consecutive match | ✓ | Breaks on first miss (line 49–53). Matches the paged-KV invariant: block K's attention state depends on blocks `0..K-1`. |
| Tie-break determinism | ✓ | `p.spec.pod_id < best_pod` on equal match; lowest id wins (line 54–58). |
| No-match fallback | ✓ | `best_len <= 0` guard routes to least-requests (line 61–68). `best_len` is initialized to `-1`, so the guard fires when all candidates return match=0 too. |
| No cluster/KV mutation | ✓ | Contract test `test_policy_does_not_mutate_cluster` covers this. Also verified by inspection: no `install`, `touch`, or field assignment on `PodRuntime`/`KVCacheState`. |
| Empty cluster | ✓ | Returns `Decision("__none__", "__none__", "no-prefill-capable-pod")`. Contract test covers. |
| Prefix-key path | ✓ | When `request.prefix_key` is set, `_hashes` returns a single-element list; everything downstream works identically. |

### 1.3 Findings

**F1 (low): `queued` counter is a mirror of `active_prefill`.** The fallback
sort key `(active_prefill + queued, pod_id)` (line 64) double-counts the same
work. The engine's `_apply_side_effects` (`engine.py:257–258`) does
`pod.active_prefill += 1; pod.queued += 1` at dispatch and retires both
symmetrically. `queued` is therefore always equal to `active_prefill` for
pods in steady state. The ordering of the fallback is unchanged (the tie-break
collapses to pure `active_prefill`), so *ranking is preserved*, but the
absolute value is 2× what the variable name implies. This leaks into
six other policies that also read `queued` (`least_request`, `least_busy_time`,
`pd`, `session_affinity`, `vtc_basic`, and Preble's historical score). Filed
as a discovered bead for systematic cleanup — not Preble-local.

**F2 (tracking): prefix_cache has no deviation from SGLang-router-style longest
prefix match.** The policy is a thin baseline. Deliberately no hotspot
mitigation (`prefix-cache-preble` is the experimental step up).

## 2. prefix_cache_preble.py (Preble-inspired)

### 2.1 What it does

Three mechanisms match the Preble paper (Zhong et al., ICLR 2025):

1. **Exploit/explore gate (paper §E2).** If `best_match > 0` and
   `missed_tokens < cached_tokens`, bind to the prefix owner (exploit).
   Otherwise route to the lightest pod (explore). This is the paper's
   conditional — not a linear `α·match − β·load` score.
2. **Time-domain load signal.** Per-pod load is `pod.pending_work_ms`, the
   sum of stamped request latencies for in-flight requests. Implements
   Preble's `L_i` with "stamped total latency" substituted for the paper's
   `Σ (PT_r + DT_r)` regression-predicted service time.
3. **Relative-imbalance hotspot.** Exploit binds unless
   `best_load > th_bal · min_load`, in which case the request is redirected
   to the lightest pod regardless of match. Matches the paper's load-shifting
   trigger; `th_bal=1.5` is the paper's suggested default.

### 2.2 Correctness

| Check | Status | Evidence |
|---|---|---|
| Exploit/explore gate shape | ✓ | Line 99: `if best_match > 0 and missed_tokens < cached_tokens:` — conditional, not linear. |
| Time-domain load | ✓ | `_load_ms` returns `pod.pending_work_ms` (line 63–65). Engine maintains: `+= observed_latency_ms` at dispatch (engine.py:259), `-= service_ms` at retirement (engine.py:100, clamped to ≥0). |
| Relative hotspot trigger | ✓ | Line 103: `if best_load > self.th_bal * min_load:` — ratio, not absolute. |
| Hotspot redirect target is lightest pod | ✓ | Line 104–106: `min(pod_data, key=lambda t: (t[1], t[2].spec.pod_id))`. Matches paper; no `match > 0` filter (docstring note: prior gate was the binding constraint under mono-homing). |
| Explore fallback picks lightest | ✓ | Line 122–128: same min on `(load, pod_id)`. |
| No mutation of cluster/kv_cache | ✓ | Contract test. |
| Cold start | ✓ | All `pending_work_ms=0`, all match=0 → explore → lightest → deterministic smallest pod_id tie-break. |
| Empty cluster | ✓ | Returns `__none__` sentinel. |
| Cache capacity exceeded | ✓ | LRU eviction happens at engine install-time, after the policy decides. Next request's decide sees the post-eviction state and re-decides accordingly. No stale cache view in the policy. |
| Concurrent routing races | **n/a (out of scope)** | Simulator is sequential. Preble paper assumes a serialized global router. Production deployments (Modal, real vLLM replicas) should be verified separately. |

### 2.3 Findings

**F3 (medium, Preble-local): asymmetric tie-break between exploit and
explore branches.** Line 90–93: the exploit branch picks the exploit
candidate via
```python
best_match, best_load, best_pod = max(
    pod_data, key=lambda t: (t[0], -t[1], t[2].spec.pod_id)
)
```
On pods with tied `(match, load)`, `max` over `pod_id` picks the
**lexicographically largest** pod_id. The explore and hotspot-redirect
branches (line 104, 123) use `min(..., key=lambda t: (t[1], t[2].spec.pod_id))`
— smallest pod_id wins. Two requests with identical state can pick different
pods depending on which branch fires, even when the true ranking is a true
tie. Determinism is preserved *within* a branch but not *across* branches.
This is a minor correctness wart under uniform workloads; under Zipf-heavy
workloads the tie-break rarely fires because match lengths vary. Filed as
discovered bead; recommended fix is to reverse the exploit tiebreak to
`-t[2].spec.pod_id` (so `max` picks smallest).

**F4 (low, Preble-local): `missed_tokens` can go negative through the
`prefix_key` path.** Line 96–97:
```python
cached_tokens = best_match * self.block_size
missed_tokens = len(request.prompt_tokens) - cached_tokens
```
When the request uses `request.prefix_key`, `_prefix_hashes` returns a single
opaque hash, so `best_match ∈ {0, 1}`. If `best_match=1` and the prompt is
shorter than one block (`len(prompt_tokens) < block_size`), then
`cached_tokens > len(prompt_tokens)` → `missed_tokens < 0 < cached_tokens` → the
exploit gate fires. This is *probably* the intended behavior (a full hit →
exploit), but the formula is a category mismatch: `cached_tokens` is a
block-granularity estimate, `len(prompt_tokens)` is a real count. Contract
tests don't catch this (all fixtures have `len ≥ block_size`). Filed as
discovered bead; recommended fix is to clamp:
`missed_tokens = max(0, len(request.prompt_tokens) - cached_tokens)`.

**F5 (medium, simulation): `pending_work_ms` inflation via nested
`queueing_ms`.** The engine adds `cost.total_ms` to `pending_work_ms` at
dispatch (`engine.py:259`). `cost.total_ms` already includes
`queueing_ms` — an M/M/1 wait estimate computed from the current
`pod.active_prefill` (`cost_model.py:171–175`). When a pod is busy, each new
arrival's stamped latency embeds a wait-for-priors term that then accrues to
`pending_work_ms` for subsequent decisions, which raises the next arrival's
queueing estimate, and so on. The signal moves in the right direction (busy
pods accumulate faster), but the absolute magnitude exceeds the paper's
`Σ (PT + DT)` service-time sum. `th_bal` is therefore implicitly calibrated
against an inflated signal. Because we compare *ratios* (max/min), the
inflation largely cancels when pods have similar arrival histories; it
biases the ratio *upward* when one pod has deeper queue-feedback, which
makes the hotspot trigger *more eager* — typically the desired direction,
but the calibration of `th_bal` is coupled to the M/M/1 wait formula. Not a
bug per se; flagged so that anyone retuning `th_bal` is aware the number has
an implicit dependency on `cost_model._mm1_wait_ms`. Filed as discovered bead
for future consideration.

**F6 (low, utility): `enumerate_prefix_hashes` raises on `block_size=0`.**
Line 171 of `kv_cache.py`: `if i % block_size == 0:` → `ZeroDivisionError`.
No input validation. Both `prefix_cache` and `prefix_cache_preble` default to
`block_size=16`, and the config surface constrains this at the dataclass
level, but misconfigured sweeps would crash with an opaque traceback. Filed
as discovered bead.

### 2.4 Revalidation of the commits referenced in the bead

| Commit | Change | Audit result |
|---|---|---|
| `fc039e4` | Incremental blake2b | Output byte-identical to legacy `b",".join(...)` format. Verified by existing `test_kv_cache::test_enumerate_prefix_hashes_matches_legacy` (line 49-58 of test_kv_cache.py); re-verified here with a regression test (`test_enumerate_prefix_hashes_stable_across_block_boundaries`). |
| `98465a6` | Minimum viable Preble + queued fix | `queued` is now kept in lockstep with `active_prefill` by the engine. The "dead counter" Problem A from the divergence doc is resolved in bookkeeping terms (see F1 for naming caveat). Preble policy no longer reads `queued`. |
| `4540cca` | Throughput starvation + sweep v4 | Out of scope for prefix-aware (affects `throughput` policy). Noted; no cross-effect on prefix-aware. |
| `50f5913` | pending_work_ms replaces EWMA proxy | Faithful-to-paper. Preble's `L_i` definition is `Σ (PT+DT)` over window `H`; we substitute "stamped total latency of in-flight requests" (no window). See F5 for the calibration caveat. |

### 2.5 Paper-fidelity summary

| Dimension | Preble paper | Our impl | Status |
|---|---|---|---|
| Load signal | Σ regression PT+DT over window `H` | `Σ stamped total_ms for in-flight requests` (no window) | **Faithful-ish** (time-domain; no window) |
| Hotspot trigger | `max > Th_bal · min` (ratio) | `best > th_bal · min` (ratio) | **Faithful** |
| Route rule | `exploit iff missed < cached else minimize L_i` | Same conditional | **Faithful** |
| Eviction cost `M_i` | Explicit term | Not modeled | **Deferred** (docstring note) |
| Prefix auto-scaling | Replicate on `queue time · 2` in `H` | Absent | **Deferred** (docstring note) |
| Radix tree | Global, per-node GPU-set | Per-pod block-hash LRU | **Deferred** (no `M_i` → no need) |
| Migration | None | None | **Match** |
| Priority-group local scheduler | Per-pod | Not modeled | **Deferred** (engine is one-pass) |
| Queueing-denominator consistency | n/a (time, not slots) | Time-domain in policy; M/M/1 in cost model. Disagreement removed. | **Resolved** |

## 3. Test coverage assessment

### 3.1 Already covered (before this audit)

- Random policy determinism under seed (`test_random_is_deterministic_under_seed`)
- Least-request min-load pick (`test_least_request_picks_min_load`)
- Prefix-cache routes to owner (`test_prefix_cache_routes_to_owner`)
- Preble hotspot redirect under mono-homing (`test_prefix_cache_preble_avoids_hotspot`)
- Preble exploit binds to owner when not a hotspot (`test_prefix_cache_preble_exploit_binds_to_owner`)
- Preble explore picks lightest on insufficient cache (`test_prefix_cache_preble_explore_picks_lightest`)
- Contract: registered, decides on nonempty cluster, handles empty cluster, no mutation
- `enumerate_prefix_hashes` matches legacy format (`test_kv_cache::test_enumerate_prefix_hashes_matches_legacy`)
- `owners_of` consecutive residency (`test_kv_cache::test_owners_of_requires_consecutive_residency` — from commit 156ff23)

### 3.2 Gaps filled by this audit

New file `tests/unit/test_prefix_aware_audit.py`:

- Cold start (no cache, all loads zero) → explore → lowest pod_id
- Single-pod cluster → never redirects regardless of load
- Explore-branch tie-break (two pods identical load) → lowest pod_id
- Exploit-branch tie-break (two pods identical match + load) → documents current (highest-id) behavior, with an `# F3` annotation so future fixers find the test
- Prefix-key path for both policies (short prompt via `prefix_key`)
- Exploit gate boundary (exactly half-cached → explore; just over half → exploit)
- Hotspot boundary (`load == th_bal * min_load` exactly → no redirect; `+ε` → redirect)
- Cache capacity exceeded — install evicts an entry, subsequent decide re-routes
- No-match fallback in `prefix_cache` reduces to least-requests
- `prefix_cache` pure-tie on matches picks lowest pod_id
- Stability under sequential dispatch: repeated identical requests converge on the owner (exploit stays sticky) in the absence of hotspot load

## 4. Modal smoke config

New file `configs/smoke_prefix_aware_modal.yaml` — a mixed-prefix workload
designed to force the cache-hit-vs-load tradeoff into every dispatch
regime:

- 4-family Zipf (s=1.3) workload → one dominant family
- 3 colocated pods, small KV budget → cache pressure
- 200 requests @ 8 QPS → ~25 seconds of wall-clock traffic
- Two runs: `prefix-cache` (baseline) and `prefix-cache-preble` (th_bal=1.5)
- Budget target: ≤ $1 Modal spend (3 pods × ~30s active = ~1.5 pod-minutes on
  A10G / L4 class; well under the epic's per-smoke budget)

Expected signatures in the output (based on sweep v4 §6.2):

- `prefix-cache-preble` **p95 margin** over `prefix-cache`: ≥ **1000 ms lower**
  at this QPS (paper regime at 8 QPS showed −3,745 ms; scaled-down smoke
  should show directionally the same effect).
- Skew (σ of per-pod request counts) should be ≥ **2× lower** under
  Preble than under plain prefix-cache.
- If either signal is absent, escalate — it would indicate a regression in
  the hotspot mitigation path.

The config is marked `# HUMAN-GATED` per epic protocol; a follow-on bead
(`go-3j8`) tracks actually executing Modal runs.

## 5. Discovered-from beads (filed)

- **F1**: `queued` counter is a mirror of `active_prefill` — cross-policy cleanup.
- **F3**: Preble tie-break asymmetry between exploit and explore branches.
- **F4**: `missed_tokens` can go negative via short-prompt + `prefix_key`.
- **F5**: `pending_work_ms` inflation via nested `queueing_ms`; `th_bal`
  calibration is coupled to `cost_model._mm1_wait_ms`.
- **F6**: `enumerate_prefix_hashes` `ZeroDivisionError` on `block_size=0`.

## 6. Verdict

- `prefix-cache`: **pass** with F1 observation (cross-policy cleanup).
- `prefix-cache-preble`: **pass** on paper-fidelity for the three
  minimum-viable mechanisms. F3 (tie-break) and F4 (negative missed_tokens)
  are minor; F5 is a simulation-calibration note; F6 is a hash utility
  hardening item. None block the sweep-v4 conclusions reproduced in
  `research/reports/routing-comparison.md` §6.

Ready for Modal smoke (via `configs/smoke_prefix_aware_modal.yaml`, gated
on human approval).
