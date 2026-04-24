# Policy audit: load-balancing (stateless) group

> **Bead:** go-651 (child of epic go-czi).
> **Policies:** `random` (`src/routing_harness/policies/random.py`),
> `least-request` (`least_request.py`),
> `least-busy-time` (`least_busy_time.py`),
> `least-latency` (`least_latency.py`),
> `least-kv-cache` (`least_kv_cache.py`),
> `throughput` (`throughput.py`).
> **Baseline commit:** `6b54ae3` (tip of `main` before this audit).

## 0. TL;DR

Four of the six policies (`random`, `least-request`, `least-busy-time`,
`least-latency`) are correct under the simulator's assumptions and rotate
traffic in the expected regimes. Two have bugs that are reproducible against
the full engine on a tiny trace:

- **`throughput` (F7, HIGH):** the starvation fix in commit `4540cca` is
  *incomplete*. `ewma_throughput_tps` is initialized to `0.0` and only
  advances via `_apply_side_effects` at dispatch — so cold pods never
  accumulate a positive score. `max(score)` picks the first-warm pod on
  every subsequent request. Reproduced: 60/0/0 dispatch across 3 colocated
  pods at 50 QPS on a synthetic trace (§2.6.3). The 4540cca change
  normalizes by `1 + active`, which helps only after all pods have
  non-zero EWMA; under the realistic init path they never do.
- **`least-kv-cache` (F10, MEDIUM):** under shared-prefix workloads,
  install-after-cache-hit is a byte-level no-op, so the pod that was first
  warmed holds its `free = cap - bytes_used` value forever. Combined with
  the policy's `max(..., pod_id)` tie-break (F9), the pod with the
  **largest** pod_id keeps winning. Reproduced: 58/1/1 skew at 50 QPS on
  three colocated pods sharing a single prefix family (§2.5.3).

Plus one cross-cutting wart (F11: inconsistent tie-break direction across
the group) and three information-only items on signal staleness,
oversubscription, and the `queued` mirror (F1 from the prefix-aware
audit).

Counts in this audit continue from the prefix-aware audit's
`F1`-`F6`; new findings start at `F7`.

## 1. Method

Each policy was read top-to-bottom and cross-checked against the engine
(`src/routing_harness/simulator/engine.py`) to confirm the provenance of
every signal it reads. Where a bug was suspected, it was reproduced on the
real engine on a 30–60-request synthetic trace (see §2.5.3 and §2.6.3).
All six policies were exercised end-to-end through the
`AnalyticCostModel` + `SimulationEngine` path, with the default fixture
of three colocated pods.

## 2. Per-policy findings

### 2.1 `random.py`

#### 2.1.1 What it does

Uniform-random pick over `cluster.prefill_capable()`, using a seeded
`random.Random` instance so two instances with the same seed produce
identical decisions.

#### 2.1.2 Correctness

| Check | Status | Evidence |
|---|---|---|
| Deterministic under seed | ✓ | `__post_init__` constructs `random.Random(self.seed)`. Existing test `test_random_is_deterministic_under_seed`. |
| Empty cluster → `__none__` | ✓ | Lines 36–38. Contract test covers. |
| Single-pod cluster | ✓ | `_random.choice([only])` returns that pod. |
| No cluster/kv mutation | ✓ | Contract test. Policy reads but never writes. |
| Distribution on N→∞ | ✓ | Confirmed empirically: 60-request trace, 3 pods → 15/23/22 (§2.6.3). Within the expected binomial spread. |

#### 2.1.3 Findings

None. `random` is the taxonomy floor and behaves as a correct floor.

---

### 2.2 `least_request.py`

#### 2.2.1 What it does

`min(cands, key=lambda p: (active_prefill + active_decode + queued, pod_id))`.
Rotates traffic toward the least-loaded pod; ties break on lexicographically
smallest `pod_id`.

#### 2.2.2 Correctness

| Check | Status | Evidence |
|---|---|---|
| Min-load selection | ✓ | Existing test `test_least_request_picks_min_load`. |
| Deterministic tie-break | ✓ | `pod_id` appended to sort key. |
| Signal freshness | ✓ | `_retire_up_to(now)` at `engine.py:112` drains completed requests *before* `decide`, so active counts reflect the current instant. |
| Starvation under high load | ✓ | 60/3 → 20/20/20 rotation at 50 QPS, 60 req (§2.6.3). |
| Starvation under low load | ✗ (expected, not a bug) | At 5 QPS with ~50 ms service, all requests retire before the next arrives → active=0 everywhere → tie-break on `pod_id` → `p0` wins every time. Correct behavior under an uncontended cluster; load is low enough that the skew has no performance cost. Flagged as §4.2 observational note, not a defect. |

#### 2.2.3 Findings

**F1 (carried over from prefix-aware audit, LOW):** the sort key
double-counts `active_prefill` because the engine keeps `queued` in
lockstep with `active_prefill` (see `engine.py:258`). Ranking is
preserved because the double-count is uniform across pods, but the score
value reported via `Decision.score` is 2× the expected. No new action;
tracked under the prefix-aware audit's discovered-bead cleanup.

---

### 2.3 `least_busy_time.py`

#### 2.3.1 What it does

`min(cands, key=lambda p: (ewma_latency_ms * (active_prefill + active_decode + queued), pod_id))`.
Scales raw load count by the EWMA-smoothed per-request latency so that
heterogeneous request sizes cost proportionally more.

#### 2.3.2 Correctness

| Check | Status | Evidence |
|---|---|---|
| Product of load × latency is the right envelope | ✓ | `busy_time ≈ concurrent_requests × service_time` is the "pod is busy for this many ms" envelope. |
| Deterministic tie-break | ✓ | `pod_id` appended. |
| Cold start (all active=0) | ✓ | Product is zero for all; tie-break picks `p0` (smallest id). Matches `least-request`. |
| Warm state rotation | ✓ | 20/20/20 at 50 QPS (§2.6.3). |
| EWMA latency staleness | ✓ (self-correcting) | EWMA updates only at dispatch (`engine.py:249`). If a pod's load drains without new dispatches, its EWMA stays elevated — but the `(active+queued)` multiplier falls to 0, zeroing out busy_time regardless. So stale EWMA cannot starve a drained pod. |

#### 2.3.3 Findings

None new. The product formulation makes the stale-EWMA concern benign
(§2.3.2 last row).

---

### 2.4 `least_latency.py`

#### 2.4.1 What it does

`min(cands, key=lambda p: (ewma_latency_ms, pod_id))`. Pure EWMA-latency
ranking.

#### 2.4.2 Correctness

| Check | Status | Evidence |
|---|---|---|
| Cold start is survivable | ✓ | `engine.py:63–65` initializes `ewma_latency_ms = initial_warm_latency_ms` (default 5.0 ms) for every pod whose EWMA is 0. All pods tied at 5.0 → tie-break picks `p0`. Without the warm-init, cold pods with `ewma=0` would monopolize traffic. |
| Does NOT monotonically starve | ✓ | After `p0` serves a real request, its EWMA rises above `5.0` (because observed latency includes routing + queueing + compute + network → ~60–80 ms). Next decide: `p1` has `ewma=5.0 < p0.ewma` → `p1` wins. Reproduced: 20/20/20 rotation at 50 QPS (§2.6.3). |
| Deterministic tie-break | ✓ | `pod_id` appended. |
| Oscillation bound | ✓ | The policy cycles pods at the cadence of EWMA decay: with `alpha=0.2`, a dispatched pod's EWMA is pulled ~20% toward the observation and away from `5.0`. Three pods cycle roughly round-robin until all three EWMAs converge, at which point the tie-break (lowest `pod_id`) takes over. No indefinite starvation. |

#### 2.4.3 Findings

None new. Depends critically on the `initial_warm_latency_ms` warm-init;
dropping that init back to `0.0` would reintroduce the "cold pod wins
all" failure mode.

---

### 2.5 `least_kv_cache.py`

#### 2.5.1 What it does

`max(cands, key=lambda p: (free(p), pod_id))` where `free(p) = cap -
kv_cache.size_bytes(pod_id)`. Tries to route toward pods with the most
free KV-cache headroom to avoid forcing evictions.

#### 2.5.2 Correctness (the good parts)

| Check | Status | Evidence |
|---|---|---|
| `free` computation handles missing pod | ✓ | Line 37 guards with `if pod_id in kv_cache.pods`. Missing → `used=0`. |
| No mutation | ✓ | Contract test. |
| Empty cluster | ✓ | `__none__` sentinel. |
| Unique-prefix workload rotation | ✓ | 60 unique-prefix requests → 20/20/20 at 50 QPS (§2.6.3). Each dispatch installs new bytes, `free` decreases, round-robin follows. |

#### 2.5.3 Findings

**F9 (MEDIUM, least-kv-cache): tie-break to largest `pod_id` is
inconsistent with the rest of the group.** `max(..., key=(free,
pod_id))` selects the largest `pod_id` on a tie. Every other
load-balancing policy in this group (and the prefix-aware group) uses
`min(...)` with `pod_id` appended, yielding the *smallest* `pod_id`
on a tie. Harmless on its own, but combined with F10 (below) it
determines which pod monopolizes traffic under shared-prefix workloads.
Fix: negate the tie-break to `(free(p), -ord?)`/`-pod_id_sortable`, or
restructure to `min(-free, pod_id)`. Filing as discovered.

**F10 (MEDIUM, least-kv-cache): shared-prefix workloads cause 2/3 pod
starvation.** When all requests share a prefix family, the first dispatch
installs it on some pod P. Subsequent requests routed to P are byte-level
no-ops (the hashes already exist in `pod.entries`; `PrefixEntry.install`
replaces in place). So `P.free` does not change, and if `P.pod_id` is
the lex-largest (F9), `P` keeps winning the tie-break. Other pods never
get warmed, so their `free` stays at `cap` and is tied with `P.free`
only when P is also at `cap`. Once any byte-level delta occurs on a
non-P pod, P will re-win forever once the whole cluster is warm.
**Reproduced**: 60-request shared-prefix trace @ 50 QPS →

```
least-kv-cache dispatch (shared prefix): {'p2': 58, 'p1': 1, 'p0': 1}
```

Under unique-prefix traffic the pathology vanishes (20/20/20), so the
bug is workload-dependent. It is a real concern for realistic GORGO-style
traffic (conversations with heavy shared system prompts). Suggested fix:
add a secondary load signal (e.g., `active_prefill`) to the sort key so
the policy falls back to load-balancing when free-byte tied:

```python
pick = max(cands, key=lambda p: (free(p), -p.active_prefill, pod_id))
```

Or, more principled, rank by a blended metric `free_bytes −
β·active_prefill·avg_tokens·kv_bytes_per_token` so that free-capacity
and demand are commensurate.

Filing as discovered.

**F12 (INFO, cross-cutting): no policy in this group rejects an
over-saturated pod.** `cluster.prefill_capable()` only filters on `role`;
nothing checks `active_prefill ≥ max_concurrent_prefill`. The engine
docstring (`engine.py:10–15`) declares this a soft limit, but a policy
that wanted admission control would need to layer it on top. Not a bug;
recorded so future policies know the contract.

---

### 2.6 `throughput.py`

#### 2.6.1 What it does

```python
pick = max(
    cands,
    key=lambda p: (
        p.ewma_throughput_tps / (1 + p.active_prefill + p.active_decode),
        -ord(p.spec.pod_id[0]),
    ),
)
```

Ranks pods by *available* throughput (EWMA tokens/s normalized by current
concurrency). Commit `4540cca` added the `/(1+active)` term to fix a
single-pod monopoly seen in sweep v4.

#### 2.6.2 Correctness (the good parts)

| Check | Status | Evidence |
|---|---|---|
| Empty cluster | ✓ | `__none__` sentinel. |
| No mutation | ✓ | Contract test. |
| Normalization direction | ✓ | As `active` grows, score drops → the loaded pod eventually loses to an equally-warm cold-ish pod. This is the 4540cca intent and it is directionally correct. |

#### 2.6.3 Findings

**F7 (HIGH, throughput): cold-start starvation — the 4540cca fix is
incomplete.** `pod.ewma_throughput_tps` is initialized to `0.0` (the
default in `PodRuntime`, `core.py:90`). The engine's `__post_init__`
warms **only** `ewma_latency_ms`, not `ewma_throughput_tps`
(`engine.py:63–65`). `ewma_throughput_tps` is updated *exclusively*
inside `_apply_side_effects` at dispatch — a cold pod that never gets a
request never accumulates a non-zero score.

At cold start, every pod has score `0.0 / (1 + 0) = 0.0`. `max` breaks
the tie by the second key `-ord(p.spec.pod_id[0])`; for canonical ids
like `p0`, `p1`, `p2`, the first character is identical (`'p'`) across
all pods, so the secondary key is also tied and `max` falls back to
iteration order — `p0` wins the first dispatch.

After dispatch: `p0.ewma_throughput_tps ≈ 0.2 · throughput > 0`, while
`p1.ewma_throughput_tps == p2.ewma_throughput_tps == 0`. Next decide:
`p0.score = 0.2·T / (1 + N_p0) > 0`, others `= 0`. `p0` wins again.
**The normalization cannot pull `p0`'s score below zero**, so other
pods never catch up.

**Reproduced** on the full engine (`SimulationEngine`, analytic cost
model, three colocated pods, `max_concurrent_prefill=2`):

```
throughput dispatch, 30 req @ 5 QPS:  {'p0': 30}
throughput dispatch, 60 req @ 50 QPS: {'p0': 60}
```

In both regimes `p0` captures 100% of traffic; this matches the
pre-4540cca failure mode. The 4540cca normalization is a partial
mitigation that works only *after* every pod has been warmed at least
once — it does not break the initial exploitation loop.

Suggested fixes (any one suffices; F7a is simplest and least invasive):

- **F7a:** also warm-init `ewma_throughput_tps` in `engine.__post_init__`
  to a positive floor, analogous to `initial_warm_latency_ms`. Set it
  to `tokens_per_req / (initial_warm_latency_ms/1000)` so the initial
  implied throughput matches the warm-latency assumption.
- **F7b:** treat `ewma_throughput_tps == 0` as infinite available
  throughput inside the policy (i.e., route to any zero-EWMA pod first).
  Strict explore-exploit interpretation.
- **F7c:** replace the ranking with one that does not have a monotone
  accumulator per pod — e.g., rank by `ewma_throughput_tps - λ · active`
  where λ is calibrated so active drives the decision in the uninformed
  regime.

Filing F7 as discovered.

**F8 (LOW, throughput): tie-break uses only the first character of
`pod_id`.** Line 41: `-ord(p.spec.pod_id[0])`. For canonical ids
`p0,p1,p2,...` all pods share the first character `'p'` and the tie-break
collapses. The effective tie-break then depends on Python's `max`
iteration order (which preserves the first occurrence). Fix is trivial:
use full `pod_id` (lex order, negated to match the `min` convention of
peer policies) or `p.spec.pod_id` ascending.

Under F7's starvation the F8 ambiguity is never exercised (scores are
never actually tied after the first dispatch). But if F7 is fixed by
warm-init, F8 becomes load-bearing: without a full-id tie-break, the
rotation across equally-warm cold pods would revert to iteration order,
which is technically deterministic but fragile to dict insertion order.

Filing F8 as discovered.

---

## 3. Cross-policy findings

**F11 (LOW, cross-policy): tie-break direction is inconsistent across
the load-balancing group.** Five policies (`least-request`,
`least-busy-time`, `least-latency`, plus `prefix-cache` /
`prefix-cache-preble` explore branches from the prior audit) pick the
**smallest** `pod_id` on a pure tie. `least-kv-cache` (F9) picks the
**largest**. `throughput` (F8) breaks on the first *character* only, and
the sign is `-ord` meaning "pick smallest first char" — which is
equivalent to smallest-id for distinct first chars, but collapses for
canonical `p0/p1/p2` ids. A consistent group-wide rule
(*smallest* `pod_id` wins ties) would remove a class of workload-
sensitive routing surprises. Filing as discovered (tracked jointly with
F3 from the prefix-aware audit, which flagged the same wart on
`prefix-cache-preble`'s exploit branch).

**Signal freshness, group-wide assessment.**

| Signal | Source | Refresh cadence | Staleness risk |
|---|---|---|---|
| `active_prefill`, `active_decode`, `queued` | `_apply_side_effects` (dispatch) + `_retire_up_to` (start of every `decide`) | Fresh-as-of-`now` at every decide | None |
| `ewma_latency_ms` | `_apply_side_effects` at dispatch only | Only updates on dispatch; a retirement does not push the EWMA | Low — `least-busy-time` zeroes via the `(active+queued)` multiplier; `least-latency` self-corrects because observed > warm-init |
| `ewma_throughput_tps` | same | same | **High** (F7) — no warm init, no retirement update |
| `kv_cache.size_bytes` | `install` at dispatch (and LRU eviction at install-time) | Fresh for each decide | None — but F10 shows the signal is non-informative under shared-prefix workloads |
| `pending_work_ms` | `+observed` at dispatch, `-service` at retirement, clamped ≥0 | Fresh at every decide | Used only by `prefix-cache-preble`; see prior audit F5 |

---

## 4. Test coverage assessment

### 4.1 Already covered

- `test_random_is_deterministic_under_seed`
- `test_least_request_picks_min_load`
- Contract tests (registration, empty cluster, decide on nonempty, no
  mutation) run for every policy including all six in this group.
- Cost-model, fabric, KV-cache unit tests are independent.

### 4.2 Gaps filled by this audit

New file `tests/unit/test_load_balancing_audit.py`:

- `test_random_uniform_distribution` — seeded, 300 requests, σ within
  3× of uniform-expected.
- `test_least_request_rotates_under_load` — at high QPS, dispatch
  σ should be close to zero (exact rotation).
- `test_least_request_idle_ties_to_lowest_id` — under uncontended
  traffic, all pods tie at 0 → `p0` wins (current behavior, documents
  §2.2.2 note).
- `test_least_busy_time_zero_load_ties` — all pods at `active=0` →
  product is 0 → smallest pod_id.
- `test_least_busy_time_rotates_under_load` — rotation analogous to
  least-request.
- `test_least_latency_warm_init_prevents_cold_pod_monopoly` — verify
  that `initial_warm_latency_ms` actually gates cold-start behavior;
  would fail if a regression removed the warm-init.
- `test_least_latency_rotates_under_load` — 50 QPS, 60 req, dispatch
  σ within ±2 across 3 pods.
- `test_least_kv_cache_rotates_on_unique_prefixes` — unique-prefix
  workload → balanced dispatch.
- `test_least_kv_cache_starves_on_shared_prefix` — **documents F10
  with an `# F10` annotation**; asserts the current (buggy) behavior
  (one pod ≥ 50/60 dispatches) so the test will fail when F10 is fixed,
  forcing the fixer to update it. Negative-regression test.
- `test_least_kv_cache_tiebreak_picks_max_id` — **documents F9**;
  asserts current behavior (largest pod_id wins). Same negative-
  regression pattern.
- `test_throughput_cold_start_starvation` — **documents F7**; asserts
  one pod captures ≥ 90% of dispatches on a tiny trace. Forces attention
  when F7 is fixed.
- `test_throughput_tiebreak_uses_first_char_only` — **documents F8**;
  asserts pods named `a0` and `b0` tie-break differently from `p0` vs
  `p1`. Forces attention when F8 is fixed.
- `test_all_load_balancing_policies_empty_cluster` — the six of us,
  loop-asserting the `__none__` sentinel.
- `test_all_load_balancing_policies_single_pod` — loop-asserting the
  single-pod pick on the lone candidate.

Negative-regression (buggy-behavior-asserting) tests are marked with
`# NEGATIVE: documents F<n>` comments so that a future fixer sees the
pin, updates the test, and closes the bead in lockstep. This is the
same pattern used in the prefix-aware audit for F3.

---

## 5. Modal smoke config

New file `configs/smoke_load_balancing_modal.yaml` — exercises the
signal-freshness question against real traffic on three colocated pods.

**Design rationale.**

- The signal-freshness question resolves to: "do the EWMA and count
  signals converge fast enough to prevent starvation under real vLLM
  timing?" In simulation we already have two real starvation signatures
  (F7, F10). The smoke's job is to check whether the simulation bug
  reproduces on Modal, which tells us whether the cause is our
  simulator's idealization (retirement exact at `now + total_ms`) or
  the policy itself.
- Workload: 4-family Zipf(s=1.3), 200 requests @ 6 QPS. Shared prefix
  of 1024 tokens ensures F10's no-op-install regime is exercised; Zipf
  skew keeps one family dominant so the prefix-awareness signal is
  non-trivial for downstream comparisons.
- Six policies run back-to-back against the same topology:
  `random`, `least-request`, `least-busy-time`, `least-latency`,
  `least-kv-cache`, `throughput`.
- Per-pod dispatch skew (σ of per-pod request counts normalized by
  `N/npods`) is the primary signal.

**Expected outcomes (post-run acceptance).**

| Policy | Expected skew (σ / mean) |
|---|---|
| `random` | ≤ 0.15 (law of large numbers on N=200) |
| `least-request` | ≤ 0.05 (near-perfect rotation under load) |
| `least-busy-time` | ≤ 0.10 |
| `least-latency` | ≤ 0.20 (EWMA-driven cycling; may lag) |
| `least-kv-cache` | 0.40–0.70 — **if F10 reproduces**; 0.10–0.20 if Modal's engine makes installs cost something extra (e.g., per-dispatch write-through) that our simulator misses |
| `throughput` | ≥ 0.80 — **if F7 reproduces**; if Modal shows balanced traffic, the sim's warm-init idealization is hiding a different failure mode |

A `throughput` σ below 0.4 on Modal would be *more* alarming than
reproduction: it would indicate that our simulator and production
disagree on initial EWMA populations and that further auditing of the
warm-init contract is warranted.

**Budget.** Mirrors the prefix-aware smoke: 3 pods × 6 runs × ~35 s
active wall-clock ≈ 10.5 pod-minutes on A10G/L4 class. Estimated < $1
under Modal's per-run ceilings, matching the epic's ≤$1-per-smoke
guardrail. Config is marked `# HUMAN-GATED` and defers actual
execution to `go-3j8`.

---

## 6. Discovered-from beads (filed)

- **F7**: `throughput` cold-start starvation — 4540cca fix incomplete;
  warm-init `ewma_throughput_tps` or treat zero-EWMA as infinite-
  available.
- **F8**: `throughput` tie-break uses only `pod_id[0]`; collapses on
  canonical `p0/p1/p2` ids.
- **F9**: `least-kv-cache` tie-break direction inconsistent with the
  rest of the load-balancing group.
- **F10**: `least-kv-cache` starves 2/3 pods under shared-prefix
  workloads (byte-level install no-op + F9 tie-break).
- **F11**: cross-group tie-break direction inconsistency (tracks
  jointly with prefix-aware F3).
- **F12** (info, no bead): no policy admits on `max_concurrent_prefill`.
  Recorded in the report for future policy authors; not a fix request.

---

## 7. Verdict

| Policy | Verdict |
|---|---|
| `random` | **pass** — correct floor. |
| `least-request` | **pass** — correct; §2.2.2 low-load note is informational. |
| `least-busy-time` | **pass** — product formulation neutralizes EWMA staleness. |
| `least-latency` | **pass** — depends on `initial_warm_latency_ms` warm-init; covered by new regression. |
| `least-kv-cache` | **fail (correctness, MEDIUM)** — F10 reproducible on shared-prefix workloads. F9 is a contributing tie-break bug. Negative-regression tests pin current behavior. |
| `throughput` | **fail (correctness, HIGH)** — F7 reproducible. `4540cca` fix does not eliminate the pathology it advertises fixing. F8 is a secondary tie-break bug. |

Ready for Modal smoke (via `configs/smoke_load_balancing_modal.yaml`,
gated on human approval). The two "fail" entries do not block the
existing sweep conclusions in `research/reports/routing-comparison.md`
— those sweeps did not use a cold-start single-family trace against
`throughput`, and their `least-kv-cache` results were downweighted per
§7.4 of that report — but they do flag F7 and F10 for fix-before-ship
on any real-traffic deployment that would route through these policies.
