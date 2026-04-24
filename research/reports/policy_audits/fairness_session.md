# Policy audit: fairness / session group

> **Bead:** go-sz5 (child of epic go-czi).
> **Policies:** `session-affinity` (`src/routing_harness/policies/session_affinity.py`),
> `vtc-basic` (`src/routing_harness/policies/vtc_basic.py`).
> **Baseline commit:** `5ff7886` (post–F3/F7a merges; tip of `main` before this audit).
> **Companion audits (continue finding-ID numbering):** go-5m8 (prefix-aware, F1–F6),
> go-651 (load-balancing, F7–F11). New findings in this audit start at **F12**.

## 0. TL;DR

Both policies are correct in their narrow scope — `session-affinity` delivers
deterministic stickiness, and `vtc-basic` deterministically steers a tenant's
next request toward the pod that has historically served them the least.
Neither has a latent bug that changes dispatch outcomes for the workloads in
`research/reports/routing-comparison.md` §6.

**However, `vtc-basic` is materially different from the VTC paper** (Sheng et
al., "Fairness in Serving Large Language Models," OSDI 2024). The paper
schedules *which queued request to admit next* to enforce a bounded max-min
fairness gap `U`; our implementation picks *which pod to route to* and has no
scheduling or admission control. As a result it cannot provide the paper's
starvation bound. This is a paper-fidelity gap, not a correctness bug — but it
must be flagged so nobody cites the policy name as shorthand for the paper's
guarantees.

Six findings (F12–F17) filed; none are release-blocking. The most important
are **F15** (VTC is not paper-faithful) and **F16** (no decay/window on the
token counter). F17 (in-flight work is invisible to the fairness signal) is
mitigated by the `busy()` tie-breaker but worth documenting.

## 1. Method

Each policy was read top-to-bottom. Every external signal (`cluster.pods`,
`PodRuntime.active_*`, `pending_work_ms`, `ewma_latency_ms`, `_bindings`,
`counters`, `pod_tenant_tokens`) was cross-referenced with the engine
(`simulator/engine.py`) to confirm when and how it is written. Hash-stability
questions for `session-affinity` were answered by static analysis — the
policy's binding is a direct `dict[session_id] -> pod_id`, not a hash — so
empirical tests focused on the observable: pod addition, pod removal, and TTL
behavior.

For `vtc-basic` the implementation was compared section-by-section against
Sheng et al. (§3 definition of VTC, §4.2 max-min fairness guarantee, §5
admission control). Behavior under a heavy-user-plus-light-user mix was
exercised through the full simulator engine on a short synthetic trace (see
test `test_vtc_spreads_heavy_tenant_across_pods_under_engine`).

## 2. `session_affinity.py`

### 2.1 What it does

Maintains a `dict[session_id] -> (pod_id, bound_ts)`. On each request:
1. If the session has a prior binding AND the bound pod still exists AND
   `arrival_ts - bound_ts <= stickiness_ttl_s`, reuse it.
2. Otherwise, pick the least-loaded prefill-capable pod (`min` on
   `active_prefill + active_decode + queued`, tie-break `pod_id`) and record a
   fresh binding.

### 2.2 Correctness

| Check | Status | Evidence |
|---|---|---|
| Deterministic under fixed arrival order | ✓ | Python dict assignment is ordered-overwrite; no randomness. |
| Empty cluster → `__none__` | ✓ | Line 38–40. |
| Single-pod cluster | ✓ | Fallback `min` returns the only pod; binding maps session to that pod. |
| Same-session stickiness | ✓ | Covered by `test_session_affinity_sticks`; re-verified in audit tests. |
| Different-session independence | ✓ | New session falls through `bindings.get() is None`; takes fallback. |
| Pod-removal handling | ✓ | `pod_id in cluster.pods` guard (line 44) — if bound pod is gone, rebinds via fallback. |
| TTL expiry | ✓ | `arrival_ts - bound_ts <= stickiness_ttl_s` (line 44). Strict `>` rebinds. |
| No cluster/KV mutation | ✓ | Contract test. Policy never touches cluster or kv_cache. |
| Hash-stability across pod-set changes | **n/a** | Not a hash-based design — direct bindings. See F14. |
| Brand-new session | ✓ | Binds via fallback min. |
| Pod failure mid-session | ✓ | Removing pod from `cluster.pods` triggers rebinding on next arrival. Audit test covers. |

### 2.3 Findings

**F12 (low, documentation): docstring references a non-existent `available`
flag.** `session_affinity.py:5` reads "Evicts stickiness if the sticky pod has
been unhealthy (modeled by an 'available' flag) or if stickiness_ttl seconds
have elapsed." `PodRuntime` has no `available` field (see `core.py:78–93`), and
`decide` does not check one. In the current engine, pod removal (not
unhealthy-flag flipping) is the mechanism for eviction via the `pod_id in
cluster.pods` guard. Either add an availability signal to `PodRuntime` and
honor it, or fix the docstring. Filed as discovered bead.

**F13 (low, memory): `_bindings` grows without bound.** TTL is checked on
read, but never-returning sessions leave stale entries forever. For a 1M-
session trace this is ~100 MB of dict overhead. Not a correctness issue, but
worth a periodic purge (or an LRU cap) for long-running Modal smoke or
production deployments. Filed as discovered bead.

**F14 (informational, paper-design): session-affinity is not a consistent-
hashing design.** The bead checklist asks about "hash-stability across pod
set changes." The policy uses explicit bindings, not hashing. Consequences:
- Adding a pod mid-run: **zero existing sessions rebind.** New pod only gets
  new-session arrivals. This is fine for cache warm-up but slow-to-balance.
- Removing a pod: **only sessions bound to that pod rebind.** Their fallback
  pick is "lightest currently", so they spread — but not via hash buckets.
- The `session_id` → `pod_id` map is the full state; two instances seeded
  identically but with different arrival interleaving can produce different
  bindings (unlike true consistent hashing).
Documented here so readers don't assume ring-hash semantics.

**F1 inheritance (cross-policy): `queued` doubles `active_prefill` in the
fallback sort key.** `session_affinity.py:48`
(`p.active_prefill + p.active_decode + p.queued`) reads the same mirrored
counter called out in the prefix-aware audit (F1). Ranking is preserved (prefill
is counted twice but only once does it matter since `queued` and
`active_prefill` move in lockstep); no new bead — tracked under F1.

### 2.4 Paper-fidelity summary

There is no canonical "session-affinity" paper. The design matches common
sticky-session implementations (Envoy/Istio, SGLang-router). The docstring
correctly notes this is not a fairness-balancing design — it's a cache
warm-up mechanism.

## 3. `vtc_basic.py`

### 3.1 What it does

Keeps two counters:
- `counters[k]`: global tokens served for fairness-key `k` (session_id by
  default, overridable to any metadata key).
- `pod_tenant_tokens[pod_id][k]`: tokens served for tenant `k` on pod `pod_id`.

Both advance in `observe_completion` by `tokens_consumed` (engine passes
`len(prompt_tokens) + max_output_tokens`). `decide` picks the pod minimizing
`(tenant_debt(pod, k), busy(pod), pod_id)` where `busy = ewma_latency_ms *
(active_prefill + active_decode + queued)`. The returned `Decision.score` is
`-vtc_score` (negative of the global counter) — for observability only; it
does not affect ranking.

### 3.2 Correctness (within stated semantics)

| Check | Status | Evidence |
|---|---|---|
| Deterministic across two seeded instances | ✓ | No randomness. |
| Empty cluster → `__none__` | ✓ | Line 63–65. |
| Single-pod cluster | ✓ | `min` returns the only pod. |
| Cold start (all counters=0, all load=0) | ✓ | Ties break on `pod_id`; audit test pins lowest pod_id. |
| Fairness-key plumbing (session_id vs. metadata) | ✓ | `_key` at line 36–39; falls back to `str(session_id)` if metadata key missing. |
| `observe_completion` updates both counters symmetrically | ✓ | Lines 53–55. |
| No cluster/KV mutation | ✓ | Contract test. |
| Score is global counter (diagnostic only) | ✓ | `score=-vtc_score` line 80. |

### 3.3 Findings

**F15 (medium, paper-fidelity): `vtc-basic` is not VTC as defined in the
paper.** The OSDI 2024 VTC paper schedules *admission order among queued
requests* with a continuous-time virtual counter, providing a bounded max-
min fairness guarantee `|U_i - U_j| ≤ U` between any two active clients. Our
implementation:

- **Has no admission queue.** Requests are dispatched in arrival order; the
  policy only chooses a pod.
- **Provides no fairness bound.** A heavy tenant is *spread* across pods but
  never *deferred*. Two tenants with wildly different arrival rates get
  whatever they submit.
- **Operates over pods, not over clients.** The paper assumes a single
  serving engine. Our adaptation (per-pod × per-tenant token count) is a
  reasonable first approximation for multi-replica deployments but is not
  validated against the paper's proofs.

The policy *docstring* is factually accurate ("Tracks per-tenant token
consumption and penalizes pods that are currently serving heavy tenants"),
but the policy *id* `vtc-basic` invites confusion. Recommended remediation
paths:

1. Rename to `per-tenant-load-balance` (or similar) and document that the VTC
   name is not claimed.
2. Add an `order=` parameter and implement paper-true admission-order
   scheduling for a new `vtc-admission` policy.

Either is fine. Filed as discovered bead (medium priority).

**F16 (medium, fairness): no decay, no window, counters are monotonic.** The
VTC paper uses a sliding window `W` (default 60s in §5.2) beyond which token
consumption ages out. Our `counters` and `pod_tenant_tokens` are append-only
`defaultdict(float)`. Consequences:

- A tenant who was heavy early in the trace is steered away from
  early-warmed pods forever, even if they've since gone idle.
- On long Modal runs (>1h), a new tenant looks nominally "catchable" but
  older tenants' counters dominate — `tenant_debt` is orders of magnitude
  asymmetric by the end.
- There is no "reset" knob for test / experiment isolation; each new
  instance starts fresh because the counters live on the policy instance,
  but within one run there's no way to clear history.

Recommended fix: add a configurable half-life or sliding window. For the
current simulator regime (≤ 30s traces in sweep-v4) this has no observable
impact, but it must be fixed before any claim of paper-fidelity. Filed as
discovered bead.

**F17 (low, fairness signal freshness): in-flight requests do not contribute
to `tenant_debt`.** `pod_tenant_tokens[pod][k]` only advances in
`observe_completion`, which fires at retirement. During a burst of N
requests from the same tenant, decisions 2..N all see the same
`tenant_debt` state as decision 1 — identical tenant_debt across pods. The
only differentiator is the `busy()` tie-breaker, which uses live
`active_prefill + active_decode + queued`. This is fine in practice (load
still spreads) but the fairness signal technically lags observed
consumption by one round-trip per request. Documented; no code fix
recommended (a mid-flight estimate would add complexity for marginal
benefit). Filed for awareness.

**F1 inheritance (cross-policy): `queued` doubles `active_prefill` in
`busy()`.** Same as F12's inheritance — ranking preserved; no new bead.

### 3.4 Paper-fidelity summary

| Dimension | VTC paper (Sheng et al., OSDI'24) | Our impl | Status |
|---|---|---|---|
| Scheduling axis | Admission order (which queued request to serve next) | Pod selection (which replica to dispatch to) | **Mismatch** |
| Fairness metric | Continuous-time virtual counter per client | Monotonic token counter per `(pod, tenant)` | **Partial** (no decay) |
| Guarantee | Bounded `|U_i − U_j| ≤ U` for active clients in window | None | **Missing** |
| Starvation bound | Yes (via admission deferral) | No | **Missing** |
| Window / decay | Sliding window `W` | None | **Missing** |
| Per-client vs. per-pod | Per-client (single engine) | Per-pod × per-tenant | **Different design** |

## 4. Test coverage assessment

### 4.1 Already covered (before this audit)

- `test_session_affinity_sticks`: same session on two requests → same pod.
- `test_vtc_fairness_annotation`: heavy tenant's `score` goes below light
  tenant's; heavy tenant's second decision differs from their first (unless
  1-pod cluster).
- Contract tests: registered, decides on nonempty cluster, empty cluster,
  no mutation.

### 4.2 Gaps filled by this audit

New file `tests/unit/test_fairness_session_audit.py`:

**session-affinity:**
- Brand-new session lands on lightest pod (cold start).
- Binding survives across many requests (no TTL breach).
- TTL expiry rebinds to lightest-now.
- Pod removal mid-run forces rebind; other sessions unaffected.
- Pod addition mid-run: existing bindings unchanged (F14 behavioral pin).
- Fallback `min` breaks ties on pod_id.
- Two sessions with different arrival orders do not share a binding.
- No cluster/KV mutation over a 20-request trace.
- Different `stickiness_ttl_s` values observe correct boundary
  (= TTL → sticky; > TTL → rebind).

**vtc-basic:**
- Cold-start routes to lowest pod_id on all-zero state.
- Observed completion raises global counter *and* `pod_tenant_tokens`.
- Heavy tenant's burst (no completions) spreads only via `busy()`, not via
  fairness signal (F17 pin).
- After completions, heavy tenant is steered to minority-load pod.
- Light tenant routed under same conditions is indifferent (all-zero debt).
- `fairness_key="tenant"` with metadata uses metadata; missing metadata
  falls back to `str(session_id)`.
- Determinism: two instances with identical input sequence produce
  identical decisions.
- Unbounded monotonic counters: audit test demonstrates counter growth
  across many completions (F16 pin; documents current behavior).
- Integration against `SimulationEngine`: heavy + light mix on 3-pod
  cluster, heavy tenant's pod-count skew is bounded (`max-min` ≤ a small
  constant) — proves the policy does *spread* even if it doesn't *throttle*.

## 5. Modal smoke config

New file `configs/smoke_fairness_session_modal.yaml` — a heavy-user +
light-user mix designed to surface fairness dynamics:

- 8 sessions: 2 heavy (80% of tokens), 6 light (20% split).
- 3 colocated pods, generous KV (no cache pressure).
- 150 requests @ 5 QPS → ~30s wall clock.
- Three runs back-to-back: `random` (floor), `session-affinity`, `vtc-basic`.
- Budget ≤ $1 Modal spend.

**Expected signatures:**

- `session-affinity`: heavy tenants monopolize their bound pods; per-pod
  request-count σ should be noticeably *higher* than `random`. (Stickiness
  does not balance.)
- `vtc-basic`: heavy tenants' request counts should spread across pods
  more evenly than `session-affinity` — σ comparable to or lower than
  `random` for per-tenant-per-pod distribution.
- **If `vtc-basic` shows per-tenant skew worse than `random`**, escalate: it
  would indicate the `tenant_debt` signal is being overpowered by
  `busy()` (possible if `ewma_latency_ms` saturates) and the policy is
  effectively reducing to least-busy-time.

Config is marked `# HUMAN-GATED` per epic protocol; execution tracked under
`go-3j8`.

## 6. Discovered-from beads (filed)

- **F12**: `session_affinity` docstring references non-existent
  `available` flag on `PodRuntime` — docs/code mismatch.
- **F13**: `session_affinity._bindings` grows without bound — add LRU/TTL
  purge for long-running deployments.
- **F14**: session-affinity is binding-based, not hash-based — documented;
  informational, no code change proposed.
- **F15**: `vtc-basic` is not VTC as defined in Sheng et al. (OSDI'24) —
  rename or implement admission-order variant.
- **F16**: `vtc-basic` counters are monotonic; add decay/window to approach
  paper-fidelity.
- **F17**: `vtc-basic` in-flight requests not reflected in `tenant_debt` —
  one-round-trip lag in fairness signal; informational.

## 7. Verdict

- `session-affinity`: **pass.** Correct within stated (non-balancing, cache-
  warm-up) semantics. F12 is a docs bug; F13 is a long-run memory item; F14
  is informational.
- `vtc-basic`: **pass-with-caveats.** Correct within its implemented
  semantics (per-pod × per-tenant steering). Diverges from the VTC paper on
  the two core axes (scheduling vs. routing; bounded vs. unbounded
  fairness gap). F15/F16 must be addressed before the name can be cited as
  paper-accurate.

Ready for Modal smoke (via `configs/smoke_fairness_session_modal.yaml`,
gated on human approval under `go-3j8`).
