# Policy audit: PD topology policy

> **Bead:** go-h0b (child of epic go-czi).
> **Policy:** `pd` (`src/routing_harness/policies/pd.py`).
> **Baseline commit:** `808b0a7` (tip of `main` before this audit; includes
> F9/F10 least-kv-cache fixes and the fairness/session audit).
> **Companion audits (continue finding-ID numbering):** go-5m8 (prefix-aware,
> F1–F6), go-651 (load-balancing, F7–F11), go-sz5 (fairness/session,
> F12–F17). New findings in this audit start at **F18**.

## 0. TL;DR

`pd` is functionally correct in its role-split contract: it partitions pods
into prefill-capable and decode-capable pools, picks prefill by
cache-affinity and decode by busy-score, and returns `__none__` when either
pool is empty. The two headline mechanisms work — prefill routes preferentially
to cache owners under the sticky-prompt test, and decode rotates across the
decode pool as `active_decode` moves.

Three substantive findings, four informational.

- **F18 (medium, correctness): `best_prefill` scores by *non-consecutive*
  block membership** rather than longest *consecutive* prefix. Inconsistent
  with `prefix_cache.py`, `KVCacheState.owners_of`, and the engine's own
  `captured` computation. Under partially-evicted or scattered-install cache
  states, the policy routes to a pod that delivers zero real reuse over a
  pod that would deliver a genuine prefix. Reproduced in §2.3.
- **F19 (medium, determinism): prefill and decode tie-breaks pull in
  opposite directions.** Prefill uses `max(..., pod_id)` → largest pod_id
  wins; decode uses `min(..., pod_id)` → smallest pod_id wins. In
  colocated-fallback mode (all `Phase.BOTH`) perfect ties force a cross-pod
  handoff for every request. Reproduced: 10/10 requests on a 3-pod BOTH
  cluster routed (p2, p0) and charged `pd_handoff_bytes` needlessly.
- **F20 (medium, signal staleness): `ewma_latency_ms` on pure-DECODE pods
  is never updated.** The engine's `_apply_side_effects` advances
  `pod.ewma_latency_ms` only for the *prefill* pod. `busy(p) = ewma_latency_ms
  * (active_decode + queued)` therefore reduces to `warm_constant *
  active_decode` on dedicated DECODE pods. The docstring's "least-busy-time"
  framing is misleading; it's effectively least-active-decode with a
  constant multiplier. Reproduced in §2.4.

Plus four informational items: F21 (cross-pool `queued`-mirror coupling),
F22 (docstring vs behavior mismatch for colocated-fallback), F23 (`peer_ids`
ignored), F24 (hard fail when one pool empties), F25 (inherited plain-
prefix-cache hotspot risk on prefill pool).

None block the sweep-v4 conclusions. F18 and F20 are candidates for a
follow-on fix bead; F19 is a one-character tie-break reversal.

## 1. Method

The policy was read top-to-bottom and cross-referenced with:

- **`simulator/engine.py`** — to confirm which signals are actually
  maintained for each pod role.
- **`cost_model.py`** — to confirm how `pd_handoff_bytes` and
  `kv_transport_ms` are charged when prefill ≠ decode.
- **`prefix_cache.py`, `kv_cache.py`** — to compare the consecutive-prefix
  contract used elsewhere in the codebase.

Each concerning behavior was reproduced on a minimal synthetic trace
through the real `SimulationEngine` rather than the policy in isolation.
Conftest-provided `pd_specs` (1 prefill + 1 decode, peered) and a larger
`2×2` PD topology (used in §2.4) cover the shared-cache and disaggregated
regimes respectively.

## 2. Findings in detail

### 2.1 Shape recap

```python
prefill = [p for p in cluster.pods.values() if p.spec.role in (PREFILL, BOTH)]
decode  = [p for p in cluster.pods.values() if p.spec.role in (DECODE,  BOTH)]
if not prefill or not decode:
    return Decision("__none__", "__none__", "pd-pools-empty")

best_prefill = max(prefill, key=lambda p: (
    sum(1 for h in hashes if kv_cache.has(p.spec.pod_id, h)),
    -p.active_prefill,
    p.spec.pod_id,
))
best_decode  = min(decode,  key=lambda p: (p.ewma_latency_ms * (p.active_decode + p.queued),
                                           p.spec.pod_id))
```

### 2.2 Correctness matrix

| Check | Status | Evidence |
|---|---|---|
| Pool partitioning is role-based, includes BOTH in both pools | ✓ | Lines 46–47. Mirrors `ClusterState.prefill_capable()` / `decode_capable()`. |
| Empty-pool → `__none__` | ✓ | Line 48–49. Contract test `test_policy_handles_empty_cluster` covers the single case; see F24 for the *one-pool-empty* nuance. |
| No cluster/KV mutation | ✓ | Contract test `test_policy_does_not_mutate_cluster`. Policy reads `active_prefill`, `active_decode`, `queued`, `ewma_latency_ms`, `kv_cache.has` — all reads. |
| `prefix_key` path | ✓ | `_prefix()` returns `[request.prefix_key]` when set; `kv_cache.has` works over any opaque hash. |
| Pairing uses `peer_ids` | **no** | F23. Prefill and decode are chosen independently. |
| Consecutive-prefix match semantics | **no** | F18. Uses raw membership count. |
| Tie-break direction consistency | **no** | F19. Prefill-max vs decode-min. |
| Decode pool latency signal is fresh | **no** | F20. `ewma_latency_ms` frozen at warm value on pure-DECODE pods. |
| Single-pod Phase.BOTH degrades correctly | ✓ | `prefill_pool == decode_pool == [only]` → `max` and `min` over a 1-element list both return that element → colocated. |
| One PREFILL + one DECODE | ✓ | Trivially — 1-element pools. |
| Imbalanced pools (1 prefill, 3 decode) | ✓ | Policy considers every decode candidate via `min`; no bucketing. Verified by inspection. |
| PD-disaggregated KV handoff accounting | ✓ | Engine, not policy: `decode_pod_id != prefill_pod_id` → `pd_handoff_bytes = len(prompt_tokens) * kv_bytes_per_token` (engine.py:168–175). Charge is correct when the policy deliberately splits; F19 is about when the split is accidental. |

### 2.3 F18 — non-consecutive prefill scoring

`prefix_cache.py` computes the longest *consecutive* prefix per pod (line
49–53 of that file: breaks on first miss). `KVCacheState.owners_of`
enforces the same invariant when the engine chooses a peer-pull source
(kv_cache.py:86–115). The engine's own `captured` counter (engine.py:130–
135) is consecutive: `for h in hashes: if has(pod, h): captured += 1 else:
break`.

`pd.py` scores with `sum(1 for h in hashes if kv_cache.has(p, h))` — it
counts *all* block-hash hits regardless of position. A pod holding blocks
{0, 2, 4} out of a 5-block request scores 3, beating a pod holding the
consecutive {0, 1} that scores 2.

**Downstream impact.** Under a paged-KV invariant, block K's attention
state depends on blocks 0..K-1, so scattered blocks *cannot* be reused.
The engine's `captured` computation on the chosen prefill pod will stop at
the first miss: pod {0, 2, 4} gets 1 block of reuse (only block 0), pod
{0, 1} would have gotten 2. The policy actively steers the request toward
lower reuse.

**Reproduction (local, §0 commit):**

```text
pfA holds scattered blocks {0, 2, 4};  pfB holds consecutive {0, 1}
pd picks prefill = pfA
prefix-cache picks prefill = pfB  (correct)
```

Engine-level impact: `cached_prefix_tokens` in the pfA case is 16 (1
block × 16 tok), vs 32 in the pfB case — a 2× regression on the reuse
metric for this one request.

**Realistic trigger.** Scattered residency arises naturally when:

- A pod's cache has partially evicted mid-prefix (LRU drops block 1 but
  block 0 is still hot; later a distinct request shares block 2 and warms
  only that block on the same pod).
- A peer pull installs only the blocks it pulled (`pull_blocks` window in
  engine.py:314–325), without the intermediate blocks the prefill pod
  already had evicted.
- A workload with multiple overlapping prefix families trains the cache
  on a shared subset of blocks rather than a clean leading prefix.

**Fix.** Replace `sum(...)` with a break-on-first-miss loop, mirroring
`prefix_cache.py`:

```python
def _match_len(pod_id: str) -> int:
    n = 0
    for h in hashes:
        if not kv_cache.has(pod_id, h):
            break
        n += 1
    return n

best_prefill = max(prefill, key=lambda p: (_match_len(p.spec.pod_id),
                                           -p.active_prefill,
                                           -_pod_id_sort_key(p)))  # see F19
```

### 2.4 F20 — stale `ewma_latency_ms` on pure-DECODE pods

`engine._apply_side_effects` (lines 254–260) updates `pod.ewma_latency_ms`
for **only the prefill pod** of the decision. Dedicated DECODE pods
(Phase.DECODE-only) never serve prefill, so their EWMA never advances past
the warm-start `initial_warm_latency_ms` (5.0 by default). Under PD
disaggregation the `busy` function:

```python
busy(p) = p.ewma_latency_ms * (p.active_decode + p.queued)
```

reduces to `5.0 * active_decode` for the decode pool (both `ewma_latency_ms`
and `queued` are constants — `queued` is 0 since DECODE pods never see
prefill traffic). The policy is effectively **least-active-decode** for the
decode pool, with no latency awareness.

**Reproduction.** 50 requests through the standard `2×2` PD topology:

```text
pfA role=prefill  ewma_latency_ms=5.00    active_decode=0  active_prefill=0
pfB role=prefill  ewma_latency_ms=500.29  active_decode=0  active_prefill=0
dcA role=decode   ewma_latency_ms=5.00    active_decode=0  active_prefill=0
dcB role=decode   ewma_latency_ms=5.00    active_decode=0  active_prefill=0
```

`pfB`'s prefill latency has converged to a realistic ~500 ms; `dcA` and
`dcB` are still pinned at 5.0 after the full trace. (Side observation:
every request went to `pfB` because of F18+F19 — see also F25.)

**Options.**

- (a) Engine fix: in `_apply_side_effects`, also EWMA-update
  `decode_pod.ewma_latency_ms` (the *decode* pod's own latency contribution
  is arguably the decode portion `compute_decode_ms` rather than
  `total_ms`; picking the right proxy is the open design question).
- (b) Policy fix: drop the `ewma_latency_ms` multiplier on the decode pool
  and use `min(decode, key=lambda p: (p.active_decode, p.spec.pod_id))` —
  renaming the policy's decode signal to "least-active-decode" to match
  reality.
- (c) Hybrid: keep the multiplier but substitute a signal that is actually
  maintained on the decode pod, e.g. a derived `pending_work_ms`-like term
  keyed on decode completions.

**Recommendation.** Pursue (b) for minimum risk. The analytic cost model
models decode latency as roughly constant per token, so `active_decode`
already approximates a time-domain signal under batched continuous
decoding. This matches the taxonomy label `load` on the decode pool.
Option (a) is the cleaner long-term fix if the engine gains real decode
latency instrumentation.

### 2.5 F19 — cross-branch tie-break asymmetry

```python
best_prefill = max(prefill, key=lambda p: (match_count, -p.active_prefill, p.spec.pod_id))
best_decode  = min(decode,  key=lambda p: (busy(p), p.spec.pod_id))
```

In a **perfectly tied** cluster state (all pods equal on the primary key,
equal on the secondary key), `max(..., pod_id)` picks the
**lexicographically largest** `pod_id`; `min(..., pod_id)` picks the
**smallest**. Under colocated-fallback (all pods are `Phase.BOTH`, so the
same pod-list feeds both pools) this guarantees prefill ≠ decode whenever
there are ≥ 2 pods — and therefore the engine charges `pd_handoff_bytes =
len(prompt_tokens) * kv_bytes_per_token` for **every** dispatch.

**Reproduction.** 3-pod colocated BOTH cluster, 10 identical-shape cold
requests, synthetic trace:

```text
colocated pick: prefill=p2  decode=p0
pd handoff bytes per request: [32768] × 10
migrated flags:                [True]  × 10
```

That is 32 tokens × 1 KiB/tok × 10 requests = 320 KiB of gratuitous fabric
traffic on a cluster that had no reason to disaggregate. In the real
smoke regime (prompt_len ≈ 1 KiB–4 KiB, `kv_bytes_per_token ≈ 128 KiB`),
the per-request unnecessary handoff would be **≥ 128 MiB** per request.
This is not theoretical: colocated fallback is the explicit use case
named in the policy docstring.

**Fix.** One-line reversal. Match the prefix-aware audit's F3 remediation
pattern: negate the pod_id in the prefill `max` key so both pools settle
on the lexicographically smallest `pod_id` under ties:

```python
best_prefill = max(prefill, key=lambda p: (match_count, -p.active_prefill, -_pod_sort(p)))
```

where `_pod_sort(p)` returns a negatable key (e.g. `(0, p.spec.pod_id)`
via a small helper, since Python `max` on strings via negation needs a
wrapping class — the simplest concrete fix is to sort decode with `max`
on `(–busy(p), -pod_id)` or prefill with an explicit secondary pass that
prefers colocated-with-decode-pick. See `test_pd_colocated_no_handoff`
in the audit tests for the expected post-fix behavior.

**Alternative fix (stronger).** In colocated-fallback mode specifically,
collapse the choice: if `prefill == decode` pool exactly, pick one pod
and set both fields to it. This would encode the docstring's intent
directly and eliminate the tie-break class of bug entirely. The cost is a
branch in `decide`. Recommended if the consecutive-prefill F18 fix is
also landing (so the tie-break fix and the match-semantics fix move
together).

### 2.6 Informational findings

**F21 (low, cross-pool coupling): `busy(p)` reads `p.queued`,** which per
the prefix-aware audit's F1 mirrors `active_prefill`. For pure-DECODE pods
this is always 0 (harmless). For Phase.BOTH pods in colocated-fallback a
prefill-hot pod gets deprioritized as a decode target — usually desirable
but undocumented, and inherits the F1 double-count naming wart. Fix moves
with F1.

**F22 (low, docstring vs behavior): docstring** says "the two pools
collapse to the same set and the policy still picks deterministically —
prefix-match for prefill, busy-time for decode — over that shared set."
The reader naturally interprets "over that shared set" as implying the
*same* pick for both fields (colocated). In practice they're chosen
independently and diverge under F19. Either fix F19 or update the
docstring to note: "prefill and decode are still chosen independently and
may land on different BOTH pods; when they do, the engine charges a
same-cluster KV handoff."

**F23 (info, topology): `peer_ids` on `PodSpec` is ignored.** Real PD
fabrics often wire specific prefill pods to specific decode pods via
fast interconnects (NVLink islands, RDMA groups). The current simulator
charges all inter-pod transfers uniformly via `inter_pod_bandwidth_gbps`,
so this simplification is hidden in the cost model — but a future
fabric-aware cost model would need the policy to respect `peer_ids` (or
the policy fix is to filter decode candidates by
`decode_pod_id in prefill_pod.peer_ids`, with a fallback if no peers are
set). Filed as discovered-from bead.

**F24 (info, failure mode): one-pool-empty returns `__none__` for every
request.** When e.g. every DECODE pod is unreachable but PREFILL (or
BOTH) pods are available, the policy cannot fall back to colocated
execution on the remaining BOTH pods. The engine silently drops every
request (engine.py:120–121) which in a real deployment would collapse
throughput to zero rather than degrade gracefully. A soft fallback
("if decode pool empty, reuse the prefill pod") is discussable but
implicit — outside this audit's scope. Filed as discovered-from bead.

**F25 (info, hotspot risk on prefill pool): PD uses plain prefix-cache
style cache-affinity on prefill,** not Preble-style hotspot deflection.
Under high-skew workloads the prefill pool can cache-lock onto one pod:
the first pod chosen (by F19 tie-break) warms up, and subsequent
identical-prompt requests then win on match count regardless of load. In
the 50-request identical-prompt reproduction in §2.4, every request
routed to `pfB`. This is the same hotspot pathology that motivates
`prefix-cache-preble`. Resolved by shipping the `pd-preble` variant
(`src/routing_harness/policies/pd_preble.py`, bead go-caz): applies the
exploit/explore gate and relative-imbalance hotspot deflection to the
prefill pool while keeping `pd`'s peer-aware decode selection. `pd`
itself is retained unchanged; operators select between the two per
workload.

## 3. Paper-fidelity summary

The PD policy does not cite a specific paper; the taxonomy assignment in
`research/reports/routing-comparison.md` §3 Table 1 places it as:

| Dimension | Intended | Observed | Status |
|---|---|---|---|
| Selection | `composite` (phase-split: cache-affinity + load) | Same | **Match** (F18 aside) |
| State | `stateless` | Stateless (no policy-private fields survive `decide`) | **Match** |
| Fairness | `best-effort` | No tenant weighting | **Match** |
| Topology | `pd-aware` | Exploits role split; tolerates colocated (with F19/F22 caveats) | **Match** |
| Migration | `none` | No rebind; engine handles handoff post-decide | **Match** |

Prior art — NVIDIA Dynamo, Mooncake Conductor, SGLang router with PD —
also picks prefill by cache and decode by load. Our implementation
matches that family conceptually; F18 is a point-implementation deviation
from the natural paged-KV semantics, and F20 is a simulator-layer
instrumentation gap.

## 4. Test coverage assessment

### 4.1 Already covered (before this audit)

- Registered, decides on nonempty cluster, handles empty cluster, no mutation
  (contract tests over all policies, parameterized).
- Role separation under PD topology (`test_pd_separates_roles` in
  `tests/unit/test_policies_individual.py`): confirms prefill pod has
  `Phase.PREFILL` and decode pod has `Phase.DECODE` on the standard
  `pd_specs` fixture.
- PD-handoff fabric contention timing
  (`tests/unit/test_fabric_contention.py`): exercises the engine's
  `pd_handoff_bytes` path with `pd` policy, confirms fabric queue
  accounting under overlapping transfers. Exercises the policy indirectly.

### 4.2 Gaps filled by this audit

New file `tests/unit/test_pd_audit.py`:

- `test_pd_empty_decode_pool_returns_none` — PREFILL pods present, DECODE
  pool empty → `__none__`; and symmetric.
- `test_pd_single_pod_colocated` — one Phase.BOTH pod → prefill=decode,
  no handoff.
- `test_pd_consecutive_prefix_match` — **pins current F18 behavior**:
  scattered-block pod outranks consecutive-prefix pod. Marked with an
  `# F18` comment so a future fixer finds the test and inverts the
  assertion on fix.
- `test_pd_colocated_fallback_tie_triggers_handoff` — **pins current F19
  behavior** on 3 BOTH pods: prefill=`p2`, decode=`p0`, inequality. Marked
  `# F19` — flip to `prefill == decode` on fix.
- `test_pd_decode_pool_ewma_is_stale` — **pins current F20 behavior**:
  after a real trace through the engine, `dcA.ewma_latency_ms == warm`.
  Marked `# F20`.
- `test_pd_prefill_cache_match_routes_to_owner` — positive: when one
  prefill pod holds the full prefix and no other does, prefill routes to
  that pod.
- `test_pd_decode_busy_picks_least_loaded` — positive: higher
  `active_decode` deprioritized.
- `test_pd_imbalanced_pools_1_prefill_3_decode` — coverage for 1×3
  asymmetric topology.
- `test_pd_peer_ids_ignored` — documents F23 by showing the policy will
  pair an unpeered (prefill, decode) combination.
- `test_pd_deterministic_under_repeated_calls` — same input → same
  decision, no hidden state.
- `test_pd_prefix_key_path` — `Request.prefix_key` short-prompt path
  (single opaque hash) produces a valid decision with a cached owner.

## 5. Modal smoke config

New file `configs/smoke_pd_modal.yaml` — a 2-prefill + 2-decode PD
topology designed to:

- Exercise the PD handoff path on every request (prefill ≠ decode by
  construction).
- Force the prefill-cache-hit signal into a high-skew regime so F18 could
  manifest on real hardware if eviction patterns scatter.
- Force the decode-load signal to actually rotate, giving F20 a regime
  where the dispatched distribution would *differ* if decode latency were
  measured (left as an observation rather than an acceptance criterion —
  on the real vLLM runner, `ewma_latency_ms` wiring is different from the
  simulator, so the pathology may or may not reproduce).
- Budget: 4 pods × ~45 s active ≈ 3 pod-minutes on A10G/L4 class ⇒ well
  under $1 Modal spend.

Expected signatures (calibration-grade, not exact):

- `pd` over `prefix-cache` on this topology: comparable *cache-capture*
  rate (PD restricts prefill-pool size, which reduces owner spread); the
  interesting signal is `p95 end-to-end` under decode saturation, where
  PD should win by offloading the memory-bound decode phase.
- Per-pod request-count distribution within each pool — skew should be
  small (not zero, per F25) under a moderate Zipf (s=1.1).
- KV-transport accounting: `kv_transport_bytes` per record should be
  close to `len(prompt_tokens) * kv_bytes_per_token` for every request
  — a sanity check that engine.py's handoff accounting is firing on PD.

The config is marked `# HUMAN-GATED` per epic protocol; the follow-on
Modal smoke bead (`go-3j8`) tracks the actual run.

## 6. Discovered-from beads (filed)

- **F18** (`go-dub`): `pd` scores prefill by non-consecutive block
  membership; inconsistent with paged-KV semantics and the rest of the
  codebase.
- **F19** (`go-5dt`): Prefill/decode tie-breaks pull in opposite
  directions; triggers gratuitous PD handoff on colocated BOTH clusters.
- **F20** (`go-vix`): `ewma_latency_ms` on pure-DECODE pods is never
  updated; the PD decode-pool `busy` signal is effectively constant-
  multiplied active_decode.
- **F22** (`go-op0`): docstring vs behavior mismatch on colocated
  fallback (bundled with F19 fix or docstring-only amendment).
- **F23** (`go-6i2`): `PodSpec.peer_ids` is ignored by the `pd` policy.
- **F24** (`go-997`): `pd` has no partial-availability fallback when one
  pool empties.
- **F25** (`go-caz`): PD inherits plain-prefix-cache hotspot risk on the
  prefill pool. Resolved by the `pd-preble` variant
  (`src/routing_harness/policies/pd_preble.py`).

F21 is a cross-policy `queued`-mirror coupling; it rides on the existing
F1 cleanup bead rather than getting its own ticket.

## 7. Verdict

- `pd`: **conditional pass.** The policy meets the taxonomy contract and
  works on the standard fixtures. F18 (non-consecutive match) and F20
  (stale decode EWMA) are real and reproducible, but neither changes
  dispatch outcomes under the specific regime driving
  `research/reports/routing-comparison.md` §6. F19 (tie-break asymmetry)
  is a sharp-edged edge case — irrelevant under PD disaggregation proper,
  catastrophic under colocated-fallback with a uniform cold workload.

Recommended next step before the Modal smoke: fix F19 (single-line
tie-break reversal). F18 and F20 are follow-on fixes; pair F20 with a
decision on whether the fix lands in the policy (swap to
`active_decode`) or the engine (instrument decode pod latency).

Ready for Modal smoke (via `configs/smoke_pd_modal.yaml`, gated on human
approval) once F19 is addressed, or as-is if the smoke is explicitly
PD-disaggregated (where F19's colocated trigger is not reachable).
