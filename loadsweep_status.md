# apr7 Load-Sweep — Status & Resume Notes

_Last updated: 2026-06-22 ~00:13 PDT — **SWEEP COMPLETE (ts1/ts2/ts3 all done)**_

## Hypothesis
Gorgo loses to `simple-session-affinity` on the high-diversity **apr7** window. Suspected cause: the high-diversity windows are also **high-throughput**, so the 3-replica fleet **saturates** and queueing (not cache/RTT) dominates — the regime where load-agnostic session-affinity wins. Test: replay the *same* apr7 requests at decreasing arrival rate (`time_scale`) and see whether gorgo's p95-TTFT vs SSA improves as load drops. Only the arrival rate varies; requests/order/prompts/weights are identical.

## Fixed config (all three ts runs)
- **Window:** `glm5_decoded_apr7_1945_to_2015` (apr7 19:45–20:15, decoded synthetic, 727 users, ~12.5k req at full rate)
- **Manifest:** `specs/c64/manifests/manifest_glm5_decoded_apr7_1945_2015.json`
- **Policies:** `simple-session-affinity`, `least-request`, `gorgo-static-p95-2d`
- **Gorgo cost fn (non-physical-rate 2D):** `score = 0.276·rtt_ms + uncached + 0.5·queued`
  - hyperparameters: `rtt_weight=0.276, queue_weight=0.5, prefill_rate=1.0, queue_rate=1.0`
  - source: `results/2d_v9_tune/learned_weights.json` (decoded_v9 apr5 tuning — best high-div config)
  - verified applied via proxy trace `hyperparameters_at_decision`
- **Fleet:** 3 regions (ap-seoul-1, eu-frankfurt-1, us-ashburn-1), 3 replicas/policy; concurrency 64; max_tokens 128; max_input_tokens 24000; arrival_mode open-loop; metrics refresh 30s.
- **Specs:** `specs/c64/loadsweep_apr7/ts{1,2,3}.json` (differ only in `time_scale` + `run_id`)
- **Constraint:** only **5 containers/region** → run **ONE ts at a time** (sequential).

## Status — SWEEP COMPLETE

| Run | time_scale | arrival rate | status | regime | gorgo TTFT p95 | SSA TTFT p95 | gorgo vs SSA |
|---|---|---|---|---|---|---|---|
| **ts1** | 1.0 | ~4.8 rps (full) | ✅ **DONE** | **over capacity (all policies)** | 8,260 ms† | 1,835 ms† | n/a (overload) |
| **ts2** | 2.0 | ~2.4 rps | ✅ **DONE** | within capacity | 1,822 ms | 1,707 ms | −6.7% (competitive) |
| **ts3** | 3.0 | ~1.6 rps | ✅ **DONE** | within capacity | **1,378 ms** | 1,556 ms | **+11.4% (gorgo wins)** |

**Verdict: SATURATION CONFIRMED.** Below the fleet's capacity ceiling, gorgo's standing improves monotonically as load drops — competitive at ts2, clean sweep at ts3. At/above the ceiling (ts1) the comparison is invalid (see slip note).

> †**CRITICAL — ts1 is NOT a clean routing comparison.** At full load the 3-replica fleet is over capacity for *every* policy: the open-loop replay backs up by minutes waiting for concurrency slots. Client-side **scheduling-slip p95** (scheduled-arrival→dispatch, *separate* from `ttft_ns` = dispatch→first-token): **SSA 243s, gorgo 448s, least-request 491s.** The TTFT numbers below are `ttft_ns` and exclude that backlog; true user latency at ts1 is slip-dominated and unusable for all three. The old "15,452 ms" figure came from a now-deleted live workload-stats dir and is not reproducible — do not cite it. (Even on `ttft_ns`, gorgo is *worst* at ts1: with zero spare capacity, cache-greedy SSA minimizes total prefill work while gorgo's off-cache diversions add work.)

> **Note on harvest:** the aggregated sweep-matrix did **not** write for any v2 run. All numbers were computed directly from each policy's `proxy_traces/<run>/.../requests.jsonl` (`ttft_ns`/`total_ns`, status==200), not from a `sweep_matrix.json`. ts2/ts3 slip is negligible (p95 ≤ 80ms for SSA/gorgo), so those `ttft_ns` numbers equal true TTFT.

### ts1 full results (DONE) — full load `time_scale=1.0` (OVER CAPACITY — `ttft_ns`, excludes minutes of slip)
| Policy | TTFT p50 | TTFT p95 | TTFT p99 | E2E p50 | E2E p95 | slip p95 | n |
|---|---:|---:|---:|---:|---:|---:|---:|
| simple-session-affinity | 540 | 1,835 | 3,539 | 3,925 | 17,940 | 243s | 8,663 |
| gorgo-static-p95-2d | 1,765 | 8,260 | 14,677 | 6,716 | 15,759 | 448s | 8,663 |
| least-request | 1,346 | 6,138 | 8,866 | 12,670 | 18,615 | 491s | 8,663 |
→ All three over capacity (slip = minutes). Not a routing result — marks the saturation ceiling. gorgo worst on `ttft_ns` here (no slack for load-spreading to exploit; off-cache diversions add prefill work).

### ts2 full results (DONE) — half load `time_scale=2.0`
| Policy | TTFT p50 | TTFT p95 | TTFT p99 | E2E p50 | E2E p95 | n |
|---|---:|---:|---:|---:|---:|---:|
| simple-session-affinity | 515 | 1,707 | 2,442 | 2,198 | 15,504 | 8,663 |
| gorgo-static-p95-2d | 537 | 1,822 | 2,758 | 2,290 | 5,648 | 8,663 |
| least-request | 938 | 4,361 | 6,214 | 4,636 | 16,847 | 8,663 |
→ gorgo within 7% of SSA on TTFT p95; **wins E2E p95 ~2.7×** (5.6s vs 15.5s). Cache-hit 0.815 vs SSA 0.875; chosen-queue 23k vs SSA 62k (SSA piles onto a saturated replica while leaving one near-empty).

### ts3 full results (DONE) — third load `time_scale=3.0`
| Policy | TTFT p50 | TTFT p95 | TTFT p99 | E2E p50 | E2E p95 | n |
|---|---:|---:|---:|---:|---:|---:|
| **gorgo-static-p95-2d** | **392** | **1,378** | **2,124** | **1,665** | **3,248** | 8,663 |
| simple-session-affinity | 455 | 1,556 | 4,589 | 1,757 | 4,041 | 8,663 |
| least-request | 546 | 1,805 | 2,623 | 2,100 | 4,917 | 8,663 |
→ gorgo **sweeps all six metrics**; largest margin is TTFT p99 (2,124 vs 4,589, **+53.7%**), where SSA's single-replica concentration (49.8%) finally damages its own tail. Cache-hit 0.825 vs 0.876; chosen-queue 8.9k vs SSA 18k.

## How to CHECK
```
export MODAL_ENVIRONMENT=alessio-dev
modal app list | grep -i gorgo | grep ephemeral          # is it still running?
# results (sweep matrix written at completion):
modal volume ls GORGO-bench-results policy_matrix_sweep/c64/glm5_c64_loadsweep_apr7_ts2_v2
# per-policy saved files:
modal volume ls GORGO-bench-results workload_runs/glm5_c64_loadsweep_apr7_ts2_v2
```
Read p95s from `<run>/glm5_c64_loadsweep_apr7_ts2_sweep_matrix.json` →
`results[0].manifest.results[*].workload.stats.ttft_seconds` / `request_e2e_seconds`.

## How to RESUME (launch the next ts) — ONE AT A TIME
Wait until the current ts shows `stopped` / has a `sweep_matrix.json`, then launch the next.
Use `.venv/bin/modal` (the global `modal` lacks `httpx`).

```
cd /Users/alessio/GORGO && MODAL_ENVIRONMENT=alessio-dev .venv/bin/modal run --detach \
  experiment_runner/policy_matrix_app.py::main \
  --base-spec-path specs/c64/loadsweep_apr7/ts3.json \
  --sweep-manifest-path specs/c64/manifests/manifest_glm5_decoded_apr7_1945_2015.json \
  --experiment-id glm5_c64_loadsweep_apr7_ts3_v2 \
  --output-dir /results/policy_matrix_sweep/c64/glm5_c64_loadsweep_apr7_ts3_v2 \
  --start-index 0 --top-k 1
```
(For ts2, same command with `ts2` substituted — already launched.)

## Gotchas
- **Drain-hang risk:** eval can stall draining a few dead in-flight requests at the end. If a run hangs (app alive, no progress, `workload/status` stuck in `replay-running`), salvage by POSTing stop to each policy's proxy:
  `curl -s -X POST "<proxy_url>/workload/stop" -d '{}'` — it finalizes partial stats (~99.8% complete) so results still save.
- `"Successfully canceled input"` in app logs at the end = **normal teardown**, not a kill.
- Launch with `--detach` (survives local disconnect). Don't run two ts at once (5/region budget).

## Result (CONFIRMED)
Within the fleet's capacity (ts2/ts3, slip negligible), gorgo's TTFT p95 vs SSA improves monotonically as load drops: **−6.7% (ts2) → +11.4% (ts3)**, crossing below SSA and **winning every metric at ts3** (incl. TTFT p99 2,124 vs 4,589 = +53.7%). At full load (ts1) the fleet is over capacity for *all* policies (slip = minutes), so it only marks the saturation ceiling. ⇒ **Saturation confirmed**: gorgo's apparent TTFT weakness on high-diversity windows is a queueing/overload artifact, not a structural routing failure. The mechanism (continuous batching shields TTFT from load, so SSA's concentration is "free" on TTFT until the fleet saturates) is documented with the per-request decomposition in `paper.md` → "Load Sweep (apr7)" section.

Results written to `paper.md` (load-sweep section: crossover table, ts2 decomposition, ts3 full table, PD-disaggregation discussion).

## Side note: midrange-window scan (separate, DONE)
`find_best_window.py` midrange scan finished: **0 windows** passed `rps∈[2.0,4.5] ∧ users≥250 ∧ top_user≤30% ∧ median_tok≥500`. ⇒ In the 7-day data, high-diversity/long-context windows are **all** high-throughput; no natural moderate-load high-diversity window exists. To get more candidates, relax filters (e.g. `--midrange-rps-max 6.0 --midrange-min-median-tokens 200`) and re-run, else rely on the `time_scale` sweep.
