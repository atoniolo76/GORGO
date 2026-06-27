# Results + Discussion restructure proposal (go-9nd)

Feedback (Rome, 2026-06-27): *"make the results and discussion less numbers and
more high level, including comparing to WildChat and LMSYS in the Appendix, using
all the figures, demonstrating where and why GORGO is better than alternatives."*

This is a **proposal**, not an applied edit — the live Results tables carry open
`\cmt{}` reviewer comments, so the consolidation below is staged in
`research/proposals/` for Rome to integrate rather than ripped into `paper.tex`.

---

## 1. Diagnosis

- **Number recitation in prose.** Every Results paragraph re-states p50/p95/p99 +
  E2E values that are already in the table directly above it
  (lines 448–461, 502–509, 533–542, 620–697). The reader reads each number twice.
- **No dedicated Discussion.** "Where/why GORGO wins" is implicit, scattered
  across Results prose, Limitations, and three appendices.
- **Six result tables + four per-window TTFT bar figures** for what is really
  two stories (p95-objective, p50-objective) over three windows.
- **The regime argument is buried.** The single most important "why" — GORGO's
  advantage exists *because* ARTChat-411k is high-reuse + long-context + loaded,
  and *vanishes* on short low-reuse public traces — lives only in Appendix A
  (WildChat) and §2, never stated as the thesis.

## 2. Principle

> Tables and figures carry the numbers; prose carries the **claim** and the
> **mechanism**. Keep at most the 2–3 numbers per paragraph that a reader would
> quote.

## 3. Concrete consolidation (already built)

| Replace | With | File |
|---|---|---|
| `tab:w1` + `tab:w2` + `tab:w2b` (p95) | one window-blocked `tab:p95_all` | `combined_results_tables.tex` |
| `tab:w1_p50` + `tab:w2_p50` + `tab:w2b_p50` | one `tab:p50_all` | `combined_results_tables.tex` |
| `ttft_bars_w1` + `ttft_bars_eval0` + `ttft_bars` (W2b) | one grouped multibar | `ttft_multibar.png` (+ `_p50`,`_p99`) |

The combined tables also **recompute column-min bolding**, which resolves the
W1 E2E-p99 and W2a TTFT-p99 bolding the live `\cmt{}` notes flag, and document
the missing W2a-p50 `random` row instead of leaving it unexplained.

## 4. Target prose (replaces per-window recitation)

**§5.1 p95 objective** — one paragraph, not three:
> Across all three windows GORGO posts the lowest TTFT p50 and p95
> (Table~\ref{tab:p95_all}, Fig.~\ref{fig:ttft_multibar}). The margin over the
> strongest baseline **widens with load**: from ~9% (p95) on the held-out
> nighttime window to ~12–25% at midday, because the learned `load_weight`=6.38
> rebalances traffic that hash-based session-affinity cannot. GORGO pays only a
> tuning-time p99 tax (ES exploration), which disappears once `gorgo-static`
> deploys frozen weights.

**§5.2 p50 objective** — one paragraph: same win on the *median*, but the
RTT-first operating point trades a small (~6–9%) E2E-p95 cost — the mirror image
of the p95 policy, which wins both. (Keep `tab:weights_compare` — it is the
"two operating points" payoff and is already compact.)

**New §6 "When does GORGO win?" (Discussion)** — the missing thesis, ~2 short
paragraphs, no new numbers:
1. *Mechanism.* GORGO wins by buying cache locality **without** the load
   concentration that cache-greedy policies incur — Fig.
   `cache_and_concentration` shows it lands in the low-latency / moderate-
   concentration corner while session-affinity gets the highest cache hit rate
   but the worst p95. That is the "where" (loaded, high-reuse) and the "why"
   (it prices queueing that prefix-only routing ignores).
2. *Regime boundary (the WildChat/LMSYS control).* Point at Appendix A: on
   short-prompt, low-reuse public traces the cache lever is near-zero, so GORGO
   ties `random` — evidence that its gains are **caused by** reuse+load
   structure, not by the routing machinery per se. This is the honest "better
   than alternatives **here, and why**" framing.

## 5. Use every figure (one beat each)

| Figure | Beat / where |
|---|---|
| `dataset_combined.png` (Fig 1) | §2 — ARTChat-411k is the high-reuse, long-context, concentrated-user regime → routing matters |
| `rtt_timeseries.png` | §4 — 18× cross-region RTT spread → network-aware term justified |
| `tune_convergence_hillclimb.png` | §5.1 — ES drives `load_weight`→6.38: load avoidance is the dominant lever |
| **`ttft_multibar.png` (NEW)** | §5 — headline: GORGO lowest p95 in every window, margin widens midday (replaces 3–4 per-window bar figs) |
| `cache_and_concentration.png` | §6 Discussion — the mechanism: cache benefit without pathological concentration |
| `cache_convergence.png` | Appendix B — cache reuse converges (supporting) |
| `load_weight_ablation.png` | Appendix C — `load_weight`=0 → single-replica concentration → E2E blow-up (reward-hack control) |
| `ttft_bars_wildchat.png` | Appendix A — regime control: advantage vanishes off-regime |

## 6. Net effect

6 result tables → 2; 4 TTFT bar figures → 1 multibar; per-window number
recitation → 2 claim-led paragraphs + 1 Discussion subsection that finally
states *where* (loaded, high-reuse) and *why* (load-aware cache routing) GORGO
beats each alternative. No data changes — only density and framing.
