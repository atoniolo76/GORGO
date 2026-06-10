# GORGO Paper — Review Questions & Future Research Directions

Companion to the green-colored review pass on `paper.tex` (rome branch,
June 2026). Green (`\rev`/`\cmt`) marks this pass; dark blue (`\chg`) is the
earlier pass. Items below either need an author decision, a data check, or are
out of scope for inline edits.

## 1. Data-integrity questions (resolve before submission)

These are the items a careful reviewer would treat as red flags. Each has an
inline green comment at the relevant spot in `paper.tex`.

1. **Headline claim vs. tables.** The abstract, contribution bullet, and
   conclusion claimed the GORGO family wins *every* percentile in *all*
   windows. Table 2 (W1) shows `random` winning p99 (1.996 s vs. hillclimb's
   2.660 s) and Table 3 (W2a) shows `least-request` winning p99 (1.758 s vs.
   gorgo-static's 1.844 s). I narrowed the claim throughout to "p50 and p95
   everywhere, full sweep in the midday window." If the broader claim is
   believed true via some other run, the tables need to change instead.
2. **Phantom p99 numbers in Limitations.** The noise-floor paragraph quoted
   gorgo-static 1.626 s and prefix-cache 1.702 s — values that appear in no
   table. Where did they come from? An earlier run? Confirm which run is
   canonical and regenerate every number in the paper from it.
3. **Wrong bold in Table 3 (W2a p99).** 1.785 was bolded as the column minimum
   but least-request's 1.758 is lower. I corrected the bold, but the underlying
   data should be re-verified — a wrong bold sometimes means a transcription
   error in the cell, not the marker.
4. **Arrival counts vs. load narrative.** §2 says W1/W2a/W2b contain
   ~4,800/4,200/3,100 arrivals, which makes W2b the *smallest* window — yet
   W2b is consistently described as "heavier midday load" with "47% more
   requests." The stated req/s rates (1.2 night, 1.7 midday) match the
   *completed* counts (2,095/2,207/3,071), not the arrival counts. Likely the
   4,800/4,200 figures are pre-filter totals. Verify against run manifests and
   state which they are.
5. **Identical n across policies.** Result tables report one n per window for
   all policies, but n is defined as per-policy completions under a per-policy
   concurrency limit. Are results truncated to a common completed prefix? Say
   so explicitly; otherwise the identical n looks like an error.
6. **gorgo-static missing from the W1 table.** §3.3 says gorgo-static ran W1
   with manual starter weights, and §4 says all eight policies ran in parallel,
   but Table 2 lists six policies and gorgo-static is not among them. Add the
   row or explain the exclusion — its absence invites the suspicion that the
   starter weights performed poorly.
7. **`random` missing from the p50-objective W2a table** (Table 6) while
   present in every other window. Add or explain.
8. **Compute accounting.** The checklist claimed 16 fleet-runs / 24 GPU-hours,
   which only covers the original two windows. With p50-objective re-runs,
   W3/W4 stress, and WildChat, it's roughly 70 fleet-runs / ~210 GPU-hours.
   I updated the estimate — verify against actual billing.

## 2. Methodology questions

1. **Single trace, single day.** All main results come from one production
   trace on one day (April 2, 2026). How stable are the learned weights across
   days, weeks, or traffic-mix shifts? Even one replication on a different day
   would substantially strengthen the generalization claim.
2. **No confidence intervals on percentiles.** The paper hand-derives one p99
   SE in Limitations, but no table carries error bars. Bootstrap CIs per
   percentile (resampling completed requests) are cheap and would let readers
   judge every margin, not just the one discussed. Paired comparisons are
   natural here since all policies replay identical arrivals.
3. **ES seed variance.** Each window has a single (1+1)-ES run. The ES is
   stochastic — how much do the converged weights (and the resulting TTFT)
   vary across seeds? A small seed study would tell us whether
   w = (0.39, 1.88, 6.38) is a basin or an accident.
4. **Window/hop sensitivity.** The 64-sample window and 16-sample hop are
   asserted, not justified. How sensitive are convergence speed and stability
   to these? (Related: bead go-x47 covers update-interval characterization.)
5. **Why does c=64 yield 100% success in main windows but ~90% in stress
   windows at the same c=64?** "Denser arrival bursts" needs quantification —
   what exactly differs between W1 and W4, which cover the same wall-clock
   period? (Inline comment in the stress appendix's W4 description.)
6. **WildChat at c=32 vs. GLM-5.1 at c=64.** Why a different concurrency for
   the control experiment? If incidental, note it; if necessary, explain.
7. **Output cap at 128 tokens.** The cap prevents decode from dominating, but
   E2E, ITL, and decode-throughput columns are then measured in an
   artificially decode-light regime. How do conclusions change with realistic
   output lengths? (Acknowledged in Limitations; worth an experiment — bead
   go-lzs touches this.)
8. **Additive cost model linearity.** Prefill cost is modeled as linear in
   uncached tokens; continuous batching and chunked prefill make this
   non-linear at high load. Validate or replace the additive assumption
   (bead go-46q).

## 3. Claims & positioning

1. **"First single-router policy" priority claim** — softened to "to our
   knowledge" inline. Worth a systematic check against 2025–2026 systems
   (e.g., recent AIBrix/production-gateway releases) before submission.
2. **GORGO acronym is never expanded.** Reviewers will ask. Expand it or call
   it a codename.
3. **Reviewer-bead overlap.** The bd tracker already carries reviewer asks
   that this pass did not resolve (they need experiments, not edits):
   go-obp (separate cost-function contribution from architecture), go-rbm
   (quantify observability overhead), go-5bv (KV-offload related work), go-ud0
   (per-component ablation). The related-work expansion (go-5bv) is the only
   one doable without new runs.

## 4. Presentation suggestions (not done inline, low priority)

- Notation drift between Eq. 1 and Eq. 2: `rtt(u)` vs. `rtt_u`, `c_u(t_r)` vs.
  `c_u`, set-difference vs. cardinality-difference forms. Harmonize.
- The commented-out Reproducibility Statement names four artifact paths;
  either restore it (NeurIPS allows it post-checklist) or delete the comment
  block before camera-ready.
- Figure 4's caption says "on W2" — disambiguate W2a/W2b (assumed W2a inline).
- Table 1 reports WildChat-4.8M as 3,199,860 requests while the name says
  4.8M (presumably 4.8M turns vs. 3.2M conversations) — one clarifying phrase
  would prevent a reviewer nitpick.

## 5. Future research directions

1. **Architecture ablation in a unified workload** (go-obp): distributed
   GORGO vs. centralized static vs. centralized autotuned in the same sweep,
   separating the cost-function contribution from the architecture.
2. **Safe online tuning.** The W3 stress result (hillclimb p99 = 7.186 s)
   shows live ES is unsafe under load. Natural directions: trust-region
   constraints on weight proposals, shadow-mode scoring (evaluate candidates
   on logged decisions before deploying), or constrained bandits with a p99
   guardrail. This would turn the "freeze weights" recommendation into a
   spectrum rather than a binary.
3. **Multi-objective tuning.** The p50-objective run improves TTFT but
   consistently degrades E2E — a visible Pareto trade-off. Scalarized or
   constrained multi-objective ES (e.g., minimize p95 TTFT s.t. E2E p95 ≤
   budget) is a small step with practical payoff.
4. **Drift and re-tuning cadence.** How long do frozen weights stay good?
   Replaying a week of traffic with weights frozen on day 1 would directly
   answer the "dedicated calibration window" claim in the conclusion.
5. **KV-offload tiers** (go-whr, go-5bv): position GORGO as the
   which-replica decision layered over Mooncake/AttentionStore-style
   where-does-KV-live mechanisms, then measure the composition.
6. **Fixing gorgo-autotune.** The rate-inversion failure has three proposed
   fixes in Limitations (denominator filtering, regularized minimum, warm-up
   estimation). Implementing one would rehabilitate the most interpretable
   variant — physical rates are auditable in production where black-box
   weights are not.
7. **Per-request contextual weighting.** GORGO learns one global weight
   vector. Long-prompt requests plausibly want cache-first routing while
   short ones want RTT-first — exactly the regime split the p50/p95 objective
   comparison exposes. A contextual policy (weights as a function of request
   length / predicted reuse) could win both.
8. **Heterogeneous fleets and PD-disaggregation.** Both are excluded in
   Limitations; both add cost terms that the additive model can in principle
   absorb. The interesting question is whether the ES still converges when the
   cost surface gains these dimensions.
