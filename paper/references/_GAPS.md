# Citation gaps in main.tex (paper v4)

Sweep of `paper/main.tex` (~920 lines) against `references.bib`. Below are non-trivial claims that currently have no citation, plus four citations that exist in `references.bib` but are never invoked in the body. Conservative: each item is either a load-bearing factual claim or a missing attribution to a canonical source. Optional/uncertain items are flagged.

Numbering follows order of appearance in main.tex.

---

## 1. Diurnal cross-region demand (§Introduction, line 138–142)

**Claim**: "Operators distribute model replicas across multiple geographies for three reasons: to follow diurnally shifting demand from a globally distributed user base, to provide redundancy against regional capacity loss or partition, and to aggregate accelerator capacity that no single region currently supplies \citep{skywalker,skyserve}."

**Status**: Skywalker's abstract explicitly covers "regional diurnal patterns" — verified. Existing cite is defensible. Optional: add `benson2010imc` as a measurement-paper anchor.

**Suggested edit (optional)**: `\citep{skywalker,skyserve,benson2010imc}`.

**KB entry**: `benson2010imc.md`.

---

## 2. Hash-based session affinity (§Introduction, line 166)

**Claim**: "...NGINX-style least-request and least-connections \citep{nginx2024}, AIBrix-style longest-prefix-match \citep{aibrix2024}, hash-based session affinity, and so on."

**Gap**: "Hash-based session affinity" has no citation. The canonical reference is consistent hashing.

**Suggested bibkey**: `karger1997` (already in references.bib, never cited).

**Suggested edit**: change `hash-based session affinity, and so on.` to `hash-based session affinity \citep{karger1997}, and so on.`

**KB entry**: `karger1997.md`.

---

## 3. Least-request / least-load theoretical foundation (§Background lines 257; §Baselines line 469)

**Claim**: `\textsc{least-request}` and `\textsc{least-load}` baselines are introduced as folklore. Their theoretical foundation is the power-of-two-choices result.

**Suggested bibkey**: `mitzenmacher` (already in references.bib, never cited).

**Suggested edit**: at §Baselines line 469, change `\citep{nginx2024}` to `\citep{nginx2024,mitzenmacher}`.

**KB entry**: `mitzenmacher2001.md`.

---

## 4. Production gateways treat inference as HTTP (§Background, line 234)

**Claim**: "AWS Bedrock cross-region inference \citep{aws_bedrock_xregion} and GKE multi-cluster Gateways \citep{gke_gateway} provide global routing and failover but treat inference as an HTTP endpoint."

**Status**: cited. No gap. (Listed for completeness — the user's hot-spot list flagged this.)

---

## 5. (1+1)-Evolution strategy algorithm description (§GORGO line 442)

**Claim**: "...runs a Gaussian $(1{+}1)$-evolution strategy \citep{rechenberg1973} over them in log-space..."

**Gap**: `rechenberg1973` is a 1973 German monograph. Modern English-language readers expect a survey-level theoretical reference.

**Suggested bibkey**: `beyer2002es`.

**Suggested edit**: change `\citep{rechenberg1973}` to `\citep{rechenberg1973,beyer2002es}`.

**KB entry**: `beyer2002es.md`.

---

## 6. 1/5-success rule (§GORGO line 449)

**Claim**: "The step size $\sigma$ adapts according to Rechenberg's 1/5-success rule \citep{rechenberg1973}..."

**Gap**: The 1/5 rule has joint origins in Schumer & Steiglitz 1968 and Rechenberg 1973. Citing only Rechenberg is historically incomplete; co-citing Beyer & Schwefel 2002 also covers the modern theoretical analysis.

**Suggested bibkeys**: `beyer2002es` (primary) and optionally `schumer1968` (historical accuracy).

**Suggested edit**: change `\citep{rechenberg1973}` to `\citep{rechenberg1973,beyer2002es}` (or `\citep{rechenberg1973,beyer2002es,schumer1968}` if the author wants full attribution).

**KB entries**: `beyer2002es.md`, `schumer1968.md`.

---

## 7. Bandit schedulers (§Related Work line 772)

**Claim**: "bandit and RL schedulers \citep{decima} and Bayesian-optimization configuration tuners \citep{cherrypick}..."

**Gap**: Decima (Mao et al., SIGCOMM 2019) is RL, not bandit. The "bandit" half of the claim is uncited.

**Suggested bibkey**: `auer2002bandits`.

**Suggested edit**: change `bandit and RL schedulers \citep{decima}` to `bandit \citep{auer2002bandits} and RL schedulers \citep{decima}`.

**KB entry**: `auer2002bandits.md`.

---

## 8. Distributed prefix-aware scheduler (§Related Work line 762)

**Claim**: The "Routing and scheduling" paragraph enumerates AIBrix, k-LPM, DLPM, Llumnix, and CacheBlend as related routing/scheduling work but omits Preble, the earliest distributed prefix-aware scheduler. The task prompt assumed Preble was already cited; it is not.

**Suggested bibkey**: `preble` (already in references.bib, never cited; author list also looks incorrect — see KB entry for the corrected entry).

**Suggested edit**: insert a clause naming Preble alongside AIBrix in the "Routing and scheduling" paragraph, e.g.: `AIBrix \citep{aibrix2024} provides prefix-aware load balancing as a baseline; Preble \citep{preble} is an earlier distributed prefix-aware scheduler combining locality with load-aware reassignment; k-LPM \citep{klpm2025} ...`

**KB entry**: `preble.md`.

---

## 9. (Verified, not a gap) Low-context regime makes TTFT indistinguishable (§Dataset lines 296–298)

**Claim**: "...prefill cost down enough that TTFT distributions are nearly indistinguishable across cache-aware and cache-blind policies---an observation also made for cross-region serving more broadly \citep{skywalker}."

**Status**: Skywalker is the right co-citation for the *cross-region* version of this claim; Mooncake is already cited elsewhere for the routing-cost decomposition. Existing cite is defensible. The author may optionally add `qin2024mooncake` here as well, but no new KB entry needed.

---

## 10. (Optional) Global optimizer / convergence claim

The original task brief mentioned "(1+1)-ES is a global optimizer in continuous parameter spaces" as a possible gap. **This sentence is not in the current main.tex** — only "Gaussian (1+1)-evolution strategy" is invoked. Item 5 (Beyer & Schwefel 2002) covers the modern reference. No additional KB entry needed.

---

## Summary table

| # | Section / line | Claim | Suggested bibkey | New entry needed? |
|---|---|---|---|---|
| 1 | Intro / 142 | Diurnal cross-region demand | `benson2010imc` | optional |
| 2 | Intro / 166 | Hash-based session affinity | `karger1997` | already in .bib |
| 3 | Baselines / 469 | Least-request / least-load theory | `mitzenmacher` | already in .bib |
| 5 | GORGO / 442 | (1+1)-ES algorithm | `beyer2002es` | yes |
| 6 | GORGO / 449 | 1/5 rule | `beyer2002es` (+ optional `schumer1968`) | yes |
| 7 | Related / 772 | Bandit schedulers | `auer2002bandits` | yes |
| 8 | Related / 762 | Preble (distributed prefix-aware scheduler) | `preble` | already in .bib (author list needs fix) |

Gap count: **6 substantive gaps + 1 optional + 4 already-in-bib-never-cited (`mitzenmacher`, `karger1997`, `preble`, plus `vtcsheng` and `sarathi_serve` are cited).**

---

## Consolidated BibTeX block (paste into references.bib)

The four entries below are the new additions. `mitzenmacher`, `karger1997`, and `preble` are already present in references.bib; only their citation sites in main.tex need to be added (and Preble's author list could be corrected — see `preble.md`).

```bibtex
@article{beyer2002es,
  title   = {Evolution strategies---A comprehensive introduction},
  author  = {Beyer, Hans-Georg and Schwefel, Hans-Paul},
  journal = {Natural Computing},
  volume  = {1},
  number  = {1},
  pages   = {3--52},
  year    = {2002},
  doi     = {10.1023/A:1015059928466}
}

@article{schumer1968,
  title   = {Adaptive step size random search},
  author  = {Schumer, M. A. and Steiglitz, K.},
  journal = {IEEE Transactions on Automatic Control},
  volume  = {13},
  number  = {3},
  pages   = {270--276},
  year    = {1968},
  doi     = {10.1109/TAC.1968.1098903}
}

@article{auer2002bandits,
  title   = {Finite-time Analysis of the Multiarmed Bandit Problem},
  author  = {Auer, Peter and Cesa-Bianchi, Nicol{\`o} and Fischer, Paul},
  journal = {Machine Learning},
  volume  = {47},
  number  = {2--3},
  pages   = {235--256},
  year    = {2002},
  doi     = {10.1023/A:1013689704352}
}

@inproceedings{benson2010imc,
  title     = {Network Traffic Characteristics of Data Centers in the Wild},
  author    = {Benson, Theophilus and Akella, Aditya and Maltz, David A.},
  booktitle = {Proceedings of the 10th ACM SIGCOMM Conference on Internet Measurement (IMC)},
  pages     = {267--280},
  year      = {2010},
  doi       = {10.1145/1879141.1879175}
}
```

## Optional one-line edits to main.tex (for the author to apply)

```
Line 166: hash-based session affinity, and so on.
       -> hash-based session affinity \citep{karger1997}, and so on.

Line 442: \citep{rechenberg1973}
       -> \citep{rechenberg1973,beyer2002es}

Line 449: \citep{rechenberg1973}
       -> \citep{rechenberg1973,beyer2002es}      (or +schumer1968 for full historical attribution)

Line 469: \citep{nginx2024}.
       -> \citep{nginx2024,mitzenmacher}.

Line 772: bandit and RL schedulers \citep{decima}
       -> bandit \citep{auer2002bandits} and RL schedulers \citep{decima}

Line 762 (Related Work): insert "Preble \citep{preble} is an earlier distributed
        prefix-aware scheduler combining locality with load-aware reassignment;"
        between the AIBrix and k-LPM clauses.
```
