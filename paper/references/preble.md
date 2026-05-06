# Preble: Efficient Distributed Prompt Scheduling for LLM Serving

- **Bibkey**: `preble` (already in references.bib)
- **Authors**: Vikranth Srivatsa, Zijian He, Reyna Abhyankar, Dongming Li, Yiying Zhang
- **Venue / Year**: arXiv:2407.00023, 2024 (subsequently presented at ICLR 2025)
- **arXiv / DOI**: arXiv:2407.00023
- **URL**: https://arxiv.org/abs/2407.00023

## One-paragraph summary
Preble is a distributed LLM serving system that targets prompt-prefix scheduling across multiple replicas. It introduces an E2 scheduling algorithm that combines prefix-cache locality with load-aware reassignment: replicas exchange summary metadata, and the global scheduler decides whether the locality benefit of routing to a warm replica outweighs the queueing cost. Preble reports up to 1.5–14.5x improvement on average and tail latency over single-replica prefix-aware schedulers, and is one of the earliest papers to formulate prefix-aware request routing as a distinct problem from intra-replica scheduling. The paper also discusses the coordination cost of cross-replica prefix-cache state.

## Supports our claim
> §Background, lines 227–238: discussion of "request-level routing as its own layer," which currently cites only AIBrix as a representative.
> §Related Work, around line 762: the "Routing and scheduling" paragraph names AIBrix, k-LPM, DLPM, Llumnix, and CacheBlend but not Preble.

`preble` is already in references.bib but is never cited in main.tex. The prompt for this task explicitly notes "Preble is cited" — that appears to be an outdated assumption from a prior draft. The current main.tex does not cite it, leaving a gap in the related-work coverage of cross-replica prefix-aware scheduling.

## Suggested citation site
§Related Work / "Routing and scheduling", around line 762, in the sentence enumerating prior cross-replica schedulers. Concretely, append Preble to the existing list:
`AIBrix \citep{aibrix2024} provides prefix-aware load balancing as a baseline; Preble \citep{preble} is an earlier distributed prefix-aware scheduler that combines locality with load-aware reassignment; k-LPM \citep{klpm2025} formulates ...`

## BibTeX
Already present in references.bib as `@misc{preble,...}` — no addition needed; only a citation site needs to be added in main.tex.

Note: the existing entry's author list (`Srivatsa, Varun and Gao, H. and Patel, M. and Sivaraman, A. and Rastogi, A.`) does not match the actual paper. The corrected entry is:

```
@misc{preble,
  title         = {{Preble}: Efficient Distributed Prompt Scheduling for {LLM} Serving},
  author        = {Srivatsa, Vikranth and He, Zijian and Abhyankar, Reyna and Li, Dongming and Zhang, Yiying},
  year          = {2024},
  eprint        = {2407.00023},
  archivePrefix = {arXiv}
}
```
