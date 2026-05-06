# Adaptive step size random search

- **Bibkey**: `schumer1968`
- **Authors**: M. A. Schumer and Kenneth Steiglitz
- **Venue / Year**: IEEE Transactions on Automatic Control, vol. 13, no. 3, pp. 270–276, 1968
- **arXiv / DOI**: 10.1109/TAC.1968.1098903
- **URL**: https://ieeexplore.ieee.org/document/1098903/

## One-paragraph summary
Introduces the Adaptive Step Size Random Search (ASSRS), the algorithm now widely recognized as the (1+1)-Evolution Strategy with the one-fifth success rule. The paper proposes that the step size of a random-search optimizer should be increased when too many proposed moves are accepted and decreased otherwise; the authors empirically observe that an acceptance rate of roughly 1/5 yields the fastest convergence on a class of test functions. The 1/5 rule that the modern evolution-strategies literature attributes to Rechenberg's 1973 monograph in fact has a co-origin in this 1968 paper.

## Supports our claim
> "The step size $\sigma$ adapts according to Rechenberg's 1/5-success rule \citep{rechenberg1973}..." (line 449)

The 1/5 rule is jointly attributable to Rechenberg (1973) and Schumer & Steiglitz (1968). Citing only Rechenberg understates the historical record; the fairer attribution adds Schumer & Steiglitz alongside.

## Suggested citation site
§GORGO, around line 449, sentence introducing the 1/5 rule: change `\citep{rechenberg1973}` to `\citep{rechenberg1973,schumer1968}`. (Optional / scholarly accuracy — not load-bearing for the paper's claims.)

## BibTeX
```
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
```
