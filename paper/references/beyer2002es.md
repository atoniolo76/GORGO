# Evolution strategies – A comprehensive introduction

- **Bibkey**: `beyer2002es`
- **Authors**: Hans-Georg Beyer and Hans-Paul Schwefel
- **Venue / Year**: Natural Computing, vol. 1, pp. 3–52, 2002
- **arXiv / DOI**: 10.1023/A:1015059928466
- **URL**: https://link.springer.com/article/10.1023/A:1015059928466

## One-paragraph summary
A canonical, English-language survey of evolution strategies (ES). Starting from the historical roots of ES in 1960s Germany (Rechenberg, Schwefel), the article systematically presents the design principles of variation and selection operators across the (1+1)-, (mu+lambda)-, and (mu,lambda)-ES families, and surveys theoretical results on convergence and step-size adaptation, including the Rechenberg 1/5-success rule and self-adaptive sigma. It is the most-cited theoretical reference for ES and is appropriate whenever a paper invokes the optimality, convergence, or step-size-control behavior of (1+1)-ES.

## Supports our claim
> "...runs a Gaussian $(1{+}1)$-evolution strategy \citep{rechenberg1973} over them in log-space..." (line 442)
> "The step size $\sigma$ adapts according to Rechenberg's 1/5-success rule \citep{rechenberg1973}..." (line 449)

The two `\citep{rechenberg1973}` citations on §GORGO point to a 1973 German monograph that is hard to verify and that does not, in modern form, give the convergence-rate analysis used in the paper. Beyer & Schwefel 2002 is the canonical English-language source for the (1+1)-ES algorithm description and the 1/5 rule, and is a more accessible co-citation.

## Suggested citation site
§GORGO, around lines 442 and 449: at both invocations of `\citep{rechenberg1973}`, change to `\citep{rechenberg1973,beyer2002es}` so a reader can verify the algorithm and the 1/5 rule from a modern reference.

## BibTeX
```
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
```
