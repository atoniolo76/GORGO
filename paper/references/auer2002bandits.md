# Finite-time Analysis of the Multiarmed Bandit Problem

- **Bibkey**: `auer2002bandits`
- **Authors**: Peter Auer, Nicolò Cesa-Bianchi, and Paul Fischer
- **Venue / Year**: Machine Learning, vol. 47, pp. 235–256, 2002
- **arXiv / DOI**: 10.1023/A:1013689704352
- **URL**: https://link.springer.com/article/10.1023/A:1013689704352

## One-paragraph summary
The canonical reference for finite-time regret analysis of the stochastic multi-armed bandit. Introduces the UCB1 and UCB2 algorithms and proves logarithmic regret bounds that hold uniformly over time and over all bounded-support reward distributions, without prior knowledge of the reward gaps. UCB-style methods are the standard "bandit" baseline against which any online-learning scheduler is compared, and the paper is the appropriate citation when a paper says "bandit scheduling" without naming a specific system.

## Supports our claim
> "bandit and RL schedulers \citep{decima} and Bayesian-optimization configuration tuners \citep{cherrypick}" (line 772)

The current sentence attributes both "bandit" and "RL" schedulers to a single citation, `\citep{decima}`. Decima (Mao et al., SIGCOMM 2019) is an RL scheduler, not a bandit method; the "bandit" half of the claim is currently uncited. Auer et al. 2002 is the canonical bandit reference; pairing it with Decima fixes the attribution.

## Suggested citation site
§Related Work / "Online and learned routing", around line 772: change
`bandit and RL schedulers \citep{decima}` to
`bandit \citep{auer2002bandits} and RL schedulers \citep{decima}`.

## BibTeX
```
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
```
