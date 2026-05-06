# The Power of Two Choices in Randomized Load Balancing

- **Bibkey**: `mitzenmacher` (already in references.bib)
- **Authors**: Michael Mitzenmacher
- **Venue / Year**: IEEE Transactions on Parallel and Distributed Systems, vol. 12, no. 10, pp. 1094–1104, 2001
- **arXiv / DOI**: 10.1109/71.963420
- **URL**: https://ieeexplore.ieee.org/document/963420/

## One-paragraph summary
The canonical analysis of the "power of two choices" load-balancing protocol: instead of routing each task to a uniformly random server (which yields a maximum load of $\Theta(\log n / \log\log n)$), route to the less-loaded of two uniformly random servers, which improves the maximum load to $\log\log n / \log d + O(1)$. The result is the theoretical foundation for least-request and least-load policies in production load balancers (NGINX, HAProxy, Envoy) and is the standard citation whenever a paper invokes "least-loaded-of-k" routing.

## Supports our claim
> "Standard production heuristics implement one of these signals at a time: NGINX-style least-request and least-connections \citep{nginx2024}, ..." (lines 164–165)
> "\textbf{least-request} routes to the replica with the fewest in-flight requests..." (line 469)
> "\textbf{least-load} routes to the replica with the lowest aggregate token-weighted load..." (line 473)

The least-request and least-load baselines are presented as if they are folklore; their theoretical justification is the power-of-two-choices result. `mitzenmacher` is already in references.bib but is never cited in main.tex.

## Suggested citation site
Two reasonable sites:

(a) §Background / "Common policies", around line 257, on the line introducing `least-request` and `least-load`: append `\citep{mitzenmacher}`.

(b) §Baselines, around line 469 (definition of `least-request`): append `\citep{mitzenmacher}` to acknowledge the underlying theory.

Single-sentence edit: at line 469, change
`scoring by $\max(\mathrm{sgLang\_running\_reqs}_u, \mathrm{proxy\_inflight}_u)$ to bridge the $\sim$10\,s staleness between metrics scrapes \citep{nginx2024}.`
to
`...staleness between metrics scrapes \citep{nginx2024,mitzenmacher}.`

## BibTeX
Already present in references.bib as `@article{mitzenmacher,...}` — no addition needed; only a citation site needs to be added in main.tex.
