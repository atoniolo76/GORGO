# Consistent Hashing and Random Trees

- **Bibkey**: `karger1997` (already in references.bib)
- **Authors**: David Karger, Eric Lehman, Tom Leighton, Rina Panigrahy, Matthew Levine, Daniel Lewin
- **Venue / Year**: Proceedings of the 29th Annual ACM Symposium on Theory of Computing (STOC '97), 1997
- **arXiv / DOI**: 10.1145/258533.258660
- **URL**: https://dl.acm.org/doi/10.1145/258533.258660

## One-paragraph summary
Introduces consistent hashing, the technique that maps requests (keys) to a small, stable set of servers via a hash-ring abstraction so that adding or removing a server only displaces $O(1/n)$ of the keys. Consistent hashing is the standard mechanism behind "session-affinity" and "sticky" routing in production HTTP and CDN systems (Akamai, Memcached, Cassandra, NGINX `hash`/`ip_hash` modes). It is the canonical citation whenever a paper invokes hash-based routing of a request to a fixed replica.

## Supports our claim
> "...hash-based session affinity, and so on." (line 166)
> "\textbf{simple-session-affinity} hashes the first 256 token IDs of the prompt modulo the number of replicas. The same prefix always maps to the same replica..." (lines 483–485)

Hash-based session affinity is invoked as a known production policy without citation. `karger1997` is already in references.bib but is never cited in main.tex.

## Suggested citation site
Two reasonable sites:

(a) §Introduction / "These three TTFT terms pull in different directions" paragraph, around line 166: change `hash-based session affinity, and so on.` to `hash-based session affinity \citep{karger1997}, and so on.`

(b) §Baselines, around line 483, definition of `simple-session-affinity`: append `\citep{karger1997}` to the first sentence.

## BibTeX
Already present in references.bib as `@inproceedings{karger1997,...}` — no addition needed; only a citation site needs to be added in main.tex.
