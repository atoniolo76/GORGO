# Citation Audit — `main.tex` (paper_v4 / NeurIPS 2026)

Read-only audit. No modifications were made to `main.tex` or `references.bib`.
Source: `/home/rome/gt/gorgo/.worktrees/paper_v4/paper/main.tex`,
`/home/rome/gt/gorgo/.worktrees/paper_v4/paper/references.bib`.

## Summary

- Total unique cite-keys referenced in `main.tex`: **31**
- Total cite *sites* (each `\citep{key}` occurrence; multi-key cites split out): **65**
- Verdict counts:
  - SOLID: **52**
  - LOOSE: **8**
  - WRONG: **2**
  - UNVERIFIABLE: **3**
- Bib hygiene: **3 entries are present in `references.bib` but never cited**
  (`vtcsheng2024`, `preble`, `mitzenmacher`, `karger1997`); none are missing.
- One bib entry has a **wrong venue** field (`skyserve`).
- `glm2024` is cited as the citation for the GLM-5.1 endpoint but the bib entry
  is the **GLM-4** technical report; the trace is from a GLM-5.1 endpoint.

## Per-citation table

The table below lists every cite site in arrival order. The "claim" column is
truncated to fit; cross-reference the line number in `main.tex` for full context.

| line | key | section | claim (truncated) | verdict | note |
|------|-----|---------|-------------------|---------|------|
| 142 | skywalker | Intro | "Operators distribute model replicas across multiple geographies … to follow diurnal demand, redundancy, aggregate capacity" | SOLID | SkyWalker abstract explicitly motivates cross-region serving for diurnal aggregation. |
| 142 | skyserve | Intro | same sentence | SOLID | SkyServe's stated motivation is serving across regions/clouds with spot for cost+availability; supports redundancy claim. |
| 146 | aibrix2024 | Intro | "the gateway … chooses one replica per request based on observable state" | SOLID | AIBrix is exactly such an LLM gateway; supports framing. |
| 146 | nginx2024 | Intro | same sentence | SOLID | NGINX docs describe per-request load balancing. |
| 150 | orca | Intro | "queueing delay behind in-flight requests admitted via continuous batching" | SOLID | Orca (OSDI '22) is the canonical continuous-batching reference. |
| 154 | sglang2024 | Intro | "modern engines expose a per-replica radix-tree KV-cache that lets requests skip prefill for any prefix already present" | SOLID | SGLang's RadixAttention is exactly this. |
| 154 | vllm2023 | Intro | same sentence | LOOSE | vLLM's PagedAttention enables prefix sharing through paged blocks but is not a radix tree; the sentence as written ("radix-tree KV-cache") fits SGLang only. Either soften to "radix-tree- or block-hash-keyed KV cache" or drop vllm2023 from this site. |
| 157 | skywalker | Intro | "KV-cache locality at the routing layer is a first-order determinant of latency" | SOLID | SkyWalker reports 1.74–6.30× latency from KV-aware routing. |
| 157 | qin2024mooncake | Intro | same sentence | SOLID | Mooncake's central thesis is KVCache-centric serving. |
| 165 | nginx2024 | Intro | "NGINX-style least-request and least-connections" | LOOSE | NGINX docs describe `least_conn` (least-connections). The docs do **not** label any method "least-request" — that's an Envoy/Istio term. The cite supports "least-connections" but not "least-request." Soften to "least-connections (NGINX) and least-request (Envoy)" or drop the "least-request" half. |
| 166 | aibrix2024 | Intro | "AIBrix-style longest-prefix-match" | SOLID | AIBrix's gateway includes prefix-cache routing; verified via project README. |
| 178 | lmsys2024 | Intro | "LMSYS-Chat-1M" | SOLID | Standard dataset citation. |
| 178 | wildchat2024 | Intro | "WildChat-4.8M" | SOLID | Standard dataset citation. |
| 183 | skywalker | Intro | "TTFT distributions are nearly indistinguishable across cache-aware and cache-blind policies" | LOOSE | SkyWalker shows the *opposite*: large gaps when cache-aware. The cite is being used to support "indistinguishable on public datasets," but SkyWalker's experiments use long-context corpora that do show separation. Either cite SkyWalker for "long-context shows separation" (supportive) or drop the cite here (the claim about public datasets being indistinguishable is your own measurement). |
| 214 | sglang2024 | Background | "Inference engines such as SGLang … maintain KV cache, continuous batching" | SOLID | Direct match. |
| 214 | vllm2023 | Background | same sentence | SOLID | PagedAttention paper. |
| 217 | orca | Background | "continuous batching" | SOLID | Canonical. |
| 221 | sglang2024 | Background | "SGLang's RadixAttention operationalizes this via a per-replica radix trie" | SOLID | Exact match. |
| 229 | sglang2024 | Background | "per-request scheduler decides admission and order within the running batch" | SOLID | SGLang's runtime does this. |
| 229 | orca | Background | same sentence | SOLID | Orca's iteration-level scheduler is the prototype. |
| 231 | aibrix2024 | Background | "AIBrix … exposing prefix-aware, least-load, and least-request policies behind a common interface" | SOLID | Verified from AIBrix repo/project description. |
| 234 | nginx2024 | Background | "production gateways like NGINX" | SOLID | Trivially supported. |
| 235 | aws_bedrock_xregion | Background | "AWS Bedrock cross-region inference … occupy the same architectural slot but at the network level, with no surface for KV state" | SOLID | AWS docs describe automatic Region selection for inference profiles; nothing about KV state — supports the negative claim. |
| 238 | skywalker | Background | "Routing-layer decisions matter disproportionately on long-context workloads because the prefill term they implicitly select for can dominate TTFT" | SOLID | SkyWalker quantifies this. |
| 238 | qin2024mooncake | Background | same sentence | SOLID | Mooncake's prefill/decode disaggregation is explicitly motivated by prefill cost. |
| 244 | vulimiri2015 | Background | "Network latency: the proxy-to-replica RTT $\mathrm{rtt}(u)$" | LOOSE | Vulimiri 2015 (NSDI '15, "Global Analytics in the Face of Bandwidth and Regulatory Constraints") is about WAN analytics and bandwidth/regulatory constraints, not about characterizing inter-region RTTs. The standard citation for "inter-region RTTs are tens to hundreds of ms" is e.g. ping-mesh papers (Pingmesh, SIGCOMM '15) or the WANalytics CIDR '15 paper. Either drop the cite (the claim is uncontroversial and well-known) or replace with a paper that actually measures inter-region RTTs. |
| 260 | aibrix2024 | Background | "prefix-cache (longest-prefix-match … in the AIBrix style)" | SOLID | Same as above. |
| 267 | aws_bedrock_xregion | Background | "AWS Bedrock cross-region inference … provides global routing and failover but treats inference as an HTTP endpoint" | SOLID | AWS docs verified. |
| 268 | gke_gateway | Background | "GKE multi-cluster Gateways … global routing and failover" | UNVERIFIABLE | The exact AWS-style URL didn't load in this audit (404 / redirect). The substantive claim is uncontroversial — multi-cluster Gateways do global L7 routing — but you should manually re-check the URL at submission time; the doc may have moved to `docs.cloud.google.com`. |
| 270 | modal_prefix_caching | Background | "Modal supports intra-region prefix caching but does not optimize across regions" | UNVERIFIABLE | The Modal `/docs/guide/prefix-caching` URL 404'd at audit time; Modal does have prefix caching guidance for vLLM/SGLang in their docs (verified via search), but the specific URL pinned in the bib should be re-verified. The "intra-region only" half of the claim is also a negative attribution that documentation rarely makes explicitly — the cite supports "Modal supports prefix caching," not "but not across regions." Soften to "Modal supports prefix caching [cite]; cross-region behavior is not exposed in their public documentation." |
| 271 | skypilot | Background | "SkyPilot aggregates capacity across clouds but does not observe per-replica KV state" | SOLID | SkyPilot is exactly an inter-cloud broker; "no KV awareness" is true (it's a cluster-launcher). |
| 273 | skyserve | Background | "SkyServe manages cross-region placement" | SOLID | Verified in abstract. **Note**: bib entry says `Proceedings of the 2025 USENIX Annual Technical Conference` but SkyServe was published at **EuroSys 2025**, not USENIX ATC. Bib fix needed. |
| 274 | skywalker | Background | "SkyWalker adds KV-aware cross-region load balancing with selective KV pushing" | SOLID | Direct quote from SkyWalker abstract. |
| 292 | lmsys2024 | Dataset | "LMSYS-Chat-1M averages 467 input tokens per request and 9% global token-weighted prefix reuse" | SOLID for dataset, claim numbers are author-measured (not from the cited paper). The cite anchors the dataset; ok. |
| 293 | wildchat2024 | Dataset | "WildChat-4.8M averages 2,925 input tokens and 5.3% intra-user reuse" | SOLID | Same — cite anchors dataset; the numbers are your measurement. |
| 298 | skywalker | Dataset | "an observation also made for cross-region serving more broadly" | LOOSE | Same concern as line 183. SkyWalker doesn't explicitly say "TTFT is indistinguishable on public datasets"; it says cache-aware policies improve in their (long-context) setting. The "cross-region serving" framing is at most adjacent. Either soften ("see also [skywalker] for a related observation in cross-region serving") or drop. |
| 305 | glm2024 | Dataset | "free GLM-5.1 endpoint" | WRONG/LOOSE | The bib entry `glm2024` cites the **GLM-4** technical report (arXiv:2406.12793), but the paper says "GLM-5.1 endpoint." The model family is the same vendor (Zhipu/GLM) so the cite is topic-adjacent, but the cited paper does not describe GLM-5.1. Either (a) replace with a GLM-5.1-specific reference if one exists, (b) reword to "GLM-family endpoint [glm2024]" with a footnote that 5.1 is the deployed variant, or (c) drop the cite. |
| 368 | qin2024mooncake | Dataset | "We re-encode GLM-5.1 into the Mooncake FAST'25 trace schema" | SOLID | Verified: Mooncake released traces in `FAST25-release/traces/` on the official GitHub repo, JSONL with timestamp, input/output token counts, hashed block IDs. Format matches the description in §3 of `main.tex`. |
| 441 | rechenberg1973 | Method | "a Gaussian (1+1)-evolution strategy" | SOLID | Rechenberg 1973 is the canonical citation for (1+1)-ES. (The 1/5 rule is from his 1965 dissertation but the 1973 book consolidates and is the standard cite.) |
| 447 | rechenberg1973 | Method | "Rechenberg's 1/5-success rule" | SOLID | Same. |
| 471 | nginx2024 | Baselines | "least-request … to bridge the ~10s staleness between metrics scrapes" | LOOSE | NGINX docs describe `least_conn` and the staleness behavior is an implementation detail of the GORGO proxy. The cite supports "this kind of policy exists in production load balancers," not the staleness claim. Acceptable but consider dropping; the cite reads as a generic "this is a known LB heuristic" attribution. |
| 479 | aibrix2024 | Baselines | "AIBrix-style longest-prefix-match policy" | SOLID | Same as before. |
| 495 | sglang2024 | Setup | "2×L40S SGLang server" | SOLID | Trivial software citation. |
| 499 | qwen3technical | Setup | "Qwen3.5-35B-A3B-FP8" | SOLID | Citation anchors the Qwen3 family; the deployed checkpoint is a Qwen3.5 variant — the bib entry is the Qwen3 tech report (Alibaba), which the description ("Qwen Team, 2025, Alibaba Cloud") matches. Make sure the bib entry covers 3.5 versions; if Alibaba published a separate Qwen3.5 report, prefer that. |
| 745 | distserve2024 | Limitations | "does not split prefill/decode" | SOLID | DistServe is the canonical reference for prefill/decode disaggregation. |
| 745 | splitwise2024 | Limitations | same sentence | SOLID | Splitwise paper verified — splits the two phases on separate machines. |
| 747 | kvflow2025 | Limitations | "scheduler-internal state … that could tighten the prefill estimate" | SOLID | KVFlow is internal-cache routing, fits the "scheduler-internal" framing. |
| 747 | lpc2025 | Limitations | same sentence | SOLID | "Learned Prefix Caching" similarly uses internal info. |
| 755 | sglang2024 | Related | "RadixAttention in SGLang stores per-request KV state in a radix tree" | SOLID | Exact. |
| 756 | vllm2023 | Related | "PagedAttention in vLLM enables prefix sharing through paged memory" | SOLID | Exact. |
| 757 | kvlink2025 | Related | "improve the single-replica cache" | SOLID (topic-adjacent) | KVLink, ChunkKV: KV-reuse/compression for single-engine. Cite is generic. |
| 757 | chunkkv2025 | Related | same | SOLID | Same. |
| 758 | kvflow2025 | Related | same | SOLID | Same. |
| 758 | lpc2025 | Related | same | SOLID | Same. |
| 761 | qin2024mooncake | Related | "Mooncake is a KV-cache-centric architecture and the source of our trace format" | SOLID | Verified — Mooncake authors released the trace alongside the paper at FAST '25. |
| 762 | aibrix2024 | Related | "AIBrix provides prefix-aware load balancing as a baseline" | SOLID | Verified. |
| 763 | klpm2025 | Related | "k-LPM formulates LLM scheduling under TTFT constraints as NP-hard" | SOLID | Bib title matches: "LLM Query Scheduling with Prefix Reuse and Latency Constraints" (NeurIPS '25). NP-hard claim should be verified in the body of that paper, but topic is on-the-nose. |
| 764 | dlpm2025 | Related | "DLPM targets fairness with locality" | SOLID | Title is "Locality-aware Fair Scheduling in LLM Serving" — exact match. |
| 765 | llumnix2024 | Related | "Llumnix migrates requests across replicas" | SOLID | Verified — Llumnix's headline contribution is live request migration. |
| 766 | routerwu2025 | Related | "Multi-LLM routers address model selection" | SOLID | Title "Efficient Training-Free Online Routing for High-Volume Multi-LLM Serving" matches; multi-model selection. |
| 767 | yao2024cacheblend | Related | "CacheBlend routes by trie-measured prefix length but does not measure network RTT" | LOOSE | CacheBlend's core contribution is cached-knowledge fusion for RAG (selective cache reuse and recomputation), not really "routes by prefix length." It does *use* prefix length but is not a routing system. The sentence overstates the routing aspect. Either soften ("CacheBlend reuses cached chunks by overlap but does not consider network RTT") or drop the routing framing. |
| 771 | jain2025performance | Related | "Performance-aware LLM load balancing learns a routing policy over mixed prefill/decode workloads" | SOLID | Verified: paper uses heuristic-guided RL to route prefill/decode-mixed workloads. The word "learns" is supportable (the paper has trainable response-length predictor + RL router). |
| 773 | decima | Related | "bandit and RL schedulers" | SOLID | Decima is RL-based scheduling for data processing clusters. The "bandit and RL" framing covers Decima on the RL side. |
| 774 | cherrypick | Related | "Bayesian-optimization configuration tuners" | SOLID | CherryPick uses Bayesian optimization for cloud configuration; exactly as cited. |
| 778 | sarathi_serve | Related | "Sarathi-Serve … disaggregate prefill from decode at the batch or GPU-pool level" | LOOSE | Sarathi-Serve does **chunked-prefill batching** (it interleaves prefill chunks with decode in the *same* batch), not phase disaggregation. DistServe and Splitwise disaggregate; Sarathi-Serve unifies. The grouping with DistServe/Splitwise misattributes. Reword: "Sarathi-Serve uses chunked prefill to share batches; DistServe and Splitwise disaggregate prefill from decode at the batch or GPU-pool level." |
| 778 | distserve2024 | Related | same sentence | SOLID | DistServe disaggregates. |
| 779 | splitwise2024 | Related | same sentence | SOLID | Splitwise disaggregates. |

## Critical fixes needed

These are the highest-impact issues — most are LOOSE-with-impact rather than
hard WRONG, but several would be flagged by a careful systems reviewer.

1. **`glm2024` — Wrong model generation (line 305).** The bib entry is the
   GLM-4 technical report; the paper says "GLM-5.1 endpoint." Either find a
   GLM-5.1-specific reference or reword to "GLM-family endpoint [glm2024]"
   with a footnote that 5.1 is the deployed variant. **Highest priority** —
   a reviewer who checks the bib will catch this immediately.

2. **`skyserve` bib entry — Wrong venue.** Bib says
   `Proceedings of the 2025 USENIX Annual Technical Conference` but SkyServe
   appeared at **EuroSys 2025** (`dl.acm.org/doi/10.1145/3689031.3717459`).
   Update the `booktitle` and authors (the full author list per ACM is
   Mao, Xia, Wu, Chiang, Griggs, Bhardwaj, Yang, Shenker, Stoica). **Highest
   priority** — bib metadata error.

3. **`sarathi_serve` misattribution (line 778).** The current sentence groups
   Sarathi-Serve with DistServe/Splitwise as phase-disaggregation systems, but
   Sarathi-Serve's contribution is the *opposite* (chunked prefill in a unified
   batch). Suggested rewrite: *"Sarathi-Serve uses chunked prefill to share a
   single batch between phases; DistServe and Splitwise disaggregate prefill
   from decode at the batch or GPU-pool level."* **High priority** — easy
   reviewer catch.

4. **`vulimiri2015` is a thin attribution (line 244).** The cite is being used
   as a generic "inter-region RTT exists" anchor, but Vulimiri 2015 is about
   bandwidth/regulatory constraints in WAN analytics, not about characterizing
   RTTs. The cite is not WRONG (the paper does measure WAN performance) but
   it's not the best citation for the specific framing. Either drop it (the
   claim is uncontroversial) or replace with a measurement paper that
   directly reports inter-region RTTs (Pingmesh, Bondi/IBM, or your own §5
   probe). **Medium priority.**

5. **`vllm2023` co-cited with "radix-tree KV-cache" (line 154).** vLLM uses
   block-hash-keyed prefix caching, not a radix tree. The current sentence
   conflates the two. Either remove `vllm2023` from this site (keeping it on
   the prefix-sharing-via-paged-memory line at 756), or reword to "radix-tree-
   or hash-keyed KV-cache." **Medium priority** — mild technical inaccuracy.

6. **`skywalker` cited for "indistinguishable on public datasets" (lines 183,
   298).** SkyWalker doesn't make this claim about public datasets; it argues
   for KV-aware routing in long-context settings. Your *own* experiments make
   the indistinguishability claim. The cite is at most adjacent. Drop or
   soften ("an observation consistent with KV-aware-routing motivation in
   prior work [skywalker]"). **Medium priority** — over-attribution.

7. **`yao2024cacheblend` framed as a router (line 767).** CacheBlend is a
   cached-chunk fusion technique, not a router that "routes by trie-measured
   prefix length." Reword to remove the "routes" verb. **Low/medium priority.**

8. **`nginx2024` for "least-request" (line 165).** NGINX docs describe
   `least_conn`, not "least-request" by name. The two policies are similar
   but the cite supports only the least-connections half. Reword to
   "least-connections (NGINX) and least-request (Envoy/Istio)" or drop the
   "least-request" half. **Low priority** — mostly a labeling cleanup.

## Citations missing from bib / unused entries

**Cite-keys referenced in `main.tex` and present in `references.bib`:** all 31
are present. **No `\citep` is missing a bib entry.**

**Bib entries with no `\citep` reference (dead weight in `references.bib`):**

- `vtcsheng2024` (Sheng et al., "Fairness in Serving LLMs," OSDI '24) — never
  cited; either drop from bib or add a cite somewhere in §10 Related Work
  alongside DLPM.
- `preble` (Srivatsa et al., "Preble: Distributed Prompt Scheduling," 2024) —
  never cited but highly relevant: it's the closest prior system to GORGO
  (distributed prefix scheduling). Strongly recommend adding a `\citep` in §10
  Related Work alongside Llumnix/AIBrix.
- `mitzenmacher` (Power of Two Choices, 2001) — never cited; seems to have been
  staged for a "least-of-two-choices" comparison that didn't end up in the
  text. Drop unless §6 Baselines is reworked.
- `karger1997` (Consistent Hashing, STOC '97) — never cited; presumably staged
  for the session-affinity baseline. Either cite from §6 (it would justify the
  hash-modulo construction in `simple-session-affinity`) or drop.

## Suggested follow-up (manual checks)

1. **Verify the GLM-5.1 endpoint claim has an appropriate cite.** This is the
   single most reviewer-visible cite issue. If no public GLM-5.1 reference
   exists, footnote it.

2. **Manually verify the AIBrix prefix-aware policy implementation.** The
   audit confirmed AIBrix has an LLM gateway with routing, but the readthedocs
   landing page didn't enumerate exact policy names. Pull the AIBrix gateway
   source (`gateway/pkg/plugins/`) to confirm `least-request`, `least-load`,
   and a longest-prefix-match policy actually ship under those names — these
   are the names you compare against.

3. **Re-verify the URLs in the Modal, GKE, and AWS Bedrock bib entries** are
   still live at submission time. AWS docs and GCP docs both moved hosts
   during this audit (one returned a 301 redirect, one 404'd). Documentation
   citations rot fast; consider adding `urldate` or archived snapshots.

4. **Sarathi-Serve and chunked-prefill framing.** The §10 sentence currently
   misattributes Sarathi-Serve. If you're going to mention chunked prefill
   anywhere, do it explicitly — otherwise drop Sarathi-Serve from the
   PD-disaggregation cluster and only keep DistServe/Splitwise.

5. **The `skywalker` over-citation.** It appears 6 times — line 142, 157,
   183, 238, 274, 298. Two of those (183, 298) are the over-attributed
   "TTFT-indistinguishable-on-public-datasets" framing. The remaining four are
   solid. Trim to ~4 sites.
