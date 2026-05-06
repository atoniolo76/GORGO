OpenReview.net
Notifications
Notifications
Activity
Tasks
Alessio Ricci Toniolo 
back arrowBack to the profile of Alessio Ricci Toniolo
GORGO: Maximizing KV-Cache Reuse While Minimizing Network Latency in Cross-Region LLM Load Balancing
Download PDF
Alessio Ricci Toniolo, Abinaya Dinesh, Rome Thorstenson
23 Jan 2026 (modified: 30 Apr 2026)
Submitted to ICML 2026
Conference, Senior Area Chairs, Area Chairs, Reviewers, Authors
Notifications
BibTeX
CC BY 4.0
Verify Author List: I have double-checked the author list and understand that additions and removals will not be allowed after the abstract submission deadline.
TL;DR: Our cross region load balancing policy dynamically routes inference tasks with network-aware selective pushing to maximize TTFT.
Abstract:
Distributing LLM inference across geographical regions can improve Time-to-First-Token (TTFT) by regionalizing service deployments. While existing multi-region load balancers save prefill computation by prioritizing Key-Value (KV) Cache hit rate, they ignore cluster networking latency, a critical factor in routing decisions. In this work, we introduce GORGO, a novel method for minimizing TTFT by optimizing total serving cost expressed as a function of available compute, network latency, and prefix caching. We benchmark GORGO against three baselines: (1) naive least-load routing, which ignores prefix-cache overlap; (2) prefix-similarity routing, which selectively pushes requests to the replica with the highest cached-prefix overlap; and (3) a centralized HTTP proxy that runs the GORGO policy while tracking requests across all nodes. Using extensive profiling on custom infrastructure, we provide a novel analysis of component-level latency bottlenecks. We demonstrate that GORGO reduces P99 TTFT through network-aware routing, improves average TTFT by preventing pathological cross-region forwarding, and provides consistent performance across regions under diverse network topologies.

Primary Area: Deep Learning->Large Language Models
Keywords: Load Balancing, LLM Inference, Multi-Region Distributed System, Networking, Cloud Computing
Ethics Agreement: I certify that all co-authors of this work have read and are committed to adhering to the Call for Papers, Author Instructions, Research Ethics, and Peer-review Ethics.
LLM Policy: This submission requires Policy A.
Proceedings-only Option: If this paper is accepted, the authors tentatively plan to present it in person at the conference (as a poster and, if selected, as an oral).
Reciprocal Reviewing Status: None of the authors are qualified according to the definition given in the Peer Review FAQ.
Reciprocal Reviewing Exemption Reason: None of the authors have two or more submitted peer review papers in the fields of ML/DL.
Reciprocal Reviewing Author:  Abinaya Dinesh
Submission Number: 25738
Filter by reply type...

Filter by author...




12 / 12 replies shown
Add:
Paper Decision
Decisionby Program Chairs30 Apr 2026 at 09:13 (modified: 30 Apr 2026 at 11:34)Program Chairs, AuthorsRevisions
Decision: Reject
Comment:
While reviewers agreed that the paper studies a relevant systems problem and shows some promising empirical gains, they also raised serious concerns about originality, clarity, and the strength of the evaluation. In particular, the main technical idea is viewed as a simple combination of known signals, the paper does not clearly validate the robustness of its cost model, and the experiments remain too limited in scope to support the broader claims. I have read the authors’ rebuttal and taken it into account, but the response mainly points to revisions and additional experiments that would be needed rather than resolving the current evidence gap. Therefore, I support rejection at this time, while encouraging the authors to revise and resubmit after strengthening the evaluation, sharpening the contribution, and improving the presentation.

Official Review of Submission25738 by Reviewer YPRp
Official Reviewby Reviewer YPRp16 Mar 2026 at 05:22 (modified: 06 Apr 2026 at 20:52)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer YPRpRevisions
Summary:
This paper addresses the routing problem for LLM inference requests in geo-distributed GPU clusters, with the goal of minimizing Time-to-First-Token (TTFT). Existing load balancers treat KV cache prefix reuse and network latency as independent optimization objectives. However, the authors present an example showing that a high cache hit rate in a remote region does not necessarily result in better TTFT than a lower local hit rate, demonstrating that the two must be jointly optimized. To this end, the authors propose an additive cost model that combines measured cross-region round-trip latency, prefill savings from prefix overlap, and the current queue depth into a single per-request routing decision. They implement two versions: a distributed per-region load balancer (GORGO) and a centralized middleware proxy (GORGO-proxy), and evaluate them on a real GPU infrastructure spanning three intercontinental regions against two baselines: least-load routing and prefix-similarity routing. GORGO-proxy achieves approximately 2.5× lower median TTFT compared to both baselines.

Strengths And Weaknesses:
Strengths:

The paper addresses a real system-level problem in LLM deployment practice: how to trade off cache locality, cross-region latency, and admission/queueing delays to optimize TTFT. This issue has significant practical relevance for geo-distributed services.
The work includes real multi-region experiments rather than purely synthetic simulations. As described in Section 4.1, the experiments span the US West Coast, Germany, and Israel, providing more convincing evidence than simulations alone.
The paper’s comparison between centralized and distributed architectures yields counterintuitive but valuable empirical insights. The performance of GORGO-proxy overturns the common assumption in cross-region inference that centralized coordination is prohibitively costly, offering a new reference point for future system design.
Weaknesses:

The two external baselines provided in Section 4.4 are overly simplistic. Are "least-load" and "PREFIXTRIE," both of which have obvious flaws, strategies currently employed in real-world production environments, or are they merely failure modes hypothesized by the authors? The paper should consider including more sophisticated, existing methods that attempt to balance multiple signals as comparative baselines.
Experimental results indicate that GORGO-proxy outperforms the distributed GORGO in both TTFT and throughput. This creates ambiguity regarding the paper's core contribution: is it the "jointly optimized cost function" or the "distributed architecture"? The authors should clarify in the discussion under what specific conditions the distributed scheme would possess an advantage over the centralized proxy.
In Tables 2-5, the count of completed requests varies significantly, ranging from 57 to 141. The paper does not explain whether the missing requests were dropped, timed out, or simply did not finish. If a specific policy completes fewer requests during periods of overload, comparisons of median and tail metrics become inherently biased.
Table 1 performs a linear regression on only 87 WildChat samples to derive a prefill rate () of 0.0938 ms/token. In practice, the prefill rate is sensitive to batch composition, prompt lengths, hardware status, and the structure of cache hits. Since the routing optimization objective depends heavily on this scalar, the authors should provide a stability or sensitivity analysis for .
Section 4.5 summarizes the conclusions in only two sentences, and the main figures lack numerical annotations. Even if constrained by page limits, the main text should include at least one concise comparison table covering the median TTFT and P99 of all methods. Correspondingly, the content in the Future Work section is excessively lengthy, taking up half a page, and this part could be streamlined accordingly.
Soundness: 2: fair
Presentation: 2: fair
Significance: 2: fair
Originality: 2: fair
Key Questions For Authors:
See Weaknesses

Limitations:
yes

Overall Recommendation: 2: Reject: For instance, a paper with technical flaws, weak evaluation, inadequate reproducibility, incompletely addressed ethical considerations, or writing so poor that it is not possible to understand its key claims.
Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
Compliance With LLM Reviewing Policy: Affirmed.
Code Of Conduct Acknowledgement: Affirmed.
Rebuttal by Authors
Rebuttalby Authors (Alessio Ricci Toniolo, Rome Thorstenson, Abinaya Dinesh)31 Mar 2026 at 01:31 (modified: 31 Mar 2026 at 07:15)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, AuthorsRevisions
Rebuttal:
Baselines and comparative methods. Consistent hashing is the traditional approach, where users are tied to replicas — comparably naive in many ways (Xia et al., 2025). Least-load is a core native algorithm for NGINX (least_conn) and AWS Elastic Load Balancing, where it is recommended for workloads with variable processing times (Nawazdhandala, 2026; AWS, 2025). Modern inference engines like SGLang and vLLM utilize radix-trie structures (RadixAttention) to manage KV-cache reuse (Zheng et al., 2024), and prefix-aware routing is the primary mechanism in distributed systems like Preble (Srivatsa et al., 2024). GORGO benchmarks against least-load and PREFIXTRIE — policies identical to their counterparts in academia and industry, not hypothesized failure modes.

Distributed GORGO vs. GORGO-proxy. GORGO-proxy shows an over 2× improvement in P99 TTFT by jointly capturing remote network latency, local queueing latency, and estimated prefill cost in a centralized environment (Appendix A). The proxy is able to track running states of all replicas in a cluster, whereas with PREFIXTRIE, regional load balancers only track remote KV-Cache states for requests forwarded from their replicas. The distributed scheme retains advantages in deployments where a centralized proxy introduces unacceptable single-point-of-failure risk or where inter-region bandwidth constraints make full state synchronization impractical. We will clarify this tradeoff in the discussion.

Prefill rate sensitivity (Table 1). The linear regression on 87 WildChat samples yields a prefill rate of 0.0938 ms/token with R² = 0.98, consistent with the well-documented near-linear relationship between input token count and prefill time on fixed hardware (Sarathi-Serve, Agrawal et al., 2024). In practice, users of this method would compute this scalar based on their own data, hardware, and cache structure. We will add a sensitivity analysis showing routing decision stability across a range of prefill rate values.

Presentation of results and paper structure. The numerical data is included in Tables 2–5; we will add a concise comparison table covering median and P99 TTFT for all methods in the main text and restructure conclusions and future work to strike a better balance.

Official Review of Submission25738 by Reviewer GroY
Official Reviewby Reviewer GroY12 Mar 2026 at 21:41 (modified: 06 Apr 2026 at 20:52)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer GroYRevisions
Summary:
The paper focuses on geo-distributed LLM inference when requests can be served locally or forwarded across regions. It introduces GORGO, a distributed load balancer that selects the serving region by minimizing an estimated Time-to-First-Token (TTFT) cost combining prefix-cache reuse, measured inter-region round-trip time, and instantaneous queue state. Experiments on custom infrastructure compare GORGO to least-load routing, prefix-similarity routing, and a centralized proxy approach, showing improved average and P99 TTFT via network-aware routing.

Strengths And Weaknesses:
S1. The problem is interesting. Tail latency of TTFT in cross-region LLM serving is indeed an issue that deserves study and careful consideration.

S2. The experiments are generally consistent with the paper’s hypotheses.

W1. Contribution is very limited. The proposed methods are mostly straight-forward, and most of the ideas lack novelty.

W2. The presentation is poor. The paper contains a lot of bullet points without proper explanation. The paper looks more like a draft than a finalized product.

W3. The assumptions are overly simplistic. The proposed cost model is too simplified that may not hold under realistic serving dynamics such as varying batching behavior, memory pressure, and non-linear interactions between load and latency.

W4. The experimental evaluation is insufficient. It only tests three regions with a fixed model and fixed hardware, resulting in an overly idealized setting. The paper does not evaluate whether GORGO remains effective in more realistic scenarios where multiple factors jointly vary, such as different batch sizes, concurrency levels, KV-cache hit rates etc.

Soundness: 2: fair
Presentation: 1: poor
Significance: 2: fair
Originality: 1: poor
Key Questions For Authors:
Q1. Cost model validity. How robust is the additive TTFT cost model across different batching regimes, concurrency levels, and GPU memory pressure?

Q2. How do results change with larger models, longer contexts, and more diverse prompts?

Q3. What is the synchronization mechanism and update interval for cross-region state exchange?

Limitations:
L1. Over-simplified performance model. The TTFT model assumes an additive decomposition (network + residual prefill + queueing). In practice, continuous batching, batch composition, memory pressure, and cache eviction can introduce non-linear interactions that may break the model and degrade routing quality.

L2. Limited experimental scope and realism. Evaluation is restricted to three regions with a fixed model and fixed hardware. This may not reflect real deployments where region count is larger, hardware is heterogeneous, and network conditions vary more widely.

L3. Insufficient coverage of key serving factors: The experiments do not systematically vary important parameters such as batch size, concurrency, KV-cache hit rate, context length distributions, and decoding settings. It is unclear whether GORGO remains effective when these factors change jointly.

Overall Recommendation: 2: Reject: For instance, a paper with technical flaws, weak evaluation, inadequate reproducibility, incompletely addressed ethical considerations, or writing so poor that it is not possible to understand its key claims.
Confidence: 3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
Ethical Review Concerns:
n.a.

Compliance With LLM Reviewing Policy: Affirmed.
Code Of Conduct Acknowledgement: Affirmed.
Final Justification:
The authors acknowledged the concerns. The paper needs a major revision that cannot be accomplished in this submission cycle.

Rebuttal by Authors
Rebuttalby Authors (Alessio Ricci Toniolo, Rome Thorstenson, Abinaya Dinesh)31 Mar 2026 at 01:28 (modified: 31 Mar 2026 at 07:15)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, AuthorsRevisions
Rebuttal:
We thank the reviewer for their detailed feedback. We address each concern below.

W1 (Limited novelty). We respectfully disagree that the contribution is straightforward. While the individual signals GORGO uses (network RTT, prefix overlap, queue depth) are individually known, the core contribution is their joint formulation as an additive cost model for geo-distributed TTFT optimization — a problem setting that prior work (e.g., SkyWalker) addresses only partially by optimizing cache reuse or load independently, never together with wide-area latency. Furthermore, the GORGO-proxy result — demonstrating that a centralized router outperforms decentralized coordination by 2.5× on median TTFT, contradicting the prevailing assumption in the literature (Xia et al., 2025) that centralized coordination is too expensive — is, to our knowledge, a novel empirical finding in the cross-region LLM serving space.

W2 (Presentation quality). We acknowledge this concern and will revise the paper to convert bullet-point-heavy sections into connected prose, particularly in §3 (cost model) and §4 (system design). We will also add a running example that traces a single request through the GORGO routing decision to improve clarity.

W3 (Simplified cost model). We want to clarify the intended scope of the cost model. As stated in §3, the model is not intended to perfectly predict per-request latency; rather, it operationalizes a consistent set of tradeoffs (forwarding latency, residual prefill, admission delay) that can be evaluated and tuned within a deployment. That said, we agree that the paper should better characterize where the additive assumption breaks down. We will add a discussion of non-linear interactions (e.g., memory pressure causing non-linear queueing) and include sensitivity analysis showing cost model accuracy across different batching regimes.

W4 (Insufficient evaluation). We agree that broader evaluation would strengthen the paper. We are running additional experiments including: (1) larger models (e.g. Qwen-3.5-35B-A3B); (2) longer-context datasets (lmsys-chat-1m); (3) varying concurrency levels and batch sizes; and (4) additional regions beyond the current three. We will include these results in the revision.

Q1 (Cost model robustness across batching/concurrency/memory pressure). GORGO factors queueing time as a key signal in routing decisions, which surfaces GPU memory pressure indirectly: when memory pressure causes requests to queue at a replica, GORGO detects the elevated admission delay and intelligently forwards requests to alternative replicas (including cross-region) based on the additive cost. We will add explicit experiments varying concurrency and batching to characterize robustness.

Q2 (Larger models, longer contexts, diverse prompts). As noted above, we are actively benchmarking on alternative and larger models and lmsys-chat-1m. We will report these results in the revision.

Q3 (Cross-region synchronization mechanism). Each GORGO load balancer maintains a lightweight synchronization object containing per-replica queueing wait times and measured inter-region RTT. These are propagated via low-bandwidth periodic updates (not full state replication), and the measured RTT is included as a state variable when evaluating remote regions. An anonymous version of the codebase is available here: https://anonymous.4open.science/r/gotoni-212D.

 Replying to Rebuttal by Authors
Rebuttal Acknowledgement by Reviewer GroY
Rebuttal Acknowledgementby Reviewer GroY01 Apr 2026 at 23:30Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors
Acknowledgement: (c) Partially resolved or unresolved, but the remaining concerns are not easily addressed in a short rebuttal - Please select this option sparingly and only when you believe that your questions concern the core tenets of the work, and addressing them requires a significant update to the paper.
Reasons:
The authors acknowledged the concerns. The paper needs a major revision that cannot be accomplished in this submission cycle.

Official Review of Submission25738 by Reviewer bWgP
Official Reviewby Reviewer bWgP11 Mar 2026 at 11:44 (modified: 09 Apr 2026 at 00:01)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer bWgPRevisions
Summary:
This paper addresses the challenge of request routing in cross-region distributed LLM serving setting, It emphasizes that decisions should consider network latency and queue status in addition to KV cache reuse rates. So, proposed GORGO routes requests to the region to yield the lowest TTFT based on prefill costs, inter-region RTT, and admission/queue states. The results show that GORGO outperforms existing methods, particularly in improving median TTFT.

Strengths And Weaknesses:
Soundness
While this paper seems technically reasonable overall, further clarification and evidence are required to fully substantiate its claims.
The evaluation was conducted using Mistral-7B-Instruct-v0.3 across 8xA100 nodes in the US West Coast, Germany, and Israel, but prefill speeds in this setup will almost always exceed the network latency caused by geographical dispersion. As a result, this experimental environment is insufficient to show the diverse operational dynamics of GORGO policy.
To estimate the core parameter , the authors performed a linear regression based correlation between input tokens and TTFT using only 87 samples from the Wildchat dataset in \S 4.2. Can it be generalized? Although the authors propose online feedback-based automatic tuning as future work, it remains unclear whether the current design can maintain robustness across broader scenarios.
The overall evaluation requires further elaboration, and the ablation study for the contribution of each design element should have been more clearly decoupled through an ablation study. For example, exactly where do the improvements in TTFT, inter-token latency, or TPOT originate from?
The paper mainly models request handling as a choice between reusing an existing KV cache and recomputing the prefill when no suitable cache is available. However, it gives little consideration to other possible designs and optimizations, such as KV-cache offloading to memory, SSD, or shared storage systems (e.g., mooncake /FAST25).
Presentation
The abstract claims an improvement in P99 TTFT, yet no corresponding P99 results are presented in the main evaluation section.
The definitions of core concepts and variables are generally unclear. Specifically, it is not properly explained what "GPU availability" refers to, nor is there a clear description of the "peer+local" parameters within PrefillCost.
One of the authors' claimed contributions is that GORGO operates under low-latency and low-bandwidth observability. However, no specific experimental results or metrics are provided in the text to support this claim.
Evaluations on the core predictive components' accruacy, such as PrefillCost or QueueWaitTime, are required.
Significance
The paper correctly points out a frequently overlooked limitation in existing methods: that a region with a high cache hit rate is not always the optimal choice.
Originality
The paper’s strongest originality seems to be its attempt to jointly capture KV-cache locality, network latency, and queue/admission state within a single routing objective.
It would be better to have new mechanism beyond well integrating known signals or techniques for the problem.
Soundness: 2: fair
Presentation: 1: poor
Significance: 2: fair
Originality: 2: fair
Key Questions For Authors:
The abstract claims an improvement in P99 TTFT, yet no corresponding P99 results are presented in the main evaluation section.
The definitions of core concepts and variables are generally unclear. Specifically, it is not properly explained what "GPU availability" refers to, nor is there a clear description of the "peer+local" parameters within PrefillCost.
One of the authors' claimed contributions is that GORGO operates under low-latency and low-bandwidth observability. However, no specific experimental results or metrics are provided in the text to support this claim.
Evaluations on the core predictive components' accruacy, such as PrefillCost or QueueWaitTime, are required.
The evaluation was conducted using Mistral-7B-Instruct-v0.3 across 8xA100 nodes in the US West Coast, Germany, and Israel, but prefill speeds in this setup will almost always exceed the network latency caused by geographical dispersion. As a result, this experimental environment is insufficient to show the diverse operational dynamics of GORGO policy.
The paper mainly models request handling as a choice between reusing an existing KV cache and recomputing the prefill when no suitable cache is available. However, it gives little consideration to other possible designs and optimizations, such as KV-cache offloading to memory, SSD, or shared storage systems (e.g., mooncake /FAST25).
Limitations:
Yes

Overall Recommendation: 3: Weak reject: A paper with clear merits, but also some weaknesses, which overall outweigh the merits. Papers in this category require revisions before they can be meaningfully built upon by others. Please use sparingly.
Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
Compliance With LLM Reviewing Policy: Affirmed.
Code Of Conduct Acknowledgement: Affirmed.
Final Justification:
Some concerns from the review have been solved, but I believe there exist further tasks to incorporate. I will maintain my score.

Rebuttal by Authors
Rebuttalby Authors (Alessio Ricci Toniolo, Rome Thorstenson, Abinaya Dinesh)31 Mar 2026 at 01:37 (modified: 31 Mar 2026 at 07:15)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, AuthorsRevisions
Rebuttal:
​​We appreciate the reviewer's concern about the generalizability of the t_p estimate. The regression was conducted specifically on WildChat samples matching the prompt-length distribution used in evaluation, yielding an R² of 0.986. Although we had a small sample size, this reflects the well-documented near-linear relationship between input token count and prefill time on a fixed hardware configuration (Sarathi-Serve, Agrawal et al., 2024). We agree that t_p will drift across batch compositions, load conditions, and hardware configurations. This is precisely why §8 proposes online feedback-based automatic tuning of t_p as a direction for future work. Within the scope of our fixed Mistral-7B / 8xA100 deployment, the high model fit supports the validity of the estimate for the experiments presented. Extending to heterogeneous accelerators or model families would require per-deployment calibration, which we acknowledge as a limitation in §7.

In terms of our novelty, GORGO's primary contribution is not a new algorithmic primitive but a systems contribution: the identification that jointly optimizing network latency and KV-cache reuse in an additive per-request cost model — and doing so in a per-region, lightweight distributed control plane — meaningfully changes routing outcomes. The empirical finding that GORGO-proxy achieves 2.5x lower median TTFT over two standard and employed baselines (least-load and prefix-trie) demonstrates that the integration itself produces non-trivial improvements.

Thank you for flagging potentially imprecise definitions. "GPU availability" refers to the SGLang runtime's admission state — whether running requests are below a configured threshold and KV-cache occupancy is below a memory limit, as polled via SGLang's /get_server_info endpoint. "PrefillCost(peer+local)" refers to the residual prefill time at the candidate region after subtracting prefix overlap estimated by the local trie — i.e., (L_p − L_hit) · t_p. We will add explicit definitions for both terms in §3.4 of the revision.

Regarding low-latency, low-bandwidth observability: the channel exchanges only periodic RTT heartbeats and lightweight peer state summaries (queue depth and prefix-locality estimates) — never raw KV-cache contents or prompt text. This keeps the bandwidth constant.

Regarding accuracy evaluations of core predictive components: we did profile per-request TTFT via Perfetto tracing (Figure 4) and validated that the linear model captures prefill latency well (R² = 0.986). Since the cost model is intended as a consistent ranking signal rather than a precise predictor (§5), we believe aggregate TTFT outcomes are the appropriate evaluation target. We will include a scatter plot of predicted versus observed TTFT components in the revision.

In terms of our evaluation, GuideLLM maintains 10 concurrent requests in flight over a 60-second window, driving sustained load against a single SGLang instance per region. Under Poisson and throughput workloads, arrival rate intentionally exceeds single-region admission capacity at peak, causing queue buildup reaching hundreds to thousands of queued tokens — as reflected in the motivating example in §2.4 (6,500 running tokens at US-West, ~300 ms admission delay). When local queue delay exceeds the cross-region RTT to a lower-load peer, GORGO's forwarding decisions produce meaningful improvements. The P99 results in the appendix capture this tail behavior: GORGO-proxy reduces P99 TTFT from over 18 seconds (least-load) to 436 ms by avoiding pathological forwarding to saturated replicas. Under light load, the inter-region tradeoff is less pronounced; the 10-concurrent-request configuration provides sufficient pressure to observe meaningful queue buildup while remaining reproducible across three regions.

Finally, in terms of other designs and optimizations, others have done great work in this direction, including NVIDIA Dynamo. GORGO currently assumes KV-cache state is local to each region's GPU memory, and the routing decision is made accordingly. Systems like Mooncake's Transfer Engine enable globally accessible KV-cache state with explicit transfer cost modeling — which maps naturally onto GORGO's additive cost framework by adding a transfer latency term alongside the existing network and prefill terms. A future version of GORGO could evaluate globally available cache state, factor in the round-trip latency between the Transfer Engine and a candidate replica, and potentially forward requests to a region whose cache is warm via TE even if it lacks local prefix matches.

 Replying to Rebuttal by Authors
Rebuttal Acknowledgement by Reviewer bWgP
Rebuttal Acknowledgementby Reviewer bWgP02 Apr 2026 at 22:45Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors
Acknowledgement: (c) Partially resolved or unresolved, but the remaining concerns are not easily addressed in a short rebuttal - Please select this option sparingly and only when you believe that your questions concern the core tenets of the work, and addressing them requires a significant update to the paper.
Reasons:
I thank the authors for the responses. I have several unresolved concerns.

I am still not convinced about the generalizability of the t_p estimate. The rebuttal highlights a strong fit based on limited samples and suggests online auto-tuning of t_p as future work. This leaves some inconsistency, and resolving it would likely require more extensive validation than can be provided in a rebuttal.
My concerns about the evaluation have been somewhat addressed. However, the response did not fully cover the question of observability under low-latency, low-bandwidth conditions, or discuss alternative designs and optimizations in this setting.
While the responses were helpful, I believe the paper would benefit from further work to address these points.
Official Review of Submission25738 by Reviewer TbMi
Official Reviewby Reviewer TbMi10 Mar 2026 at 08:24 (modified: 08 Apr 2026 at 08:59)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer TbMiRevisions
Summary:
This paper proposes a load balancer to redirect LLM inference requests across geo-distributed sites to minimize TTFT. The proposes a simple heuristic that jointly considers remote network latency, local queueing latency, and total prefill cost to decide whether to redirect a request or not. Evaluation of this heuristic of a three-site setup and a single model (Mistral-7B-Instruct-v0.3) shows latency improvements.

Strengths And Weaknesses:
This is the beginning of a research project but no where near any satisfactory completion. The paper doesn't take any systematic approach to the problem and proposes a strawman heuristic. The writing isn't on par with top-quality conferences either. Even then it completely ignores the decoding part of the problem; nor does it consider prefix-related challenges when prefixes aren't exactly the same. The evaluation is small and insubstantial. It doesn't have breadth or depth and does not answer important questions beyond TTFT (TBT/ITL, Throughput, Bandwidth usage, GPU utilization, etc.)

Soundness: 1: poor
Presentation: 1: poor
Significance: 1: poor
Originality: 1: poor
Key Questions For Authors:
What's the impact of deployment scale?
What's the impact of network topology as well as bandwidth, in addition to pairwise latency?
What's the impact on different model size?
What do TBT/ITL distributions look like?
What happens if network conditions dynamically or frequently change?
Limitations:
Yes

Overall Recommendation: 2: Reject: For instance, a paper with technical flaws, weak evaluation, inadequate reproducibility, incompletely addressed ethical considerations, or writing so poor that it is not possible to understand its key claims.
Confidence: 5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.
Compliance With LLM Reviewing Policy: Affirmed.
Code Of Conduct Acknowledgement: Affirmed.
Final Justification:
The rebuttal addressed some of my concerns, but my overall assessment about the paper remains unchanged at this time.

Rebuttal by Authors
Rebuttalby Authors (Alessio Ricci Toniolo, Rome Thorstenson, Abinaya Dinesh)31 Mar 2026 at 01:24 (modified: 31 Mar 2026 at 07:15)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, AuthorsRevisions
Rebuttal:
We thank the reviewer for their detailed critique. We respond to each point below.

On the paper being "nowhere near satisfactory completion" and proposing a "strawman heuristic." We believe this assessment is too strong. The contribution operates at two levels: (1) a joint cost model that co-optimizes KV-cache reuse, wide-area latency, and admission queueing for geo-distributed TTFT — signals that prior work (SkyWalker, CacheBlend, production gateways) each handle in isolation; and (2) the GORGO-proxy architecture, which achieves a 2.5× median TTFT improvement over decentralized coordination, directly refuting the assumption in Xia et al. (2025) that centralized cross-region coordination is too expensive. The additive cost model is not a strawman — it is a deliberately interpretable formulation that lets operators inspect which factor dominates any given routing decision, and it demonstrably outperforms the baselines. We do agree the paper should be reframed to foreground the proxy architecture as the primary contribution.

On writing quality. Acknowledged. We will convert bullet-heavy sections into connected prose, add a worked routing example, and tighten the overall narrative.

On ignoring decoding (TBT/ITL). We have collected TBT data that was only included in . Key results: GORGO achieves a tight TBT distribution (σ = 0.21 ms, range 10.71–12.84 ms) versus Least-Load (σ = 1.34 ms, range 0.01–16.11 ms). GORGO-proxy shows a higher-centered but still well-bounded distribution (σ = 0.98 ms, range 9.60–19.07 ms), reflecting the decode cost of sustaining larger batch sizes. We will add full TBT/ITL distributions along with throughput, bandwidth, and GPU utilization metrics.

On exact prefix matching limitations. Fair point. The current Radix Tree implementation requires exact prefix matches. We will discuss partial prefix matching and similarity-threshold routing as extensions and characterize sensitivity to prefix alignment.

Q: Deployment scale. The proxy stores text-based representations of each region's KV-Cache, scaling linearly with context length (Xu, 2026). Under bursty workloads, this introduces queueing overhead at the proxy itself. We will quantify this scaling behavior and discuss mitigations such as hash-based prefix representations and LRU eviction.

Q: Network topology beyond pairwise latency. GORGO monitors pairwise latency dynamically and adapts routing in real time. A natural extension is using point-to-point latency to select proximal replicas as relay points while registering forwarded request text in local Radix Trees for improved global cache awareness. We will discuss this in the revision.

Q: Model size impact. GORGO's advantage scales with model size because prefill cost dominates at larger scales. Under a 70B model (~0.5 ms/tok prefill), our §2.4 analysis shows that Israel — despite 183 ms RTT — achieves an estimated TTFT of ~283 ms through 80% cache overlap, while US-West (5% overlap, ~478 ms) and Germany (60% overlap + 281 ms RTT, ~481 ms) pay much more. At 7B, the local option dominates because prefill is cheap. This suggests GORGO's relative advantage over cache-unaware baselines increases with scale. We are validating this with experiments on Qwen-3.5-35B-A3B and will include results in the revision.

Q: Dynamic network conditions. GORGO emits periodic heartbeat probes between replicas, maintaining a running estimate of inter-region RTT that feeds directly into routing decisions. The implementation is available in our anonymized codebase: https://anonymous.4open.science/r/gotoni-212D/README.md

 Replying to Rebuttal by Authors
Rebuttal Acknowledgement by Reviewer TbMi
Rebuttal Acknowledgementby Reviewer TbMi31 Mar 2026 at 20:07Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors
Acknowledgement: (a) Fully resolved - My concerns have been adequately addressed. If you select this option, please consider adjusting your score accordingly.
Reasons:
Thank you for providing aditional data and acknowledging the limitations. I think the paper has some future but needs additional work as pointed out across the board. I am increasing a my score to 2.

About OpenReview
Hosting a Venue
All Venues
Contact
Sponsors
Donate
FAQ
Terms of Use / Privacy Policy
News
OpenReview is a long-term project to advance science through improved peer review with legal nonprofit status. We gratefully acknowledge the support of the OpenReview Sponsors. © 2026 OpenReview