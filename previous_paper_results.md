4. Experimental Evaluation
This section describes the experimental methodology used
to evaluate GORGO. Our goal is to assess whether incor-
porating measured wide-area latency, prefix-cache locality,
and admission/queue state into a single per-request decision
improves user-visible latency, as measured by TTFT.
4.1. Setup
For these trials, we leverage three geographically distributed
regions, each consisting of an 8xA100 node: West Coast
(US), Germany (EU), and Israel (ME). Each instance in the
region runs one GORGO load balancer and one SGLang in-
ference server (§3), allowing it to make decisions leveraging
regional and cross-regional compute. Models were provided
by HuggingFace and held constant throughout experiments.
All results here were recovered from inference requests on
Mistral-7B-Instruct-v0.3.
4.2. Parameter Tuning
In order to measure the tp variable, we benchmark our
SGLang server on 87 Wildchat dataset samples and measure
the linear correlation between input tokens and TTFT.
Table 1. Linear regression analysis of TTFT relative to input token
count. The high R2 confirms a strong linear relationship.
METRIC VALUE
BASE LATENCY (INTERCEPT) 150.72 MS
PREFILL RATE (SLOPE) 0.0938 MS/TOK
MODEL FIT (R2 ) 0.9863
N 87
MODEL: y = 150.72 + 0.0938x
4
Confidential reviewer copy. This manuscript is under double-blind review by ICML 2026. Unauthorized sharing, redistribution, or disclosure is
strictly prohibited.
GORGO: Maximizing KV-Cache Reuse While Minimizing Network Latency in Cross-Region LLM Load Balancing
220
221
222
223
224
225
226
227
228
229
230
231
232
233
234
235
236
237
238
239
240
241
242
243
244
245
246
247
248
249
250
251
252
253
254
255
256
257
258
259
260
261
262
263
264
265
266
267
268
269
270
271
272
273
274
4.3. Workload
Our work leverages GuideLLM, WildChat, and a geoprox-
imal integration between the two to benchmark GORGO.
This encompasses a synthetic mixed request distribution
across regions with variable prompt lengths, including
shared system prompts to induce non-trivial prefix overlap.
In WildChat, requests follow multi-turn dialogue patterns
similar to prior public traces (Zhao et al., 2024; Zheng et al.,
2023), and are informed by common LLM inference work-
load patterns (Modal, 2026c). In GuideLLM, we are able
to vary workloads between four major types: concurrent,
poisson, throughput, and sweep based.
1. Concurrent workloads enable us to compute the max-
imal sustainable throughput of each load-balancing
policy, as it maintains a fixed number of parallel re-
quests in flight and consistently replaces completed
requests.
2. Poisson workloads model human-like behavior by
sending requests at a Poisson-distributed rate, mim-
icking bursty traffic patterns. Testing at a variable rate
enables us to measure improvements at tail latencies
(p99).
3. Throughput based request-firing enables us to zoom
in on load balancing behavior when there are greater
requests/second than what is available in the server
running, available, and waiting queues.
4. Sweep Based workloads progressively increase request
rate until latency degrades or errors appear.
Finally, we integrate WildChat with GuideLLM to bench-
mark our load balancing policies against simple geo-
proximal routing. Since GuideLLM only supports targeting
a single endpoint, we leverage the location metadata in-
cluded in each WildChat conversation. First, we preprocess
WildChat to create a lookup table mapping hash(prompt)
→ (lat, lon) and export prompts as a JSONL dataset. Then,
we run a local geo-routing tunnel that GuideLLM targets.
When the proxy receives a request, it hashes the prompt,
looks up the user’s location, and forwards the request to the
geographically nearest cluster node.
4.4. Baselines
We compare against four routing policies that represent
common design points:
1. Least-load routing naively routes each request to the
instance with the highest availability.
2. Prefix-similarity routing (PREFIXTRIE) routes to
maximize cached-prefix overlap, without accounting
for inter-cluster network latency and gpu availability.
3. GORGO accounts for local estimated TTFT (based on
queue sizes), point to point latency, KV cache overlap
to other peers, and GPU availability §3.
4. GORGO-proxy acts as a middleware node by serv-
ing a http-proxy that runs the GORGO policy with
centralized information about incoming requests and
instance-based metrics.
4.5. Results
On median TTFT (ms), GORGO-proxy (A. Table 5)
achieves a 2.5x reduction in median TTFT (A. Table 2)
compared to least-load, 2.5x faster than prefix-trie (A. Table
3), and 2.4x faster than GORGO. (A. Table 4). Figures
2 and 3 show the median and mean metrics, respectively,
across methods.
Figure 2. Median latency and throughput metrics across methods.
4.6. Protocol and instrumentation
Each run lasts 60 seconds and attempts to process 20 re-
quests/second across the three regions. We record per-
request timestamps at each routing stage (load-balancer
ingress/egress and inter-region forwarding) to compute
TTFT with microsecond-resolution timestamps. Our bench-
marking and measurement practices are informed by prior
guidance on LLM serving benchmarking (Modal, 2026b).