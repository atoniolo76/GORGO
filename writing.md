# Introduction

Modern LLM inference services proxy client requests to engine replicas spanning inter-continental regions.
Load-balancing policies must jointly account for factors including KV-cache locality, replica load, and variable network latency.
However, existing systems only evaluate a subset of these metrics in their cost model, leading to uneven concentrations of load and
KV-cache across replicas. We present GORGO, a proxy architecture that holistically factors for network latency, prefill cost, and queueing delay
using tunable parameters. Since Open-source chat datasets such as LMSYS-Chat-1M and WildChat-4.8M lack
long-context, high prefix-reuse data, we fabricate a synthetic dataset, ART-Chat-411K, from long-context production metadata.
On a tuning window from ART-Chat-411K, evolutionary strategies guide the GORGO policy's parameters to directly optimize P95 TTFT by X%
over simple session affinity. During held-out evaluation windows, we fix parameter values learned from tuning and
notice a downstream improvement of E2E latency by Y%, imputed to the balanced request concentration of GORGO's cost model.



In LLM serving systems, perceived latency to the user is dominated by the time-to-first-token (TTFT). On a single replica, TTFT is dominated by three costs: prefill time, round trip time (RTT) from client (proxy) to replica, and queueing delay behind in-flight requests. Prefix-caching, which is enabled in modern inference engines SGLang and vLLLM, eliminates the prefill cost of previous turns in a multi-turn conversation (cite sglang and vllm). As LLM context windows increase in length, the time saved by prefix caching 90% of a prompt with 100,000 tokens reduces the prefill cost to 10,000 tokens and decreases TTFT substantially.

Since modern LLM deployments proxy requests to inference engines across regions, the cost savings of prefix-caching depend on choosing a replica with the request session's prefix. Popular routing policies such as consistent hashing and prefix reuse aim to distribute load evenly while creating affinity between a session's requests and replica(s) to maximize KV-cache reuse (Skywalker paper citation, other citations for consistent hashing in llm load balancing). In compute-constrained regimes, bursty workloads can saturate a high-affinity replica, causing head-of-line (HOL) queueing delays from decode memory contention and negating cost savings from prefix-cache reuse. Routing policies that holistically evaluate all costs related to TTFT can maintain high prefix-cache reuse while minimizing the negative effects of load saturation and heterogenuous network latency. 

Existing load balancing policies such as least-load, session affinity, and prefix-reuse may account for replica load or KV-cache hit rate; however, no existing policies consider network latency in cross-region scenarios, which can range on the order of 10ms to 1s (cite figure 2 with the network latency plot). GORGO's routing policy accounts for all three TTFT costs, normalizes the units of measurement via tunable parameters, and jointly optimizes parameters through online tuning on real user workloads. To tune and stress-test different routing policies, in [insert below section] we compile a LLM traffic trace from real production requests with high prefix-reuse and long-context prompts. The trace follows Mooncake's FAST'25 format containing per-request timestamps, which can be linearly scaled to simulate variable saturation profiles. 

We benchmark GORGO on a series of user workloads from ART-Chat-411K, our sensitized production Mooncake trace, and sweep across variable time scales to effectively saturate replicas without simulating unrealisitic HOL queueing delay. Over existing load balancing policy baselines, GORGO jointly balances optimal TTFT with request concentration across replicas. Under the continuous batching paradigm, the ES-driven hillclimb tuner exploits warmer replicas close to the proxy and dramatically reduces TTFT at the cost of end-to-end (E2E) latency. We map the pareto frontier of request concentration across replicas, which explodes E2E latency for unbalanced distributions, and TTFT. Our contributions help contextualize the performance of LLM proxy routing policies in real-world user workloads, and [section x] lists a case-by-case scenario of when one would want to use aforementioned baseline policies, the online GORGO policy, and offline GORGO with held-out weights. Finally, we show how conditioning the GORGO cost model on proxy-recorded replica load mitigates subversion of TTFT via continuous batching and slashes request latency across a panoply of metrics.

# Cost Model
TTFT is known to consist of three different costs: network latency, prefill time, and queueing delay (BanaServe citation).
$ TTFT = T_network + T_queue + T_prefill $
In a cross-region deployment setting, the cost of TTFT per reach replica can be further defined as:
$ TTFT = T_network(i) + T_queue(i) + T_prefill(x_r, i) $
where x_r is the input sequence of tokens for a request r and i is an inference engine replica. 

The KV-Cache, which stores key-value pairs of input sequences, removes redundant prefill computation for user sequences sharing a prefix with previously computed sequences (cite original sglang/kv-cache paper). For a replica holding a set of cached token prefixes c_i, the prefill time of a request r changes with the the set difference of (x_r \set_difference c_i), or the set of all tokens in x_r not belonging in c_i. 
$ TTFT = T_network(i) + T_queue(i) + T_prefill(x_r \set_difference c_i) $

Theoretically, T_network and T_queue correspond with round trip time from a client to server and the
duration of request processing ahead of the incoming request r. However, modern inference engines supports batching requests continuously in order to minimize waiting for completion of previous request processing (Orca, sglang again?). While continous batching allows admission of a new request into the currently running batch, the batch size is still limited by the maximum number of concurrent requests. Without this knob, HBM thrashes with the KV-cache from all running requests. In a distributed system, the temporal delay of retrieving queueing metrics from an engine replica
make cost evaluation challenging. We represent the input to T_queue as the total number of tokens
of requests without a completion event on the client proxy. T_network is measured more trivially as the exponential weighted moving average of a ping's round trip time from client proxy to server. 
$ TTFT = T_network(RTT_i) + T_queue(i, \summation over j to n_r of x_j) + T_prefill(x_r \set_difference c_i) $

# GORGO Proxy Design
The GORGO proxy routes client requests to the engine replica with the minimum calculated cost from equation [insert equation from above], parameterized by weights $W_rtt$, $W_prefill$, and $W_queue$. These parameters weight the inputs $T_network$, $T_queue$, and T_prefill, which unfairly mixes the units of time and tokens, to normalize the correlated costs of latency, prefill cost, and queueing time on TTFT. The weight of T_prefill is fixed to 1 because GORGO proxy makes routing decisions based on relative replica cost: only the ratio of weights matter in this design.

$ TTFT = W_rtt * T_network(RTT_i) + W_queue * T_queue(i, \summation over j to n_r of x_j) + T_prefill(x_r \set_difference c_i) $

GORGO proxy uses a simple (1+1) evolutionary strategy to tune weights $W_rtt$ and $W_queue$ on the objective function, P95 TTFT.
Each parent weight \x_t,k is perturbed multiplicatively in log-space by a normal random variable \z_k times step size \sigma, and the new offspring weight x_k' is clamped
to values in the hyperparameter range $[lo_k, hi_k]$ where lo_k > 0 and hi_k > 0. 

$ \x_k' = clip(exp(ln(x_t,k) + \sigma_t * z_k), [lo_k, hi_k]) $

When \x' beats the parent weight \x_t on the objective metric, the incumbent weight \x_t+1 is updated to \x', and \sigma is adjusted to maintain Rechenburg's 1/5 success
rate of 1 accepted offspring for every 5 offspring. 

# Dataset
Existing LLM chatbot datasets lack two critical components for benchmarking cache-aware policies: (i) prefill-bound requests with long-context prompts and (ii) multi-turn workloads with high prefix-reuse between requests. For example, we measure the average request length and global prefix reuse of LMSYS-Chat-1M and WildChat-4.8M, two popular LLM datasets derived from public chatbot demos [link to figure/table]. WildChat-4.8M contains hashed IPs per request, allowing categorization of cross-user and intra-user reuse while LMSYS-Chat-1M lacks user identification. Additional results from benchmarking GORGO on WildChat-4.8M can be found in [link to appendix]. Cache-aware policies provide no measurable gains over simple baseline policies like random when routing requests with a length of <3,000 tokens.

ART-Chat-2.5M is a long-context, multi-turn dataset synthetically generated from a week-long metadata trace of production inference traffic with the same prefix-reuse structure as the original workload. We release a replay-ready trace in the Mooncake FAST'25 format, which contains per-request timestamps, request metadata, and synthetically generated chat completion data [cite mooncake paper]. By stored request timestamps, one can linearly scale the time between requests to control replica load.

# Experimental Setup
## Baseline Policies
We benchmark the GORGO policy and compare performance of both online and static modes to the below baselines. All SGLang metrics are scraped every 30 seconds from the engine's prometheus endpoint `/metrics`.

1. least-load minimizes the sum of proxy-tracked queued requests with SGLang metrics `num_running_reqs`, `num_queue_reqs`, and `num_used_tokens`. Queued requests to the proxy are defined as recently dispatched requests without a token response and `num_used_tokens` are the currently occupied per-token KV slots.
2. least-request chooses the replica with the fewest in-flight requests from the proxy.
3. prefix-cache matches the request's prefix to the replica with the highest prefix-cache overlap, tracked on the proxy-side by a prefix trie of dispatched requests [cite aibrix].
4. simple-session affinity hashes the first 256 token from a request and routes to the replica with that prefix hash.

## Proxy and Engine Configuration
Each policy runs the GORGO proxy on a small CPU worker in us-ashburn and controls a dedicated SGLang inference engine in each the following regions: us-ashburn, eu-frankfurt, and ap-seoul. The engines contain two L40S GPUs each and serve the Qwen3.5-35B-A3B model in FP8 format [link qwen3.5 paper citation]. All policies are benchmarked on the same workload in parallel to rule out any variance in network conditions. The round trip time between the us-ashburn proxy and engines during the tuning window is plotted in [insert ref to timeseries figure]. Due to the Qwen model's limited context length of 32,768 tokens, we filter out any requests that contain >24,000 tokens to leave adequate KV headroom. In SGLang, we set `max_concurrent_requests` to 64 and `max_output_tokens` to 128 to limit unnecessary decode while simulating adequate load on the replica. All workloads run alongside the proxy, dispatching requests to a local chat completion endpoint.

## Tuning and Evaluation Windows
We pick three 30 minute windows with high-user diversity from the ART-Chat-2.5M trace: Apr 5th 16:15–16:45, Apr 6 15:05–15:35, and Apr 7 19:45–20:15. Statistics on each of these windows is found in [link table with the winow stats]. We assign Apr 5th as the window where we tune GORGO's weights online to minimize the P95 TTFT of a rolling 128 request window with hop size 32. $W_rtt$ and $W_queue$ are each initialized to 0.5 and 0.1 and restricted to ranges [0.05, 2.0] and [0.05, 0.5]. These values are hand-picked from the paradigm of continuous batching in SGLang, where incoming requests can be scheduled into the current batch, affecting TTFT less significantly than a fixed floor of network latency between regions. The Apr 6-7 windows fix weight values GORGO learned on the Apr 5 tuning window. Due to the greater number of requests in Apr 6-7, we linearly scale the time between requests to control replica saturation [link characteristics table of these dinwos].

# Results
Table X reports TTFT, E2E latency, and inter-token latency (ITL) for policies in all three windows. While the GORGO policy's weights are updated to minimize P95 TTFT during tuning, the held-out, fixed weight evaluation shows generalization of learned values across days with GORGO improving P95 TTFT by X% and E2E latency by Y% over the next-best policy, sesssion-affinity. GORGO slightly underperforms baseline policies on the tuning window because the evolutionary strategy actively explores the space of parameters and tests worse weights than the learned solution, which converges after X samples [link to the hyperparameter convergence window]

## Related work: 
Prefix-cache aware policies