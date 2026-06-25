# Introduction

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

# GORG Routing Policy
