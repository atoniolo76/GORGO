# Routing Policies

How each routing policy picks a replica for an incoming request. All policies receive the same per-replica metrics snapshot (scraped from SGLang's `/metrics` every ~10s) and the proxy's local radix trie of cached prefixes.

## Baselines

### `random`

Uniform random pick. No metrics, no state. The lower bound — any policy that loses to random is actively harmful.

### `least-request`

Pick the replica with the fewest in-flight requests.

```
score(replica) = max(sglangs_running_reqs, proxy_inflight_counter)
```

Uses `max` of SGLang's scrape-time count and the proxy's own real-time dispatch counter to avoid herding between scrapes (when the SGLang number is frozen, the proxy counter keeps incrementing). This is the standard NGINX `least_conn` / AWS ELB algorithm.

### `least-load`

Pick the replica with the lowest aggregate token-weighted load.

```
score(replica) = running_reqs + queue_reqs + used_kv_tokens + proxy_queued_tokens
```

Heavier signal than `least-request` — a replica running one 24k-token request scores higher than a replica running one 50-token request. Ignores cache locality entirely.

### `prefix-cache`

Pick the replica whose radix trie has the longest cached prefix match for the incoming prompt.

1. If running-request imbalance across replicas exceeds a threshold (default 8), fall back to `least-request` to prevent overload.
2. Otherwise, find replicas with the best prefix match.
3. Among those, pick the one with `running_reqs < mean + 2×std`.
4. If no match at all, fall back to `least-request`.

Maximizes KV-cache hit rate but can herd traffic onto a single cache-warm replica.

### `simple-session-affinity`

Hash the first 256 token IDs of the prompt and mod by replica count.

```
replica = replicas[hash(token_ids[:256]) % len(replicas)]
```

Same prompt prefix always hits the same replica. Maximizes intra-user cache reuse with zero overhead. Can't rebalance when sessions are unevenly distributed.

### `vtc`

Virtual Token Counter — fairness-weighted load balance.

```
score(replica) = combined_load + 10 × utilization
```

Optimizes for fair resource distribution across replicas rather than minimizing any latency metric. Included in some benchmarks for completeness but not in the current experiment set (fairness ≠ TTFT).

## GORGO

All GORGO variants use the same additive cost model and pick the replica with the minimum score:

```
score(replica) = rtt_weight × network_rtt
               + prefill_weight × (input_tokens − cached_prefix_tokens)
               + load_weight × (queued_tokens + used_kv_tokens)
```

| Term | What it captures | Source |
| --- | --- | --- |
| `rtt_weight × network_rtt` | Weighted round-trip to the replica | EWMA of a dedicated `GET /` probe, isolated from `/metrics` handler load |
| `prefill_weight × uncached_tokens` | Estimated prefill time for the tokens NOT already in the replica's KV cache | Radix trie lookup at routing time; `prefill_weight` is a per-token rate |
| `load_weight × load` | Cost of waiting behind other requests | Proxy-side in-flight token counter + SGLang's `num_used_tokens` |

The three variants differ only in how `rtt_weight`, `prefill_weight`, and `load_weight` are set:

### `gorgo-static`

Fixed hyperparameters (set in the spec, never changed during the run). Tests the cost model's value independent of any online tuning.

### `gorgo-autotune`

**Fit mode.** Every 16 new request samples, recomputes `prefill_weight` and `load_weight` per replica by taking the median of observed `(TTFT − RTT) / uncached_tokens` rates over the last 64 samples. Adapts to per-replica hardware/RTT differences automatically.

### `gorgo-hillclimb`

**Online-ES mode.** Gaussian (1+1)-Evolution Strategy with Rechenberg's 1/5 success rule. Directly minimizes `neg_p95_ttft` over the rolling 64-sample window by perturbing `rtt_weight`, `prefill_weight`, and `load_weight` in log-space. Doesn't try to estimate physical rates — just finds whatever values produce the best p95 TTFT.

The ES cycle:
1. Score the current window (p95 TTFT of last 64 requests)
2. Accept/reject the last candidate vs the incumbent
3. Adapt step size: if >20% of recent trials succeeded, widen search; if <20%, narrow
4. Propose: perturb each param by `exp(log(best) + sigma × N(0,1))`
5. Apply the proposal as the new defaults
6. Wait for 16 more samples, repeat

## Policy registry

All policies are registered in `policy/base.py::POLICY_REGISTRY`. The proxy looks up a policy by name via `POST /policy {"policy": "gorgo"}` and calls the corresponding `route_*` function on every incoming request.
