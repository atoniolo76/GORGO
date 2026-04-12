# Config Reference

Every experiment-defining field is required. Harness-internal knobs
(under `engine.*`) have documented defaults.

## Top-level keys (RunConfig)

| Key          | Type       | Description |
|--------------|------------|-------------|
| `name`       | string     | Human-readable run name. |
| `policy`     | object     | See [Policy](#policy). |
| `topology`   | object     | See [Topology](#topology). |
| `compute`    | object     | See [Compute](#compute). |
| `network`    | object     | See [Network](#network). |
| `scheduler`  | object     | See [Scheduler](#scheduler). |
| `engine`     | object     | Optional. See [Engine](#engine). |
| `workload`   | object     | See [Workload](#workload). |
| `seeds`      | list[int]  | Seeds to run (≥1). Each seed is a separate run. |
| `output_dir` | string     | Directory for result bundles. Created if missing. |

## Policy

```yaml
policy:
  policy_id: "prefix-cache"
  params: {block_size: 16}
```

`policy_id` must be one of `routing-harness list-policies`. `params` are
forwarded to the policy constructor.

## Topology

```yaml
topology:
  pods:
    - pod_id: "p0"
      role: "both"                 # "prefill" | "decode" | "both"
      gpu_count: 1
      kv_cache_bytes: 4294967296
      max_concurrent_prefill: 4
      max_concurrent_decode: 16
      peer_ids: []                 # optional, used by PD
```

## Compute

```yaml
compute:
  prefill_ms_per_token: 0.08       # calibrate per model × GPU
  decode_ms_per_token: 6.0
  prefill_overhead_ms: 5.0
  decode_overhead_ms: 2.0
```

## Network

```yaml
network:
  client_rtt_ms: 5.0
  inter_pod_rtt_ms: 0.2
  inter_pod_bandwidth_gbps: 100.0
  kv_bytes_per_token: 131072       # model-specific
  serialization_overhead_ms: 0.5
```

## Scheduler

```yaml
scheduler:
  base_routing_ms: 0.2
  per_pod_consideration_us: 5.0
```

## Engine (optional)

Defaults documented; override only with a reason.

```yaml
engine:
  kv_ewma_alpha: 0.2
  block_size: 16
  initial_warm_latency_ms: 5.0
```

## Workload

Two supported `kind`s. Schema is `kind`-specific under `params`.

### Synthetic

```yaml
workload:
  kind: "synthetic"
  params:
    n_requests: 2000
    arrival_rate_qps: 8.0
    n_prefix_families: 64
    zipf_s: 1.1
    prompt_len_min: 256
    prompt_len_max: 2048
    max_output_tokens: 128
    n_sessions: 200
```

### lmsys-chat-1m

```yaml
workload:
  kind: "lmsys"
  params:
    local_path: "/data/lmsys-chat-1m.jsonl"
    max_conversations: 1000
    language_filter: ["en"]
    min_turns: 1
    max_turns: 16
    arrival_rate_qps: 8.0
    tokens_per_char: 0.25
    max_output_tokens: 256
```

## Sweep config

```yaml
name: "my-sweep"
base: "example_run.yaml"
grid:
  "policy.policy_id": ["random", "prefix-cache"]
  "workload.params.arrival_rate_qps": [4.0, 8.0, 16.0]
```

Dotted paths navigate the base config. Cartesian product expands.

## Reproducibility

- `run_id` = `blake2b(config_snapshot)` — content-addressed.
- Full config is snapshotted into `results/<run_id>/config.json`.
- Synthetic workloads are deterministic per seed.
- lmsys workloads are deterministic per (local file contents, seed).
- No network access after config parse.
