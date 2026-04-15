# GORGO

## Production deployment (Modal + SGLang)

```bash
REGION=us MODEL_ORG="Qwen" MODEL_NAME="Qwen3.5-35B-A3B-FP8" MIN_CONTAINERS=1 modal run --env=alessio-dev engine/modal_sglang.py
```

Full list of regions [here](https://modal.com/docs/guide/region-selection).

The production proxy currently randomizes routing across replicas
(`proxy/modal_proxy.py`). The routing harness below exists to study
what should replace that.

## Modal / Shadow scaffold (legacy)

```
modal run --env=alessio-dev models/qwen/model.py
```

## LLM Inference Routing Harness

A dependency-light, test-first experiment harness for comparing LLM
inference routing strategies — especially KV-cache-aware routing —
under a common workload, cluster, and cost model. Simulates; does not
execute real inference.

Architecture: [`docs/harness_overview.md`](docs/harness_overview.md)

### Usage (once runs are permitted)

```bash
pip install -e ".[dev]"

# list registered policies
routing-harness list-policies

# single run
routing-harness run --config configs/example_run.yaml

# PD-disaggregated run
routing-harness run --config configs/example_pd_run.yaml

# parameter sweep (11 policies × 4 QPS × 3 Zipf × 3 seeds)
routing-harness sweep --config configs/example_sweep.yaml

# tests
pytest
```

Results land under `results/<run_id>/` with a content-addressed id
derived from a hash of the full config snapshot.

### Extend

- Add a policy: [`docs/how_to_add_a_policy.md`](docs/how_to_add_a_policy.md)
- Add a dataset: [`docs/how_to_add_a_dataset.md`](docs/how_to_add_a_dataset.md)
- Config reference: [`docs/config_reference.md`](docs/config_reference.md)

### Research

GRD scaffold lives in [`research/`](research/):

- [`research/grd.yaml`](research/grd.yaml) — project descriptor
- [`research/reports/routing-comparison.md`](research/reports/routing-comparison.md)
  — draft report with placeholders for quantitative results and an
  explicit "gaps to be filled by running" section.

### Policies shipped

| id | one-liner |
|------------------------|--------------------------------------------------|
| `random`               | uniform over prefill-capable pods (baseline) |
| `least-request`        | min active+queued |
| `throughput`           | max EWMA tokens/s |
| `prefix-cache`         | longest prefix match, fallback to LRQ |
| `least-busy-time`      | min EWMA(latency) × load |
| `least-kv-cache`       | max free KV bytes |
| `least-latency`        | min EWMA latency |
| `prefix-cache-preble`  | match − load, with hotspot-threshold deflection |
| `vtc-basic`            | fairness-aware tenant counter + LBT |
| `pd`                   | disaggregated: prefix-match for prefill, LBT for decode |
| `session-affinity`     | sticky by session_id with TTL |
