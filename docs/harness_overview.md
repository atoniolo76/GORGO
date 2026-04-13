# Routing Harness — Architectural Overview

## Purpose

Profile and compare LLM inference routing strategies — especially KV-cache
aware routing — through a flexible, deterministic experiment harness. The
harness simulates request traces against a model of a cluster (pods with
GPUs, KV caches, queues, network) under a pluggable routing policy, and
emits machine-readable results + plot-ready tables for a GRD report.

Everything is **implementation-only** in this iteration: no benchmarks are
run, no datasets are downloaded, no networks are measured. Code is built
to be testable and reproducible so that runs can be kicked off later.

## Repo state prior to harness

- `models/qwen/model.py` — Modal + Shadow scaffold for a Qwen endpoint on
  an A10G (unrelated to the harness; left untouched).
- `pyproject.toml` — only declares `modal` as a dependency. We extend it.
- No existing tests, CI, or config system. We introduce `pytest` as the
  test base and a YAML config system — aligned with the minimal-dep rule
  (only `PyYAML` is added as a new runtime dep; `pytest` is dev-only).

> The task prompt says "assume the repo already has some testing
> structure; reuse and extend it rather than replacing it." There is no
> such structure. We add the minimum viable one.

## Layering

```
configs/        YAML configs and sweep definitions
src/routing_harness/
  core.py           Typed dataclasses: Request, Response, Pod, Decision
  policy.py         RoutingPolicy protocol + registry
  cluster.py        ClusterState + Pod behavior
  kv_cache.py       KVCacheState: prefix trie, eviction, reuse accounting
  cost_model.py     CostModel: compute/network/scheduling cost estimation
  workload/
    trace.py        WorkloadTrace (iterator of Requests with arrivals)
    lmsys.py        lmsys-chat-1m adapter (stubbed download)
    synthetic.py    Synthetic trace generator (Poisson / Zipf prefixes)
  simulator/
    engine.py       Discrete-event simulator: routes, queues, serves
    metrics.py      Metrics collector (latency percentiles, hit rate, …)
  policies/
    random.py, least_request.py, throughput.py, prefix_cache.py,
    least_busy_time.py, least_kv_cache.py, least_latency.py,
    prefix_cache_preble.py, vtc_basic.py, pd.py, session_affinity.py
  config/
    schema.py       Dataclass-backed config loaders + validation
    sweep.py        Cartesian sweep expansion
  cli.py            `routing-harness run|sweep|list-policies`

tests/
  contract/         Policy-interface contract tests (apply to every policy)
  unit/             Per-module unit tests (cluster, kv_cache, cost_model, …)
  e2e/              Toy-cluster end-to-end runs (tiny deterministic traces)
  fixtures/         Small synthetic traces + topology YAML

research/
  reports/routing-comparison.md   GRD-ready report with result placeholders
  figures/                        (empty; plotting added later)
  grd.yaml                        GRD project config
```

## Execution flow (once runs are permitted)

1. Load YAML config → validate → expand sweeps.
2. Build `ClusterState` from topology; build `WorkloadTrace` from dataset
   adapter or synthetic generator.
3. For each (policy × params × seed):
   - Instantiate policy with its policy-specific config.
   - Run `Simulator.run(trace, policy, cluster, cost_model)`.
   - Collect metrics; write result bundle (config snapshot + metrics)
     under `results/<run_id>/`.
4. `results/index.json` accumulates runs for report aggregation.

## Key design choices

- **Discrete-event simulator, not a real server.** All latency is
  estimated from the cost model; this keeps the harness deterministic,
  fast, and dependency-free. Real-server measurement hooks live in
  `cost_model.py::InstrumentedCostModel` as placeholders.
- **Policy plugin registry.** Policies register themselves by id; adding
  a new one means one file + one contract test pass, no edits to the
  runner.
- **Cost model is separate from policy.** A policy returns a
  `Decision(pod_id, rationale)`; the cost model owns how long it will
  take. This lets us swap cost models (e.g., Preble-style vs Mooncake-
  style transport costs) without touching policies.
- **Fabric contention under fluid fair-share.** Concurrent KV
  transfers share the inter-pod fabric: the engine keeps a heap of
  in-flight transfers and feeds the cost model the sum of overlapping
  bytes, so an individual transfer's time becomes
  `rtt + Σbytes_in_flight / bandwidth`. A lone transfer (nothing else
  on the fabric) reduces to the uncontended `rtt + bytes/B` formula.
- **KV-cache state is a prefix trie with eviction and transport events.**
  Reuse accounting distinguishes *available reuse* (some pod has the
  prefix) from *captured reuse* (the pod the request was routed to has
  the prefix), which is central to answering the core research question.
- **No silent defaults.** Every config section has an explicit required
  schema; omitting a field is an error, not a default.

## What this harness intentionally does NOT do (yet)

- Real GPU measurement or Modal deploy integration.
- Real dataset download (stubbed; user provides local path).
- Real plotting (tables are emitted; figures are a follow-up).
- Claim parity with Mooncake / SGLang / Dynamo / AIBrix / Ant AI Gateway —
  policies are *baselines inspired by* those systems with documented
  assumptions, not reimplementations.
