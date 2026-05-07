# GORGO

## Goal
Decrease TTFT from standard methods for LLM load balanacing (least-load, consistent-hasing, vtc-basic) by >2X using GORGO + tuning.

## Getting Started

Setup your Python environment:
```
uv venv
source .venv/bin/activate
uv sync
```

Note that for the following `run` commands, you can optionally specify a `--env=<your-modal-env>` following the `run` subcommand.

To launch a model replica:
```bash
REGION=us-east GPU_TYPE=H100 MODEL_ORG=Qwen MODEL_NAME=Qwen3.5-35B-A3B-FP8 modal deploy engine/modal_sglang.py
``` 

> Full list of regions [here](https://modal.com/docs/guide/region-selection).

To launch the proxy:
```bash
REGION=us-east modal run proxy/modal_proxy.py::proxy
```

> See the list of API routes [here](https://github.com/Arcadia-Research-Team/GORGO/blob/main/proxy/openapi.yaml)

To run an example workload on lmsys-chat-1m:
```bash
modal run proxy/workload.py --proxy-url https://... \
  --source hf --preset lmsys --num-requests 1000 --stream true
```

Or on Wildchat-4.8M with a specified data path:
```bash
modal run proxy/workload.py --proxy-url https://... \
  --source hf --data-path /datasets/datasets/allenai__WildChat-4.8M --stream true --num-requests 1000
```

> Note that the Modal volume names are specified in [app.py](https://github.com/Arcadia-Research-Team/GORGO/blob/main/app.py)


## Project Structure

- *proxy*: Request handling, workload streaming, and parameter tuning code that all runs on CPU instances in the same region.
- *engine*: LLM inference engine backend. Currently sglang is supported with DeepGEMM kernels built into the image and volume weight loading.
- *data_processing*: Scripts for reading from HF/local volumes and saving data + statistics + serialized radix trees to volumes.
- *policy*: Various load-balancing policies constructed from both Arcadia Research's GORGO paper and vLLM's AI-Brix model gateway.
- *utils*: Helpful util classes including RadixTrie, which is used for storing KV-cache state across sglang servers in the proxy.

## Running Policy Experiments

The experiment runner spawns isolated per-policy engine fleets + proxies across regions, replays a Mooncake trace, and saves results. See `experiment_runner/BENCHMARK_PLAN.md` for full methodology.

```bash
# 1. Build a trace (GLM-5.1 example; also supports --source lmsys / --source wildchat)
modal run --env=alessio-dev data_processing/build_mooncake_trace.py::main \
  --source glm5 --start-time 2026-04-01T00:30:00 --end-time 2026-04-01T01:00:00 \
  --num-requests 200000 --include-bodies --max-input-tokens 24000 --time-scale 1.0 \
  --output-path mooncake_traces/my_trace/with_bodies/glm5.jsonl

# 2. Launch (spec defines policies/regions/concurrency; manifest points at the trace)
modal run --detach --env=alessio-dev experiment_runner/policy_matrix_app.py::main \
  --base-spec-path specs/policy_matrix_abstract_night.json \
  --sweep-manifest-path specs/manifest_glm5_0030_0100.json \
  --experiment-id my_experiment_v1 --start-index 0 --top-k 1 \
  --output-dir /results/policy_matrix_sweep/my_experiment_v1

# 3. Monitor / early-stop
python scripts/experiment_status.py --experiment-id my_experiment_v1 --env alessio-dev
python scripts/stop_experiment.py --experiment-id my_experiment_v1  # saves partial results

# 4. Pull + analyze
modal volume get --env=alessio-dev --force GORGO-bench-results /workload_runs results/
python scripts/analyze_results.py --prefix <run_prefix> --label "My Run"
python scripts/plot_policy_summary.py --results-dir results --run-prefix <run_prefix> --out results/analysis/summary.png
```

## Tuning Parameters

The tuning script is now a lightweight client for the running proxy. It
submits a batch tuning request to the proxy's embedded `/tuning/*` API; the
proxy runs the workload locally against `http://127.0.0.1:8000` so the tuning
metric does not include client-to-proxy tunnel latency. Start the proxy with
the `GORGO-glm5-completions` and `GORGO-bench-results` volumes available (the
default `proxy/modal_proxy.py::proxy` deployment does this) and set the active
policy to `gorgo` before launching a run.

The tuning script will present a TUI allowing you to adjust default parameters
before starting the proxy-managed workload steps.
```bash
modal run -q proxy/tuning.py::tune_interactive --proxy-url https://your-proxy.modal.run
```

Alternatively, you can specify the specific parameters/settings via CLI args:
```bash
modal run proxy/tuning.py::tune_cli --proxy-url https://your-proxy.modal.run \
  --start-time 2026-04-01T12:00:00 \
  --num-requests 200 \
  --concurrency 32 \
  --metric output_throughput \
  --algorithm gaussian-es \
  --max-steps 16 \
  --seed 0 \
  --t-prefill-min 1e-4 --t-prefill-max 0.1 \
  --queued-tokens-weight-min 1e-3 --queued-tokens-weight-max 0.05
```