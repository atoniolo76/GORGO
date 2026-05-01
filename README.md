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

## Notes
- [*] Do some smoke tests on the hyperparameter ranges in tuning.py
  - 0.001 < x < 0.1 is a good range
- [*] Validate on-the-fly parameter tuning for proxy and adjust window size + hop rate to suitable values
  - Program runs successfully but no validation with GORGO policy yet
- [*] Test the TUI-based tuning utility for running tuning workloads and adjusting defaults + parameters manually