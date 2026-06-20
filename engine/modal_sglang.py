import json
import os
import subprocess
import time
import urllib.error
import urllib.request

import modal

from app import app, ENVIRONMENT_NAME

replicas = modal.Dict.from_name(
    "GORGO-replicas", create_if_missing=True, environment_name=ENVIRONMENT_NAME
)

sglang_image = (
    modal.Image.from_registry("lmsysorg/sglang:nightly-dev-cu13-20260411-0011d2ae")
    .run_commands("rm -rf /root/.cache/huggingface")
    .entrypoint(
        []  # silence chatty logs on container start
    )
)
# NOTE: the local source is added as the LAST image layer (see below, after
# compile_deep_gemm). compile_deep_gemm doesn't need our code, and baking the
# source in before it would make every code edit invalidate that ~20-min
# compile layer. Keeping the copy last means edits only rebuild the cheap
# final layer.

REGION = os.getenv("REGION", "us-east")
GPU_TYPE = os.getenv("GPU_TYPE", "H100")
MODEL_ORG = os.getenv("MODEL_ORG", "Qwen")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen3.5-35B-A3B-FP8")
HF_REPO_ID = f"{MODEL_ORG}/{MODEL_NAME}"
MODEL_REVISION = (  # pin revision id to avoid nasty surprises!
    "0b2752837483aa34b3db6e83e151b150c0e00e49"  # latest commit as of 2026-04-03, from release
)
N_GPUS = os.getenv("N_GPUS", 1)
GPU = f"{GPU_TYPE}:{N_GPUS}"
PORT = 8000
# Pinning ``--context-length`` lets ``proxy/workload.py`` auto-detect a
# safe input-token cap from ``/get_server_info`` (which would otherwise
# return ``context_length: null`` and force operators to remember
# ``--max-input-tokens N`` on every workload/tuning run). Default is
# Qwen3's native 32k window; bump via env var if the GPU has the KV
# headroom to support longer prompts.
CONTEXT_LENGTH = int(os.getenv("CONTEXT_LENGTH", 32768))
HF_CACHE_VOL = modal.Volume.from_name(
    f"{MODEL_NAME}-huggingface-cache", create_if_missing=True, environment_name=ENVIRONMENT_NAME
)
HF_CACHE_PATH = "/root/.cache/huggingface"
FULL_MODEL_NAME = f"{MODEL_ORG}/{MODEL_NAME}"
MODEL_PATH = f"{HF_CACHE_PATH}/{FULL_MODEL_NAME}"
# Default to scale-to-zero: ``modal deploy`` parks the function with no
# warm replicas, the first inbound /v1/chat/completions cold-starts a
# container and ``wait_ready`` does an in-process warmup before
# registering the tunnel URL in ``replicas[REGION]`` (the workload
# client also has its own client-side warmup phase). Once warm, the
# container survives ``SCALEDOWN_WINDOW_SECONDS`` of idle so back-to-back
# experiments avoid paying cold-start again. Set ``MIN_CONTAINERS`` to a
# positive integer if you want a permanently-warm pool instead.
MIN_CONTAINERS = os.getenv("MIN_CONTAINERS", 0)
SCALEDOWN_WINDOW_SECONDS = int(os.getenv("SCALEDOWN_WINDOW_SECONDS", 15 * 60))
WAIT_READY_TIMEOUT = os.getenv("WAIT_READY_TIMEOUT", 1200)

sglang_image = sglang_image.env(
    {
        "HF_HUB_CACHE": HF_CACHE_PATH,
        "HF_XET_HIGH_PERFORMANCE": "1",
        "SGLANG_ENABLE_JIT_DEEPGEMM": "1",
    }
)
sglang_image = sglang_image.run_commands(
    f"python3 -m sglang.compile_deep_gemm --model-path {FULL_MODEL_NAME} --revision {MODEL_REVISION} --tp {N_GPUS}",
    # Do not mount the DeepGEMM cache here; compiled kernels should be written
    # into the image layer. The HF cache remains a volume so model files are not
    # baked into the image.
    volumes={HF_CACHE_PATH: HF_CACHE_VOL},
    gpu=GPU,
)
# Local source goes LAST so editing app/engine only rebuilds this cheap copy
# layer, not the expensive compile_deep_gemm step above.
sglang_image = sglang_image.add_local_python_source("app", "engine", copy=True)


@app.function(
    image=sglang_image,
    timeout=24 * 60 * 60,
    region=REGION,
    gpu=GPU,
    min_containers=int(MIN_CONTAINERS),
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    volumes={HF_CACHE_PATH: HF_CACHE_VOL},
)
def model_endpoint(registry_key: str = REGION):
    import os

    os.environ["SGLANG_JIT_DEEPGEMM_FAST_WARMUP"] = "1"
    cmd = [
        "python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        HF_REPO_ID,
        "--revision",
        MODEL_REVISION,
        "--served-model-name",
        HF_REPO_ID,
        "--host",
        "0.0.0.0",
        "--port",
        f"{PORT}",
        "--tp",  # use all GPUs to split up tensor-parallel operations
        f"{N_GPUS}",
        "--cuda-graph-max-bs",  # only capture CUDA graphs for batch sizes we're likely to observe
        f"{10 * 2}",
        "--enable-metrics",  # expose metrics endpoints for telemetry
        "--decode-log-interval",  # how often to log during decoding, in tokens
        "100",
        "--mem-fraction",  # leave space for speculative model
        "0.8",
        "--context-length",  # surfaces in /get_server_info; drives workload pre-filter
        f"{CONTEXT_LENGTH}",
    ]

    # SGLang exposes OpenAI-compatible routes plus control endpoints; RadixAttention
    # KV state can be cleared with POST /flush_cache on this server (same port).
    with modal.forward(PORT) as tunnel:
        print(f"tunnel.url        = {tunnel.url}")
        print(f"tunnel.tls_socket = {tunnel.tls_socket}")
        process = subprocess.Popen(cmd)
        try:
            wait_ready(process)
            replicas[registry_key] = tunnel.url
            print(replicas[registry_key])
            process.wait()
        finally:
            if replicas.get(registry_key) == tunnel.url:
                replicas[registry_key] = ""
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def _check_process(process: subprocess.Popen):
    if (rc := process.poll()) is not None:
        raise subprocess.CalledProcessError(rc, cmd=process.args)


def wait_ready(process: subprocess.Popen, timeout: int = WAIT_READY_TIMEOUT):
    deadline = time.time() + timeout

    while time.time() < deadline:
        _check_process(process)
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/model_info"):
                break
        except urllib.error.URLError:
            time.sleep(2)
    else:
        raise TimeoutError(f"SGLang server not ready within {timeout} seconds")

    warmup_body = json.dumps(
        {
            "model": HF_REPO_ID,
            "messages": [{"role": "user", "content": "warmup"}],
            "max_tokens": 1,
        }
    ).encode()
    warmup_req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=warmup_body,
        headers={"Content-Type": "application/json"},
    )
    while time.time() < deadline:
        _check_process(process)
        try:
            with urllib.request.urlopen(warmup_req):
                return
        except (urllib.error.URLError, urllib.error.HTTPError):
            time.sleep(2)
    raise TimeoutError(f"SGLang server not ready within {timeout} seconds")


if __name__ == "__main__":
    model_endpoint.remote(REGION)
