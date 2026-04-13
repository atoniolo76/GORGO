import json
import os
import subprocess
import time
import urllib.error
import urllib.request

import modal

app = modal.App(name="GORGO")

sglang_image = modal.Image.from_registry(
    "lmsysorg/sglang:nightly-dev-cu13-20260411-0011d2ae"
).run_commands("rm -rf /root/.cache/huggingface").entrypoint(
    []  # silence chatty logs on container start
)

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
HF_CACHE_VOL = modal.Volume.from_name(f"{MODEL_NAME}-huggingface-cache", create_if_missing=True)
HF_CACHE_PATH = "/root/.cache/huggingface"
FULL_MODEL_NAME = f"{MODEL_ORG}/{MODEL_NAME}"
MODEL_PATH = f"{HF_CACHE_PATH}/{FULL_MODEL_NAME}"
MIN_CONTAINERS = os.getenv("MIN_CONTAINERS", 2)
WAIT_READY_TIMEOUT = os.getenv("WAIT_READY_TIMEOUT", 1200)
DG_CACHE_VOL = modal.Volume.from_name("deepgemm-cache", create_if_missing=True)
DG_CACHE_PATH = "/root/.cache/deepgemm"

sglang_image = sglang_image.env({"HF_HUB_CACHE": HF_CACHE_PATH, "HF_XET_HIGH_PERFORMANCE": "1", "SGLANG_ENABLE_JIT_DEEPGEMM": "1"})
sglang_image = sglang_image.run_commands(
    f"python3 -m sglang.compile_deep_gemm --model-path {FULL_MODEL_NAME} --revision {MODEL_REVISION} --tp {N_GPUS}",
    volumes={DG_CACHE_PATH: DG_CACHE_VOL, HF_CACHE_PATH: HF_CACHE_VOL},
    gpu=GPU,
)

@app.function(
    image=sglang_image,
    timeout=3600,
    region=REGION,
    gpu=GPU,
    min_containers=int(MIN_CONTAINERS),
    volumes={HF_CACHE_PATH: HF_CACHE_VOL},
)
def model_endpoint():
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
    ]

    with modal.forward(PORT) as tunnel:
        print(f"tunnel.url        = {tunnel.url}")
        print(f"tunnel.tls_socket = {tunnel.tls_socket}")
        process = subprocess.Popen(cmd)
        wait_ready(process)
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

    warmup_body = json.dumps({
        "model": HF_REPO_ID,
        "messages": [{"role": "user", "content": "warmup"}],
        "max_tokens": 1,
    }).encode()
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
    model_endpoint.remote()
