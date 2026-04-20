from dataclasses import dataclass
import json
import random
import modal
import asyncio
import httpx
import tiktoken

from app import app, replicas


@app.function(
    image=modal.Image.debian_slim().pip_install("httpx", "uvicorn").add_local_python_source("app")
)
def proxy():
    import json

    import httpx
    import uvicorn

    local_replicas = {}
    for k, v in replicas.items():
        local_replicas[k] = v

    replica_urls = [v for v in local_replicas.values()]

    async def asgi_app(scope, receive, send):
        if scope["type"] == "http":
            req = await receive()
            if scope["path"] == "/v1/chat/completions" and req["type"] == "http.request":
                body = req.get("body", b"") or b""
                data = json.loads(body.decode())
                try:
                    async with httpx.AsyncClient() as client:
                        resp = await client.post(
                            f"{random.choice(replica_urls)}/v1/chat/completions",
                            json=data,
                            timeout=60,
                        )
                except httpx.ConnectError:
                    err = json.dumps({"error": "upstream replica unreachable"}).encode()
                    await send(
                        {
                            "type": "http.response.start",
                            "status": 502,
                            "headers": [(b"content-type", b"application/json")],
                        }
                    )
                    await send(
                        {
                            "type": "http.response.body",
                            "body": err,
                        }
                    )
                    return
                response_headers = [
                    (k.lower().encode(), v.encode()) for k, v in resp.headers.items()
                ]
                await send(
                    {
                        "type": "http.response.start",
                        "status": resp.status_code,
                        "headers": response_headers,
                    }
                )
                await send(
                    {
                        "type": "http.response.body",
                        "body": resp.content,
                    }
                )
                return
            # fallback: 404
            await send(
                {
                    "type": "http.response.start",
                    "status": 404,
                    "headers": [(b"content-type", b"text/plain")],
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": b"Not found",
                }
            )

    with modal.forward(8000) as tunnel:
        print(f"proxy listening at {tunnel.url}")
        # TODO: replace uvicorn with a faster reverse-proxy (e.g. nginx, envoy, or Rust-based)
        uvicorn.run(asgi_app, host="0.0.0.0", port=8000)


@dataclass
class replica_state:
    num_running_reqs: int
    num_queue_reqs: int
    num_used_tokens: int
    latency: float


async def fetch_replica_metrics(endpoints: list[str]) -> dict[str, replica_state]:
    import time

    async def _fetch(client: httpx.AsyncClient, endpoint: str) -> tuple[str, replica_state]:
        t0 = time.monotonic()
        resp = await client.get(f"{endpoint}/metrics")
        latency = time.monotonic() - t0

        metrics = {
            parts[0].split("{")[0]: float(parts[1])
            for line in resp.text.splitlines()
            if not line.startswith("#")
            and len(parts := line.rsplit(" ", 1)) == 2
            and parts[1]
            .replace(".", "", 1)
            .replace("e+", "", 1)
            .replace("e-", "", 1)
            .lstrip("-")
            .isdigit()
        }

        return endpoint, replica_state(
            num_running_reqs=int(metrics.get("sglang:num_running_reqs", 0)),
            num_queue_reqs=int(metrics.get("sglang:num_queue_reqs", 0)),
            num_used_tokens=int(metrics.get("sglang:num_used_tokens", 0)),
            latency=latency,
        )

    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*[_fetch(client, ep) for ep in endpoints])
        return dict(results)


def tokenize_input(messages: list[dict]) -> int:
    return len(tiktoken.encoding_for_model("gpt-4o").encode(json.dumps(messages)))


async def ai_brix_power_of_two(endpoints_queued_tokens: list[str, str]) -> list[str]:
    random_endpoints = random.sample(endpoints_queued_tokens.keys(), 2)

    metrics = await fetch_replica_metrics(random_endpoints)

    if (
        metrics[random_endpoints[0]].num_used_tokens + endpoints_queued_tokens[random_endpoints[0]]
        > metrics[random_endpoints[1]].num_used_tokens
        + endpoints_queued_tokens[random_endpoints[1]]
    ):
        return [random_endpoints[1]]
    else:
        return [random_endpoints[0]]


async def gorgo_multi_objective(
    request_tokens: int,
    endpoints_queued_tokens: list[str, str],
    replica_metrics: dict[str, replica_state],
    hyperparameters: dict[str, float],
) -> list[str]:
    keys = endpoints_queued_tokens.keys()

    if set(keys) != set(replica_metrics.keys()):
        raise ValueError("Endpoints and replica metrics keys do not match")

    scores = {}
    for key in keys:
        network_latency = replica_metrics[key].latency
        num_used_tokens = request_tokens * hyperparameters["t_prefill"]
        num_queued_tokens = (
            endpoints_queued_tokens[key] + replica_metrics[key].num_used_tokens
        ) * hyperparameters["queued_tokens_weight"]

        score = network_latency + num_used_tokens + num_queued_tokens
        scores[key] = score

    return min(scores, key=scores.get)
