import random
import modal

from app import app, replicas

@app.function(image=modal.Image.debian_slim().pip_install("httpx", "uvicorn").add_local_python_source("app"))
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
                    await send({
                        "type": "http.response.start",
                        "status": 502,
                        "headers": [(b"content-type", b"application/json")],
                    })
                    await send({
                        "type": "http.response.body",
                        "body": err,
                    })
                    return
                response_headers = [
                    (k.lower().encode(), v.encode())
                    for k, v in resp.headers.items()
                ]
                await send({
                    "type": "http.response.start",
                    "status": resp.status_code,
                    "headers": response_headers,
                })
                await send({
                    "type": "http.response.body",
                    "body": resp.content,
                })
                return
            # fallback: 404
            await send({
                "type": "http.response.start",
                "status": 404,
                "headers": [(b"content-type", b"text/plain")],
            })
            await send({
                "type": "http.response.body",
                "body": b"Not found",
            })

    with modal.forward(8000) as tunnel:
        print(f"proxy listening at {tunnel.url}")
        # TODO: replace uvicorn with a faster reverse-proxy (e.g. nginx, envoy, or Rust-based)
        uvicorn.run(asgi_app, host="0.0.0.0", port=8000)
