from dataclasses import dataclass
import json
import random
import modal
import asyncio
import httpx
import tiktoken

from app import app, replicas
from utils.radix_trie import RadixNode, RadixTrie


SUPPORTED_POLICIES = {"random", "power_of_two", "gorgo"}
DEFAULT_POLICY = "random"
DEFAULT_GORGO_HYPERPARAMETERS = {"t_prefill": 1.0, "queued_tokens_weight": 1.0}
ALLOWED_HYPERPARAM_KEYS = set(DEFAULT_GORGO_HYPERPARAMETERS)
METRICS_REFRESH_INTERVAL_SECONDS = 1.0
METRICS_FETCH_TIMEOUT_SECONDS = 2.0
# SGLang may wait until idle; allow a generous read window for POST /flush_cache.
FLUSH_UPSTREAM_TIMEOUT_SECONDS = 120.0

# Headers that are connection-local and must not be forwarded verbatim when
# we proxy upstream responses back to the client. Re-emitting these would
# either confuse uvicorn (e.g. it wants to manage ``transfer-encoding`` and
# ``content-length`` itself when we stream chunks) or leak upstream-specific
# keep-alive semantics to the client. We ask the upstream not to compress
# via ``accept-encoding: identity`` so ``content-encoding`` is also safe to
# drop unconditionally.
_HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "transfer-encoding",
        "upgrade",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "content-encoding",
        "content-length",
    }
)

_ENCODER = None


def _get_encoder():
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = tiktoken.encoding_for_model("gpt-4o")
    return _ENCODER


def _message_text(content) -> str:
    """Extract the text payload from an OpenAI chat-completions ``content`` field.

    ``content`` is either a string, or a list of content blocks
    (``{"type": "text", "text": "..."}``, ``{"type": "image_url", ...}``, ...).
    Non-text blocks are dropped -- they don't contribute to the input token
    count we care about for routing decisions.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text = block.get("text", "") or ""
                    if text:
                        parts.append(text)
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return ""


def tokenize_input(messages: list[dict]) -> list[int]:
    """Tokenize an OpenAI chat-completions ``messages`` payload into a flat
    list of token ids.

    Each message's text content is encoded individually via tiktoken's
    batched encoder and the per-message ids are concatenated in order. This
    isn't an exact match for the server-side count (we ignore per-message
    formatting tokens and tool-call arguments) but it's directionally
    correct and cheap enough to run on every request for both routing
    decisions (``len(ids)``) and prefix-sharing tracking (the ids
    themselves).
    """
    if not isinstance(messages, list) or not messages:
        return []
    enc = _get_encoder()
    texts: list[str] = []
    for msg in messages:
        if isinstance(msg, dict):
            text = _message_text(msg.get("content"))
            if text:
                texts.append(text)
        elif isinstance(msg, str):
            texts.append(msg)
    if not texts:
        return []
    encoded = enc.encode_batch(texts, num_threads=4, disallowed_special=())
    out: list[int] = []
    for ids in encoded:
        out.extend(ids)
    return out


@app.function(
    image=modal.Image.debian_slim()
    .pip_install("httpx[http2]", "uvicorn", "tiktoken")
    .add_local_python_source("app", "utils"),
    timeout=(24 * 60 * 60),
)
def proxy():
    import json
    import time

    import httpx
    import uvicorn

    local_replicas = {}
    for k, v in replicas.items():
        local_replicas[k] = v

    replica_urls = [v for v in local_replicas.values()]

    # Routing state. Kept in a dict so the asgi_app closure can mutate it in
    # place from the /policy handler; uvicorn is single-process / single-loop
    # so we don't need a lock (all writes happen on the asyncio thread).
    #
    # ``upstream_client`` is a long-lived httpx.AsyncClient shared across all
    # handlers (metrics refresh + chat completions). Keeping it alive means
    # TCP + TLS handshakes to each replica happen once and keepalive
    # connections are reused for every subsequent request, which is a big
    # TTFT win on cold proxies. Created in lifespan.startup so it binds to
    # the right event loop; closed in lifespan.shutdown.
    state = {
        "policy": DEFAULT_POLICY,
        "hyperparameters": dict(DEFAULT_GORGO_HYPERPARAMETERS),
        "upstream_client": None,
    }
    endpoints_queued_tokens: dict[str, int] = {url: 0 for url in replica_urls}

    # Live radix trie of every prompt we've forwarded. Each node along a
    # sequence's insertion path is tagged with the replica URL that received
    # that prompt, so the trie doubles as an approximate "which replica
    # currently has this prefix cached in KV" index. Updated inline in the
    # /v1/chat/completions handler after the upstream request is dispatched.
    #
    # Note: currently unbounded; a long-running proxy will accumulate state
    # until the container is recycled. That's fine for the evaluation setup;
    # a production deployment would need LRU/TTL eviction.
    radix_trie = RadixTrie()

    # Default timeout used by the shared client. ``read=None`` is critical for
    # chat completions: httpx treats ``read`` as "seconds between bytes", so a
    # generation that thinks for a while before emitting its first token would
    # otherwise trip the timeout. Metrics calls pass their own shorter
    # override via the per-request ``timeout=`` kwarg.
    DEFAULT_UPSTREAM_TIMEOUT = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0)
    # Keep enough warm connections around to saturate concurrent load without
    # re-handshaking. ``keepalive_expiry=None`` means "never expire idle
    # connections"; Modal will eventually tear down the container anyway.
    UPSTREAM_LIMITS = httpx.Limits(
        max_connections=200, max_keepalive_connections=100, keepalive_expiry=None
    )

    def _new_upstream_client() -> httpx.AsyncClient:
        # ``http2=True`` negotiates HTTP/2 via ALPN on HTTPS replicas; if the
        # upstream (SGLang-on-uvicorn) only speaks HTTP/1.1, httpx transparently
        # falls back. Either way we keep the keep-alive pool benefit.
        return httpx.AsyncClient(
            http2=True,
            timeout=DEFAULT_UPSTREAM_TIMEOUT,
            limits=UPSTREAM_LIMITS,
        )

    # Live mirror of each replica's /metrics output. A background task refreshes
    # this every ``METRICS_REFRESH_INTERVAL_SECONDS``; policy functions read
    # snapshots of it per request instead of fetching synchronously.
    live_metrics: dict[str, replica_state] = {}
    metrics_meta = {
        "last_refresh_monotonic": 0.0,
        "last_refresh_errors": {},  # url -> str
    }

    async def _refresh_one(client: httpx.AsyncClient, url: str) -> None:
        t0 = time.monotonic()
        try:
            # Override the client's generous default timeout -- if a replica
            # can't answer /metrics in 2s it's effectively down for routing
            # purposes and we'd rather fall back than block the refresh loop.
            resp = await client.get(f"{url}/metrics", timeout=METRICS_FETCH_TIMEOUT_SECONDS)
            resp.raise_for_status()
        except Exception as e:
            metrics_meta["last_refresh_errors"][url] = repr(e)
            return
        latency = time.monotonic() - t0
        parsed = {
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
        live_metrics[url] = replica_state(
            num_running_reqs=int(parsed.get("sglang:num_running_reqs", 0)),
            num_queue_reqs=int(parsed.get("sglang:num_queue_reqs", 0)),
            num_used_tokens=int(parsed.get("sglang:num_used_tokens", 0)),
            latency=latency,
        )
        metrics_meta["last_refresh_errors"].pop(url, None)

    async def _refresh_metrics_once(client: httpx.AsyncClient) -> None:
        if not replica_urls:
            return
        await asyncio.gather(
            *[_refresh_one(client, url) for url in replica_urls],
            return_exceptions=True,
        )
        metrics_meta["last_refresh_monotonic"] = time.monotonic()

    async def _flush_upstream_replica(client: httpx.AsyncClient, base_url: str) -> tuple[str, dict]:
        url = f"{base_url.rstrip('/')}/flush_cache"
        try:
            resp = await client.post(
                url,
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=FLUSH_UPSTREAM_TIMEOUT_SECONDS,
                    write=30.0,
                    pool=10.0,
                ),
            )
            out = {"ok": resp.is_success, "status_code": resp.status_code}
            if not resp.is_success:
                body = (resp.text or "")[:512]
                if body:
                    out["body_preview"] = body
            return base_url, out
        except Exception as e:
            return base_url, {"ok": False, "error": repr(e)}

    async def _metrics_refresh_loop() -> None:
        try:
            while True:
                client = state["upstream_client"]
                if client is not None:
                    try:
                        await _refresh_metrics_once(client)
                    except Exception as e:
                        print(f"[proxy] metrics refresh iteration failed: {e}")
                await asyncio.sleep(METRICS_REFRESH_INTERVAL_SECONDS)
        except asyncio.CancelledError:
            pass

    async def _read_body(receive) -> bytes:
        chunks: list[bytes] = []
        while True:
            msg = await receive()
            if msg["type"] != "http.request":
                continue
            chunks.append(msg.get("body", b"") or b"")
            if not msg.get("more_body"):
                break
        return b"".join(chunks)

    async def _send_json(send, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": body})

    async def _select_endpoint(token_ids: list[int]) -> str:
        policy = state["policy"]
        request_tokens = len(token_ids)
        if policy == "random" or len(replica_urls) <= 1:
            return random.choice(replica_urls)

        # Snapshot live_metrics so mid-request refreshes can't mutate our view.
        # Skip replicas we've never successfully scraped.
        metrics_snapshot = {url: live_metrics[url] for url in replica_urls if url in live_metrics}
        if len(metrics_snapshot) < len(replica_urls):
            missing = [u for u in replica_urls if u not in metrics_snapshot]
            print(
                f"[proxy] live metrics missing for {len(missing)} replica(s); "
                f"falling back to random for this request"
            )
            return random.choice(replica_urls)

        if policy == "power_of_two":
            chosen = await ai_brix_power_of_two(endpoints_queued_tokens, metrics_snapshot)
            return chosen[0] if isinstance(chosen, list) else chosen
        if policy == "gorgo":
            # One trie walk per request that answers "how many leading
            # tokens of this prompt are already cached on each replica?".
            # Skipped for empty prompts -- the batched walk would just
            # return zeros but we'd rather avoid the set-allocs.
            if token_ids:
                endpoints_cached_tokens = radix_trie.cached_prefix_lengths(token_ids, replica_urls)
            else:
                endpoints_cached_tokens = {url: 0 for url in replica_urls}
            chosen = await gorgo_multi_objective(
                request_tokens=request_tokens,
                endpoints_queued_tokens=endpoints_queued_tokens,
                replica_metrics=metrics_snapshot,
                hyperparameters=state["hyperparameters"],
                endpoints_cached_tokens=endpoints_cached_tokens,
            )
            return chosen[0] if isinstance(chosen, list) else chosen
        return random.choice(replica_urls)

    async def asgi_app(scope, receive, send):
        if scope["type"] == "lifespan":
            while True:
                msg = await receive()
                if msg["type"] == "lifespan.startup":
                    state["upstream_client"] = _new_upstream_client()
                    # Prime with a synchronous first pass so the first request
                    # that lands while the loop is still running doesn't hit
                    # empty live_metrics and fall back to random.
                    try:
                        await _refresh_metrics_once(state["upstream_client"])
                    except Exception as e:
                        print(f"[proxy] initial metrics refresh failed: {e}")
                    state["_metrics_task"] = asyncio.create_task(_metrics_refresh_loop())
                    print(
                        f"[proxy] metrics refresh loop started "
                        f"(interval={METRICS_REFRESH_INTERVAL_SECONDS}s, "
                        f"{len(replica_urls)} replicas); "
                        f"upstream client: http2=True, "
                        f"max_connections={UPSTREAM_LIMITS.max_connections}, "
                        f"max_keepalive_connections={UPSTREAM_LIMITS.max_keepalive_connections}"
                    )
                    await send({"type": "lifespan.startup.complete"})
                elif msg["type"] == "lifespan.shutdown":
                    task = state.get("_metrics_task")
                    if task is not None:
                        task.cancel()
                        try:
                            await task
                        except (asyncio.CancelledError, Exception):
                            pass
                    client = state.get("upstream_client")
                    if client is not None:
                        try:
                            await client.aclose()
                        except Exception as e:
                            print(f"[proxy] upstream client close failed: {e}")
                        state["upstream_client"] = None
                    await send({"type": "lifespan.shutdown.complete"})
                    return

        if scope["type"] != "http":
            return

        path = scope["path"]
        method = scope["method"]

        if path == "/policy":
            if method == "GET":
                await _send_json(
                    send,
                    200,
                    {
                        "policy": state["policy"],
                        "supported": sorted(SUPPORTED_POLICIES),
                        "hyperparameters": state["hyperparameters"],
                    },
                )
                return
            if method == "POST":
                body = await _read_body(receive)
                try:
                    data = json.loads(body.decode()) if body else {}
                except json.JSONDecodeError:
                    await _send_json(send, 400, {"error": "invalid JSON body"})
                    return
                name = data.get("policy") or data.get("name")
                if name not in SUPPORTED_POLICIES:
                    await _send_json(
                        send,
                        400,
                        {
                            "error": f"unknown policy {name!r}",
                            "supported": sorted(SUPPORTED_POLICIES),
                        },
                    )
                    return
                state["policy"] = name
                print(f"[proxy] routing policy set to {name!r}")
                await _send_json(
                    send,
                    200,
                    {"policy": state["policy"], "hyperparameters": state["hyperparameters"]},
                )
                return
            await _send_json(send, 405, {"error": "method not allowed"})
            return

        if path == "/replicas":
            if method == "GET":
                await _send_json(
                    send,
                    200,
                    {"replicas": list(replica_urls), "count": len(replica_urls)},
                )
                return
            if method == "POST":
                body = await _read_body(receive)
                try:
                    data = json.loads(body.decode()) if body else {}
                except json.JSONDecodeError:
                    await _send_json(send, 400, {"error": "invalid JSON body"})
                    return
                if isinstance(data, list):
                    raw = data
                elif isinstance(data, dict):
                    raw = data.get("replicas") or data.get("endpoints")
                else:
                    raw = None
                if not isinstance(raw, list) or not all(isinstance(u, str) for u in raw):
                    await _send_json(
                        send,
                        400,
                        {
                            "error": (
                                "body must be a JSON array of endpoint URLs "
                                'or an object like {"replicas": [...]}'
                            )
                        },
                    )
                    return

                seen: set[str] = set()
                normalized: list[str] = []
                invalid: list[str] = []
                for u in raw:
                    u = u.strip().rstrip("/")
                    if not u:
                        continue
                    if not (u.startswith("http://") or u.startswith("https://")):
                        invalid.append(u)
                        continue
                    if u not in seen:
                        seen.add(u)
                        normalized.append(u)
                if invalid:
                    await _send_json(
                        send,
                        400,
                        {
                            "error": "all endpoints must start with http:// or https://",
                            "invalid": invalid,
                        },
                    )
                    return

                old = set(replica_urls)
                new = set(normalized)
                added = sorted(new - old)
                removed = sorted(old - new)

                # Mutate the same list/dict objects so the background refresh
                # loop and _select_endpoint closure see the update.
                replica_urls.clear()
                replica_urls.extend(normalized)
                for u in added:
                    endpoints_queued_tokens[u] = 0
                for u in removed:
                    endpoints_queued_tokens.pop(u, None)
                    live_metrics.pop(u, None)
                    metrics_meta["last_refresh_errors"].pop(u, None)

                print(
                    f"[proxy] replicas updated: +{len(added)} -{len(removed)} "
                    f"(total={len(replica_urls)})"
                )
                await _send_json(
                    send,
                    200,
                    {
                        "replicas": list(replica_urls),
                        "count": len(replica_urls),
                        "added": added,
                        "removed": removed,
                    },
                )
                return
            await _send_json(send, 405, {"error": "method not allowed"})
            return

        if path == "/trie" and method == "GET":
            # Summary stats only -- the full trie is too large to serialize
            # on every request. ``coverage`` counts how many nodes are tagged
            # with each replica URL, which is a useful sanity check that
            # routing is producing the shape of prefix-sharing we expect.
            coverage: dict[str, int] = {url: 0 for url in replica_urls}
            stack = [radix_trie.root]
            tagged_nodes = 0
            while stack:
                node = stack.pop()
                if node.replica_endpoints:
                    tagged_nodes += 1
                    for url in node.replica_endpoints:
                        coverage[url] = coverage.get(url, 0) + 1
                stack.extend(node.children.values())
            await _send_json(
                send,
                200,
                {
                    "num_sequences": radix_trie.num_sequences,
                    "total_tokens_inserted": radix_trie.total_tokens_inserted,
                    "unique_token_count": radix_trie.unique_token_count(),
                    "node_count": radix_trie.node_count(),
                    "tagged_node_count": tagged_nodes,
                    "replica_coverage": coverage,
                },
            )
            return

        if path == "/replica_metrics" and method == "GET":
            now = time.monotonic()
            last = metrics_meta["last_refresh_monotonic"]
            await _send_json(
                send,
                200,
                {
                    "refresh_interval_seconds": METRICS_REFRESH_INTERVAL_SECONDS,
                    "last_refresh_age_seconds": (now - last) if last else None,
                    "errors": metrics_meta["last_refresh_errors"],
                    "metrics": {
                        url: {
                            "num_running_reqs": m.num_running_reqs,
                            "num_queue_reqs": m.num_queue_reqs,
                            "num_used_tokens": m.num_used_tokens,
                            "latency_seconds": m.latency,
                        }
                        for url, m in live_metrics.items()
                    },
                    "endpoints_queued_tokens": endpoints_queued_tokens,
                },
            )
            return

        if path == "/hyperparameters":
            if method == "GET":
                await _send_json(
                    send,
                    200,
                    {
                        "hyperparameters": state["hyperparameters"],
                        "allowed_keys": sorted(ALLOWED_HYPERPARAM_KEYS),
                        "defaults": DEFAULT_GORGO_HYPERPARAMETERS,
                    },
                )
                return
            if method in ("POST", "PATCH", "PUT"):
                body = await _read_body(receive)
                try:
                    data = json.loads(body.decode()) if body else {}
                except json.JSONDecodeError:
                    await _send_json(send, 400, {"error": "invalid JSON body"})
                    return
                if not isinstance(data, dict):
                    await _send_json(
                        send, 400, {"error": "body must be a JSON object of hyperparameters"}
                    )
                    return
                unknown = sorted(k for k in data if k not in ALLOWED_HYPERPARAM_KEYS)
                if unknown:
                    await _send_json(
                        send,
                        400,
                        {
                            "error": f"unknown hyperparameter(s): {unknown}",
                            "allowed_keys": sorted(ALLOWED_HYPERPARAM_KEYS),
                        },
                    )
                    return
                try:
                    updates = {k: float(v) for k, v in data.items()}
                except (TypeError, ValueError):
                    await _send_json(send, 400, {"error": "hyperparameter values must be numeric"})
                    return
                if method == "PUT":
                    merged = dict(DEFAULT_GORGO_HYPERPARAMETERS)
                    merged.update(updates)
                    state["hyperparameters"] = merged
                else:
                    state["hyperparameters"].update(updates)
                print(f"[proxy] hyperparameters updated: {state['hyperparameters']}")
                await _send_json(send, 200, {"hyperparameters": state["hyperparameters"]})
                return
            await _send_json(send, 405, {"error": "method not allowed"})
            return

        if path == "/flush":
            if method != "POST":
                await _send_json(send, 405, {"error": "method not allowed"})
                return
            radix_trie.clear()
            client = state["upstream_client"]
            replica_results: dict[str, dict] = {}
            if client is None:
                for u in replica_urls:
                    replica_results[u] = {
                        "ok": False,
                        "error": "upstream client not yet initialized",
                    }
            elif replica_urls:
                pairs = await asyncio.gather(
                    *[_flush_upstream_replica(client, u) for u in replica_urls],
                )
                replica_results = dict(pairs)
            await _send_json(
                send,
                200,
                {
                    "radix_trie_cleared": True,
                    "replicas": replica_results,
                },
            )
            return

        if path == "/v1/chat/completions" and method == "POST":
            body = await _read_body(receive)
            try:
                data = json.loads(body.decode()) if body else {}
            except json.JSONDecodeError:
                await _send_json(send, 400, {"error": "invalid JSON body"})
                return
            if not isinstance(data, dict):
                await _send_json(send, 400, {"error": "body must be a JSON object"})
                return

            messages = data.get("messages", [])
            if not isinstance(messages, list):
                await _send_json(send, 400, {"error": "'messages' must be a list"})
                return

            token_ids = tokenize_input(messages)
            request_tokens = len(token_ids)

            try:
                target = await _select_endpoint(token_ids)
            except Exception as e:
                print(f"[proxy] policy {state['policy']!r} failed ({e}); falling back to random")
                target = random.choice(replica_urls)

            endpoints_queued_tokens[target] = (
                endpoints_queued_tokens.get(target, 0) + request_tokens
            )
            client = state["upstream_client"]
            if client is None:
                # Shouldn't happen outside of a race between startup and the
                # first inbound request; be defensive anyway.
                await _send_json(send, 503, {"error": "upstream client not yet initialized"})
                if target in endpoints_queued_tokens:
                    endpoints_queued_tokens[target] = max(
                        0, endpoints_queued_tokens[target] - request_tokens
                    )
                return

            # Stream the upstream response straight through to the client so
            # the first SSE event / first response chunk arrives as soon as
            # SGLang emits it (critical for TTFT measurements).
            #
            # We forward the raw request bytes (``content=body``) instead of
            # re-serializing ``data`` -- we already parsed once to tokenize
            # and pick a route, no reason to pay json.dumps() again. The
            # ``application/json`` content-type is set explicitly because
            # httpx won't infer it from raw bytes.
            #
            # ``accept-encoding: identity`` tells the upstream not to
            # compress, so we don't have to juggle ``content-encoding`` on
            # the way out to the client.
            upstream_headers = {
                "accept-encoding": "identity",
                "content-type": "application/json",
            }
            headers_sent = False
            try:
                async with client.stream(
                    "POST",
                    f"{target}/v1/chat/completions",
                    content=body,
                    headers=upstream_headers,
                ) as upstream:
                    # At this point the request body has been sent upstream
                    # and the response headers have come back, so the replica
                    # has ingested the prompt and (at minimum) started
                    # prefill. Record the prefix on the trie so concurrent
                    # requests arriving during our streaming phase can see
                    # that ``target`` now caches this prefix.
                    #
                    # Only tag successful dispatches -- a 4xx/5xx likely
                    # means the replica never got far enough to cache
                    # anything meaningful.
                    if token_ids and 200 <= upstream.status_code < 300:
                        try:
                            radix_trie.insert(token_ids, endpoint=target)
                        except Exception as e:
                            # Trie bookkeeping must never break forwarding.
                            print(f"[proxy] radix trie insert failed: {e}")

                    response_headers = [
                        (k.lower().encode(), v.encode())
                        for k, v in upstream.headers.items()
                        if k.lower() not in _HOP_BY_HOP_HEADERS
                    ]
                    await send(
                        {
                            "type": "http.response.start",
                            "status": upstream.status_code,
                            "headers": response_headers,
                        }
                    )
                    headers_sent = True
                    async for chunk in upstream.aiter_raw():
                        if not chunk:
                            continue
                        await send(
                            {
                                "type": "http.response.body",
                                "body": chunk,
                                "more_body": True,
                            }
                        )
                    await send({"type": "http.response.body", "body": b"", "more_body": False})
            except (httpx.ConnectError, httpx.ConnectTimeout) as e:
                if not headers_sent:
                    await _send_json(send, 502, {"error": f"upstream replica unreachable: {e}"})
            except httpx.HTTPError as e:
                if headers_sent:
                    # Already committed to a status; best we can do is close
                    # the body cleanly so the client sees a truncated stream.
                    print(f"[proxy] upstream stream error mid-response: {e}")
                    try:
                        await send({"type": "http.response.body", "body": b"", "more_body": False})
                    except Exception:
                        pass
                else:
                    await _send_json(send, 502, {"error": f"upstream stream error: {e}"})
            finally:
                # Only decrement if the replica is still registered; if it was
                # removed mid-request via /replicas POST, don't leak its key
                # back into the queue-tokens dict.
                if target in endpoints_queued_tokens:
                    endpoints_queued_tokens[target] = max(
                        0, endpoints_queued_tokens[target] - request_tokens
                    )
            return

        await send(
            {
                "type": "http.response.start",
                "status": 404,
                "headers": [(b"content-type", b"text/plain")],
            }
        )
        await send({"type": "http.response.body", "body": b"Not found"})

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


async def ai_brix_power_of_two(
    endpoints_queued_tokens: dict[str, int],
    replica_metrics: dict[str, replica_state],
) -> list[str]:
    """Pick the less-loaded of two randomly sampled replicas.

    Reads ``num_used_tokens`` from the caller-provided ``replica_metrics``
    snapshot (kept live by the proxy's background refresh loop); does no I/O
    of its own.
    """
    candidates = [u for u in endpoints_queued_tokens if u in replica_metrics]
    if len(candidates) < 2:
        return candidates[:1] if candidates else []
    a, b = random.sample(candidates, 2)
    load_a = replica_metrics[a].num_used_tokens + endpoints_queued_tokens.get(a, 0)
    load_b = replica_metrics[b].num_used_tokens + endpoints_queued_tokens.get(b, 0)
    return [b] if load_a > load_b else [a]


async def gorgo_multi_objective(
    request_tokens: int,
    endpoints_queued_tokens: dict[str, int],
    replica_metrics: dict[str, replica_state],
    hyperparameters: dict[str, float],
    endpoints_cached_tokens: dict[str, int],
) -> str:
    """Score each replica and pick the lowest.

    ``endpoints_cached_tokens[url]`` is the number of leading tokens of the
    current request that the proxy's live radix trie believes ``url`` has
    already cached in its KV (i.e. tokens that won't need to be re-prefilled
    there). We subtract it from the prefill cost so a replica that already
    has most of the prompt cached is strongly preferred -- the whole point
    of the Gorgo routing policy.
    """
    keys = list(endpoints_queued_tokens.keys())
    if set(keys) != set(replica_metrics.keys()):
        raise ValueError("Endpoints and replica metrics keys do not match")

    scores: dict[str, float] = {}
    for key in keys:
        network_latency = replica_metrics[key].latency
        cached = endpoints_cached_tokens.get(key, 0)
        effective_prefill_tokens = max(0, request_tokens - cached)
        prefill_cost = effective_prefill_tokens * hyperparameters["t_prefill"]
        queue_cost = (
            endpoints_queued_tokens[key] + replica_metrics[key].num_used_tokens
        ) * hyperparameters["queued_tokens_weight"]
        scores[key] = network_latency + prefill_cost + queue_cost

    return min(scores, key=scores.get)
