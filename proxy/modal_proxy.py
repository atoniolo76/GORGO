import asyncio
import json
import os
import random
from collections import deque

import httpx
import modal
import tiktoken

from app import app, replicas
from proxy.measure import (
    NS_PER_S,
    consume_sse_stream,
    recommend_hyperparameters,
    recommend_hyperparameters_per_target,
    summarize_samples,
)
from policy import (
    POLICY_REGISTRY,
    ROUTING_POLICIES,
    PolicyDef,
    ReplicaSnapshot,
    RouteContext,
    normalize_policy,
)
from policy.gorgo import (
    ALLOWED_HYPERPARAM_KEYS,
    DEFAULT_GORGO_HYPERPARAMETERS,
    make_default_store,
    merge_update,
    prune_per_target,
    validate_update,
)
from utils.radix_trie import RadixTrie

# Match engine/modal_sglang.py so the proxy lands in the same datacenter as
# its replicas (and as the workload generator in proxy/workload.py).
REGION = os.getenv("REGION", "us-east")

DEFAULT_POLICY = "random"
METRICS_REFRESH_INTERVAL_SECONDS = 1.0
METRICS_FETCH_TIMEOUT_SECONDS = 2.0
# SGLang may wait until idle; allow a generous read window for POST /flush_cache.
FLUSH_UPSTREAM_TIMEOUT_SECONDS = 120.0

# On-the-fly tuning. The proxy keeps a bounded ring buffer of per-request
# samples (TTFT / total / token counts / per-token rates); when the
# auto-tuner is enabled (``POST /tune``) it sliding-window-recomputes
# ``gorgo`` hyperparameters from this buffer via
# ``proxy.measure.recommend_hyperparameters`` -- the same primitive
# ``proxy/calibrate.py`` uses for one-shot calibration.
#
# The auto-tuner is a stateful toggle, *not* a one-shot endpoint: once
# enabled it keeps recomputing every ``hop_size`` new samples until
# explicitly disabled (``POST /tune {"enabled": false}``). Subsequent
# POSTs while it's running atomically reconfigure the live tuner.
MAX_REQUEST_SAMPLES = 1000
DEFAULT_TUNE_WINDOW_SIZE = 100
DEFAULT_TUNE_HOP_SIZE = 50  # recompute every N new samples once warm
DEFAULT_TUNE_APPLY = True  # default to actually mutating hyperparameters

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


def _parse_metrics_text(text: str) -> dict[str, float]:
    """Parse a Prometheus exposition snippet into ``{metric_name: value}``,
    dropping label suffixes and skipping non-numeric or commented lines."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        value = parts[1]
        if (
            not value.replace(".", "", 1)
            .replace("e+", "", 1)
            .replace("e-", "", 1)
            .lstrip("-")
            .isdigit()
        ):
            continue
        out[parts[0].split("{")[0]] = float(value)
    return out


@app.function(
    image=modal.Image.debian_slim()
    .pip_install("httpx[http2]", "uvicorn", "tiktoken")
    .add_local_python_source("app", "proxy", "policy", "utils"),
    region=REGION,
    timeout=(24 * 60 * 60),
)
def proxy():
    import time

    import httpx
    import uvicorn

    replica_urls: list[str] = list(replicas.values())

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
    state: dict = {
        "policy": DEFAULT_POLICY,
        # Structured GORGO hyperparameter store (see policy/gorgo.py).
        # ``defaults`` applies to every replica; ``per_target`` keys
        # specific replicas to override individual scalars. The auto-
        # tuner writes per-target overrides automatically once enough
        # live samples accumulate per replica; offline tools (calibrate,
        # tuning.py) typically write only ``defaults``.
        "hyperparameters": make_default_store(),
        "upstream_client": None,
        "metrics_task": None,
        # Total samples appended over the proxy lifetime. Doesn't
        # saturate at ``MAX_REQUEST_SAMPLES`` (unlike ``len(samples)``)
        # so it can be diffed across snapshots to count arrivals.
        "total_samples_appended": 0,
        # Live auto-tuner config. Reconfigured atomically by ``POST
        # /tune``; consumed by ``_record_request_sample`` after every
        # successful sample append. ``enabled=False`` means the
        # background recompute is dormant and the rest of the dict is
        # frozen at its last value.
        "auto_tune": {
            "enabled": False,
            "window_size": DEFAULT_TUNE_WINDOW_SIZE,
            "hop_size": DEFAULT_TUNE_HOP_SIZE,
            "apply": DEFAULT_TUNE_APPLY,
            # Counter zeroed on enable / on apply, incremented per
            # sample. Triggers a recompute when it crosses ``hop_size``.
            "samples_since_last_apply": 0,
            # Diagnostics surfaced by GET /tune.
            "applied_count": 0,
            "last_applied_at_monotonic": None,
            "last_recommendation": None,
            "enabled_at_monotonic": None,
        },
    }
    endpoints_queued_tokens: dict[str, int] = {url: 0 for url in replica_urls}

    # Bounded ring buffer of per-request samples produced by the
    # SSE-tee in ``_handle_chat_completions``. Each entry has the same
    # shape as ``proxy.measure.measure_chat_completion`` returns, plus
    # ``target`` (chosen replica) and ``recorded_at_monotonic`` so /tune
    # consumers can do their own time-windowing if desired.
    samples: deque[dict] = deque(maxlen=MAX_REQUEST_SAMPLES)

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
    default_upstream_timeout = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0)
    # Keep enough warm connections around to saturate concurrent load without
    # re-handshaking. ``keepalive_expiry=None`` means "never expire idle
    # connections"; Modal will eventually tear down the container anyway.
    upstream_limits = httpx.Limits(
        max_connections=200, max_keepalive_connections=100, keepalive_expiry=None
    )

    # Live mirror of each replica's /metrics output. A background task refreshes
    # this every ``METRICS_REFRESH_INTERVAL_SECONDS``; policy functions read
    # snapshots of it per request instead of fetching synchronously.
    live_metrics: dict[str, ReplicaSnapshot] = {}
    metrics_meta: dict = {
        "last_refresh_monotonic": 0.0,
        "last_refresh_errors": {},  # url -> str
    }

    # ---------- ASGI helpers ----------

    async def _read_json_body(receive) -> dict | list:
        """Drain the ASGI request body and parse it as JSON.

        Combines body assembly with JSON decode so handler call sites stay
        a single ``try / except`` instead of two. Returns ``{}`` for an
        empty body. Raises ``json.JSONDecodeError`` on malformed input
        (handlers convert that to a 400).
        """
        chunks: list[bytes] = []
        while True:
            msg = await receive()
            if msg["type"] != "http.request":
                continue
            chunks.append(msg.get("body", b"") or b"")
            if not msg.get("more_body"):
                break
        body = b"".join(chunks)
        return json.loads(body.decode()) if body else {}

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

    # ---------- Upstream client + metrics refresh ----------

    def _new_upstream_client() -> httpx.AsyncClient:
        # ``http2=True`` negotiates HTTP/2 via ALPN on HTTPS replicas; if the
        # upstream (SGLang-on-uvicorn) only speaks HTTP/1.1, httpx transparently
        # falls back. Either way we keep the keep-alive pool benefit.
        return httpx.AsyncClient(
            http2=True,
            timeout=default_upstream_timeout,
            limits=upstream_limits,
        )

    async def _refresh_one(client: httpx.AsyncClient, url: str) -> None:
        """Scrape one replica's /metrics into ``live_metrics[url]``."""
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
        parsed = _parse_metrics_text(resp.text)
        live_metrics[url] = ReplicaSnapshot(
            num_running_reqs=int(parsed.get("sglang:num_running_reqs", 0)),
            num_queue_reqs=int(parsed.get("sglang:num_queue_reqs", 0)),
            num_used_tokens=int(parsed.get("sglang:num_used_tokens", 0)),
            latency=latency,
            gen_throughput=float(parsed.get("sglang:gen_throughput", 0.0)),
            utilization=float(parsed.get("sglang:utilization", 0.0)),
        )
        metrics_meta["last_refresh_errors"].pop(url, None)

    async def _refresh_all(client: httpx.AsyncClient | None) -> None:
        """One pass: refresh every registered replica in parallel."""
        if client is None or not replica_urls:
            return
        await asyncio.gather(
            *[_refresh_one(client, url) for url in replica_urls],
            return_exceptions=True,
        )
        metrics_meta["last_refresh_monotonic"] = time.monotonic()

    async def _metrics_refresh_loop() -> None:
        """Background task: refresh every ``METRICS_REFRESH_INTERVAL_SECONDS``.
        Cancelled in lifespan.shutdown. Per-iteration exceptions are
        logged so a single transient failure doesn't kill the loop."""
        try:
            while True:
                try:
                    await _refresh_all(state["upstream_client"])
                except Exception as e:
                    print(f"[proxy] metrics refresh iteration failed: {e}")
                await asyncio.sleep(METRICS_REFRESH_INTERVAL_SECONDS)
        except asyncio.CancelledError:
            pass

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

    # ---------- Routing ----------

    def _select_endpoint(token_ids: list[int]) -> str:
        """Pick an upstream URL using the policy registry in :mod:`policy`.

        For policies that don't need ``/metrics`` data (random, pd*,
        simple-session-affinity) we skip the snapshot entirely so the proxy
        can keep routing during a metrics-refresh outage. For metrics-using
        policies we filter ``live_metrics`` to currently-registered
        replicas; if any replica has no snapshot yet (cold start, /metrics
        timeout) we fall back to random rather than passing a partial view
        to the policy.
        """
        if not replica_urls:
            raise ValueError("no replicas configured")
        if len(replica_urls) == 1:
            return replica_urls[0]

        pdef: PolicyDef = POLICY_REGISTRY[normalize_policy(state["policy"])]

        if pdef.needs_metrics:
            # Filter to replicas with a live snapshot. Both this code and the
            # background refresh run on the same asyncio thread so there's
            # no race; the dict comprehension just drops missing entries.
            metrics = {u: live_metrics[u] for u in replica_urls if u in live_metrics}
            if len(metrics) < len(replica_urls):
                missing = len(replica_urls) - len(metrics)
                print(
                    f"[proxy] live metrics missing for {missing} replica(s); "
                    f"falling back to random for this request"
                )
                return random.choice(replica_urls)
        else:
            metrics = {}

        return pdef.fn(
            RouteContext(
                replica_urls=replica_urls,
                metrics=metrics,
                endpoints_queued_tokens=endpoints_queued_tokens,
                radix_trie=radix_trie,
                token_ids=token_ids,
                request_tokens=len(token_ids),
                hyperparameters=state["hyperparameters"],
            )
        )

    # ---------- JSON route handlers ----------
    #
    # Each handler returns ``(status, payload)``; the dispatcher reads
    # the body, calls the handler, and serializes the response. Body
    # parsing / 400-on-invalid-json is handled by ``_dispatch_json`` so
    # individual handlers can focus on validation and side effects.
    #
    # All handlers are nested closures because they read / mutate state
    # captured in this ``proxy()`` invocation -- ``replicas``,
    # ``live_metrics``, ``samples``, ``radix_trie``,
    # ``endpoints_queued_tokens``, ``state``. Lifting them to module
    # scope would require threading a ``ProxyContext`` through every
    # call. The naming convention matches the other request-shaped
    # closures already living here (``_handle_chat_completions``,
    # ``_handle_lifespan``).

    async def _handle_get_policy(_data) -> tuple[int, dict]:
        return 200, {
            "policy": state["policy"],
            "supported": sorted(ROUTING_POLICIES),
            "hyperparameters": state["hyperparameters"],
        }

    async def _handle_post_policy(data) -> tuple[int, dict]:
        raw = (data or {}).get("policy") or (data or {}).get("name")
        if not isinstance(raw, str):
            return 400, {"error": "body must include string policy or name"}
        name = normalize_policy(raw)
        if name not in ROUTING_POLICIES:
            return 400, {
                "error": f"unknown policy {raw!r}",
                "supported": sorted(ROUTING_POLICIES),
            }
        state["policy"] = name
        print(f"[proxy] routing policy set to {name!r}")
        return 200, {"policy": state["policy"], "hyperparameters": state["hyperparameters"]}

    async def _handle_get_replicas(_data) -> tuple[int, dict]:
        return 200, {"replicas": list(replica_urls), "count": len(replica_urls)}

    async def _handle_post_replicas(data) -> tuple[int, dict]:
        if isinstance(data, list):
            raw = data
        elif isinstance(data, dict):
            raw = data.get("replicas") or data.get("endpoints")
        else:
            raw = None
        if not isinstance(raw, list) or not all(isinstance(u, str) for u in raw):
            return 400, {
                "error": (
                    "body must be a JSON array of endpoint URLs "
                    'or an object like {"replicas": [...]}'
                )
            }

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
            return 400, {
                "error": "all endpoints must start with http:// or https://",
                "invalid": invalid,
            }

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
        # Drop any per-target hyperparameter overrides that targeted
        # a now-removed replica. Stale overrides are harmless for
        # routing (effective_hyperparameters won't look them up) but
        # they'd accumulate over a long-running proxy and confuse
        # ``GET /hyperparameters`` consumers.
        prune_per_target(state["hyperparameters"], set(replica_urls))

        print(
            f"[proxy] replicas updated: +{len(added)} -{len(removed)} (total={len(replica_urls)})"
        )
        return 200, {
            "replicas": list(replica_urls),
            "count": len(replica_urls),
            "added": added,
            "removed": removed,
        }

    async def _handle_get_trie(_data) -> tuple[int, dict]:
        # Summary stats only -- the full trie is too large to serialize on
        # every request. ``coverage`` counts how many nodes are tagged with
        # each replica URL: a useful sanity check that routing is producing
        # the prefix-sharing shape we expect.
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
        return 200, {
            "num_sequences": radix_trie.num_sequences,
            "total_tokens_inserted": radix_trie.total_tokens_inserted,
            "unique_token_count": radix_trie.unique_token_count(),
            "node_count": radix_trie.node_count(),
            "tagged_node_count": tagged_nodes,
            "replica_coverage": coverage,
        }

    async def _handle_get_replica_metrics(_data) -> tuple[int, dict]:
        now = time.monotonic()
        last = metrics_meta["last_refresh_monotonic"]
        return 200, {
            "refresh_interval_seconds": METRICS_REFRESH_INTERVAL_SECONDS,
            "last_refresh_age_seconds": (now - last) if last else None,
            "errors": metrics_meta["last_refresh_errors"],
            "metrics": {
                url: {
                    "num_running_reqs": m.num_running_reqs,
                    "num_queue_reqs": m.num_queue_reqs,
                    "num_used_tokens": m.num_used_tokens,
                    "latency_seconds": m.latency,
                    "gen_throughput": m.gen_throughput,
                    "utilization": m.utilization,
                }
                for url, m in live_metrics.items()
            },
            "endpoints_queued_tokens": endpoints_queued_tokens,
        }

    async def _handle_get_hyperparameters(_data) -> tuple[int, dict]:
        return 200, {
            "hyperparameters": state["hyperparameters"],
            "allowed_keys": sorted(ALLOWED_HYPERPARAM_KEYS),
            "defaults": dict(DEFAULT_GORGO_HYPERPARAMETERS),
        }

    async def _handle_write_hyperparameters(data, *, replace: bool) -> tuple[int, dict]:
        """Validate + apply a hyperparameter update against the
        structured store (see :mod:`policy.gorgo`).

        Two body shapes are accepted, both validated by
        ``policy.gorgo.validate_update``:

        * **Flat** -- ``{"t_prefill": X, "queued_tokens_weight": Y}``
          updates ``defaults`` only (backward-compat: this is what
          ``proxy/tuning.py`` and ``proxy/calibrate.py`` POST).
        * **Structured** -- ``{"defaults": {...}, "per_target": {url:
          {...}}}`` lets callers write per-replica overrides.

        ``replace=True`` (PUT) resets the store to factory defaults
        before applying the update; ``replace=False`` (POST/PATCH)
        layers the update on top of the existing store.
        """
        update, err = validate_update(data, known_targets=set(replica_urls))
        if err is not None:
            return 400, {
                "error": err,
                "allowed_keys": sorted(ALLOWED_HYPERPARAM_KEYS),
            }
        state["hyperparameters"] = merge_update(
            state["hyperparameters"], update or {}, replace=replace
        )
        print(f"[proxy] hyperparameters updated: {state['hyperparameters']}")
        return 200, {"hyperparameters": state["hyperparameters"]}

    async def _handle_post_hyperparameters(data) -> tuple[int, dict]:
        return await _handle_write_hyperparameters(data, replace=False)

    async def _handle_put_hyperparameters(data) -> tuple[int, dict]:
        return await _handle_write_hyperparameters(data, replace=True)

    async def _handle_post_flush(_data) -> tuple[int, dict]:
        radix_trie.clear()
        # Drop tuning samples too: post-flush per-token rates will look
        # different (cold KV cache, fresh queue depths) so mixing them
        # with pre-flush samples would bias the next auto-tune
        # recompute. The auto-tuner config (enabled / window / hop /
        # apply) is intentionally preserved -- only the buffered
        # samples + the per-window counter are reset.
        samples.clear()
        state["total_samples_appended"] = 0
        state["auto_tune"]["samples_since_last_apply"] = 0
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
        return 200, {"radix_trie_cleared": True, "replicas": replica_results}

    def _auto_tune_status() -> dict:
        """Snapshot of the live auto-tuner config + diagnostics.
        Shared by ``GET /tune``, ``GET /samples``, and the response of
        every ``POST /tune`` so callers always see what they just
        configured."""
        at = state["auto_tune"]
        return {
            "enabled": at["enabled"],
            "window_size": at["window_size"],
            "hop_size": at["hop_size"],
            "apply": at["apply"],
            "buffered_samples": len(samples),
            "samples_since_last_apply": at["samples_since_last_apply"],
            "samples_until_next_apply": (
                max(0, at["hop_size"] - at["samples_since_last_apply"]) if at["enabled"] else None
            ),
            "applied_count": at["applied_count"],
            "last_applied_at_monotonic": at["last_applied_at_monotonic"],
            "last_recommendation": at["last_recommendation"],
            "enabled_at_monotonic": at["enabled_at_monotonic"],
            "current_policy": state["policy"],
            "current_hyperparameters": state["hyperparameters"],
        }

    async def _handle_get_samples(_data) -> tuple[int, dict]:
        """Visibility into the tuning sample buffer. Returns the most
        recent samples (capped to keep the response small) plus the
        live auto-tuner status."""
        recent = list(samples)[-50:]
        return 200, {
            "buffered_samples": len(samples),
            "max_buffer_size": samples.maxlen,
            "total_samples_appended": state["total_samples_appended"],
            "auto_tune": _auto_tune_status(),
            "recent": recent,
        }

    async def _handle_get_tune(_data) -> tuple[int, dict]:
        """Live auto-tuner status (no side effects)."""
        return 200, {"auto_tune": _auto_tune_status()}

    async def _handle_post_tune(data) -> tuple[int, dict]:
        """Toggle / reconfigure the on-the-fly auto-tuner.

        The auto-tuner is *stateful*: once enabled it keeps
        recomputing ``t_prefill`` / ``queued_tokens_weight`` every
        ``hop_size`` new samples (after the first ``window_size``
        samples have buffered to fill the window) until disabled.
        Each ``POST /tune`` atomically merges the body into the live
        config; only the keys present in the body are touched.

        Body (all optional):
          * ``enabled``:     bool. When omitted, defaults to ``True``
            so a bare ``POST /tune {}`` turns the tuner on with the
            current config. Pass ``{"enabled": false}`` to disable.
          * ``window_size``: int. Trailing-sample window fed to
            ``recommend_hyperparameters``.
          * ``hop_size``:    int (>0). Recompute every N new samples.
            A small ``hop_size`` reacts faster to load shifts; a
            larger one is more stable.
          * ``apply``:       bool. When ``False`` the recommender
            still runs (and ``last_recommendation`` is updated) but
            ``state["hyperparameters"]`` is not mutated -- useful for
            shadow-mode comparison against the current settings.

        Enabling requires ``policy == 'gorgo'`` (the only policy that
        consumes the tuned scalars). Disabling works regardless of
        policy so a misconfigured tuner can always be turned off.
        """
        if not isinstance(data, dict):
            return 400, {"error": "body must be a JSON object"}

        at = state["auto_tune"]

        # Validate first; only mutate after every requested key passes.
        # Otherwise a bad ``hop_size`` after a good ``window_size``
        # would leave the tuner half-configured.
        new_window = at["window_size"]
        new_hop = at["hop_size"]
        new_apply = at["apply"]

        if "window_size" in data:
            try:
                new_window = int(data["window_size"])
            except (TypeError, ValueError):
                return 400, {"error": "window_size must be an integer"}
            if new_window <= 0:
                return 400, {"error": "window_size must be positive"}
        if "hop_size" in data:
            try:
                new_hop = int(data["hop_size"])
            except (TypeError, ValueError):
                return 400, {"error": "hop_size must be an integer"}
            if new_hop <= 0:
                return 400, {"error": "hop_size must be > 0"}
        if "apply" in data:
            new_apply = bool(data["apply"])

        # Default to enabling so a bare POST /tune {} turns it on. To
        # leave the toggle alone (e.g. just adjust window_size while
        # already running) callers can pass the current value back, but
        # in practice the explicit-default keeps the common case terse.
        new_enabled = bool(data.get("enabled", True))

        if new_enabled and normalize_policy(state["policy"]) != "gorgo":
            return 400, {
                "error": ("auto-tuning can only be enabled when the active policy is 'gorgo'"),
                "current_policy": state["policy"],
            }

        was_enabled = at["enabled"]
        at["window_size"] = new_window
        at["hop_size"] = new_hop
        at["apply"] = new_apply
        at["enabled"] = new_enabled

        if new_enabled and not was_enabled:
            # Fresh enable: zero the per-window counter so the first
            # recompute is measured from this moment, not from stale
            # samples that landed while the tuner was off.
            at["samples_since_last_apply"] = 0
            at["enabled_at_monotonic"] = time.monotonic()
            print(f"[proxy] auto-tune ENABLED window={new_window} hop={new_hop} apply={new_apply}")
        elif not new_enabled and was_enabled:
            print("[proxy] auto-tune DISABLED")
        elif new_enabled:
            # Reconfigured while running -- keep the existing counter
            # so we don't reset the "samples until next apply" clock
            # on every adjustment.
            print(
                f"[proxy] auto-tune RECONFIGURED window={new_window} "
                f"hop={new_hop} apply={new_apply}"
            )

        # Best-effort summary of the current trailing window if it's
        # already large enough; gives the caller something useful to
        # see immediately even before the next recompute fires.
        preview: dict | None = None
        if samples:
            window = list(samples)[-new_window:]
            preview = {
                "window_size_used": len(window),
                "recommendation": recommend_hyperparameters_per_target(window),
                "stats": summarize_samples(window),
            }

        return 200, {
            "auto_tune": _auto_tune_status(),
            "preview": preview,
        }

    # JSON route table. The dispatcher distinguishes 405 (method exists
    # for path but not this verb) from 404 (path unknown) by membership
    # checks, so adding a new (method, path) entry here is the only edit
    # required to expose a new JSON endpoint.
    json_routes: dict[tuple[str, str], object] = {
        ("GET", "/policy"): _handle_get_policy,
        ("POST", "/policy"): _handle_post_policy,
        ("GET", "/replicas"): _handle_get_replicas,
        ("POST", "/replicas"): _handle_post_replicas,
        ("GET", "/trie"): _handle_get_trie,
        ("GET", "/replica_metrics"): _handle_get_replica_metrics,
        ("GET", "/hyperparameters"): _handle_get_hyperparameters,
        ("POST", "/hyperparameters"): _handle_post_hyperparameters,
        ("PATCH", "/hyperparameters"): _handle_post_hyperparameters,
        ("PUT", "/hyperparameters"): _handle_put_hyperparameters,
        ("POST", "/flush"): _handle_post_flush,
        ("GET", "/samples"): _handle_get_samples,
        ("GET", "/tune"): _handle_get_tune,
        ("POST", "/tune"): _handle_post_tune,
    }

    async def _dispatch_json(method: str, path: str, receive, send) -> bool:
        handler = json_routes.get((method, path))
        if handler is None:
            if any(p == path for (_, p) in json_routes):
                await _send_json(send, 405, {"error": "method not allowed"})
                return True
            return False  # let caller emit 404

        # Body-bearing methods may include a JSON payload; GET callers
        # pass an empty dict so handlers don't have to special-case.
        if method in ("POST", "PUT", "PATCH"):
            try:
                data = await _read_json_body(receive)
            except json.JSONDecodeError:
                await _send_json(send, 400, {"error": "invalid JSON body"})
                return True
        else:
            data = {}

        status, payload = await handler(data)
        await _send_json(send, status, payload)
        return True

    # ---------- Chat completions (streaming passthrough + tuning tap) ----------

    def _record_request_sample(
        *,
        target: str,
        ttft_ns: int | None,
        total_ns: int,
        prompt_tokens: int | None,
        completion_tokens: int | None,
    ) -> None:
        """Append a per-request sample shaped like
        :func:`proxy.measure.measure_chat_completion`'s output. Skipped
        silently when token counts are missing (calibrate-style safeguard
        against corrupted per-token rates).

        ``ping_seconds`` reuses the most recent /metrics scrape latency
        for ``target`` as a stand-in for the irreducible RTT subtracted
        from TTFT -- same role ``ping_once`` plays in
        ``proxy/calibrate.py``. If no metrics snapshot exists yet
        (cold start), we record ``ping=0`` so the prefill rate is a
        slight overestimate rather than negative.
        """
        if (
            ttft_ns is None
            or prompt_tokens is None
            or prompt_tokens <= 0
            or completion_tokens is None
            or completion_tokens <= 0
        ):
            return

        snap = live_metrics.get(target)
        ping_rtt_s = snap.latency if snap is not None else 0.0

        ttft_s = ttft_ns / NS_PER_S
        total_s = total_ns / NS_PER_S
        prefill_s = max(ttft_s - ping_rtt_s, 0.0)
        decode_s = max(total_s - ttft_s, 0.0)

        samples.append(
            {
                "ping_seconds": ping_rtt_s,
                "ttft_seconds": ttft_s,
                "total_seconds": total_s,
                "prefill_seconds": prefill_s,
                "decode_seconds": decode_s,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "prefill_rate_seconds_per_token": prefill_s / prompt_tokens,
                "decode_rate_seconds_per_token": decode_s / completion_tokens,
                "target": target,
                "recorded_at_monotonic": time.monotonic(),
            }
        )
        state["total_samples_appended"] += 1

        # Auto-tune hook: only fire on the same code path that produced
        # the sample so the recompute is naturally serialized with
        # appends (single asyncio thread, no lock needed). The fast path
        # is a single bool check when the tuner is disabled.
        at = state["auto_tune"]
        if not at["enabled"]:
            return
        at["samples_since_last_apply"] += 1
        if at["samples_since_last_apply"] < at["hop_size"]:
            return
        if len(samples) < at["window_size"]:
            # Don't fire until the buffer holds at least one full window;
            # earlier recomputes would over-weight whichever short prefix
            # of the run happens to have landed first.
            return
        if normalize_policy(state["policy"]) != "gorgo":
            # Auto-tune only writes ``t_prefill`` / ``queued_tokens_weight``;
            # under any other policy the writes are inert. Skip the work
            # and keep the counter pinned so a switch back to gorgo
            # immediately resumes recomputing on the next sample.
            at["samples_since_last_apply"] = at["hop_size"]
            return

        window = list(samples)[-at["window_size"] :]
        # Per-target recommendation: pooled ``defaults`` for unseen
        # replicas, plus per-replica overrides for any replica with
        # at least ``min_samples_per_target`` observations in the
        # window. Replicas that fall below the threshold simply
        # inherit ``defaults`` instead of getting a noisy single-
        # sample median.
        recommendation = recommend_hyperparameters_per_target(window)
        if at["apply"]:
            # Use the same merge primitive as ``POST /hyperparameters``
            # so layering rules stay identical between manual writes
            # and auto-tune writes (key-level merge: per-target keys
            # not in the recommendation are preserved).
            state["hyperparameters"] = merge_update(
                state["hyperparameters"], recommendation, replace=False
            )
        at["applied_count"] += 1
        at["last_applied_at_monotonic"] = time.monotonic()
        at["last_recommendation"] = recommendation
        at["samples_since_last_apply"] = 0
        print(
            f"[proxy] auto-tune #{at['applied_count']} "
            f"window={len(window)} defaults={recommendation['defaults']} "
            f"per_target={list(recommendation['per_target'])} "
            f"(apply={at['apply']})"
        )

    async def _handle_chat_completions(receive, send) -> None:
        try:
            data = await _read_json_body(receive)
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
            target = _select_endpoint(token_ids)
        except Exception as e:
            print(f"[proxy] policy {state['policy']!r} failed ({e}); falling back to random")
            if not replica_urls:
                await _send_json(send, 503, {"error": "no replicas registered"})
                return
            target = random.choice(replica_urls)

        endpoints_queued_tokens[target] = endpoints_queued_tokens.get(target, 0) + request_tokens
        client = state["upstream_client"]
        if client is None:
            # Race between startup and the first inbound request; defensive.
            await _send_json(send, 503, {"error": "upstream client not yet initialized"})
            if target in endpoints_queued_tokens:
                endpoints_queued_tokens[target] = max(
                    0, endpoints_queued_tokens[target] - request_tokens
                )
            return

        # Serialize the original parsed JSON exactly once for the upstream
        # request. We forward bytes directly instead of letting httpx do
        # its own json= serialization to avoid a second round-trip through
        # the encoder. ``accept-encoding: identity`` tells the upstream
        # not to compress so we don't have to juggle ``content-encoding``
        # on the way out to the client.
        upstream_body = json.dumps(data).encode()
        upstream_headers = {
            "accept-encoding": "identity",
            "content-type": "application/json",
        }
        headers_sent = False
        # Captured before client.stream(...) so TTFT measured by the SSE
        # tee includes request-send + response-headers latency, matching
        # the contract in proxy/measure.py::consume_sse_stream.
        request_start_ns = time.perf_counter_ns()
        try:
            async with client.stream(
                "POST",
                f"{target}/v1/chat/completions",
                content=upstream_body,
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

                is_sse = upstream.headers.get("content-type", "").startswith("text/event-stream")

                async def _sink(chunk: bytes) -> None:
                    """Forward an upstream byte chunk to the ASGI client.
                    Used as the ``chunk_sink`` for ``consume_sse_stream``
                    so passthrough latency is unaffected by the parse loop."""
                    await send(
                        {
                            "type": "http.response.body",
                            "body": chunk,
                            "more_body": True,
                        }
                    )

                if is_sse and 200 <= upstream.status_code < 300:
                    # Tee path: parse SSE for tuning samples while the
                    # raw bytes flow straight through to the client.
                    (
                        ttft_ns,
                        output_tokens,
                        prompt_tokens,
                        completion_tokens,
                    ) = await consume_sse_stream(
                        upstream,
                        request_start_ns=request_start_ns,
                        chunk_sink=_sink,
                    )
                    total_ns = time.perf_counter_ns() - request_start_ns
                    _record_request_sample(
                        target=target,
                        ttft_ns=ttft_ns,
                        total_ns=total_ns,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=(
                            completion_tokens if completion_tokens is not None else output_tokens
                        ),
                    )
                else:
                    # Plain passthrough for non-SSE bodies (e.g. error
                    # JSON, or a caller that asked for stream=False).
                    async for chunk in upstream.aiter_raw():
                        if chunk:
                            await _sink(chunk)
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

    # ---------- Lifespan ----------

    async def _handle_lifespan(receive, send) -> None:
        while True:
            msg = await receive()
            if msg["type"] == "lifespan.startup":
                state["upstream_client"] = _new_upstream_client()
                # Prime synchronously so the first inbound request doesn't
                # see empty live_metrics and fall back to random.
                try:
                    await _refresh_all(state["upstream_client"])
                except Exception as e:
                    print(f"[proxy] initial metrics refresh failed: {e}")
                state["metrics_task"] = asyncio.create_task(_metrics_refresh_loop())
                print(
                    f"[proxy] metrics refresh loop started "
                    f"(interval={METRICS_REFRESH_INTERVAL_SECONDS}s, "
                    f"{len(replica_urls)} replicas); "
                    f"upstream client: http2=True, "
                    f"max_connections={upstream_limits.max_connections}, "
                    f"max_keepalive_connections={upstream_limits.max_keepalive_connections}"
                )
                await send({"type": "lifespan.startup.complete"})
            elif msg["type"] == "lifespan.shutdown":
                task = state.get("metrics_task")
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

    # ---------- ASGI entry point ----------

    async def asgi_app(scope, receive, send):
        if scope["type"] == "lifespan":
            await _handle_lifespan(receive, send)
            return
        if scope["type"] != "http":
            return

        path = scope["path"]
        method = scope["method"]

        if path == "/v1/chat/completions" and method == "POST":
            await _handle_chat_completions(receive, send)
            return

        if await _dispatch_json(method, path, receive, send):
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
