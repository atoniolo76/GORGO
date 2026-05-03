"""GORGO routing policy and per-target hyperparameter store.

The GORGO policy scores each replica by a closed-form cost model and
picks the minimum::

    score(u) = network_rtt(u)
             + t_prefill(u)            * effective_prefill_tokens(u)
             + queued_tokens_weight(u) * (queued_tokens(u) + used_tokens(u))

``network_rtt`` is the EWMA-smoothed RTT of a dedicated lightweight
probe (``snap.network_rtt`` populated by ``proxy/modal_proxy.py``); when
the probe hasn't completed yet it falls back to ``snap.latency`` (the
``/metrics`` scrape RTT, which is a noisy upper bound on RTT).
Critically, the same value is subtracted from observed TTFT before
fitting ``t_prefill`` in ``_record_request_sample``, so the network leg
is accounted for once -- not double-counted, not ignored.

``t_prefill`` and ``queued_tokens_weight`` have units of seconds-per-
token in this scoring function and can be calibrated either offline
(``proxy/calibrate.py``) or online (``proxy/modal_proxy.py``'s auto-
tuner via ``POST /tune``).

This module also owns the *hyperparameter store* shape, since GORGO
is currently the only policy that reads it::

    {
        "defaults":   {"t_prefill": <float>, "queued_tokens_weight": <float>},
        "per_target": {<replica_url>: {"t_prefill": ..., "queued_tokens_weight": ...}, ...}
    }

``defaults`` applies to every replica; ``per_target`` overrides
specific keys for a specific replica URL. The auto-tuner produces
per-target overrides automatically because each per-request sample
is labeled with its destination ``target``; offline tools
(``calibrate.py``, ``tuning.py``) typically only update ``defaults``
since they characterize a single homogeneous fleet.

If your replicas are deliberately heterogeneous (mixed GPU classes /
batch sizes / model variants under one proxy) the per-target slot is
where that heterogeneity lives in the routing math.
"""

from __future__ import annotations

from typing import Any

from policy.base import PolicyDef, RouteContext, route_random

# ---------------------------------------------------------------------------
# Hyperparameter schema
# ---------------------------------------------------------------------------

# Default (per-replica) values. Any replica without a ``per_target``
# override inherits these. ``1.0`` is intentionally a heavy
# overestimate -- the calibrator and the auto-tuner both bias the
# numbers downward as they collect signal.
DEFAULT_GORGO_HYPERPARAMETERS: dict[str, float] = {
    "t_prefill": 1.0,
    "queued_tokens_weight": 1.0,
}

# Allowed keys inside any ``defaults`` / ``per_target.<url>`` map.
# Used by ``proxy/modal_proxy.py``'s ``/hyperparameters`` validator.
ALLOWED_HYPERPARAM_KEYS: frozenset[str] = frozenset(DEFAULT_GORGO_HYPERPARAMETERS)


def make_default_store() -> dict[str, Any]:
    """Fresh hyperparameter store with no per-target overrides.
    Calling code should always go through this so the shape stays
    consistent (callers that build dicts by hand inevitably forget
    to seed an empty ``per_target``)."""
    return {
        "defaults": dict(DEFAULT_GORGO_HYPERPARAMETERS),
        "per_target": {},
    }


def effective_hyperparameters(store: dict[str, Any], target: str) -> dict[str, float]:
    """Resolve the *effective* hyperparameters for ``target`` by
    layering its per-target overrides on top of the global defaults.

    Missing keys fall through to defaults; missing target falls
    through to defaults entirely. Returns a fresh dict so callers
    can't accidentally mutate the store.
    """
    defaults = store.get("defaults") or {}
    per_target = store.get("per_target") or {}
    override = per_target.get(target) or {}
    merged = dict(DEFAULT_GORGO_HYPERPARAMETERS)
    merged.update(defaults)
    merged.update(override)
    return merged


def validate_update(
    data: Any,
    *,
    known_targets: set[str] | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate a body bound for ``POST /hyperparameters`` and return
    a normalized update dict (or ``(None, error)``).

    Two body shapes are accepted:

    1. **Flat** -- ``{"t_prefill": X, "queued_tokens_weight": Y}``.
       Keys are written to ``defaults`` only. This is the original
       shape and remains the path that ``proxy/tuning.py`` /
       ``proxy/calibrate.py`` POST.

    2. **Structured** -- ``{"defaults": {...}, "per_target": {url:
       {...}}}``. Either branch is optional. ``per_target`` URLs
       must be currently-registered replicas if ``known_targets`` is
       supplied (typoed URLs would otherwise silently shadow no
       traffic).

    The returned update has the same structured shape regardless of
    the input form, so the caller's merge logic only deals with one
    case.
    """
    if not isinstance(data, dict):
        return None, "body must be a JSON object"

    has_structured_keys = "defaults" in data or "per_target" in data
    flat_keys = {k for k in data if k not in {"defaults", "per_target"}}

    if has_structured_keys and flat_keys:
        return None, (
            "body cannot mix flat hyperparameter keys with 'defaults'/'per_target'; pick one shape"
        )

    out: dict[str, Any] = {"defaults": {}, "per_target": {}}

    def _coerce_block(block: Any, where: str) -> tuple[dict[str, float] | None, str | None]:
        if not isinstance(block, dict):
            return None, f"{where} must be a JSON object"
        unknown = sorted(k for k in block if k not in ALLOWED_HYPERPARAM_KEYS)
        if unknown:
            return None, f"unknown hyperparameter(s) under {where}: {unknown}"
        try:
            return {k: float(v) for k, v in block.items()}, None
        except (TypeError, ValueError):
            return None, f"hyperparameter values under {where} must be numeric"

    if has_structured_keys:
        if "defaults" in data:
            block, err = _coerce_block(data["defaults"], "defaults")
            if err:
                return None, err
            out["defaults"] = block or {}
        if "per_target" in data:
            pt = data["per_target"]
            if not isinstance(pt, dict):
                return None, "'per_target' must be a JSON object keyed by replica URL"
            for url, block in pt.items():
                if not isinstance(url, str) or not url:
                    return None, "per_target keys must be non-empty replica URL strings"
                if known_targets is not None and url not in known_targets:
                    return None, (f"per_target URL {url!r} is not a currently-registered replica")
                block, err = _coerce_block(block, f"per_target[{url!r}]")
                if err:
                    return None, err
                out["per_target"][url] = block or {}
    else:
        block, err = _coerce_block(data, "body")
        if err:
            return None, err
        out["defaults"] = block or {}

    return out, None


def merge_update(
    store: dict[str, Any],
    update: dict[str, Any],
    *,
    replace: bool,
) -> dict[str, Any]:
    """Apply a normalized ``update`` to ``store`` and return the new
    store.

    * ``replace=False`` (POST/PATCH) -- key-level merge. Defaults are
      merged into the existing defaults; each per-target dict is
      merged into any existing override for that URL. Unmentioned
      keys / URLs are preserved.
    * ``replace=True`` (PUT) -- the resulting store is built fresh:
      ``defaults`` start from :data:`DEFAULT_GORGO_HYPERPARAMETERS`,
      ``per_target`` starts empty, and the update is layered on top.
      Equivalent to "reset to factory defaults, then apply this".
    """
    if replace:
        out: dict[str, Any] = {
            "defaults": dict(DEFAULT_GORGO_HYPERPARAMETERS),
            "per_target": {},
        }
    else:
        out = {
            "defaults": dict(store.get("defaults") or DEFAULT_GORGO_HYPERPARAMETERS),
            "per_target": {
                url: dict(block) for url, block in (store.get("per_target") or {}).items()
            },
        }

    out["defaults"].update(update.get("defaults") or {})
    for url, block in (update.get("per_target") or {}).items():
        merged_block = dict(out["per_target"].get(url, {}))
        merged_block.update(block)
        out["per_target"][url] = merged_block
    return out


def prune_per_target(store: dict[str, Any], known_targets: set[str]) -> dict[str, Any]:
    """Drop per-target entries whose URL is no longer a registered
    replica. Called by the proxy after ``POST /replicas`` so stale
    overrides don't pile up indefinitely. Returns the same store
    object after in-place mutation (caller convenience)."""
    pt = store.get("per_target") or {}
    for url in list(pt):
        if url not in known_targets:
            pt.pop(url, None)
    return store


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def route_gorgo(ctx: RouteContext) -> str:
    """GORGO multi-objective routing.

    Each replica is scored independently using its own *effective*
    hyperparameters (defaults overlaid by any per-target override).
    The minimum-score replica wins; replicas without a metrics
    snapshot are skipped, falling back to random if every snapshot
    is missing.

    Read directly off ``ctx`` rather than via positional arguments
    so the registry's adapter lambda stays a one-liner. The other
    policies still take positional args for backward compat with the
    handful of tests that call them directly; gorgo is a clean break
    because its signature is changing anyway (per-target store).
    """
    if not ctx.replica_urls:
        raise ValueError("no replicas")

    endpoints_cached_tokens = (
        ctx.radix_trie.cached_prefix_lengths(ctx.token_ids, ctx.replica_urls)
        if ctx.token_ids
        else {u: 0 for u in ctx.replica_urls}
    )
    scores: dict[str, float] = {}
    for u in ctx.replica_urls:
        snap = ctx.metrics.get(u)
        if snap is None:
            continue
        eff = effective_hyperparameters(ctx.hyperparameters, u)
        cached = endpoints_cached_tokens.get(u, 0)
        effective_prefill = max(0, ctx.request_tokens - cached)
        prefill_cost = effective_prefill * eff["t_prefill"]
        queue_cost = (ctx.endpoints_queued_tokens.get(u, 0) + snap.num_used_tokens) * eff[
            "queued_tokens_weight"
        ]
        # Use the dedicated lightweight RTT probe when available; fall back
        # to scrape latency for cold-start (no probe completed yet). Same
        # source as ``_record_request_sample``'s subtraction so the network
        # leg is accounted for symmetrically: subtract once when fitting
        # ``t_prefill``, add back once when scoring routes.
        rtt = snap.network_rtt if snap.network_rtt > 0.0 else snap.latency
        scores[u] = rtt + prefill_cost + queue_cost
    if not scores:
        return route_random(ctx.replica_urls)
    return min(scores, key=scores.get)


# ---------------------------------------------------------------------------
# Registry export
# ---------------------------------------------------------------------------

GORGO_POLICIES: list[PolicyDef] = [
    PolicyDef("gorgo", needs_metrics=True, fn=route_gorgo),
]
