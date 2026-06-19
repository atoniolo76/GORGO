"""Offline calibration of the two physical rates used by the GORGO cost model.

We fit two per-token rates from a recorded proxy request trace:

- ``P`` -- prefill rate, milliseconds per UNCACHED prompt token.
- ``Q`` -- queue rate, milliseconds per QUEUED prompt token, where "queued" is
  the PROXY-side in-flight counter captured at dispatch
  (``candidate_snapshot[target]["queued_tokens"]``), NOT an engine-side FCFS
  reconstruction. This is the same load signal ``route_gorgo_2d`` scores on, so
  fitting against it keeps the cost model and the router consistent.

Trace schema (read from ``proxy/modal_proxy.py``, not modified here)
--------------------------------------------------------------------
The proxy writes a JSON-lines file per trace at
``/results/proxy_traces/<trace_id>/requests.jsonl`` (see
``_save_trace_to_volume`` / ``_write_jsonl``). Each line is one
``request_trace_event`` dict (built ~L2759). Fields we read:

- ``request_id``        : str
- ``wall_ts``           : ISO-8601 UTC string (used for windowing)
- ``monotonic_s``       : float seconds, proxy-local monotonic clock at dispatch
- ``target``            : replica URL (group key)
- ``request_tokens``    : int, prompt length seen by the proxy at dispatch
- ``cached_tokens_at_dispatch`` : int, cached prefix length on ``target``
- ``ttft_ns``           : int nanoseconds, time-to-first-token (proxy-measured)
- ``total_ns``          : int nanoseconds, total request wall time
- ``prompt_tokens``     : int (engine-reported; may be None)
- ``completion_tokens`` : int (may be None)
- ``status``            : int HTTP status (filter to 2xx)
- ``candidate_snapshot``: dict[target -> per-replica at-decision values], where
  ``candidate_snapshot[target]["queued_tokens"]`` is the proxy in-flight queued
  token counter (already excludes the current request).

Defensive / forward-compatible fields (added by a separate worker; may be
absent or None in older traces -- ALWAYS accessed with ``.get`` and None
checks):

- ``meta_info``            : dict of engine timing fields. We treat the latency
  fields as SECONDS and convert to ms (see ``UNIT ASSUMPTION`` below):
    - ``queue_time``               (seconds)
    - ``prefill_waiting_latency``  (seconds)
    - ``prefill_launch_latency``   (seconds)
    - ``e2e_latency``              (seconds)
    - ``cached_tokens``            (int)
    - ``*_ts``                     (timestamps; not used for rate fitting)
- ``load_release_at_ttft`` : top-level bool (reported only).

UNIT ASSUMPTION: ``ttft_ns`` / ``total_ns`` are NANOSECONDS (proxy uses
``time.perf_counter_ns``). All ``meta_info`` *latency* fields are assumed to be
SECONDS (SGLang reports these in seconds) and are multiplied by 1000 to get ms.
If a ``meta_info`` latency value looks like nanoseconds (>= 1e6 for a per-stage
latency) we still treat it as seconds but flag the trace as suspicious in the
diagnostics, so the assumption is auditable rather than silently wrong.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# meta_info latency fields we read, all assumed to be in SECONDS.
_META_LATENCY_FIELDS = (
    "queue_time",
    "prefill_waiting_latency",
    "prefill_launch_latency",
    "e2e_latency",
)
# Heuristic: a single prefill/queue stage latency in seconds should be well
# under this. If we see a larger value it was probably already ms or ns.
_SUSPICIOUS_SECONDS = 1.0e3


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{line_no}: invalid JSONL row: {e}") from e
            if isinstance(row, dict):
                yield row


def _status_is_2xx(status: Any) -> bool:
    try:
        code = int(status)
    except (TypeError, ValueError):
        return False
    return 200 <= code <= 299


def _parse_wall_ts(wall_ts: Any) -> float | None:
    """Parse the proxy ISO-8601 ``wall_ts`` into epoch seconds (UTC)."""
    if not isinstance(wall_ts, str) or not wall_ts:
        return None
    text = wall_ts.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.timestamp()


def _coerce_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _coerce_int(value: Any) -> int | None:
    f = _coerce_float(value)
    if f is None:
        return None
    return int(f)


class Record:
    """A single calibrated request observation, derived from a trace row."""

    __slots__ = (
        "request_id",
        "dispatch_wall_s",
        "dispatch_monotonic_s",
        "target",
        "prompt_tokens",
        "uncached_tokens",
        "queued_tokens_at_dispatch",
        "ttft_ns",
        "total_ns",
        "meta_info",
    )

    def __init__(
        self,
        *,
        request_id: str,
        dispatch_wall_s: float | None,
        dispatch_monotonic_s: float | None,
        target: str,
        prompt_tokens: int,
        uncached_tokens: int,
        queued_tokens_at_dispatch: int,
        ttft_ns: int | None,
        total_ns: int | None,
        meta_info: dict | None,
    ) -> None:
        self.request_id = request_id
        self.dispatch_wall_s = dispatch_wall_s
        self.dispatch_monotonic_s = dispatch_monotonic_s
        self.target = target
        self.prompt_tokens = prompt_tokens
        self.uncached_tokens = uncached_tokens
        self.queued_tokens_at_dispatch = queued_tokens_at_dispatch
        self.ttft_ns = ttft_ns
        self.total_ns = total_ns
        self.meta_info = meta_info

    @property
    def ttft_ms(self) -> float | None:
        if self.ttft_ns is None:
            return None
        return self.ttft_ns / 1.0e6

    def meta_latency_ms(self, field: str) -> float | None:
        """Return ``meta_info[field]`` converted from seconds to ms, or None."""
        if not isinstance(self.meta_info, dict):
            return None
        secs = _coerce_float(self.meta_info.get(field))
        if secs is None:
            return None
        return secs * 1000.0


class LoadResult:
    """Outcome of loading + filtering a trace file."""

    def __init__(self) -> None:
        self.records: list[Record] = []
        self.total_rows = 0
        self.kept = 0
        self.excluded_non_request = 0
        self.excluded_status = 0
        self.excluded_missing_fields = 0
        self.meta_info_present = 0
        self.load_release_values: set[Any] = set()
        self.suspicious_unit_rows = 0


def load_trace(path: Path) -> LoadResult:
    """Parse a proxy ``requests.jsonl`` trace into per-request ``Record``s.

    Keeps successful (2xx) request rows that carry the fields needed to fit at
    least one rate, and groups nothing here -- grouping by target happens in the
    aggregation step. Forward-compatible fields are accessed defensively.
    """
    res = LoadResult()
    for row in _iter_jsonl(path):
        res.total_rows += 1

        kind = row.get("kind")
        if kind is not None and kind != "request":
            res.excluded_non_request += 1
            continue

        if not _status_is_2xx(row.get("status")):
            res.excluded_status += 1
            continue

        target = row.get("target")
        if not isinstance(target, str) or not target:
            res.excluded_missing_fields += 1
            continue

        # Prompt length: prefer the engine-reported prompt_tokens, fall back to
        # the proxy's request_tokens (token count at dispatch).
        prompt_tokens = _coerce_int(row.get("prompt_tokens"))
        if prompt_tokens is None or prompt_tokens <= 0:
            prompt_tokens = _coerce_int(row.get("request_tokens"))
        if prompt_tokens is None or prompt_tokens <= 0:
            res.excluded_missing_fields += 1
            continue

        cached = _coerce_int(row.get("cached_tokens_at_dispatch")) or 0
        uncached_tokens = max(1, prompt_tokens - cached)

        # queued_tokens_at_dispatch = candidate_snapshot[target]["queued_tokens"]
        snapshot = row.get("candidate_snapshot")
        queued = 0
        if isinstance(snapshot, dict):
            per_target = snapshot.get(target)
            if isinstance(per_target, dict):
                queued = _coerce_int(per_target.get("queued_tokens")) or 0

        ttft_ns = _coerce_int(row.get("ttft_ns"))
        total_ns = _coerce_int(row.get("total_ns"))

        meta_info = row.get("meta_info")
        if not isinstance(meta_info, dict):
            meta_info = None
        else:
            res.meta_info_present += 1
            for field in _META_LATENCY_FIELDS:
                secs = _coerce_float(meta_info.get(field))
                if secs is not None and secs >= _SUSPICIOUS_SECONDS:
                    res.suspicious_unit_rows += 1
                    break

        # load_release_at_ttft is a top-level bool (reported, not fitted).
        if "load_release_at_ttft" in row:
            res.load_release_values.add(row.get("load_release_at_ttft"))

        res.records.append(
            Record(
                request_id=str(row.get("request_id") or ""),
                dispatch_wall_s=_parse_wall_ts(row.get("wall_ts")),
                dispatch_monotonic_s=_coerce_float(row.get("monotonic_s")),
                target=target,
                prompt_tokens=prompt_tokens,
                uncached_tokens=uncached_tokens,
                queued_tokens_at_dispatch=queued,
                ttft_ns=ttft_ns,
                total_ns=total_ns,
                meta_info=meta_info,
            )
        )
        res.kept += 1

    return res


# --------------------------------------------------------------------------- #
# Rate computation
# --------------------------------------------------------------------------- #
def prefill_delay_ms(rec: Record) -> tuple[float | None, bool]:
    """Prefill stage delay in ms for one request.

    Returns ``(delay_ms, used_fallback)``. Preferred source is
    ``meta_info.prefill_launch_latency`` (seconds -> ms). Fallback, when no
    meta_info is available, is the full ``ttft_ms``; since the trace has no RTT
    field we cannot subtract network round-trip, so the fallback is an
    OVERESTIMATE of true prefill time and is flagged.
    """
    launch_ms = rec.meta_latency_ms("prefill_launch_latency")
    if launch_ms is not None:
        return launch_ms, False
    ttft_ms = rec.ttft_ms
    if ttft_ms is None:
        return None, True
    return ttft_ms, True


def queue_stage_delay_ms(rec: Record) -> tuple[float | None, bool]:
    """Queue + prefill-wait stage delay in ms for one request.

    Returns ``(delay_ms, used_fallback)``. Preferred source is
    ``meta_info.queue_time + meta_info.prefill_waiting_latency`` (seconds -> ms).
    Fallback, when no meta_info is available, is ``ttft_ms`` (no RTT field in the
    trace to subtract), which is flagged as an overestimate.
    """
    queue_ms = rec.meta_latency_ms("queue_time")
    wait_ms = rec.meta_latency_ms("prefill_waiting_latency")
    if queue_ms is not None or wait_ms is not None:
        return (queue_ms or 0.0) + (wait_ms or 0.0), False
    ttft_ms = rec.ttft_ms
    if ttft_ms is None:
        return None, True
    return ttft_ms, True


def _summarize(values: list[float]) -> dict | None:
    if not values:
        return None
    out = {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }
    out["stdev"] = statistics.stdev(values) if len(values) > 1 else 0.0
    return out


def compute_P(records: list[Record]) -> dict:
    """Compute prefill rate P (ms / uncached token), overall and per-replica."""
    overall: list[float] = []
    per_replica: dict[str, list[float]] = defaultdict(list)
    used = 0
    excluded = 0
    fallback_used = 0
    for rec in records:
        delay_ms, used_fallback = prefill_delay_ms(rec)
        if delay_ms is None or rec.uncached_tokens <= 0:
            excluded += 1
            continue
        p_r = delay_ms / rec.uncached_tokens
        overall.append(p_r)
        per_replica[rec.target].append(p_r)
        used += 1
        if used_fallback:
            fallback_used += 1
    return {
        "overall": _summarize(overall),
        "per_replica": {t: _summarize(v) for t, v in sorted(per_replica.items())},
        "used": used,
        "excluded": excluded,
        "fallback_used": fallback_used,
    }


def compute_Q(records: list[Record]) -> dict:
    """Compute queue rate Q (ms / queued token) using the PROXY in-flight counter
    as the denominator. Requests with ``queued_tokens_at_dispatch == 0`` are
    excluded."""
    overall: list[float] = []
    per_replica: dict[str, list[float]] = defaultdict(list)
    used = 0
    excluded_zero_queue = 0
    excluded_missing_delay = 0
    fallback_used = 0
    for rec in records:
        if rec.queued_tokens_at_dispatch <= 0:
            excluded_zero_queue += 1
            continue
        delay_ms, used_fallback = queue_stage_delay_ms(rec)
        if delay_ms is None:
            excluded_missing_delay += 1
            continue
        q_r = delay_ms / rec.queued_tokens_at_dispatch
        overall.append(q_r)
        per_replica[rec.target].append(q_r)
        used += 1
        if used_fallback:
            fallback_used += 1
    return {
        "overall": _summarize(overall),
        "per_replica": {t: _summarize(v) for t, v in sorted(per_replica.items())},
        "used": used,
        "excluded_zero_queue": excluded_zero_queue,
        "excluded_missing_delay": excluded_missing_delay,
        "fallback_used": fallback_used,
    }


# --------------------------------------------------------------------------- #
# Per-window aggregation
# --------------------------------------------------------------------------- #
def compute_windows(records: list[Record], window_seconds: float) -> list[dict]:
    """Bin requests by dispatch wall time and report per-window P and Q."""
    timed = [r for r in records if r.dispatch_wall_s is not None]
    if not timed:
        return []
    t0 = min(r.dispatch_wall_s for r in timed)  # type: ignore[type-var]
    buckets: dict[int, list[Record]] = defaultdict(list)
    for rec in timed:
        idx = int((rec.dispatch_wall_s - t0) // window_seconds)  # type: ignore[operator]
        buckets[idx].append(rec)

    windows = []
    for idx in sorted(buckets):
        bucket = buckets[idx]
        p = compute_P(bucket)
        q = compute_Q(bucket)
        windows.append(
            {
                "window_index": idx,
                "window_start_s": t0 + idx * window_seconds,
                "window_end_s": t0 + (idx + 1) * window_seconds,
                "requests": len(bucket),
                "P_mean": (p["overall"] or {}).get("mean"),
                "P_used": p["used"],
                "Q_mean": (q["overall"] or {}).get("mean"),
                "Q_used": q["used"],
                "Q_excluded_zero_queue": q["excluded_zero_queue"],
            }
        )
    return windows


# --------------------------------------------------------------------------- #
# STEP 5: cross-check only (NOT used to fit Q)
# --------------------------------------------------------------------------- #
def crosscheck_ahead_load(records: list[Record]) -> dict:
    """Reconstruct a post-hoc "ahead-load" via an interval sweep and compare it to
    the raw proxy ``queued_tokens_at_dispatch``.

    For request ``r`` on target ``t`` (dispatch ``d_r``, ttft ``t_r``), count and
    sum-uncached-tokens of other requests ``r'`` on the SAME target where
    ``dispatch(r') < d_r`` AND ``dispatch(r') + ttft(r') > d_r`` -- i.e. ``r'`` was
    dispatched earlier and had not yet emitted its first token when ``r`` was
    dispatched. We compare the token-weighted ahead-load to the proxy queued
    counter to quantify in-flight-vs-queue conflation. This is a diagnostic only;
    it is never used to compute the reported Q.
    """
    by_target: dict[str, list[Record]] = defaultdict(list)
    for rec in records:
        if rec.dispatch_monotonic_s is None or rec.ttft_ns is None:
            continue
        by_target[rec.target].append(rec)

    ratios: list[float] = []
    ahead_tokens_all: list[float] = []
    proxy_queued_all: list[float] = []
    compared = 0
    skipped_no_proxy_queue = 0

    for target, recs in by_target.items():
        recs_sorted = sorted(recs, key=lambda r: r.dispatch_monotonic_s)  # type: ignore[arg-type,return-value]
        for r in recs_sorted:
            d_r = r.dispatch_monotonic_s
            ahead_tokens = 0
            ahead_count = 0
            for rp in recs_sorted:
                if rp is r:
                    continue
                d_rp = rp.dispatch_monotonic_s
                if d_rp >= d_r:  # type: ignore[operator]
                    continue
                finish_first_token = d_rp + (rp.ttft_ns / 1.0e9)  # type: ignore[operator]
                if finish_first_token > d_r:  # type: ignore[operator]
                    ahead_tokens += rp.uncached_tokens
                    ahead_count += 1
            ahead_tokens_all.append(float(ahead_tokens))
            proxy_queued_all.append(float(r.queued_tokens_at_dispatch))
            if r.queued_tokens_at_dispatch > 0:
                ratios.append(ahead_tokens / r.queued_tokens_at_dispatch)
                compared += 1
            else:
                skipped_no_proxy_queue += 1

    return {
        "compared": compared,
        "skipped_no_proxy_queue": skipped_no_proxy_queue,
        "ahead_tokens": _summarize(ahead_tokens_all),
        "proxy_queued_tokens": _summarize(proxy_queued_all),
        "ahead_over_proxy_ratio": _summarize(ratios),
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def build_report(res: LoadResult, window_seconds: float) -> dict:
    records = res.records
    return {
        "diagnostics": {
            "total_rows": res.total_rows,
            "kept_requests": res.kept,
            "excluded_non_request": res.excluded_non_request,
            "excluded_status": res.excluded_status,
            "excluded_missing_fields": res.excluded_missing_fields,
            "meta_info_present": res.meta_info_present,
            "meta_info_fraction": (res.meta_info_present / res.kept) if res.kept else 0.0,
            "suspicious_unit_rows": res.suspicious_unit_rows,
            "load_release_at_ttft_values": sorted(str(v) for v in res.load_release_values),
            "window_seconds": window_seconds,
        },
        "P": compute_P(records),
        "Q": compute_Q(records),
        "windows": compute_windows(records, window_seconds),
        "crosscheck_ahead_load": crosscheck_ahead_load(records),
    }


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _print_summary(report: dict) -> None:
    diag = report["diagnostics"]
    print("=" * 72)
    print("GORGO rate calibration")
    print("=" * 72)
    print(f"total trace rows         : {diag['total_rows']}")
    print(f"kept (2xx + fields)      : {diag['kept_requests']}")
    print(f"  excluded non-request   : {diag['excluded_non_request']}")
    print(f"  excluded by status     : {diag['excluded_status']}")
    print(f"  excluded missing fields: {diag['excluded_missing_fields']}")
    print(
        f"meta_info present        : {diag['meta_info_present']} "
        f"({diag['meta_info_fraction'] * 100:.1f}% of kept)"
    )
    if diag["meta_info_present"] == 0:
        print("  WARNING: no meta_info -> P and Q use ttft fallback (overestimates).")
    elif diag["meta_info_present"] < diag["kept_requests"]:
        print("  NOTE: meta_info present on only some rows -> mixed fallback usage.")
    if diag["suspicious_unit_rows"]:
        print(
            f"  WARNING: {diag['suspicious_unit_rows']} rows had meta_info latency "
            ">= 1000s; SECONDS unit assumption may be wrong."
        )
    lr = diag["load_release_at_ttft_values"]
    if not lr:
        print("load_release_at_ttft     : (absent in trace)")
    elif len(lr) == 1:
        print(f"load_release_at_ttft     : {lr[0]}")
    else:
        print(f"load_release_at_ttft     : MIXED {lr}  <-- WARNING: inconsistent across trace")

    def _print_rate(name: str, block: dict, unit: str) -> None:
        ov = block["overall"]
        print("-" * 72)
        print(f"{name} ({unit})")
        if ov is None:
            print("  overall: no samples")
        else:
            print(
                f"  overall: mean={_fmt(ov['mean'])} median={_fmt(ov['median'])} "
                f"stdev={_fmt(ov['stdev'])} n={ov['count']} "
                f"[min={_fmt(ov['min'])} max={_fmt(ov['max'])}]"
            )
        print(f"  used={block['used']} fallback_used={block['fallback_used']}", end="")
        if "excluded_zero_queue" in block:
            print(
                f" excluded_zero_queue={block['excluded_zero_queue']} "
                f"excluded_missing_delay={block['excluded_missing_delay']}"
            )
        else:
            print(f" excluded={block['excluded']}")
        for target, s in block["per_replica"].items():
            if s is None:
                continue
            print(f"    {target}: mean={_fmt(s['mean'])} median={_fmt(s['median'])} n={s['count']}")

    _print_rate("P  prefill rate", report["P"], "ms / uncached token")
    _print_rate("Q  queue rate", report["Q"], "ms / queued token (proxy in-flight)")

    print("-" * 72)
    print(f"per-window aggregation (window_seconds={diag['window_seconds']})")
    windows = report["windows"]
    if not windows:
        print("  no timestamped requests for windowing")
    else:
        print(f"  {'idx':>4} {'reqs':>6} {'P_mean':>10} {'P_n':>5} {'Q_mean':>10} {'Q_n':>5}")
        for w in windows:
            print(
                f"  {w['window_index']:>4} {w['requests']:>6} "
                f"{_fmt(w['P_mean']):>10} {w['P_used']:>5} "
                f"{_fmt(w['Q_mean']):>10} {w['Q_used']:>5}"
            )

    print("-" * 72)
    print("CROSS-CHECK ONLY: post-hoc 'ahead-load' vs proxy queued_tokens")
    print("  (interval sweep; NOT used to fit Q)")
    cc = report["crosscheck_ahead_load"]
    print(f"  compared={cc['compared']} skipped_no_proxy_queue={cc['skipped_no_proxy_queue']}")

    def _row(label: str, s: dict | None) -> None:
        if s is None:
            print(f"  {label:<22}: no samples")
            return
        print(
            f"  {label:<22}: mean={_fmt(s['mean'])} median={_fmt(s['median'])} "
            f"min={_fmt(s['min'])} max={_fmt(s['max'])} n={s['count']}"
        )

    _row("ahead_tokens", cc["ahead_tokens"])
    _row("proxy_queued_tokens", cc["proxy_queued_tokens"])
    _row("ahead/proxy ratio", cc["ahead_over_proxy_ratio"])
    print("=" * 72)


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def _make_fake_row(
    *,
    request_id: str,
    target: str,
    wall_s: float,
    monotonic_s: float,
    prompt_tokens: int,
    cached: int,
    queued: int,
    p_true_ms: float,
    q_true_ms: float,
    ttft_ns: int,
) -> dict:
    """Build a synthetic trace row with KNOWN P and Q baked into meta_info."""
    uncached = max(1, prompt_tokens - cached)
    prefill_launch_s = (p_true_ms * uncached) / 1000.0
    # Split the known queue-stage delay across the two meta_info fields.
    queue_stage_s = (q_true_ms * queued) / 1000.0 if queued > 0 else 0.0
    wall_ts = datetime.fromtimestamp(wall_s, tz=UTC).isoformat().replace("+00:00", "Z")
    return {
        "kind": "request",
        "request_id": request_id,
        "wall_ts": wall_ts,
        "monotonic_s": monotonic_s,
        "target": target,
        "request_tokens": prompt_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": 8,
        "cached_tokens_at_dispatch": cached,
        "ttft_ns": ttft_ns,
        "total_ns": ttft_ns + 5_000_000,
        "status": 200,
        "candidate_snapshot": {
            target: {
                "queued_tokens": queued,
                "queued_uncached_tokens": queued,
                "inflight_requests": 1 if queued > 0 else 0,
                "cached_prefix_tokens": cached,
            }
        },
        "meta_info": {
            "queue_time": queue_stage_s * 0.4,
            "prefill_waiting_latency": queue_stage_s * 0.6,
            "prefill_launch_latency": prefill_launch_s,
            "e2e_latency": (ttft_ns + 5_000_000) / 1.0e9,
            "cached_tokens": cached,
        },
        "load_release_at_ttft": True,
    }


def run_selftest() -> int:
    p_true = 0.5  # ms per uncached token
    q_true = 0.2  # ms per queued token
    rows = []
    base_wall = 1_000_000.0
    base_mono = 5000.0
    configs = [
        # (target, prompt_tokens, cached, queued)
        ("http://r1", 1000, 200, 300),
        ("http://r1", 1500, 500, 0),  # zero-queue -> excluded from Q, used for P
        ("http://r1", 800, 0, 150),
        ("http://r2", 2000, 1000, 600),
        ("http://r2", 1200, 100, 50),
        ("http://r2", 600, 0, 0),  # zero-queue
    ]
    for i, (target, prompt, cached, queued) in enumerate(configs):
        uncached = max(1, prompt - cached)
        # ttft_ns chosen so the fallback path would also be roughly sane, but the
        # selftest relies on meta_info for exact recovery.
        ttft_ns = int((p_true * uncached + q_true * queued) * 1e6)
        rows.append(
            _make_fake_row(
                request_id=f"req-{i}",
                target=target,
                wall_s=base_wall + i * 10.0,
                monotonic_s=base_mono + i * 10.0,
                prompt_tokens=prompt,
                cached=cached,
                queued=queued,
                p_true_ms=p_true,
                q_true_ms=q_true,
                ttft_ns=ttft_ns,
            )
        )

    # Load via the real loader path (write to a temp file to exercise _iter_jsonl).
    import tempfile

    res = LoadResult()
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as tf:
        for row in rows:
            tf.write(json.dumps(row) + "\n")
        tmp_path = Path(tf.name)
    try:
        res = load_trace(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    report = build_report(res, window_seconds=60.0)

    p_mean = (report["P"]["overall"] or {}).get("mean")
    q_mean = (report["Q"]["overall"] or {}).get("mean")

    failures = []
    if res.kept != len(configs):
        failures.append(f"expected {len(configs)} kept records, got {res.kept}")
    if p_mean is None or abs(p_mean - p_true) > 1e-6:
        failures.append(f"P mean {p_mean} != expected {p_true}")
    if q_mean is None or abs(q_mean - q_true) > 1e-6:
        failures.append(f"Q mean {q_mean} != expected {q_true}")
    # Two configs have queued == 0 -> excluded from Q.
    if report["Q"]["excluded_zero_queue"] != 2:
        failures.append(
            f"expected 2 zero-queue exclusions, got {report['Q']['excluded_zero_queue']}"
        )
    if report["Q"]["used"] != 4:
        failures.append(f"expected Q used=4, got {report['Q']['used']}")
    if report["P"]["used"] != len(configs):
        failures.append(f"expected P used={len(configs)}, got {report['P']['used']}")
    if report["P"]["fallback_used"] != 0 or report["Q"]["fallback_used"] != 0:
        failures.append("expected no fallback usage when meta_info is present")

    _print_summary(report)
    print()
    if failures:
        print("SELFTEST FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(
        f"SELFTEST PASSED: recovered P={_fmt(p_mean)} (true {p_true}), "
        f"Q={_fmt(q_mean)} (true {q_true}) within tolerance."
    )
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Offline calibration of P (prefill ms/uncached-token) and Q "
            "(queue ms/queued-token) from a proxy request trace JSONL."
        )
    )
    parser.add_argument(
        "--trace",
        type=Path,
        help="Path to a proxy request-trace JSONL (e.g. .../proxy_traces/<id>/requests.jsonl).",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=60.0,
        help="Window size (seconds) for per-window P/Q stability aggregation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to dump the full results as JSON.",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="Run the synthetic self-test (no trace file needed) and exit.",
    )
    args = parser.parse_args(argv)

    if args.selftest:
        return run_selftest()

    if args.trace is None:
        parser.error("--trace is required (or pass --selftest)")
    if not args.trace.exists():
        parser.error(f"trace file not found: {args.trace}")

    res = load_trace(args.trace)
    report = build_report(res, window_seconds=args.window_seconds)
    _print_summary(report)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            json.dump(report, f, indent=2)
        print(f"\nwrote results JSON -> {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
