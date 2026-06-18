"""Compute user-diversity / load stats for a fixed set of named windows.

Mirrors the metric definitions in ``find_best_window.py`` (same keep-alive
filter, same token_hash = user proxy, same multi-turn / token definitions) but
reports a specific, named set of windows rather than the global top-20, so the
candidate windows in ``data_index.md`` can be compared apples-to-apples.

Usage::

    modal run --env=alessio-dev data_processing/window_diversity_stats.py::main
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import Counter
from datetime import datetime

import modal

from app import app, completions_volume

image = modal.Image.debian_slim().pip_install("duckdb").add_local_python_source("app")

FILE_PREFIX = "llm_responses_202604"
FILE_CUTOFF = "llm_responses_20260408"

# (label, start_iso_utc, end_iso_utc). All on 5-min boundaries.
WINDOWS = [
    ("apr1 00:30-01:00 (ref night)", "2026-04-01T00:30:00", "2026-04-01T01:00:00"),
    ("apr1 01:00-01:30 (ref night)", "2026-04-01T01:00:00", "2026-04-01T01:30:00"),
    ("apr1 12:30-13:00 (midday)", "2026-04-01T12:30:00", "2026-04-01T13:00:00"),
    ("apr2 00:30-01:00 (W1)", "2026-04-02T00:30:00", "2026-04-02T01:00:00"),
    ("apr2 01:00-01:30 (W2a)", "2026-04-02T01:00:00", "2026-04-02T01:30:00"),
    ("apr2 12:30-13:00 (W2b midday)", "2026-04-02T12:30:00", "2026-04-02T13:00:00"),
    ("apr5 16:15-16:45 (diverse day)", "2026-04-05T16:15:00", "2026-04-05T16:45:00"),
    ("apr6 15:05-15:35 (eval)", "2026-04-06T15:05:00", "2026-04-06T15:35:00"),
    ("apr7 19:45-20:15 (eval)", "2026-04-07T19:45:00", "2026-04-07T20:15:00"),
]


def _percentile(sorted_vals, q):
    if not sorted_vals:
        return 0
    idx = min(len(sorted_vals) - 1, int(q * len(sorted_vals)))
    return sorted_vals[idx]


def _score_window(rows):
    """Same scoring as find_best_window._score_window (rows = (ts, user, msgs, tokens))."""
    n = len(rows)
    users = Counter(r[1] for r in rows)
    n_users = len(users)
    top_user_count = users.most_common(1)[0][1] if users else 0
    top_user_share = top_user_count / n if n > 0 else 0

    multi_turn = sum(1 for r in rows if r[2] > 2)
    multi_turn_pct = 100 * multi_turn / n if n > 0 else 0

    tokens = sorted(r[3] for r in rows if r[3] > 0)
    avg_tokens = sum(tokens) / len(tokens) if tokens else 0
    median_tokens = _percentile(tokens, 0.5)
    p95_tokens = _percentile(tokens, 0.95)

    score = (
        math.log(max(n_users, 1))
        * (multi_turn_pct / 100 + 0.1)
        * math.log(max(avg_tokens, 1))
        * (1 - top_user_share)
        * math.sqrt(n)
    )

    return {
        "n_requests": n,
        "n_users": n_users,
        "multi_turn_pct": round(multi_turn_pct, 1),
        "avg_tokens": round(avg_tokens),
        "median_tokens": median_tokens,
        "p95_tokens": p95_tokens,
        "top_user_share_pct": round(top_user_share * 100, 1),
        "top_user_count": top_user_count,
        "diversity_score": round(score, 1),
    }


@app.function(
    image=image,
    memory=1024 * 32,
    timeout=14400,
    volumes={"/data": completions_volume},
)
def scan_windows():
    import duckdb

    bounds = [
        (label, datetime.fromisoformat(s).timestamp(), datetime.fromisoformat(e).timestamp())
        for label, s, e in WINDOWS
    ]
    min_ts = min(b[1] for b in bounds)
    max_ts = max(b[2] for b in bounds)

    files = sorted(
        f
        for f in os.listdir("/data")
        if f.endswith(".parquet") and f.startswith(FILE_PREFIX) and f < FILE_CUTOFF + ".parquet"
    )
    print(f"[scan] {len(files)} parquet files to scan", flush=True)

    # rows per window label
    buckets: dict[str, list[tuple]] = {label: [] for label, _, _ in WINDOWS}

    t0 = time.perf_counter()
    con = duckdb.connect()
    for i, filename in enumerate(files):
        path = os.path.join("/data", filename)
        cursor = con.execute(
            """
            SELECT timestamp, request_metadata.token_hash AS token_hash, request, response
            FROM read_parquet(?)
            WHERE request NOT LIKE '%keep-alive%'
            """,
            [path],
        )
        while True:
            chunk = cursor.fetchmany(4096)
            if not chunk:
                break
            for ts_raw, token_hash, request_raw, response_raw in chunk:
                if isinstance(ts_raw, datetime):
                    ts = ts_raw
                elif isinstance(ts_raw, str):
                    try:
                        ts = datetime.fromisoformat(ts_raw[:26])
                    except ValueError:
                        continue
                else:
                    continue
                ts_epoch = ts.timestamp()
                if ts_epoch < min_ts or ts_epoch >= max_ts:
                    continue

                # which window?
                label = None
                for lbl, ws, we in bounds:
                    if ws <= ts_epoch < we:
                        label = lbl
                        break
                if label is None:
                    continue

                msg_count = 0
                try:
                    req = json.loads(request_raw) if isinstance(request_raw, str) else request_raw
                    if isinstance(req, dict):
                        msgs = req.get("messages", [])
                        if isinstance(msgs, list):
                            msg_count = len(msgs)
                except (json.JSONDecodeError, TypeError):
                    continue
                if msg_count == 0:
                    continue

                est_tokens = 0
                try:
                    resp = (
                        json.loads(response_raw) if isinstance(response_raw, str) else response_raw
                    )
                    if isinstance(resp, dict):
                        usage = resp.get("usage")
                        if isinstance(usage, dict):
                            pt = usage.get("prompt_tokens")
                            if isinstance(pt, int):
                                est_tokens = pt
                except (json.JSONDecodeError, TypeError):
                    pass

                buckets[label].append((ts_epoch, token_hash or "", msg_count, est_tokens))

        if (i + 1) % 20 == 0 or i == len(files) - 1:
            print(f"[scan] {i + 1}/{len(files)} files, {time.perf_counter() - t0:.0f}s", flush=True)
    con.close()

    out = []
    for label, _, _ in WINDOWS:
        rows = buckets[label]
        stats = _score_window(rows) if rows else {"n_requests": 0}
        out.append({"window": label, **stats})

    print(f"\n{'=' * 100}")
    print("WINDOW DIVERSITY / LOAD STATS")
    print(f"{'=' * 100}")
    for s in out:
        print(json.dumps(s))
    return out


@app.local_entrypoint()
def main():
    results = scan_windows.remote()
    print("\n\nFinal results:")
    print(json.dumps(results, indent=2, default=str))
