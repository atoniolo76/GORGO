"""Find the 30-minute window with the highest user diversity and multi-turn
density across the 7-day GLM-5.1 trace.

Scans raw parquets extracting only lightweight metadata (no tokenization):
timestamp, token_hash, message count, and prompt_tokens from usage stats.

Usage::

    modal run --env=alessio-dev data_processing/find_best_window.py::main
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import Counter
from datetime import datetime, timedelta

import modal

from app import app, completions_volume

image = modal.Image.debian_slim().pip_install("duckdb").add_local_python_source("app")

FILE_PREFIX = "llm_responses_202604"
FILE_CUTOFF = "llm_responses_20260408"


def _content_to_str(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return ""


@app.function(
    image=image,
    memory=1024 * 32,
    timeout=14400,
    volumes={"/data": completions_volume},
)
def scan_all_parquets():
    import duckdb

    files = sorted(
        f
        for f in os.listdir("/data")
        if f.endswith(".parquet") and f.startswith(FILE_PREFIX) and f < FILE_CUTOFF + ".parquet"
    )
    print(f"[scan] {len(files)} parquet files to scan")

    t0 = time.perf_counter()
    all_rows: list[tuple] = []  # (timestamp_s, token_hash, msg_count, est_tokens)

    con = duckdb.connect()
    for i, filename in enumerate(files):
        path = os.path.join("/data", filename)
        cursor = con.execute(
            """
            SELECT
                timestamp,
                request_metadata.token_hash AS token_hash,
                request,
                response
            FROM read_parquet(?)
            WHERE request NOT LIKE '%keep-alive%'
            ORDER BY timestamp
            """,
            [path],
        )
        file_count = 0
        while True:
            chunk = cursor.fetchmany(4096)
            if not chunk:
                break
            for ts_raw, token_hash, request_raw, response_raw in chunk:
                # Parse timestamp
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

                # Count messages (no tokenization)
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

                # Get prompt_tokens from response usage (no tokenization needed)
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

                all_rows.append((ts_epoch, token_hash or "", msg_count, est_tokens))
                file_count += 1

        if (i + 1) % 20 == 0 or i == len(files) - 1:
            elapsed = time.perf_counter() - t0
            print(
                f"[scan] {i + 1}/{len(files)} files, {len(all_rows):,} rows, {elapsed:.0f}s",
                flush=True,
            )
    con.close()

    print(f"\n[scan] total: {len(all_rows):,} rows in {time.perf_counter() - t0:.0f}s")

    # Sort by timestamp
    all_rows.sort(key=lambda r: r[0])

    # Slide 30-min window at 5-min increments
    window_sec = 30 * 60
    step_sec = 5 * 60

    if not all_rows:
        print("[scan] no rows found")
        return []

    min_ts = all_rows[0][0]
    max_ts = all_rows[-1][0]

    results = []
    window_start = min_ts

    while window_start + window_sec <= max_ts:
        window_end = window_start + window_sec

        # Binary search for window bounds
        lo = _bisect_left(all_rows, window_start)
        hi = _bisect_left(all_rows, window_end)
        window_rows = all_rows[lo:hi]

        if len(window_rows) >= 10:
            score, stats = _score_window(window_rows, window_start, window_end)
            results.append((score, stats))

        window_start += step_sec

    # Sort by score descending
    results.sort(key=lambda r: -r[0])

    print(f"\n{'=' * 80}")
    print(f"TOP 20 WINDOWS (of {len(results)} scored)")
    print(f"{'=' * 80}")
    for rank, (score, stats) in enumerate(results[:20]):
        start_dt = datetime.utcfromtimestamp(stats["start_ts"])
        end_dt = datetime.utcfromtimestamp(stats["end_ts"])
        print(
            f"\n#{rank + 1} score={score:.1f} | "
            f"{start_dt.strftime('%Y-%m-%d %H:%M')}–{end_dt.strftime('%H:%M')} UTC"
        )
        print(
            f"  requests={stats['n_requests']:,}  users={stats['n_users']}  "
            f"multi_turn={stats['multi_turn_pct']:.1f}%  "
            f"avg_tokens={stats['avg_tokens']:.0f}  "
            f"top_user_share={stats['top_user_share']:.1f}%  "
            f"median_tokens={stats['median_tokens']:.0f}"
        )

    return [{"rank": i + 1, "score": s, **st} for i, (s, st) in enumerate(results[:20])]


def _bisect_left(rows, target_ts):
    lo, hi = 0, len(rows)
    while lo < hi:
        mid = (lo + hi) // 2
        if rows[mid][0] < target_ts:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _score_window(rows, start_ts, end_ts):
    n = len(rows)
    users = Counter(r[1] for r in rows)
    n_users = len(users)
    top_user_count = users.most_common(1)[0][1] if users else 0
    top_user_share = top_user_count / n if n > 0 else 0

    multi_turn = sum(1 for r in rows if r[2] > 2)
    multi_turn_pct = 100 * multi_turn / n if n > 0 else 0

    tokens = [r[3] for r in rows if r[3] > 0]
    avg_tokens = sum(tokens) / len(tokens) if tokens else 0
    median_tokens = sorted(tokens)[len(tokens) // 2] if tokens else 0

    # Score: reward diversity, multi-turn, long prompts; penalize single-user dominance
    # log(users) to not over-reward user count vs other factors
    score = (
        math.log(max(n_users, 1))
        * (multi_turn_pct / 100 + 0.1)  # multi-turn fraction (with floor)
        * math.log(max(avg_tokens, 1))
        * (1 - top_user_share)
        * math.sqrt(n)  # mild reward for more requests
    )

    stats = {
        "start_ts": start_ts,
        "end_ts": end_ts,
        "n_requests": n,
        "n_users": n_users,
        "multi_turn_pct": multi_turn_pct,
        "avg_tokens": avg_tokens,
        "median_tokens": median_tokens,
        "top_user_share": top_user_share * 100,
        "top_user_count": top_user_count,
    }
    return score, stats


@app.local_entrypoint()
def main():
    results = scan_all_parquets.remote()
    print("\n\nFinal results:")
    print(json.dumps(results, indent=2, default=str))
