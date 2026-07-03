"""Average turns-per-conversation for the apr5/apr6/apr7 candidate windows.

Each production request carries the *full* message history up to that point, so a
conversation's depth equals the number of user-role messages in its deepest
request. There's no conversation id in the raw data, so we group requests into
conversations by ``(token_hash, sha1(opening user message))`` and take the max
user-turn depth seen per conversation, then average over conversations.

Scans only the Apr 5-7 parquet shards (by filename date) for speed.

Usage::

    modal run --env=alessio-dev data_processing/window_turn_stats.py::main
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import defaultdict
from datetime import datetime

import modal

from app import app, completions_volume

image = modal.Image.debian_slim().pip_install("duckdb").add_local_python_source("app")

FILE_PREFIX = "llm_responses_202604"
# Only the shards covering the three windows (UTC dates).
DATE_TAGS = ("20260405", "20260406", "20260407")

WINDOWS = [
    ("apr5 16:15-16:45 (diverse day)", "2026-04-05T16:15:00", "2026-04-05T16:45:00"),
    ("apr6 15:05-15:35 (eval)", "2026-04-06T15:05:00", "2026-04-06T15:35:00"),
    ("apr7 19:45-20:15 (eval)", "2026-04-07T19:45:00", "2026-04-07T20:15:00"),
]

OUT_PATH = "/data/window_turn_stats.json"


def _prompt_tokens(response_raw):
    """usage.prompt_tokens from a plain-JSON or SSE-streaming response (0 if absent)."""
    if not isinstance(response_raw, str):
        response_raw = "" if response_raw is None else str(response_raw)
    try:
        resp = json.loads(response_raw)
        if isinstance(resp, dict):
            usage = resp.get("usage")
            if isinstance(usage, dict) and isinstance(usage.get("prompt_tokens"), int):
                return usage["prompt_tokens"]
    except (json.JSONDecodeError, TypeError):
        pass
    best = 0
    for line in response_raw.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        usage = obj.get("usage") if isinstance(obj, dict) else None
        if isinstance(usage, dict) and isinstance(usage.get("prompt_tokens"), int):
            best = usage["prompt_tokens"]
    return best


def _content_to_str(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content) if content is not None else ""


def _parse_request(request_raw):
    """Return (n_messages, n_user_turns, opening_user_hash) or None."""
    try:
        req = json.loads(request_raw) if isinstance(request_raw, str) else request_raw
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(req, dict):
        return None
    msgs = req.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return None
    user_msgs = [m for m in msgs if isinstance(m, dict) and m.get("role") == "user"]
    n_user = len(user_msgs)
    if n_user == 0:
        return None
    opening = _content_to_str(user_msgs[0].get("content"))
    opening_hash = hashlib.sha1(opening.encode("utf-8", "ignore")).hexdigest()[:16]
    return len(msgs), n_user, opening_hash


def _summarize(turns_by_key, tokens_by_key, n_requests, total_user_turns):
    """turns_by_key/tokens_by_key: dict[key] -> max user turns / max prompt_tokens."""
    if not turns_by_key:
        return {"n_requests": n_requests, "n_conversations": 0}
    keys = list(turns_by_key)
    turns = sorted(turns_by_key.values())
    n_conv = len(turns)
    avg_turns = sum(turns) / n_conv
    median_turns = turns[n_conv // 2]
    p90_turns = turns[min(n_conv - 1, int(0.9 * n_conv))]
    multi_turn_convos = sum(1 for t in turns if t > 1)

    conv_lengths = [tokens_by_key[k] for k in keys]
    avg_conv_len = sum(conv_lengths) / n_conv
    # mean over conversations of (full-conversation tokens / turns)
    per_turn_ratios = [tokens_by_key[k] / turns_by_key[k] for k in keys if turns_by_key[k] > 0]
    mean_of_ratios = sum(per_turn_ratios) / len(per_turn_ratios) if per_turn_ratios else 0

    return {
        "n_requests": n_requests,
        "n_conversations": n_conv,
        "avg_turns_per_conversation": round(avg_turns, 2),
        "median_turns_per_conversation": median_turns,
        "p90_turns_per_conversation": p90_turns,
        "max_turns": turns[-1],
        "multi_turn_conversation_pct": round(100 * multi_turn_convos / n_conv, 1),
        "avg_user_turns_per_request": round(total_user_turns / n_requests, 2) if n_requests else 0,
        "avg_conversation_length_tokens": round(avg_conv_len),
        # your requested approximation: ratio of the two means
        "avg_prompt_tokens_per_turn_ratio_of_means": round(avg_conv_len / avg_turns, 1)
        if avg_turns
        else 0,
        # more robust: per-conversation ratio, then averaged
        "avg_prompt_tokens_per_turn_mean_of_ratios": round(mean_of_ratios, 1),
    }


@app.function(
    image=image,
    memory=1024 * 16,
    timeout=14400,
    volumes={"/data": completions_volume},
)
def scan_turns():
    import duckdb

    bounds = [
        (label, datetime.fromisoformat(s).timestamp(), datetime.fromisoformat(e).timestamp())
        for label, s, e in WINDOWS
    ]
    min_ts = min(b[1] for b in bounds)
    max_ts = max(b[2] for b in bounds)
    min_iso = datetime.utcfromtimestamp(min_ts).isoformat()
    max_iso = datetime.utcfromtimestamp(max_ts).isoformat()

    files = sorted(
        os.path.join("/data", f)
        for f in os.listdir("/data")
        if f.endswith(".parquet")
        and f.startswith(FILE_PREFIX)
        and any(tag in f for tag in DATE_TAGS)
    )
    print(f"[scan] {len(files)} parquet shards; span {min_iso}..{max_iso}", flush=True)

    # per-window: conversation key -> max user turns / max prompt_tokens; plus counters
    turns_by_key: dict[str, dict[tuple, int]] = {label: defaultdict(int) for label, _, _ in WINDOWS}
    tokens_by_key: dict[str, dict[tuple, int]] = {
        label: defaultdict(int) for label, _, _ in WINDOWS
    }
    n_requests = {label: 0 for label, _, _ in WINDOWS}
    total_user_turns = {label: 0 for label, _, _ in WINDOWS}

    t0 = time.perf_counter()
    con = duckdb.connect()
    cursor = con.execute(
        """
        SELECT
            epoch(CAST(timestamp AS TIMESTAMP)) AS ts_epoch,
            request_metadata.token_hash AS token_hash,
            request,
            response
        FROM read_parquet(?)
        WHERE request NOT LIKE '%keep-alive%'
          AND CAST(timestamp AS TIMESTAMP) >= CAST(? AS TIMESTAMP)
          AND CAST(timestamp AS TIMESTAMP) <  CAST(? AS TIMESTAMP)
        """,
        [files, min_iso, max_iso],
    )

    while True:
        chunk = cursor.fetchmany(8192)
        if not chunk:
            break
        for ts_epoch, token_hash, request_raw, response_raw in chunk:
            if ts_epoch is None:
                continue
            label = None
            for lbl, ws, we in bounds:
                if ws <= ts_epoch < we:
                    label = lbl
                    break
            if label is None:
                continue
            parsed = _parse_request(request_raw)
            if parsed is None:
                continue
            _, n_user, opening_hash = parsed
            key = (token_hash or "", opening_hash)
            td = turns_by_key[label]
            if n_user > td[key]:
                td[key] = n_user
            pt = _prompt_tokens(response_raw)
            kd = tokens_by_key[label]
            if pt > kd[key]:
                kd[key] = pt
            n_requests[label] += 1
            total_user_turns[label] += n_user
    con.close()
    print(f"[scan] done in {time.perf_counter() - t0:.0f}s", flush=True)

    out = []
    for label, _, _ in WINDOWS:
        out.append(
            {
                "window": label,
                **_summarize(
                    turns_by_key[label],
                    tokens_by_key[label],
                    n_requests[label],
                    total_user_turns[label],
                ),
            }
        )

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    completions_volume.commit()

    print(f"\n{'=' * 90}")
    print("TURNS-PER-CONVERSATION")
    print(f"{'=' * 90}")
    for s in out:
        print(json.dumps(s))
    print(f"\n[scan] wrote {OUT_PATH}", flush=True)
    return out


@app.local_entrypoint()
def main():
    results = scan_turns.remote()
    print("\n\nFinal results:")
    print(json.dumps(results, indent=2, default=str))
