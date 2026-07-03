"""Per-window stats over the decoded *replay* files actually benchmarked.

Reads the three decoded Mooncake files used in the Table~\\ref{tab:results}
results (apr5 tuning, apr6 eval, apr7 eval) and reports, per window: request
count, distinct users, input-token length distribution, and token-weighted
256-token block-level prefix reuse (global / intra-user), with identical block
keying to ``week_reuse_stats.py``.

Usage::

    modal run --env=alessio-dev data_processing/window_replay_stats.py::main
"""

from __future__ import annotations

import json

import modal

from app import app, completions_volume

image = modal.Image.debian_slim().add_local_python_source("app")

FILES = {
    "apr5": "/data/mooncake_traces/decoded/glm5_decoded_apr5_1615_to_1645.jsonl",
    "apr6": "/data/mooncake_traces/decoded/glm5_decoded_apr6_1505_to_1535.jsonl",
    "apr7": "/data/mooncake_traces/decoded/glm5_decoded_apr7_1945_to_2015.jsonl",
}
BLOCK_SIZE = 256


@app.function(image=image, memory=1024 * 16, timeout=3600, volumes={"/data": completions_volume})
def window_stats(block_size: int = BLOCK_SIZE) -> dict:
    import statistics

    out = {}
    for tag, path in FILES.items():
        global_seen: set[int] = set()
        intra_seen: dict[str, set[int]] = {}
        users: set[str] = set()
        n = 0
        total_tok = 0
        g_saved = 0
        i_saved = 0
        lengths: list[int] = []
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                e = json.loads(line)
                inp = e["input_length"]
                hids = e["hash_ids"]
                u = e.get("token_hash", "")
                n += 1
                total_tok += inp
                lengths.append(inp)
                users.add(u)
                nb = len(hids)
                us = intra_seen.setdefault(u, set())
                for bi, d in enumerate(hids):
                    bt = block_size if bi < nb - 1 else inp - block_size * (nb - 1)
                    key = int(d[:16], 16) if isinstance(d, str) else int(d)
                    if key in global_seen:
                        g_saved += bt
                    else:
                        global_seen.add(key)
                    if key in us:
                        i_saved += bt
                    else:
                        us.add(key)
        lengths.sort()
        out[tag] = {
            "path": path,
            "requests": n,
            "users": len(users),
            "requests_per_user": round(n / len(users), 1) if users else 0,
            "avg_input_tokens": round(total_tok / n, 1) if n else 0,
            "median_input_tokens": lengths[len(lengths) // 2] if lengths else 0,
            "p95_input_tokens": lengths[int(0.95 * (len(lengths) - 1))] if lengths else 0,
            "block_global_reuse_pct": round(100 * g_saved / total_tok, 2) if total_tok else 0,
            "block_intra_user_reuse_pct": round(100 * i_saved / total_tok, 2) if total_tok else 0,
        }
        print(f"[{tag}] " + json.dumps(out[tag]), flush=True)
    return out


@app.local_entrypoint()
def main(out_path: str = "results/decoded_v9/window_replay_stats.json"):
    import os

    result = window_stats.remote()
    print(json.dumps(result, indent=2))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {out_path}")
