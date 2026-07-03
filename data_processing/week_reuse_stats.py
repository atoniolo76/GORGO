"""Block-level reuse stats over the full-week metadata.

Token-level (radix-trie) reuse for the corpus is already computed in
``/data/prefix_trie_stats_llm_responses_202604.json`` (session-deduplicated:
53.7% intra / 55.3% global). This script computes the *block-level* reuse that
the KV cache actually keys on, over the request-row week metadata
(``/data/mooncake_traces/metadata_week``), split into intra-user vs cross-user,
token-weighted (256-token blocks, partial last block).

Single global pass in timestamp order (shards are name-sorted = time-sorted,
each shard internally sorted) so "saved = block seen earlier" is well defined.
Digests are keyed as 64-bit ints to keep the seen-sets compact.

Usage::

    modal run --env=alessio-dev data_processing/week_reuse_stats.py::main
"""

from __future__ import annotations

import json

import modal

from app import app, completions_volume

image = modal.Image.debian_slim().add_local_python_source("app")

METADATA_DIR = "/data/mooncake_traces/metadata_week"
BLOCK_SIZE = 256


@app.function(image=image, memory=1024 * 64, timeout=14400, volumes={"/data": completions_volume})
def block_stats(input_dir: str = METADATA_DIR, block_size: int = BLOCK_SIZE) -> dict:
    import os
    import time
    from collections import defaultdict

    files = sorted(f for f in os.listdir(input_dir) if f.endswith(".jsonl"))
    print(f"[block_stats] {len(files)} shards")

    global_seen: set[int] = set()
    intra_seen: dict[str, set[int]] = defaultdict(set)
    user_tokens: dict[str, int] = defaultdict(int)
    users: set[str] = set()
    n_req = 0
    total_tok = 0
    total_blocks = 0
    g_saved = 0
    i_saved = 0
    t0 = time.time()

    for fi, fname in enumerate(files, start=1):
        with open(os.path.join(input_dir, fname)) as f:
            for line in f:
                if not line.strip():
                    continue
                e = json.loads(line)
                inp = e["input_length"]
                hids = e["hash_ids"]
                u = e.get("token_hash", "")
                n_req += 1
                total_tok += inp
                users.add(u)
                user_tokens[u] += inp
                n = len(hids)
                total_blocks += n
                us = intra_seen[u]
                for bi, d in enumerate(hids):
                    bt = block_size if bi < n - 1 else inp - block_size * (n - 1)
                    key = int(d[:16], 16)  # 64-bit key
                    if key in global_seen:
                        g_saved += bt
                    else:
                        global_seen.add(key)
                    if key in us:
                        i_saved += bt
                    else:
                        us.add(key)
        if fi % 40 == 0 or fi == len(files):
            print(
                f"  {fi}/{len(files)} | {n_req:,} req | "
                f"{len(global_seen):,} unique blocks | {time.time() - t0:.0f}s",
                flush=True,
            )

    top10 = sorted(user_tokens.values(), reverse=True)[:10]
    out = {
        "requests": n_req,
        "users": len(users),
        "total_input_tokens": total_tok,
        "avg_input_tokens": round(total_tok / n_req, 1) if n_req else 0,
        "requests_per_user": round(n_req / len(users), 1) if users else 0,
        "top10_user_concentration_pct": round(100 * sum(top10) / total_tok, 2) if total_tok else 0,
        "total_blocks": total_blocks,
        "unique_blocks": len(global_seen),
        "block_global_reuse_pct": round(100 * g_saved / total_tok, 2) if total_tok else 0,
        "block_intra_user_reuse_pct": round(100 * i_saved / total_tok, 2) if total_tok else 0,
        "block_cross_user_reuse_pct": round(100 * (g_saved - i_saved) / total_tok, 2)
        if total_tok
        else 0,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    print("[block_stats] RESULT " + json.dumps(out), flush=True)
    return out


@app.local_entrypoint()
def main(out_path: str = "results/decoded_v9/week_reuse_stats.json"):
    import os

    result = block_stats.remote()
    print(json.dumps(result, indent=2))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {out_path}")
