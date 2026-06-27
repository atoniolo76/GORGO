"""Generate schema-matching SAMPLE tokenized parquets with MULTIPLE conversations
per user and KNOWN cross-conversation overlap ground truth, for local E2E testing
of ``analyze_user_reuse`` -- NO Modal, NO real GLM-5.1 data.

Schema matches build_eval_dataset.py:245-253 (one row per conversation):
  session_id (string), token_hash (string), message_count (int32),
  prompt_token_count (int32), prompt_ids (list<uint32>)

Embedded users (user = token_hash; conversation = session_id):
  userP_{0,1,2}   5 convs: share SYS prefix + TOOL middle, unique query+tail
                  -> prefix reuse AND content (middle) reuse; content > prefix
  userPrefixOnly  4 convs: share SYS prefix only (no middle)
                  -> content == prefix (no middle gap)
  userMiddleOnly  4 convs: unique head, share TOOL middle (no shared prefix)
                  -> prefix ~ 0, content > 0, big content-prefix gap, warm_prefix ~ 0
  userSingle      1 conv -> zero cross-conversation reuse (everything)
  userWhale      20 long convs: big shared SYS + TOOL -> dominates tokens (whale tier)
  userLight_{0..4} 2 short convs: small shared SYS only -> light tier, low reuse

All shared segments are reused BY REFERENCE within a user (identical token ids) and
are block-aligned (offsets/lengths multiples of 1024 for the main users) so content
blocks match at every swept size; all non-shared content is globally unique (drawn
from a wide range -> no accidental window/block collisions). Deterministic seed.

Run standalone:  python3 data_processing/tests/make_sample_user_reuse_data.py <out_root>
"""

from __future__ import annotations

import hashlib
import os
import random
import sys

import pyarrow as pa
import pyarrow.parquet as pq

FILE_PREFIX = "llm_responses_202604"  # mirrors build_eval_dataset.FILE_PREFIX
LO, HI = 1, 2_000_000  # wide id range so distinct content never collides by chance
NUM_FILES = 4


def generate(out_root: str, seed: int = 7) -> dict:
    rng = random.Random(seed)

    def blk(n: int) -> list[int]:
        return [rng.randint(LO, HI - 1) for _ in range(n)]

    sessions: list[tuple[str, list[int], str]] = []  # (token_hash, prompt_ids, cluster)

    def add(token_hash: str, ids: list[int], cluster: str):
        sessions.append((token_hash, ids, cluster))

    # userP_0/1/2 : shared SYS prefix (1024) + unique query (1024) + shared TOOL
    #               middle (1024) + unique tail (1024). TOOL offset 2048 (aligned).
    for u in range(3):
        sys_u, tool_u = blk(1024), blk(1024)
        for _ in range(5):
            add(f"userP_{u}", list(sys_u) + blk(1024) + list(tool_u) + blk(1024), "P")

    # userPrefixOnly : shared SYS (1024) + unique query (1024) + unique tail (1024).
    sys_po = blk(1024)
    for _ in range(4):
        add("userPrefixOnly", list(sys_po) + blk(1024) + blk(1024), "PREFIX_ONLY")

    # userMiddleOnly : unique head (1024) + shared TOOL (1024) + unique tail (1024).
    tool_mo = blk(1024)
    for _ in range(4):
        add("userMiddleOnly", blk(1024) + list(tool_mo) + blk(1024), "MIDDLE_ONLY")

    # userSingle : exactly one conversation -> no cross-conversation reuse.
    add("userSingle", blk(1024) + blk(1024) + blk(1024), "SINGLE")

    # userWhale : 20 long convs sharing a big SYS (2048) + TOOL (2048), TOOL offset
    #             3072 (aligned to all sizes). Dominates total tokens (whale tier).
    sys_w, tool_w = blk(2048), blk(2048)
    for _ in range(20):
        add("userWhale", list(sys_w) + blk(1024) + list(tool_w) + blk(2048), "WHALE")

    # userLight_{0..4} : 2 short convs sharing a small SYS (256) only.
    for u in range(5):
        sys_l = blk(256)
        for _ in range(2):
            add(f"userLight_{u}", list(sys_l) + blk(256) + blk(256), "LIGHT")

    # Distribute across files round-robin (order-independent metric, so any layout ok)
    buckets: list[list[tuple[int, str, list[int], str]]] = [[] for _ in range(NUM_FILES)]
    for idx, (th, ids, cl) in enumerate(sessions):
        buckets[idx % NUM_FILES].append((idx, th, ids, cl))

    tok_dir = os.path.join(out_root, f"tokenized_{FILE_PREFIX}")
    os.makedirs(tok_dir, exist_ok=True)
    written = []
    for fi, rows in enumerate(buckets):
        table = pa.table(
            {
                "session_id": pa.array(
                    [hashlib.sha256(str(idx).encode()).hexdigest() for idx, *_ in rows], pa.string()
                ),
                "token_hash": pa.array([th for _, th, _, _ in rows], pa.string()),
                "message_count": pa.array([1] * len(rows), pa.int32()),
                "prompt_token_count": pa.array([len(ids) for _, _, ids, _ in rows], pa.int32()),
                "prompt_ids": pa.array([ids for _, _, ids, _ in rows], pa.list_(pa.uint32())),
            }
        )
        path = os.path.join(tok_dir, f"{FILE_PREFIX}{fi + 1:02d}_000000.tokenized.parquet")
        pq.write_table(table, path, compression="zstd")
        written.append(path)

    # Per-user ground-truth rollup
    users: dict[str, dict] = {}
    for th, ids, cl in sessions:
        u = users.setdefault(
            th, {"token_hash": th, "cluster": cl, "num_conversations": 0, "total_tokens": 0}
        )
        u["num_conversations"] += 1
        u["total_tokens"] += len(ids)

    return {
        "out_root": out_root,
        "tok_dir": tok_dir,
        "files": written,
        "num_files": NUM_FILES,
        "num_sessions": len(sessions),
        "num_users": len(users),
        "users": users,
        "ground_truth": {
            "prefix_and_middle_users": ["userP_0", "userP_1", "userP_2"],
            "prefix_only_user": "userPrefixOnly",
            "middle_only_user": "userMiddleOnly",
            "single_conversation_user": "userSingle",
            "whale_user": "userWhale",
            "light_users": [f"userLight_{u}" for u in range(5)],
            "block_aligned_block_sizes": [16, 64, 256, 512, 1024],
        },
    }


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.getcwd(), "sample_user_reuse")
    m = generate(out)
    print(f"wrote {m['num_sessions']} conversations / {m['num_users']} users -> {m['tok_dir']}")
    for u in sorted(m["users"].values(), key=lambda x: -x["total_tokens"]):
        print(
            f"  {u['token_hash']:>16}  convs={u['num_conversations']:>2}  tokens={u['total_tokens']:>7}  [{u['cluster']}]"
        )
