"""Generate schema-matching SAMPLE tokenized parquets with KNOWN overlap ground
truth, for local E2E testing of ``analyze_overlap_structure`` -- NO Modal, NO
real GLM-5.1 data.

Schema matches ``build_eval_dataset.py:245-253`` exactly:
  session_id (string), token_hash (string), message_count (int32),
  prompt_token_count (int32), prompt_ids (list<uint32>)
One row = one session's longest-conversation prompt token stream.

Embedded clusters (all present simultaneously so AGGREGATE metrics are meaningful):
  A  shared long PREFIX then divergence      -> prefix reuse; content == chained
  B  divergent prefix, identical MIDDLE chunk -> content reuse > chained reuse
  C  many small repeated segments scattered   -> small-block reuse >> large-block
  D  "whale" token_hash, EXACT-duplicate prompts -> exercises whole-prompt dedup
  E  genuinely unique sessions (varied lengths, some < window) -> noise floor

Token-id bands are disjoint so ground truth is crisp: SHARED structural content
in [0, 50k); per-session UNIQUE content drawn from [50k, 200k) (a 64-token
window/block of random ids never collides by chance). Deterministic seed.

Run standalone to write a sample set:
  python3 data_processing/tests/make_sample_overlap_data.py <out_root>
"""

from __future__ import annotations

import hashlib
import os
import random
import sys

import pyarrow as pa
import pyarrow.parquet as pq

# Keep this module Modal-free; mirrors build_eval_dataset.FILE_PREFIX.
FILE_PREFIX = "llm_responses_202604"

SHARED_LO, SHARED_HI = 0, 50_000
UNIQUE_LO, UNIQUE_HI = 50_000, 200_000
NUM_FILES = 4

P_A_LEN = 2048  # cluster A shared prefix (block-aligned to 64/256/512/1024)
M_B_LEN = 1024  # cluster B shared middle (block-aligned)
MOTIF_LEN = 64  # cluster C small segment (== default n-gram window)
NUM_MOTIFS = 6
WHALE_DUPES = 30


def generate(out_root: str, seed: int = 1234) -> dict:
    """Write the sample parquets under ``<out_root>/tokenized_<FILE_PREFIX>/`` and
    return a manifest describing the embedded ground truth."""
    srng = random.Random(seed)  # shared (fixed) structural content
    P_A = [srng.randint(SHARED_LO, SHARED_HI - 1) for _ in range(P_A_LEN)]
    M_B = [srng.randint(SHARED_LO, SHARED_HI - 1) for _ in range(M_B_LEN)]
    MOTIFS = [
        [srng.randint(SHARED_LO, SHARED_HI - 1) for _ in range(MOTIF_LEN)]
        for _ in range(NUM_MOTIFS)
    ]

    urng = random.Random(seed + 1)  # per-session unique content

    def uniq(n: int) -> list[int]:
        return [urng.randint(UNIQUE_LO, UNIQUE_HI - 1) for _ in range(n)]

    sessions: list[tuple[str, list[int], str]] = []  # (token_hash, prompt_ids, cluster)

    # A: shared 2048-token prefix, then a divergent unique tail. 8 users x 5.
    tail_lens = [512, 1024, 1536, 2048, 3072]
    for i in range(40):
        ids = list(P_A) + uniq(tail_lens[i % len(tail_lens)])
        sessions.append((f"userA_{i % 8}", ids, "A"))

    # B: divergent (block-aligned) unique prefix, identical 1024-token MIDDLE,
    #    then a unique suffix. 10 users x 4. M_B starts at a multiple of 1024.
    pre_lens = [1024, 2048]
    suf_lens = [512, 1024]
    for i in range(40):
        ids = uniq(pre_lens[i % 2]) + list(M_B) + uniq(suf_lens[(i // 2) % 2])
        sessions.append((f"userB_{i % 10}", ids, "B"))

    # C: alternating 64-token unique filler and a shared 64-token MOTIF, x8.
    #    All segments 64-aligned; motifs recur across sessions. 10 users.
    for i in range(40):
        ids: list[int] = []
        for k in range(8):
            ids += uniq(MOTIF_LEN)
            ids += list(MOTIFS[(i + k) % NUM_MOTIFS])
        sessions.append((f"userC_{i % 10}", ids, "C"))

    # D: a whale token_hash submitting the SAME prompt many times (exact dupes).
    W = uniq(1500)
    for _ in range(WHALE_DUPES):
        sessions.append(("whale", list(W), "D"))

    # E: genuinely unique sessions, varied lengths (incl. some < default window).
    e_lens = [40, 60, 128, 256, 512, 1024, 2048, 4096]
    for i in range(30):
        sessions.append((f"userE_{i}", uniq(e_lens[i % len(e_lens)]), "E"))

    # Distribute across files round-robin, preserving global order within a file
    # (so whole-prompt dedup's first-occurrence rule is deterministic).
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

    return {
        "out_root": out_root,
        "tok_dir": tok_dir,
        "files": written,
        "num_files": NUM_FILES,
        "num_sessions": len(sessions),
        "sessions": [
            {"token_hash": th, "length": len(ids), "cluster": cl} for th, ids, cl in sessions
        ],
        "block_aligned_block_sizes": [64, 256, 512, 1024],
        "ground_truth": {
            "P_A_len": P_A_LEN,
            "M_B_len": M_B_LEN,
            "motif_len": MOTIF_LEN,
            "num_motifs": NUM_MOTIFS,
            "whale_dupes": WHALE_DUPES,
            "expected_whole_prompt_dupes_dropped": WHALE_DUPES - 1,  # 29 (1 kept)
        },
    }


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.getcwd(), "sample_glm")
    m = generate(out)
    print(f"wrote {m['num_sessions']} sessions across {m['num_files']} files -> {m['tok_dir']}")
    for p in m["files"]:
        print(f"  {p}")
