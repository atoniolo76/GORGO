"""Local unit tests for ``overlap_metrics`` -- runnable with plain pytest, NO Modal.

Each test feeds hand-constructed token-id arrays with KNOWN overlap structure
and asserts the metrics come out exactly as reasoned. The whole point is to
de-risk the (data-gated) Modal run without the GLM-5.1 data.

Token-value convention: "distinct" filler uses large, per-(seq,slot)-unique
values (>= 1_000_000) so distinct content NEVER accidentally collides with
other distinct content or with the small-valued SHARED motifs.

Run:  python3 -m pytest data_processing/tests/test_overlap_structure.py -q
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from overlap_metrics import (  # noqa: E402
    block_reuse_stats,
    content_block_digests,
    dedup_whole_prompts,
    positional_collision_profile,
    prefix_chained_block_digests,
    segment_length_histogram,
)

SHARED_A = [1, 2, 3, 4]
SHARED_B = [5, 6, 7, 8]
MOTIF = [11, 12, 13, 14]


def distinct(seq_idx, slot, n):
    """Deterministic, globally-unique filler of length n."""
    base = 1_000_000 + seq_idx * 100_000 + slot * 1_000
    return list(range(base, base + n))


# --------------------------------------------------------------------------- #
# Digest primitives
# --------------------------------------------------------------------------- #
def test_content_digest_is_position_independent():
    # Same 4-token block in two different positions -> same content digest.
    a = content_block_digests([0, 0, 0, 0] + SHARED_A, block_size=4)
    b = content_block_digests(SHARED_A + [9, 9, 9, 9], block_size=4)
    assert a[1] == b[0]  # the SHARED_A block matches regardless of offset


def test_chained_digest_depends_on_prefix():
    # Identical block, DIFFERENT preceding block -> chained digests differ,
    # but content digests are identical. This is the core prefix-vs-content
    # distinction the whole investigation hinges on.
    s1 = distinct(0, 0, 4) + SHARED_A
    s2 = distinct(1, 0, 4) + SHARED_A
    c1 = content_block_digests(s1, 4)
    c2 = content_block_digests(s2, 4)
    p1 = prefix_chained_block_digests(s1, 4)
    p2 = prefix_chained_block_digests(s2, 4)
    assert c1[1] == c2[1]  # SHARED_A: same content -> same content digest
    assert p1[1] != p2[1]  # SHARED_A: different prefix -> different chained digest


# --------------------------------------------------------------------------- #
# (a) Shared prefix then divergence
# --------------------------------------------------------------------------- #
def test_shared_prefix_content_equals_chained():
    # 3 sequences: identical 8-token prefix, then divergent 16-token suffixes.
    # For PURE prefix sharing, content-hashed and prefix-chained reuse MUST be
    # equal -- the prefix blocks are both content-identical and prefix-identical.
    prefix = SHARED_A + SHARED_B  # 2 blocks of size 4
    seqs = [prefix + distinct(i, 1, 16) for i in range(3)]
    content = block_reuse_stats(seqs, block_size=4, chained=False)
    chained = block_reuse_stats(seqs, block_size=4, chained=True)
    # 3 seqs * 6 blocks = 18 total; unique = 2 shared prefix + 12 distinct = 14.
    assert content["total_blocks"] == 18
    assert content["unique_blocks"] == 14
    assert content["block_reuse_pct"] == chained["block_reuse_pct"]
    assert content["block_reuse_pct"] > 0

    prof = positional_collision_profile(seqs, w=4, stride=1, n_buckets=4)
    head = prof["buckets"][0]["shared_pct"]
    tail = prof["buckets"][-1]["shared_pct"]
    assert head > 0  # overlap lives in the head
    assert tail == 0  # nothing shared in the tail

    hist = segment_length_histogram(seqs, w=4, stride=1)
    assert hist["on_prefix"]["runs"] >= 3  # one prefix-anchored run per seq
    assert hist["off_prefix"]["runs"] == 0  # no middle/suffix runs


# --------------------------------------------------------------------------- #
# (b) Identical MIDDLE chunk, divergent prefix  (few big middle chunks)
# --------------------------------------------------------------------------- #
def test_identical_middle_chunk_content_beats_chained():
    # distinct prefix (8) + SHARED middle (8, block-aligned) + distinct suffix (8).
    middle = SHARED_A + SHARED_B
    seqs = [distinct(i, 0, 8) + middle + distinct(i, 2, 8) for i in range(3)]

    content = block_reuse_stats(seqs, block_size=4, chained=False)
    chained = block_reuse_stats(seqs, block_size=4, chained=True)
    # Content: the 2 middle blocks repeat across 3 seqs -> reuse > 0.
    assert content["block_reuse_pct"] > 0
    assert content["unique_blocks"] == 14  # 6 prefix + 2 shared middle + 6 suffix
    # Chained: divergent prefix poisons every downstream block -> ZERO reuse.
    assert chained["block_reuse_pct"] == 0.0

    # Positional profile (alignment-robust): overlap concentrated in the MIDDLE.
    prof = positional_collision_profile(seqs, w=4, stride=1, n_buckets=4)
    head = prof["buckets"][0]["shared_pct"]
    tail = prof["buckets"][-1]["shared_pct"]
    mid = max(prof["buckets"][1]["shared_pct"], prof["buckets"][2]["shared_pct"])
    assert mid > 0
    assert head == 0.0
    assert tail == 0.0

    # Segment histogram: one OFF-prefix run per seq, length 8 (a big-ish chunk),
    # zero on-prefix runs.
    hist = segment_length_histogram(seqs, w=4, stride=1)
    assert hist["off_prefix"]["runs"] == 3
    assert hist["on_prefix"]["runs"] == 0
    assert hist["off_prefix"]["tokens"] == 24  # 3 runs * 8 tokens


# --------------------------------------------------------------------------- #
# (c) Many small repeated segments scattered through the body
# --------------------------------------------------------------------------- #
def test_many_small_segments_small_block_beats_large_block():
    # Layout per seq (offsets are multiples of 4 so MOTIF is block-aligned at 4):
    #   distinct(4) MOTIF distinct(4) MOTIF distinct(4) MOTIF distinct(4)
    # MOTIF (len 4) is identical across all seqs; appears 3x per seq.
    seqs = []
    for i in range(3):
        seqs.append(
            distinct(i, 0, 4)
            + MOTIF
            + distinct(i, 1, 4)
            + MOTIF
            + distinct(i, 2, 4)
            + MOTIF
            + distinct(i, 3, 4)
        )

    small = block_reuse_stats(seqs, block_size=4, chained=False)
    large = block_reuse_stats(seqs, block_size=16, chained=False)
    # Small blocks catch the repeated motif; large blocks straddle distinct
    # filler and miss it. This is the "many small segments" signature.
    assert small["block_reuse_pct"] > 0
    assert large["block_reuse_pct"] == 0.0
    assert small["block_reuse_pct"] > large["block_reuse_pct"]

    # Segment histogram: MANY short OFF-prefix runs (3 per seq, all length w=4),
    # none anchored at the prefix.
    hist = segment_length_histogram(seqs, w=4, stride=1)
    assert hist["off_prefix"]["runs"] == 9  # 3 motifs * 3 seqs
    assert hist["on_prefix"]["runs"] == 0
    # All runs are short -> mass in the smallest bin (<= w).
    assert hist["off_prefix"]["counts"][0] == 9


# --------------------------------------------------------------------------- #
# Whole-prompt dedup
# --------------------------------------------------------------------------- #
def test_whole_prompt_dedup():
    a = distinct(0, 0, 8)
    b = distinct(1, 0, 8)
    deduped = dedup_whole_prompts([a, list(a), b])  # a appears twice
    assert len(deduped) == 2

    # With dedup ON, two identical prompts collapse to one and (alone) produce
    # NO collisions; the distinct third prompt also collides with nothing.
    prof_dedup = positional_collision_profile(
        [a, list(a), b], w=4, stride=1, drop_whole_prompt_dupes=True
    )
    assert prof_dedup["num_sequences"] == 2
    assert prof_dedup["overall_shared_pct"] == 0.0
    # With dedup OFF, the duplicate makes every window of `a` collide.
    prof_raw = positional_collision_profile(
        [a, list(a), b], w=4, stride=1, drop_whole_prompt_dupes=False
    )
    assert prof_raw["num_sequences"] == 3
    assert prof_raw["overall_shared_pct"] > 0.0


if __name__ == "__main__":  # allow plain `python3 test_overlap_structure.py`
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
