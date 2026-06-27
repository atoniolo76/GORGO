"""End-to-end test of the REAL data path: schema-matching sample parquets ->
duckdb read -> streaming aggregation -> JSON output, validated against the
ground truth embedded by ``make_sample_overlap_data.generate``.

Runs the actual Modal function BODIES via ``.local()`` (no Modal account, no
spend, no real data). Requires duckdb + pyarrow + modal (the worktree .venv):
  .venv/bin/python -m pytest data_processing/tests/test_overlap_e2e.py -q

It also leaves the produced JSONs in a temp dir; ``make_e2e_artifacts.py`` (or
running this file as __main__) writes them to a fixed path for eyeballing.
"""

import os
import sys
import tempfile

import pytest

_THIS = os.path.dirname(os.path.abspath(__file__))
_DATA_PROC = os.path.dirname(_THIS)
_REPO_ROOT = os.path.dirname(_DATA_PROC)
for _p in (_REPO_ROOT, _DATA_PROC, _THIS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import duckdb  # noqa: F401
except ImportError:  # pragma: no cover
    pytest.skip("duckdb not installed (use the worktree .venv)", allow_module_level=True)

import analyze_overlap_structure as A  # noqa: E402
from make_sample_overlap_data import generate  # noqa: E402

BLOCK_SIZES = "64,256,512,1024"


def run_e2e(out_root: str) -> dict:
    """Generate sample data and run both stages locally; return everything."""
    manifest = generate(out_root)
    out_dir = os.path.join(out_root, "overlap_structure")
    sweep = A.block_sweep.local(
        block_sizes=BLOCK_SIZES,
        min_sequence_len=1,
        data_dir=out_root,
        output_dir=out_dir,
        commit_volume=False,
    )
    ng = A.ngram_structure.local(
        window=64,
        stride=16,
        n_buckets=4,
        drop_whole_prompt_dupes=True,
        min_sequence_len=64,
        data_dir=out_root,
        output_dir=out_dir,
        commit_volume=False,
    )
    ng_nodedup = A.ngram_structure.local(
        window=64,
        stride=16,
        drop_whole_prompt_dupes=False,
        min_sequence_len=64,
        data_dir=out_root,
        output_dir=os.path.join(out_root, "overlap_structure_nodedup"),
        commit_volume=False,
    )
    return {
        "manifest": manifest,
        "out_dir": out_dir,
        "sweep": sweep,
        "profile": ng["profile"],
        "histogram": ng["histogram"],
        "profile_nodedup": ng_nodedup["profile"],
    }


@pytest.fixture(scope="module")
def e2e():
    with tempfile.TemporaryDirectory(prefix="overlap_e2e_") as d:
        yield run_e2e(d)


def _by_bs(sweep):
    return {r["block_size"]: r for r in sweep["sweep"]}


# --------------------------------------------------------------------------- #
# Block-size sweep ground truth
# --------------------------------------------------------------------------- #
def test_outputs_written(e2e):
    for name in (
        "blocksize_sweep.json",
        "position_bucket_profile.json",
        "segment_length_histogram.json",
    ):
        assert os.path.exists(os.path.join(e2e["out_dir"], name)), name


def test_content_ge_chained_everywhere(e2e):
    # chained-equal => content-equal, so content reuse is always >= chained reuse.
    for r in e2e["sweep"]["sweep"]:
        assert r["content_token_reuse_pct"] >= r["chained_token_reuse_pct"] - 1e-9, r
        assert r["content_block_reuse_pct"] >= r["chained_block_reuse_pct"] - 1e-9, r


def test_middle_chunk_content_strictly_beats_chained(e2e):
    # The identical MIDDLE chunk (cluster B) is content-matchable but prefix-poisoned
    # for the chained hash, so content > chained at every block-aligned size.
    by = _by_bs(e2e["sweep"])
    for bs in (64, 256, 512, 1024):
        assert by[bs]["content_minus_chained_token_pct"] > 0.0, (bs, by[bs])


def test_small_segments_small_block_beats_large(e2e):
    # The 64-token scattered motifs (cluster C) show up at block_size=64 but are
    # straddled (missed) at 512 -> small-block content reuse strictly higher.
    by = _by_bs(e2e["sweep"])
    assert by[64]["content_token_reuse_pct"] > by[512]["content_token_reuse_pct"], by


def test_reuse_is_substantial(e2e):
    # Sanity: the planted structure produces clearly non-trivial reuse.
    by = _by_bs(e2e["sweep"])
    assert by[64]["content_token_reuse_pct"] > 10.0, by[64]


# --------------------------------------------------------------------------- #
# Whole-prompt dedup
# --------------------------------------------------------------------------- #
def test_whale_dupes_dropped(e2e):
    prof = e2e["profile"]
    gt = e2e["manifest"]["ground_truth"]
    assert prof["num_whole_prompt_dupes_dropped"] == gt["expected_whole_prompt_dupes_dropped"]  # 29

    # kept == (#sessions with length >= window) - dropped dupes
    n_len_ge = sum(1 for s in e2e["manifest"]["sessions"] if s["length"] >= prof["window"])
    assert prof["num_sequences_kept"] == n_len_ge - prof["num_whole_prompt_dupes_dropped"]


def test_dedup_reduces_collisions(e2e):
    # With the whale dupes left IN, overall shared fraction is higher and nothing
    # is dropped -- confirms dedup actually changes the measurement.
    assert e2e["profile_nodedup"]["num_whole_prompt_dupes_dropped"] == 0
    assert e2e["profile_nodedup"]["overall_shared_pct"] > e2e["profile"]["overall_shared_pct"]


# --------------------------------------------------------------------------- #
# Positional collision profile (alignment-robust)
# --------------------------------------------------------------------------- #
def test_positional_profile_head_and_middle(e2e):
    prof = e2e["profile"]
    assert prof["overall_shared_pct"] > 0.0
    b = prof["buckets"]
    assert b[0]["shared_pct"] > 0.0  # head: cluster A shared prefix
    # middle/late: cluster B middle chunk + cluster C scattered motifs
    assert (b[2]["shared_pct"] > 0.0) or (b[3]["shared_pct"] > 0.0), b


# --------------------------------------------------------------------------- #
# Segment-length histogram: BOTH a long tail and short mass off-prefix
# --------------------------------------------------------------------------- #
def test_segment_histogram_shapes(e2e):
    h = e2e["histogram"]
    on, off = h["on_prefix"], h["off_prefix"]
    assert on["runs"] > 0  # cluster A prefix-anchored runs
    assert off["runs"] > 0  # cluster B/C off-prefix runs
    # off-prefix short mass (cluster C 64-token motifs land in the smallest bin)
    assert off["counts"][0] > 0, off
    # off-prefix long tail (cluster B ~1024-token middle: bin index 4 = "<= 1024")
    assert sum(off["counts"][4:]) > 0, off
    # on-prefix long run (cluster A ~2048-token prefix: bin index 5 = "<= 4096")
    assert sum(on["counts"][4:]) > 0, on


if __name__ == "__main__":
    # Write artifacts to a fixed dir for eyeballing, and print a summary.
    import json

    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.getcwd(), "e2e_out")
    os.makedirs(out, exist_ok=True)
    res = run_e2e(out)
    print("\n=== block-size sweep ===")
    for r in res["sweep"]["sweep"]:
        print(
            f"  bs={r['block_size']:>4}: content_tok={r['content_token_reuse_pct']:6.2f}% "
            f"chained_tok={r['chained_token_reuse_pct']:6.2f}% "
            f"(content-chained={r['content_minus_chained_token_pct']:+6.2f}pp) "
            f"content_blk={r['content_block_reuse_pct']:5.1f}% chained_blk={r['chained_block_reuse_pct']:5.1f}%"
        )
    p = res["profile"]
    print(
        f"\n=== positional profile (kept={p['num_sequences_kept']} dropped={p['num_whole_prompt_dupes_dropped']}) ==="
    )
    for b in p["buckets"]:
        print(
            f"  pos {b['range_pct'][0]:.0f}-{b['range_pct'][1]:.0f}%: shared {b['shared_pct']:5.1f}% (n={b['total_windows']})"
        )
    h = res["histogram"]
    print("\n=== segment-length histogram ===")
    print(
        f"  on_prefix : runs={h['on_prefix']['runs']:>4} counts={h['on_prefix']['counts']} ({h['on_prefix']['labels']})"
    )
    print(f"  off_prefix: runs={h['off_prefix']['runs']:>4} counts={h['off_prefix']['counts']}")
    print(f"\nartifacts in {out}/overlap_structure/")
    print(json.dumps({"out": out}, indent=2))
