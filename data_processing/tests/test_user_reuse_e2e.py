"""End-to-end test of measurement (A): across-conversation within-user reuse.

Runs the REAL data path -- multi-conversation-per-user sample parquets -> duckdb
``ORDER BY token_hash`` grouping -> per-user trie + content blocks -> aggregate ->
JSON -- by invoking the actual Modal function body via ``.local()`` (no Modal
account, no spend, no real data), then validates against the planted ground truth.

Requires duckdb + pyarrow + modal (the worktree .venv):
  .venv/bin/python -m pytest data_processing/tests/test_user_reuse_e2e.py -q
"""

import os
import sys

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

import analyze_user_reuse as U  # noqa: E402
from make_sample_user_reuse_data import generate  # noqa: E402

ATTR = 512


def run_e2e(out_root: str) -> dict:
    manifest = generate(out_root)
    out_dir = os.path.join(out_root, "user_reuse")
    payload = U.user_reuse.local(
        block_sizes="16,64,256,512,1024",
        attribution_block_size=ATTR,
        min_sequence_len=1,
        data_dir=out_root,
        output_dir=out_dir,
        commit_volume=False,
        include_per_user=True,
    )
    by_user = {r["token_hash"]: r for r in payload["per_user"]}
    return {"manifest": manifest, "out_dir": out_dir, "payload": payload, "by_user": by_user}


@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("user_reuse_e2e"))
    return run_e2e(d)


def _content(rec, bs):
    return rec["content"][str(bs)]


# --------------------------------------------------------------------------- #
# Output + sanity
# --------------------------------------------------------------------------- #
def test_output_written_and_counts(e2e):
    assert os.path.exists(os.path.join(e2e["out_dir"], "user_reuse.json"))
    p = e2e["payload"]
    assert p["num_users"] == e2e["manifest"]["num_users"]
    assert p["num_conversations_total"] == e2e["manifest"]["num_sessions"]


# --------------------------------------------------------------------------- #
# Per-user ground truth (prefix vs content; warm attribution)
# --------------------------------------------------------------------------- #
def test_prefix_and_middle_users(e2e):
    # userP: shared SYS prefix AND shared TOOL middle -> prefix>0, content>prefix.
    for u in ("userP_0", "userP_1", "userP_2"):
        r = e2e["by_user"][u]
        assert r["prefix_savings_pct"] > 0, u
        assert _content(r, ATTR)["content_savings_pct"] > r["prefix_savings_pct"] + 1.0, u
        assert _content(r, ATTR)["cross_conv_token_pct"] > 0, u
        # per-conversation: warm-from-content (prefix+middle) > warm-from-prefix alone
        assert r["mean_warm_prefix_pct"] > 0, u
        assert r["mean_warm_content_pct"] > r["mean_warm_prefix_pct"] + 1.0, u


def test_prefix_only_user_has_no_middle_gap(e2e):
    # Pure shared prefix -> content savings == prefix savings (no content-prefix gap).
    r = e2e["by_user"]["userPrefixOnly"]
    assert r["prefix_savings_pct"] > 0
    assert abs(_content(r, ATTR)["content_savings_pct"] - r["prefix_savings_pct"]) < 1e-6
    assert r["mean_warm_content_pct"] == pytest.approx(r["mean_warm_prefix_pct"], abs=1e-6)


def test_middle_only_user(e2e):
    # Unique heads, shared middle -> prefix ~0, content>0, large gap, warm_prefix ~0.
    r = e2e["by_user"]["userMiddleOnly"]
    assert r["prefix_savings_pct"] == pytest.approx(0.0, abs=1e-9)
    assert _content(r, ATTR)["content_savings_pct"] > 10.0
    assert _content(r, ATTR)["cross_conv_token_pct"] > 0
    assert r["mean_warm_prefix_pct"] == pytest.approx(0.0, abs=1e-9)
    assert r["mean_warm_content_pct"] > 10.0


def test_single_conversation_user_zero_reuse(e2e):
    r = e2e["by_user"]["userSingle"]
    assert r["num_conversations"] == 1
    assert r["prefix_savings_pct"] == pytest.approx(0.0, abs=1e-9)
    for bs in (16, 64, 256, 512, 1024):
        assert _content(r, bs)["cross_conv_token_pct"] == pytest.approx(0.0, abs=1e-9)
    assert r["mean_warm_prefix_pct"] == pytest.approx(0.0, abs=1e-9)
    assert r["mean_warm_content_pct"] == pytest.approx(0.0, abs=1e-9)


def test_content_ge_prefix_per_user_at_small_block(e2e):
    # At the smallest (finest) block size, content dedup is a superset of prefix
    # dedup -> content savings >= prefix, per user. (At LARGER block sizes a user
    # whose only sharing is a sub-block prefix can fall below prefix -- the block
    # metric is alignment/granularity-limited; the radix trie is exact. That is
    # exactly why we report BOTH.)
    for r in e2e["payload"]["per_user"]:
        assert _content(r, 16)["content_savings_pct"] >= r["prefix_savings_pct"] - 1e-6, r[
            "token_hash"
        ]


# --------------------------------------------------------------------------- #
# Pooled + reconciliation
# --------------------------------------------------------------------------- #
def test_pooled_content_ge_prefix_and_middle_gap(e2e):
    pooled = e2e["payload"]["pooled"]
    assert pooled["prefix_savings_pct"] > 0
    # robust invariant: at the finest block size pooled content >= pooled prefix
    assert pooled["content_savings_pct"]["16"] >= pooled["prefix_savings_pct"] - 1e-6
    # block-aligned shared middles (TOOL) -> positive content-prefix gap at every size
    for bs, gap in pooled["content_minus_prefix_pct"].items():
        assert gap > 0.0, (bs, gap)


def test_reconciliation_formula_consistent(e2e):
    # pooled prefix == sum_u savings / sum_u tokens (the intra-user A definition).
    recs = e2e["payload"]["per_user"]
    T = sum(r["total_tokens"] for r in recs)
    expect = 100.0 * sum(r["prefix_savings_tokens"] for r in recs) / T
    got = e2e["payload"]["reconciliation"]["pooled_prefix_savings_pct"]
    assert got == pytest.approx(expect, rel=1e-9)
    assert e2e["payload"]["reconciliation"]["expected_intra_user_A_pct"] == pytest.approx(
        53.6708, abs=1e-3
    )


def test_distribution_flags(e2e):
    d = e2e["payload"]["per_user_distribution"]
    assert d["num_single_conversation_users"] >= 1
    assert 0.0 < d["frac_users_with_cross_conv_content_reuse"] < 1.0  # single-conv user has none


# --------------------------------------------------------------------------- #
# Activity tiers
# --------------------------------------------------------------------------- #
def test_tiers_whale_dominant(e2e):
    p = e2e["payload"]
    tiers = p["tiers"]
    assert len(tiers) >= 2
    # the whale (most tokens) is the top of the top tier
    ranked = sorted(p["per_user"], key=lambda r: r["total_tokens"], reverse=True)
    assert ranked[0]["token_hash"] == "userWhale"
    # top non-empty tier reuses more (content) than the lightest tier
    top, bottom = tiers[0], tiers[-1]
    assert top[f"pooled_content_savings_pct@{ATTR}"] > bottom[f"pooled_content_savings_pct@{ATTR}"]


if __name__ == "__main__":
    import json

    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.getcwd(), "user_reuse_out")
    os.makedirs(out, exist_ok=True)
    res = run_e2e(out)
    p = res["payload"]
    rec = p["reconciliation"]
    print(
        f"\nusers={p['num_users']} conversations={p['num_conversations_total']} tokens={p['total_tokens']:,}"
    )
    print(
        f"pooled PREFIX = {rec['pooled_prefix_savings_pct']:.2f}%  (A cross-check {rec['expected_intra_user_A_pct']:.2f}%)"
    )
    print("pooled CONTENT by block size (content / content-prefix gap):")
    for bs, v in p["pooled"]["content_savings_pct"].items():
        print(f"  bs={bs:>4}: {v:6.2f}%  gap={p['pooled']['content_minus_prefix_pct'][bs]:+6.2f}pp")
    d = p["per_user_distribution"]
    print(
        f"per-user median: prefix={d['prefix_savings_pct']['p50']:.1f}%  "
        f"content@{ATTR}={d[f'content_savings_pct@{ATTR}']['p50']:.1f}%  "
        f"warm/conv prefix={d['mean_warm_prefix_pct']['p50']:.1f}% content={d['mean_warm_content_pct']['p50']:.1f}%"
    )
    print("tiers:")
    for t in p["tiers"]:
        print(
            f"  {t['tier']:>18}: users={t['num_users']:>2} convs/user={t['mean_conversations']:.1f} "
            f"prefix={t['pooled_prefix_savings_pct']:.1f}% content@{ATTR}={t[f'pooled_content_savings_pct@{ATTR}']:.1f}%"
        )
    print(f"\nartifacts in {out}/user_reuse/")
    print(json.dumps({"out": out}, indent=2))
