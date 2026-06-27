"""Measurement (A): ACROSS-CONVERSATION within-user overlap (Modal driver).

Quantifies the reuse a session-affinity / per-user cache can exploit across a
user's DISTINCT conversations (NOT trivial within-conversation growth). Reads the
EXISTING tokenized cache ``tokenized_<FILE_PREFIX>/*.tokenized.parquet`` (cols
``token_hash``, ``session_id``, ``prompt_ids`` (list<uint32>); ONE ROW PER
CONVERSATION = the longest/max-footprint request per session, per
``build_eval_dataset.py:245-253``). user = ``token_hash``; conversation =
``session_id``. NO re-tokenize. Order-independent (the cache has no timestamp).

Two notions, both reported (per-user distribution + pooled + tiers):
  1. PREFIX  -- what session-affinity + a prefix cache captures (shared
     system/tools HEAD). Per user, a radix trie over the user's conversation
     sequences; savings = tokens - unique_trie_tokens. Pooled prefix == the
     existing intra-user A=53.67% (definitional cross-check).
  2. CONTENT/BLOCK -- what a content-addressed per-user cache captures (shared
     tool defs / RAG context / persona that aren't a clean prefix). Per user,
     content-hashed (prefix-independent) blocks; fraction of the user's tokens in
     blocks present in >= 2 of the user's conversations, swept over block sizes.
     content_savings - prefix_savings == cross-conversation MIDDLE reuse.

Plus PER-CONVERSATION attribution (order-independent): for each conversation, the
fraction of its tokens that ALSO appear in another conversation of the same user
(prefix-warm via the trie; content-warm via blocks at ``attribution_block_size``).

Memory: rows are grouped by ``token_hash`` (duckdb ``ORDER BY token_hash``) and one
user is processed at a time, so peak RAM ~ the single largest user (~85M tokens for
the top GLM whale, a few GiB) -- not the whole 8.65B-token corpus.

================================  HOW TO RUN  ================================
NO Modal spend without Rome's explicit approval. CPU-only; ~64 GiB / 8 CPU / 3 h.
The GLM-5.1 tokenized data lives ONLY in Modal env `alessio-dev`; whoever runs this
needs access there. Env-overridable via ``OVERLAP_MODAL_ENV``.

  # original data, env alessio-dev:
  MODAL_PROFILE=<profile-with-alessio-dev> modal config set-environment alessio-dev
  cd <repo> && modal run data_processing/analyze_user_reuse.py::user_reuse
  # data copied into a reachable env (e.g. GORGO):
  export OVERLAP_MODAL_ENV=GORGO
  MODAL_PROFILE=arcadia-research modal config set-environment GORGO
  modal run data_processing/analyze_user_reuse.py::user_reuse
  # or via the portable runner:  ./data_processing/run_overlap.sh user-reuse

LOCAL E2E (no Modal, no spend): the read/write paths are parameterized
(``data_dir``/``tokenized_glob``/``output_dir``/``commit_volume``) so the body runs
unchanged via ``user_reuse.local(...)``. See make_sample_user_reuse_data.py and
test_user_reuse_e2e.py.

Output: /data/user_reuse/user_reuse.json  (pooled prefix vs content by block size,
per-user percentile distribution, activity-tier breakdown, top users, and the
A=53.67% reconciliation). Verify by pulling it off the volume; then `modal app list`.
=============================================================================
"""

import glob as _glob
import os

import modal

# Standalone App + config, mirroring analyze_overlap_structure.py: we do NOT
# import app.py / build_eval_dataset.py (they bind module-level Modal objects to a
# HARDCODED env, which poisons cross-env runs). The two constants are mirrored here.
_DEFAULT_ENV = "alessio-dev"  # mirrors app.py ENVIRONMENT_NAME
FILE_PREFIX = "llm_responses_202604"  # mirrors build_eval_dataset.FILE_PREFIX

ENVIRONMENT_NAME = os.environ.get("OVERLAP_MODAL_ENV", _DEFAULT_ENV)

app = modal.App("GORGO-user-reuse")
completions_volume = modal.Volume.from_name(
    "GORGO-glm5-completions", environment_name=ENVIRONMENT_NAME
)

# Mount only the (stdlib-only) metric logic.
image = (
    modal.Image.debian_slim()
    .pip_install("duckdb")
    .env({"OVERLAP_MODAL_ENV": ENVIRONMENT_NAME})
    .add_local_python_source("overlap_metrics")
)

OUTPUT_DIR = "/data/user_reuse"
DEFAULT_BLOCK_SIZES = "16,64,256,512,1024"
DEFAULT_ATTR_BLOCK_SIZE = 512
EXPECTED_INTRA_USER_A_PCT = 53.6708211552563  # prefix_trie_results/.../stats.json:17


def _tokenized_files(data_dir: str = "/data", tokenized_glob: str | None = None) -> list[str]:
    """Resolve the tokenized parquet set (same set the trie ran on). ``data_dir``
    parameterizes the read path so the body runs under ``.local()`` on sample data."""
    if tokenized_glob:
        pattern = (
            tokenized_glob
            if os.path.isabs(tokenized_glob)
            else os.path.join(data_dir, tokenized_glob)
        )
    else:
        pattern = os.path.join(data_dir, f"tokenized_{FILE_PREFIX}", "*.tokenized.parquet")
    files = sorted(_glob.glob(pattern))
    if not files:
        raise RuntimeError(
            f"No tokenized parquets matching {pattern} (env={ENVIRONMENT_NAME}). "
            f"For production, run build_eval_dataset.py::tokenize_main first."
        )
    return files


def _iter_users(files: list[str], min_sequence_len: int, duckdb_memory_limit: str):
    """Stream conversations grouped by ``token_hash`` (one user at a time).

    Uses duckdb ``ORDER BY token_hash`` (external-sorts/spills as needed) and yields
    ``(token_hash, [(session_id, prompt_ids), ...])`` so a user's whole conversation
    set is available, but only one user is buffered at once.
    """
    import duckdb
    import re

    con = duckdb.connect()
    try:
        if duckdb_memory_limit:
            # PRAGMA can't take a bind parameter, so the value is interpolated;
            # validate it to a strict size literal first (defense-in-depth — the
            # value is operator config, never untrusted input).
            if not re.fullmatch(
                r"\d+(\.\d+)?\s*[KMGT]?B", duckdb_memory_limit.strip(), re.IGNORECASE
            ):
                raise ValueError(
                    f"invalid duckdb_memory_limit: {duckdb_memory_limit!r} (e.g. '48GB')"
                )
            con.execute(f"PRAGMA memory_limit='{duckdb_memory_limit.strip()}'")
        cur = con.execute(
            "SELECT token_hash, session_id, prompt_ids FROM read_parquet(?) ORDER BY token_hash",
            [files],
        )
        cur_hash = None
        buf: list = []
        while True:
            chunk = cur.fetchmany(2048)
            if not chunk:
                break
            for token_hash, session_id, prompt_ids in chunk:
                if not prompt_ids or len(prompt_ids) < min_sequence_len:
                    continue
                if token_hash != cur_hash:
                    if cur_hash is not None and buf:
                        yield cur_hash, buf
                    cur_hash = token_hash
                    buf = []
                buf.append((session_id, prompt_ids))
        if cur_hash is not None and buf:
            yield cur_hash, buf
    finally:
        con.close()


def _write(
    name: str, payload: dict, output_dir: str = OUTPUT_DIR, commit_volume: bool = True
) -> str:
    import json

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)
    if commit_volume:
        completions_volume.commit()
    print(f"wrote {path}")
    return path


def _analyze_user(token_hash: str, rows: list, sizes: list[int], attr_bs: int) -> dict:
    """Per-user cross-conversation reuse. ``rows`` = [(session_id, prompt_ids), ...]
    for ONE user. Pure-Python; uses overlap_metrics primitives."""
    from array import array

    from overlap_metrics import RadixTrie, content_block_digests

    total_tokens = sum(len(ids) for _, ids in rows)
    n_conv = len(rows)

    # --- prefix: per-user radix trie over the user's conversation sequences ---
    trie = RadixTrie()
    # --- content: per block size, occurrences + distinct-conversation counts ---
    occ = {bs: {} for bs in sizes}
    conv_count = {bs: {} for bs in sizes}
    blen = {bs: {} for bs in sizes}

    for _sid, ids in rows:
        trie.insert(array("I", ids))
        for bs in sizes:
            digs = content_block_digests(ids, bs)
            o, cc, ln = occ[bs], conv_count[bs], blen[bs]
            seen_in_conv = set()
            for idx, d in enumerate(digs):
                o[d] = o.get(d, 0) + 1
                if d not in ln:
                    start = idx * bs
                    ln[d] = min(bs, len(ids) - start)
                if d not in seen_in_conv:
                    seen_in_conv.add(d)
                    cc[d] = cc.get(d, 0) + 1

    unique_prefix_tokens = trie.unique_token_count()
    prefix_savings_tokens = total_tokens - unique_prefix_tokens

    content = {}
    for bs in sizes:
        o, cc, ln = occ[bs], conv_count[bs], blen[bs]
        unique_block_tokens = sum(ln.values())
        content_savings_tokens = total_tokens - unique_block_tokens
        cross_conv_tokens = sum(o[d] * ln[d] for d in o if cc[d] >= 2)
        content[bs] = {
            "content_savings_tokens": content_savings_tokens,
            "content_savings_pct": 100.0 * content_savings_tokens / total_tokens
            if total_tokens
            else 0.0,
            "cross_conv_tokens": cross_conv_tokens,
            "cross_conv_token_pct": 100.0 * cross_conv_tokens / total_tokens
            if total_tokens
            else 0.0,
        }

    # --- per-conversation attribution (order-independent) ---
    attr_cc, attr_ln = conv_count[attr_bs], blen[attr_bs]
    warm_prefix, warm_content = [], []
    for _sid, ids in rows:
        L = len(ids)
        if L == 0:
            continue
        spl = trie.shared_prefix_length(array("I", ids))
        warm_prefix.append(100.0 * spl / L)
        digs = content_block_digests(ids, attr_bs)
        cwarm = 0
        for idx, d in enumerate(digs):
            if attr_cc[d] >= 2:
                start = idx * attr_bs
                cwarm += min(attr_bs, L - start)
        warm_content.append(100.0 * cwarm / L)

    mean_warm_prefix = sum(warm_prefix) / len(warm_prefix) if warm_prefix else 0.0
    mean_warm_content = sum(warm_content) / len(warm_content) if warm_content else 0.0

    return {
        "token_hash": token_hash,
        "num_conversations": n_conv,
        "total_tokens": total_tokens,
        "prefix_savings_tokens": prefix_savings_tokens,
        "prefix_savings_pct": 100.0 * prefix_savings_tokens / total_tokens if total_tokens else 0.0,
        "content": {str(bs): content[bs] for bs in sizes},
        "attribution_block_size": attr_bs,
        "mean_warm_prefix_pct": mean_warm_prefix,
        "mean_warm_content_pct": mean_warm_content,
    }


def _aggregate(records: list, sizes: list[int], attr_bs: int) -> dict:
    from overlap_metrics import percentiles

    T = sum(r["total_tokens"] for r in records)
    n_users = len(records)
    n_conv_total = sum(r["num_conversations"] for r in records)
    bs_keys = [str(bs) for bs in sizes]

    def pooled_pct(field_path):
        num = 0
        for r in records:
            num += field_path(r)
        return 100.0 * num / T if T else 0.0

    pooled_prefix = pooled_pct(lambda r: r["prefix_savings_tokens"])
    pooled_content = {
        bs: pooled_pct(lambda r, bs=bs: r["content"][bs]["content_savings_tokens"])
        for bs in bs_keys
    }
    pooled_cross = {
        bs: pooled_pct(lambda r, bs=bs: r["content"][bs]["cross_conv_tokens"]) for bs in bs_keys
    }
    content_minus_prefix = {bs: pooled_content[bs] - pooled_prefix for bs in bs_keys}

    # per-user distribution (unweighted across users)
    dist = {
        "prefix_savings_pct": percentiles([r["prefix_savings_pct"] for r in records]),
        f"content_savings_pct@{attr_bs}": percentiles(
            [r["content"][str(attr_bs)]["content_savings_pct"] for r in records]
        ),
        f"cross_conv_token_pct@{attr_bs}": percentiles(
            [r["content"][str(attr_bs)]["cross_conv_token_pct"] for r in records]
        ),
        "mean_warm_prefix_pct": percentiles([r["mean_warm_prefix_pct"] for r in records]),
        "mean_warm_content_pct": percentiles([r["mean_warm_content_pct"] for r in records]),
        "num_single_conversation_users": sum(1 for r in records if r["num_conversations"] == 1),
        "frac_users_with_cross_conv_content_reuse": (
            sum(1 for r in records if r["content"][str(attr_bs)]["cross_conv_tokens"] > 0) / n_users
            if n_users
            else 0.0
        ),
        "conversations_per_user": percentiles([float(r["num_conversations"]) for r in records]),
    }

    # activity tiers by total-token rank (mirror top_users tiering). Boundaries are
    # NON-OVERLAPPING integer indices; tiers that collapse to empty on small N are
    # skipped (on the real 4,984-user corpus all four are populated and distinct).
    ranked = sorted(records, key=lambda r: r["total_tokens"], reverse=True)
    tier_defs = [
        ("whale (top 1%)", 0.00, 0.01),
        ("heavy (next 9%)", 0.01, 0.10),
        ("medium (next 40%)", 0.10, 0.50),
        ("light (bottom 50%)", 0.50, 1.00),
    ]
    tiers = []
    for name, lo, hi in tier_defs:
        a = int(lo * n_users)
        b = n_users if hi >= 1.0 else int(hi * n_users)
        grp = ranked[a:b]
        if not grp:
            continue
        gt = sum(r["total_tokens"] for r in grp)
        tiers.append(
            {
                "tier": name,
                "num_users": len(grp),
                "total_tokens": gt,
                "mean_conversations": sum(r["num_conversations"] for r in grp) / len(grp),
                "pooled_prefix_savings_pct": 100.0
                * sum(r["prefix_savings_tokens"] for r in grp)
                / gt
                if gt
                else 0.0,
                f"pooled_content_savings_pct@{attr_bs}": 100.0
                * sum(r["content"][str(attr_bs)]["content_savings_tokens"] for r in grp)
                / gt
                if gt
                else 0.0,
                f"pooled_cross_conv_pct@{attr_bs}": 100.0
                * sum(r["content"][str(attr_bs)]["cross_conv_tokens"] for r in grp)
                / gt
                if gt
                else 0.0,
            }
        )

    top_users = [
        {
            "token_hash": r["token_hash"],
            "tokens": r["total_tokens"],
            "num_conversations": r["num_conversations"],
            "prefix_savings_pct": r["prefix_savings_pct"],
            f"content_savings_pct@{attr_bs}": r["content"][str(attr_bs)]["content_savings_pct"],
            f"cross_conv_token_pct@{attr_bs}": r["content"][str(attr_bs)]["cross_conv_token_pct"],
        }
        for r in ranked[:10]
    ]

    return {
        "num_users": n_users,
        "num_conversations_total": n_conv_total,
        "total_tokens": T,
        "reconciliation": {
            "pooled_prefix_savings_pct": pooled_prefix,
            "expected_intra_user_A_pct": EXPECTED_INTRA_USER_A_PCT,
            "note": (
                "pooled prefix savings = (sum_u (T_u - U(R_u))) / sum_u T_u = intra-user A by "
                "construction; on the real cache it should match stats.json A=53.67%."
            ),
        },
        "pooled": {
            "prefix_savings_pct": pooled_prefix,
            "content_savings_pct": pooled_content,
            "cross_conv_token_pct": pooled_cross,
            "content_minus_prefix_pct": content_minus_prefix,
        },
        "per_user_distribution": dist,
        "tiers": tiers,
        "top_users": top_users,
    }


@app.function(
    image=image, memory=1024 * 64, cpu=8.0, timeout=10800, volumes={"/data": completions_volume}
)
def user_reuse(
    block_sizes: str = DEFAULT_BLOCK_SIZES,
    attribution_block_size: int = DEFAULT_ATTR_BLOCK_SIZE,
    min_sequence_len: int = 1,
    data_dir: str = "/data",
    tokenized_glob: str = "",
    output_dir: str = "",
    commit_volume: bool = True,
    duckdb_memory_limit: str = "48GB",
    include_per_user: bool = False,
) -> dict:
    """Across-conversation within-user reuse, per user then pooled. One user at a
    time (rows grouped by token_hash). ``data_dir``/``tokenized_glob``/``output_dir``
    /``commit_volume`` parameterize I/O so the body runs via ``.local()`` on samples.

    ``min_sequence_len=1`` matches build_prefix_trie so pooled prefix reconciles with
    intra-user A=53.67%."""
    import time

    sizes = [int(s) for s in block_sizes.split(",") if s.strip()]
    attr_bs = attribution_block_size
    if attr_bs not in sizes:
        sizes = sorted(set(sizes) | {attr_bs})
    files = _tokenized_files(data_dir, tokenized_glob or None)
    out_dir = output_dir or OUTPUT_DIR
    print(
        f"user_reuse: sizes={sizes} attr_bs={attr_bs} over {len(files)} files (env={ENVIRONMENT_NAME})"
    )

    t0 = time.time()
    records = []
    n_users = 0
    for token_hash, rows in _iter_users(files, min_sequence_len, duckdb_memory_limit):
        records.append(_analyze_user(token_hash, rows, sizes, attr_bs))
        n_users += 1
        if n_users % 500 == 0:
            print(f"  {n_users:,} users | {time.time() - t0:,.0f}s")

    if not records:
        raise RuntimeError("No users with qualifying conversations found.")

    payload = {
        "environment": ENVIRONMENT_NAME,
        "file_prefix": FILE_PREFIX,
        "num_files": len(files),
        "block_sizes": sizes,
        "attribution_block_size": attr_bs,
        "min_sequence_len": min_sequence_len,
        "elapsed_seconds": time.time() - t0,
        "interpretation": (
            "PREFIX savings = what session-affinity + a prefix cache captures across a "
            "user's conversations (shared system/tools head). CONTENT savings = what a "
            "per-user content-addressed cache captures (shared tool defs / RAG / persona "
            "anywhere). content_minus_prefix = cross-conversation MIDDLE reuse. cross_conv_"
            "token_pct = fraction of a user's tokens that also appear in >=2 of their "
            "conversations. mean_warm_*_pct (per-user mean over conversations) answers "
            "'how much of each new conversation is already warm from the user's others'."
        ),
        **_aggregate(records, sizes, attr_bs),
    }
    if include_per_user:  # lean by default; sample-E2E / debugging only
        payload["per_user"] = records
    _write("user_reuse.json", payload, out_dir, commit_volume)
    return payload


@app.local_entrypoint()
def main(
    block_sizes: str = DEFAULT_BLOCK_SIZES,
    attribution_block_size: int = DEFAULT_ATTR_BLOCK_SIZE,
):
    r = user_reuse.remote(block_sizes=block_sizes, attribution_block_size=attribution_block_size)
    rec = r["reconciliation"]
    print(
        f"\nusers={r['num_users']:,}  conversations={r['num_conversations_total']:,}  tokens={r['total_tokens']:,}"
    )
    print(
        f"pooled PREFIX savings = {rec['pooled_prefix_savings_pct']:.2f}%  "
        f"(intra-user A cross-check = {rec['expected_intra_user_A_pct']:.2f}%)"
    )
    abs_ = str(attribution_block_size)
    print("pooled CONTENT savings by block size:")
    for bs, v in r["pooled"]["content_savings_pct"].items():
        print(
            f"  bs={bs:>4}: content={v:6.2f}%  (content-prefix={r['pooled']['content_minus_prefix_pct'][bs]:+6.2f}pp)"
        )
    d = r["per_user_distribution"]
    print(
        f"per-user median: prefix={d['prefix_savings_pct']['p50']:.1f}%  "
        f"content@{abs_}={d[f'content_savings_pct@{abs_}']['p50']:.1f}%  "
        f"warm/conv prefix={d['mean_warm_prefix_pct']['p50']:.1f}% content={d['mean_warm_content_pct']['p50']:.1f}%"
    )
    print("Done. Pull /data/user_reuse/user_reuse.json to verify; then `modal app list`.")
