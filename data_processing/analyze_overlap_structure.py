"""GLM-5.1 content-overlap STRUCTURE: prefix vs. middle/overall (Modal driver).

Companion to ``build_prefix_trie.py``. The radix trie measures strict PREFIX
reuse (B = 55.30% global, 53.67% intra-user); the existing 512-token block
metric is also prefix-dependent (chained hashing, build_mooncake_trace.py:428-444).
Neither measures NON-prefix / middle overlap. This driver does, via three
content-hashed (prefix-independent) measurements -- see
``prefix_trie_results/glm-5.1-completions/overlap_structure_analysis.md`` and the
pure, unit-tested logic in ``overlap_metrics.py``:

  (i)   block-size sweep {16,64,256,512,1024}: CONTENT-INDEPENDENT block reuse
        vs. PREFIX-CHAINED block reuse. Content >> chained at small sizes =>
        many small off-prefix segments; both equal & only-large => prefix-only.
  (ii)  positional collision profile: content-hashed 64-tok rolling n-grams,
        bucketed by position-in-prompt, AFTER dropping whole-prompt duplicates.
        High shared-fraction in late buckets => genuine middle/suffix overlap.
  (iii) matched-segment-length histogram (on- vs off-prefix). Long off-prefix
        runs => few big middle chunks; many short runs => many small segments.

Reads the SAME tokenized cache the trie used:
``/data/tokenized_<FILE_PREFIX>/*.tokenized.parquet`` with columns
``token_hash`` (string) and ``prompt_ids`` (list<uint32>); see
``build_eval_dataset.py:245-253`` and ``build_prefix_trie.py:137-156``. Streams
off-disk -- never materializes the 8.65B-token corpus in RAM.

================================  HOW TO RUN  ================================
NO Modal spend without Rome's explicit approval. CPU-only; ~64 GiB / 8 CPU /
3 h timeout. The GLM-5.1 tokenized data lives ONLY in Modal environment
`alessio-dev`; this box's profiles cannot reach it. Whoever runs this needs
alessio-dev access. The script is env-overridable via ``OVERLAP_MODAL_ENV``.

# --- Case A: original data, environment `alessio-dev` (Rome / Alessio) ---
source /home/rome/.venv/bin/activate
MODAL_PROFILE=<profile-with-alessio-dev>   # the profile that can see alessio-dev
modal config set-environment alessio-dev
cd /home/rome/gt/gorgo/crew/hypatia_glm_wt
modal run data_processing/analyze_overlap_structure.py::analyze_all
# (defaults: OVERLAP_MODAL_ENV unset -> alessio-dev)

# --- Case B: data copied into a reachable env (e.g. `GORGO`) ---
# First copy the tokenized dir into the GORGO-env volume of the same name, then:
export OVERLAP_MODAL_ENV=GORGO
MODAL_PROFILE=arcadia-research
modal config set-environment GORGO
cd /home/rome/gt/gorgo/crew/hypatia_glm_wt
modal run data_processing/analyze_overlap_structure.py::analyze_all

# Run one stage / tune knobs (Modal lowercases flags; no single-letter names):
modal run data_processing/analyze_overlap_structure.py::block_sweep --block-sizes 16,64,256,512,1024
modal run data_processing/analyze_overlap_structure.py::ngram_structure --window 64 --stride 16
# If pass-1 n-gram counting OOMs at 64 GiB, bump --stride (32/64) or memory.
# If the size-16 sweep OOMs, run it alone on a bigger container:
modal run data_processing/analyze_overlap_structure.py::block_sweep --block-sizes 16
=============================================================================

Outputs (written under the volume + returned + vol.commit()'d):
  /data/overlap_structure/blocksize_sweep.json
  /data/overlap_structure/position_bucket_profile.json
  /data/overlap_structure/segment_length_histogram.json
Verify after the run by pulling them off the volume (trust the artifact, not
the logs): ``modal volume get GORGO-glm5-completions overlap_structure/... /tmp/``
and confirm no app is left running: ``modal app list`` / ``modal app stop``.

LOCAL E2E (no Modal, no spend): the read/write paths are parameterized
(``data_dir``, ``tokenized_glob``, ``output_dir``, ``commit_volume``) so the
function bodies run unchanged via ``.local()`` against sample parquets, e.g.:
    block_sweep.local(block_sizes="64,256,512,1024", data_dir="<sample>",
                      output_dir="<out>", commit_volume=False)
See data_processing/tests/make_sample_overlap_data.py and test_overlap_e2e.py.
"""

import glob as _glob
import os

import modal

# Standalone App + config. We deliberately do NOT `from app import ...` /
# `from build_eval_dataset import ...`: app.py binds module-level Modal objects
# (Dicts/Volumes) to a HARDCODED environment (alessio-dev), so importing it makes
# Modal resolve that environment at run time and crashes any run targeting a
# different env (e.g. NotFoundError: Environment 'alessio-dev' not found). The
# only things we need from those modules are two constants, mirrored here, which
# keeps this driver env-portable (overridable via OVERLAP_MODAL_ENV).
_DEFAULT_ENV = "alessio-dev"  # mirrors app.py ENVIRONMENT_NAME
FILE_PREFIX = "llm_responses_202604"  # mirrors build_eval_dataset.FILE_PREFIX

ENVIRONMENT_NAME = os.environ.get("OVERLAP_MODAL_ENV", _DEFAULT_ENV)

app = modal.App("GORGO-overlap-structure")
completions_volume = modal.Volume.from_name(
    "GORGO-glm5-completions", environment_name=ENVIRONMENT_NAME
)

# Mount only the (stdlib-only) metric logic; app.py / build_eval_dataset.py are
# intentionally NOT mounted (unused at runtime and env-poisoned, see above).
image = (
    modal.Image.debian_slim()
    .pip_install("duckdb")
    # Propagate the resolved target env INTO the container so the in-container
    # module load matches this run's env (not the alessio-dev default): keeps the
    # output JSON "environment" label and the vol.commit() handle env-correct for
    # cross-env runs. (Build steps must precede add_local_*; see Modal invariants.)
    .env({"OVERLAP_MODAL_ENV": ENVIRONMENT_NAME})
    .add_local_python_source("overlap_metrics")
)

OUTPUT_DIR = "/data/overlap_structure"
DEFAULT_BLOCK_SIZES = "16,64,256,512,1024"


def _tokenized_files(data_dir: str = "/data", tokenized_glob: str | None = None) -> list[str]:
    """Resolve the tokenized parquet set to analyze.

    ``data_dir`` defaults to the Modal volume mount ``/data`` (production) but can
    point at any local directory so the function bodies run unchanged under
    ``.local()`` against sample data. ``tokenized_glob`` overrides the pattern
    (absolute, or relative to ``data_dir``); by default it globs the tokenized
    cache dir ``<data_dir>/tokenized_<FILE_PREFIX>/*.tokenized.parquet`` -- the
    same in-window set the radix trie ran on (the cache only ever contains files
    matching FILE_PREFIX within the date window; see build_eval_dataset.py).
    """
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


def _iter_sequences(files: list[str], min_sequence_len: int):
    """Stream (token_hash, prompt_ids) off the tokenized parquets, one session
    at a time -- never holds the whole corpus in RAM. Re-callable for multi-pass."""
    import duckdb

    con = duckdb.connect()
    try:
        for path in files:
            cur = con.execute("SELECT token_hash, prompt_ids FROM read_parquet(?)", [path])
            while True:
                chunk = cur.fetchmany(2048)
                if not chunk:
                    break
                for token_hash, token_ids in chunk:
                    if not token_ids or len(token_ids) < min_sequence_len:
                        continue
                    yield token_hash, token_ids
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
        completions_volume.commit()  # REQUIRED in-container: writes vanish on exit otherwise
    print(f"wrote {path}")
    return path


# --------------------------------------------------------------------------- #
# (i) Block-size sweep: content-independent vs prefix-chained
# --------------------------------------------------------------------------- #
@app.function(
    image=image, memory=1024 * 64, cpu=8.0, timeout=10800, volumes={"/data": completions_volume}
)
def block_sweep(
    block_sizes: str = DEFAULT_BLOCK_SIZES,
    min_sequence_len: int = 1,
    data_dir: str = "/data",
    tokenized_glob: str = "",
    output_dir: str = "",
    commit_volume: bool = True,
) -> dict:
    """One streaming pass; for each block size keeps a content accumulator and a
    prefix-chained accumulator. Memory is dominated by the SMALLEST block size
    (most distinct blocks). If it OOMs, run smaller sizes in separate jobs.

    ``data_dir``/``tokenized_glob`` parameterize the read path so this body runs
    unchanged via ``.local()`` against sample data; ``output_dir`` (default
    ``OUTPUT_DIR``) and ``commit_volume`` likewise let it write locally."""
    import time

    from overlap_metrics import BlockReuseAccumulator

    sizes = [int(s) for s in block_sizes.split(",") if s.strip()]
    files = _tokenized_files(data_dir, tokenized_glob or None)
    out_dir = output_dir or OUTPUT_DIR
    print(f"block_sweep: sizes={sizes} over {len(files)} files (env={ENVIRONMENT_NAME})")

    accs = {(bs, ch): BlockReuseAccumulator(bs, ch) for bs in sizes for ch in (False, True)}
    t0 = time.time()
    n_seq = 0
    for _, token_ids in _iter_sequences(files, min_sequence_len):
        for acc in accs.values():
            acc.add(token_ids)
        n_seq += 1
        if n_seq % 50000 == 0:
            print(f"  {n_seq:,} sessions | {time.time() - t0:,.0f}s")

    rows = []
    for bs in sizes:
        c = accs[(bs, False)].result()
        p = accs[(bs, True)].result()
        rows.append(
            {
                "block_size": bs,
                "content_block_reuse_pct": c["block_reuse_pct"],
                "content_token_reuse_pct": c["token_reuse_pct"],
                "chained_block_reuse_pct": p["block_reuse_pct"],
                "chained_token_reuse_pct": p["token_reuse_pct"],
                "content_minus_chained_token_pct": c["token_reuse_pct"] - p["token_reuse_pct"],
                "content_unique_blocks": c["unique_blocks"],
                "chained_unique_blocks": p["unique_blocks"],
                "total_blocks": c["total_blocks"],
                "total_tokens": c["total_tokens"],
            }
        )
    payload = {
        "environment": ENVIRONMENT_NAME,
        "file_prefix": FILE_PREFIX,
        "num_files": len(files),
        "num_sequences": n_seq,
        "min_sequence_len": min_sequence_len,
        "elapsed_seconds": time.time() - t0,
        "interpretation": (
            "content >> chained at small block sizes => many small off-prefix "
            "segments; content == chained and only large blocks reuse => "
            "prefix-dominated, no middle overlap; content jumps only at large "
            "sizes => few big aligned chunks. Compare token_reuse_pct to the "
            "radix-trie global_savings_pct=55.30 (prefix baseline)."
        ),
        "sweep": rows,
    }
    _write("blocksize_sweep.json", payload, out_dir, commit_volume)
    return payload


# --------------------------------------------------------------------------- #
# (ii)+(iii) n-gram positional profile + segment-length histogram
# --------------------------------------------------------------------------- #
@app.function(
    image=image, memory=1024 * 64, cpu=8.0, timeout=10800, volumes={"/data": completions_volume}
)
def ngram_structure(
    window: int = 64,
    stride: int = 16,
    n_buckets: int = 4,
    drop_whole_prompt_dupes: bool = True,
    min_sequence_len: int = 64,
    data_dir: str = "/data",
    tokenized_glob: str = "",
    output_dir: str = "",
    commit_volume: bool = True,
) -> dict:
    """Streaming two-pass over the tokenized cache, using the same primitives as
    ``overlap_metrics.positional_collision_profile`` / ``segment_length_histogram``
    (which are unit-tested) but without materializing the corpus.

    Pass 1: global Counter of content-hashed n-grams (memory ~ distinct n-grams;
            tune via ``stride``). Pass 2: per-position-bucket shared fraction +
            merged shared-run length histogram, split on-/off-prefix.

    Whole-prompt duplicates are dropped identically in both passes (deterministic
    first-occurrence in sorted-file order), so middle overlap is measured among
    OTHERWISE-DISTINCT prompts -- factoring out the reuse-whale exact-dup effect.
    """
    import time
    from collections import Counter

    from overlap_metrics import (
        _bin_index,
        _empty_hist,
        _hist_bins,
        ngram_hashes,
        position_bucket,
        shared_runs,
        whole_prompt_key,
    )

    files = _tokenized_files(data_dir, tokenized_glob or None)
    out_dir = output_dir or OUTPUT_DIR
    print(
        f"ngram_structure: w={window} stride={stride} over {len(files)} files (env={ENVIRONMENT_NAME})"
    )
    t0 = time.time()

    # ---- Pass 1: global n-gram counts (with streaming whole-prompt dedup) ----
    global_counts: Counter = Counter()
    seen_prompts: set[bytes] = set()
    kept = dropped = 0
    for _, token_ids in _iter_sequences(files, min_sequence_len):
        if drop_whole_prompt_dupes:
            k = whole_prompt_key(token_ids)
            if k in seen_prompts:
                dropped += 1
                continue
            seen_prompts.add(k)
        kept += 1
        for h, _start in ngram_hashes(token_ids, window, stride):
            global_counts[h] += 1
        if kept % 50000 == 0:
            print(
                f"  pass1 {kept:,} kept ({dropped:,} dup) | {len(global_counts):,} ngrams | {time.time() - t0:,.0f}s"
            )
    del seen_prompts

    def _is_shared(h: bytes) -> bool:
        return global_counts[h] >= 2

    # ---- Pass 2: position buckets + segment-length histogram ----
    shared = [0] * n_buckets
    total = [0] * n_buckets
    edges = _hist_bins(window)
    on = _empty_hist(edges)
    off = _empty_hist(edges)
    seen_prompts2: set[bytes] = set()
    n2 = 0
    for _, token_ids in _iter_sequences(files, min_sequence_len):
        if drop_whole_prompt_dupes:
            k = whole_prompt_key(token_ids)
            if k in seen_prompts2:
                continue
            seen_prompts2.add(k)
        L = len(token_ids)
        for h, start in ngram_hashes(token_ids, window, stride):
            b = position_bucket(start, L, window, n_buckets)
            total[b] += 1
            if global_counts[h] >= 2:
                shared[b] += 1
        for first_start, span, anchored in shared_runs(token_ids, _is_shared, window, stride):
            tgt = on if anchored else off
            tgt["counts"][_bin_index(span, edges)] += 1
            tgt["runs"] += 1
            tgt["tokens"] += span
        n2 += 1
        if n2 % 50000 == 0:
            print(f"  pass2 {n2:,} | {time.time() - t0:,.0f}s")

    buckets = [
        {
            "bucket": i,
            "range_pct": [round(100.0 * i / n_buckets, 1), round(100.0 * (i + 1) / n_buckets, 1)],
            "total_windows": total[i],
            "shared_windows": shared[i],
            "shared_pct": 100.0 * shared[i] / total[i] if total[i] else 0.0,
        }
        for i in range(n_buckets)
    ]
    common = {
        "environment": ENVIRONMENT_NAME,
        "file_prefix": FILE_PREFIX,
        "num_files": len(files),
        "window": window,
        "stride": stride,
        "min_sequence_len": min_sequence_len,
        "dropped_whole_prompt_dupes": drop_whole_prompt_dupes,
        "num_sequences_kept": kept,
        "num_whole_prompt_dupes_dropped": dropped,
        "distinct_ngrams": len(global_counts),
        "elapsed_seconds": time.time() - t0,
    }
    profile = {
        **common,
        "n_buckets": n_buckets,
        "overall_shared_pct": (100.0 * sum(shared) / sum(total)) if sum(total) else 0.0,
        "interpretation": (
            "shared_pct high only in bucket 0 => pure prefix (nothing new beyond "
            "the 55% trie). shared_pct high in late buckets => genuine "
            "middle/suffix overlap that prefix metrics miss."
        ),
        "buckets": buckets,
    }
    histogram = {
        **common,
        "interpretation": (
            "off_prefix heavy tail (long runs) => few big middle chunks. "
            "off_prefix mass at short runs => many small repeated segments. "
            "on_prefix runs are the classic shared-prefix."
        ),
        "on_prefix": on,
        "off_prefix": off,
    }
    _write("position_bucket_profile.json", profile, out_dir, commit_volume)
    _write("segment_length_histogram.json", histogram, out_dir, commit_volume)
    return {"profile": profile, "histogram": histogram}


@app.local_entrypoint()
def analyze_all(
    block_sizes: str = DEFAULT_BLOCK_SIZES,
    window: int = 64,
    stride: int = 16,
    n_buckets: int = 4,
    drop_whole_prompt_dupes: bool = True,
    data_dir: str = "/data",
    tokenized_glob: str = "",
):
    print(f"GLM-5.1 overlap-structure analysis (env={ENVIRONMENT_NAME})")
    print("Stage 1/2: block-size sweep ...")
    sweep = block_sweep.remote(
        block_sizes=block_sizes, data_dir=data_dir, tokenized_glob=tokenized_glob
    )
    for r in sweep["sweep"]:
        print(
            f"  bs={r['block_size']:>4}: content_tok={r['content_token_reuse_pct']:6.2f}% "
            f"chained_tok={r['chained_token_reuse_pct']:6.2f}% "
            f"(content-chained={r['content_minus_chained_token_pct']:+6.2f}pp)"
        )
    print("Stage 2/2: n-gram positional profile + segment histogram ...")
    ng = ngram_structure.remote(
        window=window,
        stride=stride,
        n_buckets=n_buckets,
        drop_whole_prompt_dupes=drop_whole_prompt_dupes,
        data_dir=data_dir,
        tokenized_glob=tokenized_glob,
    )
    for b in ng["profile"]["buckets"]:
        print(
            f"  pos {b['range_pct'][0]:.0f}-{b['range_pct'][1]:.0f}%: shared {b['shared_pct']:5.1f}%"
        )
    print("Done. Pull JSONs off the volume to verify; then `modal app list` / `modal app stop`.")
