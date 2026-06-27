# GLM-5.1 content-overlap structure — run anywhere

Measures how much of the GLM-5.1 prompt content overlaps **as a prefix** vs.
**off-prefix** (middle/suffix), and the *shape* of the off-prefix overlap (few big
chunks vs. many small segments). Companion to the prefix radix-trie
(`build_prefix_trie.py`, which only sees strict prefixes). Full method + findings:
[`prefix_trie_results/glm-5.1-completions/overlap_structure_analysis.md`](prefix_trie_results/glm-5.1-completions/overlap_structure_analysis.md).

## TL;DR — three commands

```bash
# 0. install just what this analysis needs (not the whole GORGO stack)
pip install -r data_processing/requirements-overlap.txt

# 1. prove it works on THIS machine — sample data, no Modal account, no spend
./data_processing/run_overlap.sh verify

# 2. run for real, in the Modal environment that holds the tokenized data
#    (this BILLS that account; defaults to env `alessio-dev`)
OVERLAP_MODAL_ENV=alessio-dev MODAL_PROFILE=<your-profile> \
  ./data_processing/run_overlap.sh run
```

The runner derives all paths relative to itself, so it works from any clone on any
machine. `verify` runs the **real** parquet→aggregate→JSON code path against
synthetic sample data with known ground truth (15 assertions) — if it passes, the
logic is good on your box and the only remaining variable is the real data.

## Prerequisites

- Python ≥ 3.12 with the deps above (a repo `.venv` or `$HOME/.venv` is
  auto-detected; otherwise set `PYTHON=/path/to/python`).
- For the **real run**: `modal setup` done once, and a Modal profile/account that
  can see the environment where the tokenized cache lives. The GLM-5.1 data is in
  Modal env **`alessio-dev`** (the only place the 8.65B-token tokenized cache
  exists); a profile without that environment will fail fast with
  `Environment '…' not found`.

## Configuration (env vars, all optional)

| var | meaning | default |
|---|---|---|
| `OVERLAP_MODAL_ENV` | Modal environment holding the data; also sets `MODAL_ENVIRONMENT` for the run | `alessio-dev` (baked into the driver) |
| `MODAL_PROFILE` | named `~/.modal.toml` profile to use | active profile |
| `PYTHON` | interpreter to use | auto-detected `.venv`, else `python3` |

To run somewhere other than `alessio-dev` (e.g. data copied into another env),
just set `OVERLAP_MODAL_ENV=<that-env>` — the driver is fully env-portable and
bakes the value into the container so output labels and `vol.commit()` stay
correct. The data must actually exist in that env (volume names are per-env).

## What it produces

Three JSON files, written to `/data/overlap_structure/` on the
`GORGO-glm5-completions` volume (and echoed as a console summary):

| file | answers |
|---|---|
| `blocksize_sweep.json` | how much overlaps overall vs. prefix-only, across block sizes 16–1024 (the `content_minus_chained_token_pct` column = the non-prefix overlap) |
| `position_bucket_profile.json` | *where* shared content sits (prompt quartiles): high only in bucket 0 = pure prefix; high in late buckets = real middle/suffix overlap |
| `segment_length_histogram.json` | the *shape*: off-prefix mass in short bins = many small segments; heavy long-run tail = few big chunks |

Pull and inspect after a run:

```bash
modal volume get GORGO-glm5-completions overlap_structure /tmp/overlap_out
```

## Stages & knobs

`run_overlap.sh run` calls the `analyze_all` entrypoint (block sweep + n-gram
profile + histogram). You can pass flags through, or run a single stage:

```bash
./data_processing/run_overlap.sh run --block-sizes 16,64,256,512,1024 --stride 16
# single stages (Modal lowercases flags; no single-letter names):
<PY> -m modal run data_processing/analyze_overlap_structure.py::block_sweep --block-sizes 16
<PY> -m modal run data_processing/analyze_overlap_structure.py::ngram_structure --window 64 --stride 16
```

Resolution: the rolling n-gram measures (the authoritative middle-overlap
signal) catch exact shared runs **≥ `--window` tokens (default 64)**, placed to
the nearest prompt-quarter; the block sweep reaches down to `--block-sizes 16`
but is alignment-fragile. It is exact-match only (no fuzzy/near-dup).

## Measurement (A): across-conversation within-user reuse

A second analysis (`analyze_user_reuse.py`) quantifies the reuse a
**session-affinity / per-user cache** can exploit *across a user's distinct
conversations* (user = `token_hash`, conversation = `session_id`), **excluding**
trivial within-conversation growth. It reuses the SAME tokenized cache (no
re-tokenize), groups by `token_hash` (one user at a time, peak RAM ~ the single
largest user), and reports two notions:

- **PREFIX** — a per-user radix trie; `savings = tokens − unique_trie_tokens`
  (shared system/tools head a prefix cache captures). Pooled prefix == the
  existing intra-user **A = 53.67%** by construction (a built-in cross-check).
- **CONTENT** — per-user content-hashed (prefix-independent) blocks, swept over
  `{16,64,256,512,1024}`: the fraction of a user's tokens appearing in **≥2** of
  their conversations. `content − prefix` = cross-conversation **middle** reuse
  (shared tool defs / RAG context / persona that aren't a clean prefix).

Plus **per-conversation attribution** ("how much of each new conversation is
already warm from the user's others"), a per-user percentile **distribution**,
and an activity-**tier** breakdown (whales vs light users).

```bash
# offline E2E (already covered by `verify`; multi-conversation-per-user sample):
./data_processing/run_overlap.sh verify
# real run (BILLS the env that holds the data; defaults to alessio-dev):
OVERLAP_MODAL_ENV=alessio-dev MODAL_PROFILE=<your-profile> \
  ./data_processing/run_overlap.sh user-reuse
# single function / knobs:
<PY> -m modal run data_processing/analyze_user_reuse.py::user_reuse \
     --block-sizes 16,64,256,512,1024 --attribution-block-size 512
```

Output: `/data/user_reuse/user_reuse.json` (pooled prefix vs content by block
size, per-user distribution percentiles, tier breakdown, top users, and the
A = 53.67% reconciliation). Note: the block-content metric is alignment-limited,
so a user whose only sharing is a *sub-block* prefix can read **below** the exact
trie prefix at large block sizes — which is exactly why both are reported.

## Troubleshooting

- **`Environment '…' not found`** → your `MODAL_PROFILE` can't see
  `OVERLAP_MODAL_ENV`. Use a profile/account that has that environment.
- **OOM on the full 8.65B-token corpus** (default 64 GiB / 8 CPU): raise
  `--stride` (32/64) to shrink the pass-1 n-gram counter, and/or run the heavy
  smallest block size alone: `run --block-sizes 16`.
- **Schema mismatch** → the driver expects tokenized parquets with columns
  `token_hash` (string) and `prompt_ids` (list<uint32>); if the cache was rebuilt
  differently, adjust the one `SELECT` in `analyze_overlap_structure._iter_sequences`.
- **After any run**, confirm nothing is left billing: `modal app list` should show
  no app in `running` state; `modal app stop <id>` if so.
