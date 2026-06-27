# GLM-5.1 content-overlap structure: prefix vs. middle/overall

*Investigation for Rome's question: "how much of the GLM-5.1 content overlaps as
far as the **prefix** goes, but also **overall** — few big middle chunks, or many
small repeated segments?"*

Worktree: `hypatia_glm_wt`, branch `glm-overlap-investigation`. Every code fact below
is cited `file:line`; every number is cited to its source JSON. No raw GLM-5.1 content
was read and no Modal job was run (data is non-redistributable and lives only on a
Modal volume).

---

## 1. Answer up front

- **Prefix overlap is large and real: ~55% of GLM-5.1 prompt tokens are redundant
  under perfect prefix sharing** (radix trie `global_savings_pct = 55.30%`,
  `prefix_trie_results/glm-5.1-completions/stats.json:19`). Almost all of it is
  **intra-user** (A = 53.67%, `stats.json:17`); cross-user sharing is tiny (C = 1.63%,
  `stats.json:18`). Shape: a *within-tenant* phenomenon — the same client resubmitting
  growing/near-identical long prompts (agentic/automated traffic), **not** a shared
  template across many users (the WildChat pattern).
- **The ~84% "block reuse" number does NOT establish middle/overall overlap.** The
  block hash is **prefix-dependent (chained)** — `build_mooncake_trace.py:428-444`
  literally does `h.update(prev_digest)` so a 512-token block matches *only when the
  entire preceding prefix is byte-identical*. So 84% is still **strict prefix reuse**,
  measured at 512-token granularity on a much hotter population (a 30-minute window of
  repeat-heavy clients), not evidence of repeated middle chunks. The 55→84 gap is a
  population/granularity artifact, not the answer Rome is looking for.
- **Therefore: the existing artifacts measure prefix reuse only. Whether GLM-5.1 also
  has genuine non-prefix overlap (few big middle chunks vs. many small repeated
  segments) is currently UNMEASURED.** Answering it requires a new, content-hashed
  (prefix-independent) analysis on the raw tokens — designed in §5, flagged for Modal
  spend approval.

---

## 2. Prefix overlap (established): the 55% / 53.67% radix-trie result

**What the trie measures.** `utils/radix_trie.py` is a path-compressed radix trie over
uint32 token-id sequences. `unique_token_count()` returns "the sum of all compressed
edge lengths = KV-cache footprint after perfect prefix sharing"
(`radix_trie.py:229-238`). The savings metric is
`saved = total_tokens_inserted − unique_tokens_in_trie` (`build_prefix_trie.py:13-15`,
computed at `:195-208`). A radix trie shares **only common prefixes** — two sequences
collapse exactly up to their first differing token, then split (`radix_trie.py:119-145`).
So this number is, by construction, **strict prefix overlap** and nothing else.

**The split** (`build_prefix_trie.py:206-208`, definitions):
- **A — intra-user** `(T − Σ_u U(R_u)) / T` — prefixes shared within one user's own
  requests (grouped by `token_hash`).
- **C — cross-user** `(Σ_u U(R_u) − U(all)) / T` — extra sharing unlocked by pooling
  across users.
- **B — global** `(T − U(all)) / T = A + C`.

**GLM-5.1 results** (`prefix_trie_results/glm-5.1-completions/stats.json`), over
411,169 sessions / 8,652,547,293 tokens / 4,984 users (April week-1, files
`llm_responses_202604…` up to `…20260408`):

| metric | value | source |
|---|---:|---|
| A — intra-user prefix savings | **53.67%** | `stats.json:17` |
| C — cross-user extra | **1.63%** | `stats.json:18` |
| **B — global prefix savings** | **55.30%** | `stats.json:19` |
| total tokens T | 8.65B | `stats.json:9` |
| unique (post-share) footprint | 3.87B | `stats.json:12` |
| users | 4,984 | `stats.json:11` |

**Shape of the prefix reuse — two things stand out:**

1. **It is almost entirely intra-user (A=53.67 vs C=1.63).** GLM-5.1 reuse is one
   client resubmitting similar prompts, *not* many clients sharing a system header.
   This is the **opposite** of WildChat (C=29% cross-user, §4) and is the signature
   of agentic/automated single-tenant traffic (long prompts that grow turn-over-turn,
   or the same harness prompt fired repeatedly).

2. **Whale users with extreme self-reuse drive A.** From `top_users` (`stats.json:21-72`),
   per-user intra-savings `= (tokens − unique)/tokens`:

   | token_hash (prefix) | tokens | unique | intra-savings |
   |---|---:|---:|---:|
   | `6b094a4b…` | 41.0M | 4.57M | **88.8%** |
   | `361d1f2a…` | 45.0M | 8.34M | **81.5%** |
   | `492225852…` | 54.2M | 19.9M | 63.3% |
   | `d5b86a14…` | 47.0M | 17.9M | 61.9% |
   | `9a1457ef…` | 75.5M | 30.7M | 59.4% |
   | `ac1c66a1…` | 56.1M | 30.8M | 45.1% |
   | `f3bde45b…` | 85.1M | 52.3M | 38.6% |
   | `652d8110…` | 40.3M | 26.3M | 34.8% |
   | `f4614448…` | 45.7M | 33.4M | 26.9% |
   | `514f31de…` | 62.9M | 50.0M | 20.5% |

   This is the same **two-regime structure** the WildChat writeup names
   (`prefix_trie_results/wildchat/analysis.md:48-52`): *heavy-reuse bots* (80–89%
   self-savings — automated clients hammering near-identical prompts) vs *diverse heavy
   users* (20–35% — a busy endpoint behind many distinct requests). The top-10 users
   hold ~553M tokens ≈ **6.4% of T**, so concentration is real but moderate; the bulk
   of the 53.67% comes from the broad middle of the 4,984-user distribution, weighted
   toward the reuse-whales.

**Caveat on the unit.** The trie ingests, per session, the **longest** conversation
(max-message) for that `token_hash`+first-message fingerprint
(`build_eval_dataset.py:90-113`, `:236-241`); `prompt_ids` is all messages of that
conversation concatenated (`:117-134`). So "prefix reuse" here is across **maximal
session prompts** — a clean KV-footprint ceiling. (The staircase of turn-1, turn-2, …
within a session is collapsed to the maximal chain, so intra-user savings here is
*conservative* relative to a per-request trace.)

---

## 3. The block metric: what it actually hashes (the crux)

**Verdict: PREFIX-DEPENDENT (chained), not content-hashed.** The block-id function in
`build_mooncake_trace.py:424-445` is:

```python
prev_digest = b""
for i in range(0, len(token_ids), block_size):
    block = token_ids[i : i + block_size]
    # Prefix-aware: chain each block's digest into the next so two
    # requests with a shared K-token prefix produce identical hash_ids[:ceil(K/B)].
    h = hashlib.sha256()
    h.update(prev_digest)                 # <-- chains in ALL preceding blocks
    h.update(b"".join(t.to_bytes(4, "little", signed=False) for t in block))
    digest = h.digest()
    ...
    prev_digest = digest
```

Because block *i*'s digest folds in `prev_digest` (block *i−1*'s digest, which folded
in *i−2*, …), **block *i* matches another request's block *i* only if blocks 0..*i* are
byte-identical** — i.e. the whole prefix matches. This is exactly vLLM/SGLang
prefix-cache block hashing. **The identical chained construction is used everywhere
block reuse is computed in this repo** — there is *no* content-hashed/independent-block
path anywhere:
- `build_mooncake_trace.py:428-444` (the summaries Rome cited),
- `export_metadata_trace.py:147-159` ("Prefix-aware block hashing (same as
  build_mooncake_trace.py)"),
- `build_synthetic_trace.py:232-244`.

**What `block_reuse_pct` / `unique_token_share_pct` are** (`build_mooncake_trace.py`):
- `block_reuse_pct = 100·(total_blocks − unique_blocks)/total_blocks` (`:1005-1007`)
  — occurrence-weighted fraction of 512-token blocks that are *repeat* occurrences of
  an already-seen **prefix-chained** block.
- `unique_token_share_pct = 100·total_unique_input_tokens/total_input` (`:1011-1013`),
  where per row `unique_input_tokens = min(new_block_count·block_size, input_length)`
  (`:929-930`) — the token-weighted analog, directly comparable to the trie's B.

**Why 84% > 55% — all prefix/duplication effects, none of them middle overlap.**
`glm5_0030_to_0100.summary.json`: `block_reuse_pct = 84.28%`,
`unique_token_share_pct = 17.06%` (i.e. 82.94% of tokens prefix-saved), but this is a
**different, far hotter population**:
1. **Population/scope.** It's a single **30-minute** chronological window
   (`selection_mode: "chronological"`, `start/end 2026-04-01T00:30→01:00`), 4,817
   rows / 19.3M input tokens / **184** distinct `token_hash`es — ~0.22% of the week's
   tokens, vs. the trie's full week / 4,984 users / 8.65B tokens. Hot windows
   concentrate the reuse-whales.
2. **Time locality + duplication.** The window is dominated by a few automated clients
   firing near-identical prompts: top `token_hash 030a5259…` = 2,832 rows / 22,656
   tokens (≈8 tok/row — keep-probe-ish), `d5b86a14…` = 126 rows / 1.72M tokens
   (≈13.6k-token prompt repeated) (`glm5_0030_to_0100.summary.json:48-58`). Exact /
   near-exact prompt repeats collapse to ~zero unique blocks — which is *whole-prompt
   prefix* reuse, the strongest possible prefix case.
3. **Block quantization** rounds shared content up to whole 512-token blocks in emit
   order (`:929-930`), collapsing duplicate prompts harder than the per-token trie.
4. **Length cap** `max_input_tokens=24000` drops the 1,868 longest, most-divergent
   prompts in the window (`skipped_over_max_input`), biasing toward short repetitive
   traffic.

Across the four GLM windows the block reuse is 72.7–84.3% (`glm5_*` summaries), all
chronological, all on the same hot-window basis. **Every one of these drivers is a
prefix or whole-prompt duplication effect.** Because the hash is prefix-chained, a
512-block *cannot* be credited as reused on the strength of matching middle content
while its prefix differs. **So the 55→84 gap carries no information about middle-chunk
overlap.**

---

## 4. Comparison baselines (full-dataset radix trie, same methodology)

| dataset | T | users | A (intra) | C (cross) | **B (global prefix)** | source |
|---|---:|---:|---:|---:|---:|---|
| **GLM-5.1** | 8.65B | 4,984 | 53.67% | 1.63% | **55.30%** | `glm-5.1-completions/stats.json` |
| WildChat-4.8M | 9.36B | 1.83M | 5.30% | 29.06% | **34.35%** | `wildchat/stats.json` |
| LMSYS-Chat-1M | 466.8M | 1.00M* | 0.00%* | 8.95% | **8.95%** | `lmsys-chat-1m/stats.json` |

\* LMSYS has no user key; it falls back to `conversation_id` (unique per row), forcing
A=0 and C=B — an artifact, not a finding (`lmsys-chat-1m/analysis.md:17-26`). Only B is
comparable across datasets.

**Read:** GLM-5.1 has **far more prefix reuse than either public baseline** (55% vs 34%
vs 9%), and its reuse is **structurally different**: GLM is intra-user-dominated
(53.67/1.63), WildChat is cross-user-dominated (5.30/29.06, driven by a shared ChatGPT
proxy's system prompts and viral copy-paste prompts, `wildchat/analysis.md:22-27`).
GLM-5.1 looks like single-tenant agentic traffic; WildChat looks like a shared template
hub. Methodological notes worth borrowing for the paper: the trie metric is
**token-weighted** (a 500-token preamble shared by 10k users contributes ≈5M
cross-user tokens), so trie-C is much larger than the paper's *pairwise* cross-user
similarity (~2.5%) and must be labeled "extra savings from pooling (token-weighted KV
footprint)", not "cross-user similarity" (`wildchat/analysis.md:27,66`). The same
caveat applies to GLM's C=1.63%.

(The `wildchat_*`/`lmsys_*` *block* summaries in `results/trace_summaries/` show 5–8%
block reuse, but those are tiny eval windows with 2–25 users — not comparable to the
full-dataset trie B, and a separate population again. Use the trie B row above for
cross-dataset comparison.)

---

## 5. Verdict on Rome's hypotheses

| claim | status |
|---|---|
| Prefix overlap is substantial (~55%), almost all intra-user, driven by reuse-whale clients resubmitting near-identical long prompts | **ESTABLISHED** (§2) |
| GLM-5.1 has more prefix reuse than WildChat/LMSYS and is intra-user-shaped (vs WildChat's cross-user template sharing) | **ESTABLISHED** (§4) |
| The 84% block number proves there are big middle chunks / many small repeated segments | **REFUTED** — block hash is prefix-chained (§3); 84% is prefix reuse on a hotter window |
| "Things differ in the beginning but share big middle chunks" (few big middle chunks) | **UNTESTED** — needs §6 analysis |
| "Many small overlapping segments" scattered through the body | **UNTESTED** — needs §6 analysis |

The honest summary: **what the current artifacts establish is that GLM-5.1 reuse is
mostly a shared *prefix* that then diverges (often a whole-prompt duplicate), and that
this prefix reuse is unusually high and intra-user.** They say *nothing* about whether,
after the shared prefix diverges, the **middles/suffixes** still collide — because both
existing metrics (radix trie and prefix-chained blocks) are prefix-anchored by
construction and are blind to non-prefix-aligned overlap. To distinguish "few big
middle chunks" from "many small repeated segments" we need a **content-hashed
(prefix-independent)** pass on the raw tokens, below.

---

## 6. Proposed Modal analysis — **NEEDS ROME APPROVAL (Modal spend)**

> ⚠ **No Modal run without Rome's explicit go.** This section is a ready-to-run spec
> only. Per the modal skill: all spend (even a smoke run) requires approval; auth being
> configured is not approval. Workspace `alessio-dev` / volume `GORGO-glm5-completions`
> (the GLM data tenant, `app.py:12-14`), **not** the shared `research` volume.

### Inputs (already on the volume — no re-tokenize, no re-download)
`/data/tokenized_llm_responses_202604/*.tokenized.parquet`, schema
(`build_eval_dataset.py:245-253`): `session_id`, `token_hash`, `message_count`,
`prompt_token_count`, **`prompt_ids` (list<uint32>)**. 336 files, 411,169 sessions,
8.65B tokens — the exact set the trie ran on, so results are directly comparable to the
55% baseline.

### New file: `data_processing/analyze_overlap_structure.py`
Reuse `from app import app, completions_volume`; mount `volumes={"/data":
completions_volume}`; image `debian_slim().pip_install("duckdb","numpy",
"pyarrow").add_local_python_source("app")`. Three measurements, each designed to
*separate* the hypotheses:

**(i) Block-size sweep, content-hashed vs prefix-chained (the decisive contrast).**
For `block_size ∈ {16, 64, 256, 512, 1024}` compute two global block-dedup numbers:
- *prefix-chained* reuse (reuse the exact `_block_ids` from
  `build_mooncake_trace.py:428-444`) — the existing definition; and
- *content-independent* reuse: hash each block on **its own tokens only** (drop the
  `h.update(prev_digest)` line), dedup globally.

Report token-weighted reuse `(T − unique_blocks·B)/T` for both at each size.
**Discriminator:** if content-independent reuse ≫ prefix-chained reuse at *small*
blocks (16/64) → **many small repeated segments** scattered off-prefix. If the two
curves coincide and only large blocks reuse → **prefix-dominated, no middle overlap**.
If content-independent reuse jumps only at *large* blocks → **few big aligned chunks**.

**(ii) Post-prefix (middle+suffix) collision via content-hashed rolling n-grams
(alignment-robust — the main event).** Fixed-window `w = 64` tokens, stride `s = 16`.
Two-pass, numpy sort-based (cheap, no Python dict over billions):
- Pass 1: for every session, emit `(hash64(prompt_ids[p:p+w]), position_bucket)` for
  sampled offsets `p`, where `position_bucket ∈ {0–25%, 25–50%, 50–75%, 75–100%}` of
  the prompt length. Accumulate into a growable `uint64` array (hashes) + `uint8` array
  (buckets).
- Pass 2: `np.sort` the hashes, find values with count ≥ 2 (collides across the
  corpus), and tabulate the **shared fraction per position bucket**.

**Discriminator:** if the shared fraction is high only in bucket 0 (0–25%) → pure
prefix (matches the 55% story, nothing new). If buckets 2–3 (50–100%) also show a high
shared fraction → **genuine middle/suffix overlap** that prefix metrics miss. To
neutralize the whale-bot exact-duplicate effect, also report the bucket profile after
**dropping sessions whose entire prompt is a duplicate** of an earlier one (so we
measure middle overlap *among otherwise-distinct prompts*, which is what Rome is really
asking).

**(iii) Matched-segment-length histogram.** Merge adjacent colliding windows from (ii)
into maximal "shared runs" and histogram run lengths (in tokens), separately for
on-prefix vs off-prefix runs. **Discriminator:** a heavy tail of long off-prefix runs →
**few big middle chunks**; mass concentrated at short runs → **many small segments**.
This is the literal picture Rome asked for.

### Resources / cost / runtime
- `memory = 1024*64` (64 GiB), `cpu = 8.0`, `timeout = 10800` (3 h). Sampled `uint64`
  hash arrays at stride 16 over 8.65B tokens ≈ 540M windows ≈ 4.3 GB per array — fits
  comfortably; `np.sort` on ~0.5B uint64 is minutes, not hours. (The trie job used 32
  GiB / `timeout=7200` and ran 2,993 s, `stats.json:5,20` + `build_prefix_trie.py:46-52`;
  this is lighter per token — hashing strided windows, not full radix insertion.)
- Fan out the block-size sweep with `.starmap` over the 5 sizes (or batch files) so the
  whole thing is one wall-clock pass.
- **Runtime estimate:** ~1.5–3 h single container (or ~45 min fanned out). **Cost:**
  comparable to the ~$1–2 trie job — **order single-digit USD** on CPU. (Exact CPU
  $/core-hr: *verify-current* against Modal pricing before quoting Rome a hard number.)
- Outputs (write under `/data/overlap_structure/`, `vol.commit()` after each):
  `blocksize_sweep.json`, `position_bucket_profile.json`,
  `segment_length_histogram.json`, plus a one-page `summary.md`.
- **Teardown:** plain `@app.function` (no `keep_warm`, no `@app.cls`) tears down on its
  own; still run `modal app list` after and `modal app stop` anything lingering.

### What each output *proves*
- Sweep (i): prefix-vs-content curves diverging at small block sizes = small-segment
  middle reuse exists; coinciding = it doesn't.
- Bucket profile (ii): where in the prompt the (alignment-robust) overlap lives —
  settles "shared prefix then divergence" vs "substantial shared middles".
- Histogram (iii): the *shape* of any non-prefix overlap — few big chunks vs many small
  segments — i.e. Rome's exact dichotomy.

Together these convert "we only know the prefix is 55%" into a quantified answer for the
overall/middle question, on the same dataset, directly comparable to the 55% trie
baseline.

---

## 7. Status: script built & unit-tested — run pending `alessio-dev` access

The analysis is **implemented and locally unit-tested**; only the (data-gated) Modal
run remains. Rome approved the analysis, but there is a **hard access blocker**: the
GLM-5.1 tokenized data lives only in Modal environment **`alessio-dev`**
(`app.py:3`, `ENVIRONMENT_NAME="alessio-dev"`), which this box's profiles cannot reach
(`research-exp` → main/interp/rome; `arcadia-research` → main/GORGO). The same-named
`GORGO-glm5-completions` volume in the reachable `GORGO` env is **empty** (Modal volume
names are per-environment). **Whoever runs this needs `alessio-dev` access (Rome or
Alessio).** No Modal job has been run.

**Files added** (this branch):
- `data_processing/overlap_metrics.py` — pure, stdlib-only metric logic (block
  digests content vs prefix-chained; rolling n-grams; positional buckets; shared-run
  merge; segment histogram). No Modal/duckdb/numpy, so it is unit-testable offline.
- `data_processing/analyze_overlap_structure.py` — Modal driver. Reuses the pipeline
  `app`, streams the SAME tokenized parquets the trie used
  (`/data/tokenized_<FILE_PREFIX>/*.tokenized.parquet`, cols `token_hash`,
  `prompt_ids`), `@app.function`s at module scope (`block_sweep`, `ngram_structure`)
  with the volume mounted on each, writes JSON under `/data/overlap_structure/` and
  `vol.commit()`s. CPU-only, 64 GiB / 8 CPU / 3 h timeout. **Streams off-disk** — never
  materializes the 8.65B-token corpus. Environment is **overridable** via
  `OVERLAP_MODAL_ENV` (defaults to `app.py`'s `alessio-dev`).
- `data_processing/tests/test_overlap_structure.py` — 6 local pytest cases on
  hand-built sequences with known overlap. **All pass** (`6 passed in 0.04s`). Key
  assertions: identical-middle-chunk → `content_block_reuse > 0` while
  `chained_block_reuse == 0` (the prefix-vs-content discriminator); many-small-segments
  → small-block reuse > large-block reuse and many short off-prefix runs; shared-prefix
  → content reuse == chained reuse and only on-prefix runs.

### How to run (once `alessio-dev` is reachable)

```bash
# Case A — original data, env alessio-dev (Rome / Alessio):
source /home/rome/.venv/bin/activate
MODAL_PROFILE=<profile-that-can-see-alessio-dev> modal config set-environment alessio-dev
cd /home/rome/gt/gorgo/crew/hypatia_glm_wt
modal run data_processing/analyze_overlap_structure.py::analyze_all     # both stages

# Case B — data copied into a reachable env (e.g. GORGO under arcadia-research):
export OVERLAP_MODAL_ENV=GORGO
MODAL_PROFILE=arcadia-research modal config set-environment GORGO
cd /home/rome/gt/gorgo/crew/hypatia_glm_wt
modal run data_processing/analyze_overlap_structure.py::analyze_all

# Individual stages / knobs (Modal lowercases flags; no single-letter names):
modal run data_processing/analyze_overlap_structure.py::block_sweep --block-sizes 16,64,256,512,1024
modal run data_processing/analyze_overlap_structure.py::ngram_structure --window 64 --stride 16
# If size-16 sweep or n-gram pass-1 OOMs at 64 GiB: run size 16 alone, or raise --stride.
```

After the run: pull the JSONs to verify (trust the artifact, not the logs) and confirm
teardown — `modal volume get GORGO-glm5-completions overlap_structure/blocksize_sweep.json /tmp/`,
then `modal app list` and `modal app stop` anything left running.

### Schema assumptions to verify against the real data before running
1. **Tokenized parquet columns** `token_hash` (string) and `prompt_ids` (list<uint32>)
   exist and `prompt_ids` is the flat per-session prompt token-id stream
   (`build_eval_dataset.py:245-253`; read identically to `build_prefix_trie.py:137-156`).
   *If the cache was rebuilt with a different schema, adjust the `SELECT`.*
2. **File set / window:** raw parquets matching `FILE_PREFIX="llm_responses_202604"`
   and `< FILE_CUTOFF="llm_responses_20260408"` each have a `*.tokenized.parquet`
   sibling (`tokenized_path_for`). The driver skips missing ones and errors if none
   exist — confirm the 336-file / 411,169-session cache is present in `alessio-dev`
   (it produced the 55% trie, so it should be).
3. **Same unit as the trie:** `prompt_ids` is the *longest* conversation per session
   (`build_eval_dataset.py:90-113`), so these results are directly comparable to the
   `global_savings_pct=55.30` baseline. If a per-request (non-maximal) cache is used
   instead, the comparison shifts — note it.
4. **Memory:** at `stride=16` the n-gram pass-1 Counter and the size-16 block set are
   the heavy structures; 64 GiB is the planned envelope but is unverified against the
   real token volume — be ready to raise `--stride` / split the size-16 sweep / bump
   container memory if the first run OOMs.
