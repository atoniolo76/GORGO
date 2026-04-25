# Main-branch recon: parallel research lineage on `origin/main`

**Scope**: read-only survey of `origin/main` (sha `ec5a9e0`) vs `rome`. No
merges, no commits beyond this report, no follow-up beads filed.

**Context**: rome diverged from main early in this project. `go-5m8`
flagged that `src/routing_harness/policies/` exists only on rome; this
recon goes the other way — what's on main that rome lacks.

## Summary

`origin/main` and `rome` are essentially **non-overlapping siblings**, not a
fork-then-add pattern. Diff is `+8,610 / −99` lines from the rome side,
and `+6,554 / −1` from the main side, on completely disjoint paths.

- **rome lineage**: simulation harness — `src/routing_harness/{policies,
  simulator, workload, cost_model, kv_cache}`, full pytest suite, no live
  serving path.
- **main lineage**: live Modal-based deployment — SGLang engine launcher,
  HTTP proxy with live routing + KV-cache trie, Prometheus metrics
  scraping, plus a HuggingFace dataset prep + analysis pipeline. No
  simulator, no test suite, no cost model.

The two lineages implement overlapping *concepts* (routing policies,
prefix-cache awareness, load metrics) using completely different code,
data structures, and deployment assumptions. Names of policies often
match (`least-request`, `least-kv-cache`, `prefix-cache`, etc.) — main
borrows the Aibrix gateway taxonomy, which rome's policy library mirrors
in spirit.

## What `origin/main` is doing that rome isn't

20 commits on `main` since divergence. Three threads:

1. **Live serving stack on Modal** (Apr 2026 first half)
   - `engine/modal_sglang.py` — launches an SGLang model replica on
     Modal, configurable via env (`REGION`, `GPU_TYPE`, `MODEL_ORG`,
     `MODEL_NAME`, `N_GPUS`). Pinned image
     `lmsysorg/sglang:nightly-dev-cu13-20260411-0011d2ae`, model
     `Qwen/Qwen3.5-35B-A3B-FP8` by default.
   - `proxy/modal_proxy.py` (833 lines) — full async HTTP proxy in front
     of an SGLang replica pool. OpenAI chat-completions passthrough,
     pluggable routing policies, background metrics-refresh loop, live
     RadixTrie of forwarded prompts (each node tagged with replica URL
     to approximate "which replica has this prefix in KV"), kv-cache
     flush endpoint, runtime policy / replica / hyperparameter mutation
     via control-plane HTTP.
   - `proxy/openapi.yaml` (506 lines) — full OpenAPI 3.1 spec for the
     proxy's control plane and proxy routes.
   - `app.py` — Modal app + named volumes (replicas dict, completions,
     hf-datasets, lmsys-chat-1m).
   - `proxy/tests/sglang_metrics.txt` — sample SGLang Prometheus
     `/metrics` output captured for parser regression.

2. **Routing policy library — Aibrix flavor, scraped-metrics inputs**
   - `utils/lb_aibrix.py` (454 lines). Implements 17+ policies under
     kebab-case names matching `vllm-project/aibrix`'s
     `pkg/plugins/gateway/algorithms`:
     `random`, `gorgo`, `power-of-two`, `least-request`, `least-load`,
     `least-kv-cache`, `least-gpu-cache`, `least-latency`,
     `least-utilization`, `least-busy-time`, `throughput`, `pack-load`,
     `prefix-cache`, `queue-router`, `simple-session-affinity`,
     `vtc-basic`, `slo` family, `fallback`, plus a `pd` stub.
   - `route_gorgo` is the multi-objective baseline:
     `score = latency + (request_tokens − cached_tokens) * t_prefill +
     (queued_tokens + used_kv_tokens) * queued_tokens_weight`. Two
     hyperparameters: `t_prefill`, `queued_tokens_weight`.
   - `route_prefix_cache` mirrors Aibrix's prefix-cache router: imbalance
     check → least-request fallback; otherwise pick max-cached-prefix
     replica filtered to those under `mean_running + std_factor*std`.
   - `utils/radix_trie.py` (247 lines) — path-compressed radix trie over
     uint32 token ids (`array('I')` for 4-byte tokens vs ~28 for Python
     ints). `RadixNode.replica_endpoints` records which replicas hold KV
     for each prefix; lazy-allocated, dedup on append.

3. **HuggingFace dataset pipeline + global prefix-overlap analysis**
   - `data_processing/download_hf_dataset.py` — downloads HF datasets
     to a Modal volume.
   - `data_processing/build_hf_prefix_trie.py` (1,972 lines) — builds
     radix tries over HF `save_to_disk` datasets (LMSYS-Chat-1M,
     WildChat-4.8M); reports intra-user vs cross-user vs global prefix
     savings (`A`, `C`, `B = A + C` definitions). Multiple ingestion
     modes: `--dedup-content-prefix-sha256` (WildChat default),
     `--ingest-all-rows` (no dedup), per-IP partitioning. Tokenizes via
     tiktoken `gpt-4o`. Designed to run on Modal w/ 256 GiB.
   - `data_processing/build_prefix_trie.py` — older pipeline over
     `llm_responses_*.parquet` (April 2026 GLM-5.1 traffic).
   - `data_processing/build_eval_dataset.py` — extracts per-session
     longest conversation from April 2026 dump; computes per-message
     token counts; one row per maximum-KV-footprint session.
   - `data_processing/load_completions.py`,
     `data_processing/download_db.py`,
     `data_processing/dump_sample_requests.py`,
     `data_processing/query_wildchat_duplicate_conversations.py` —
     supporting ingest / sampling / spot-check scripts.
   - Results checked in:
     - `data_processing/prefix_trie_results/wildchat/{stats.json,
       analysis.md}`
     - `data_processing/prefix_trie_results/lmsys-chat-1m/{stats.json,
       analysis.md}`
     - `data_processing/prefix_trie_results/glm-5.1-completions/stats.json`

## Policies / routing logic

Reference: `utils/lb_aibrix.py` on main vs
`src/routing_harness/policies/` on rome.

| Concept                  | rome (`policies/*.py`)        | main (`lb_aibrix.py`)                  |
|--------------------------|-------------------------------|----------------------------------------|
| random                   | `random.py`                   | `route_random`                         |
| least-request            | `least_request.py`            | `route_least_request`                  |
| least-kv-cache           | `least_kv_cache.py`           | `route_least_kv_cache` (+gpu alias)    |
| least-latency            | `least_latency.py`            | `route_least_latency`                  |
| least-busy-time          | `least_busy_time.py`          | `route_least_busy_time` (= util)       |
| throughput               | `throughput.py`               | `route_throughput` (max gen/s)         |
| prefix-cache (Preble)    | `prefix_cache_preble.py`      | (not on main)                          |
| prefix-cache (Aibrix)    | `prefix_cache.py`             | `route_prefix_cache`                   |
| session affinity         | `session_affinity.py`         | `route_simple_session_affinity`        |
| pd-disaggregation        | `pd.py`                       | `route_pd_stub` (random; not modeled)  |
| vtc-basic                | `vtc_basic.py`                | `route_vtc_basic` (no client id)       |
| **gorgo multi-objective**| (cost-model based; `cost_model.py`, `core.py`) | `route_gorgo` (live: latency + prefill + queue) |
| power-of-two             | (not in rome)                 | `route_power_of_two`                   |
| least-load (combined)    | (not in rome)                 | `route_least_load`                     |
| pack-load                | (not in rome)                 | `route_pack_load` (median + 2×MAD cap) |
| queue-router             | (not in rome)                 | `route_queue_router`                   |
| slo family               | (not in rome)                 | `route_slo_family` (mapped to load)    |
| fallback                 | (not in rome)                 | `route_fallback` (= least-request)     |

**Headline differences**:
- main runs against **scraped SGLang `/metrics`** (`num_running_reqs`,
  `num_queue_reqs`, `num_used_tokens`, `gen_throughput`, `utilization`,
  plus proxy-side latency & queued-token bookkeeping). rome runs against
  a **simulator state** (cost model + KV cache).
- main's `route_gorgo` is a **simpler** scoring function than rome's:
  `latency + effective_prefill * t_prefill + (queued + used_kv) * w`.
  No fabric contention, no multi-component cost model, no calibration.
- main's `route_prefix_cache` ≈ Aibrix reference; rome has both this
  flavor and a Preble flavor (`prefix_cache_preble.py`) — main has no
  Preble equivalent.
- main has policies rome lacks: `power-of-two`, `least-load`,
  `pack-load`, `queue-router`, `slo` family, `fallback`. Several are
  Aibrix-specific; some (e.g. `pack-load` median+MAD cap) are non-trivial.
- main's `radix_trie.RadixTrie` is **live**, mutated per request; rome's
  `kv_cache.py` is the simulator's KV model. Different lifecycles
  (forever-growing vs LRU/sim-driven).

## Workload / dataset code

main has **no synthetic / replay workload generator** — it's serving
real traffic. Its dataset code is for **measuring** (offline analysis),
not for **driving** simulations.

rome's `src/routing_harness/workload/{lmsys, sharegpt, code_completion,
synthetic, trace}.py` has no equivalent on main.

What main has on the data side (no rome equivalent):
- A HF-dataset download → tiktoken tokenize → radix-trie ingest →
  intra/cross/global prefix-savings pipeline (`build_hf_prefix_trie.py`).
- A GLM-5.1 (`llm_responses_*.parquet`) ingest pipeline that produces
  per-session longest conversations (`build_eval_dataset.py`,
  `build_prefix_trie.py`).
- Three checked-in result snapshots (WildChat 9.36B tokens, LMSYS
  466.8M tokens, glm-5.1 8.65B tokens).

## Reports / docs worth borrowing

The two analysis writeups under
`data_processing/prefix_trie_results/*/analysis.md` are dense and
publication-style. Highlights:

- **WildChat**: `B = 34.35%` global prefix savings under perfect
  pooling, broken into `A = 5.30%` intra-IP + `C = 29.06%` cross-IP.
  Discusses why token-weighted savings ≫ pairwise-similarity numbers
  (~2.5% in the original paper), the role of shared system prompts,
  staircase artifacts, whale-IP regimes (~75–83% intra savings vs ~0.7%
  diverse heavy users). 192M tokens (~2%) come from the top 10 IPs.
- **LMSYS-Chat-1M**: `B = 8.95%` global savings; flags `A = 0%` as a
  schema artifact (no user/IP key → falls back to `conversation_id`
  which is unique per row). Cleanly explains why LMSYS A/C should not
  be reported separately. Comparison table with WildChat is tight.
- **glm-5.1-completions**: `B = 55.30%` global savings on April 2026
  GLM-5.1 traffic (411k sessions, 8.65B tokens, 4,984 users). `A =
  53.67%` dominates → first-party traffic is much more reuse-heavy
  than open chat datasets.

These would make solid "dataset_metrics_*" companions to rome's
`research/reports/dataset_metrics_*.md` series.

## Recommendations (for scout to triage — no beads filed by polecat)

Ordered by what looks most decision-relevant:

1. **Port the Aibrix policy taxonomy gaps into rome's harness.**
   `power-of-two`, `least-load` (combined running+queue+used-KV+queued-prompt),
   `pack-load` (median+2×MAD soft-cap), `queue-router`, and the `slo`
   family are absent from rome. They're cheap to implement (10–60 LoC
   each) and would make rome's policy comparison table comprehensive
   relative to the real-world Aibrix gateway. `route_pack_load` in
   particular is non-trivial and a worthy addition. — refs:
   `utils/lb_aibrix.py:179-249, 254-276` on main.

2. **Compare rome's `gorgo` cost-model policy against main's live
   `route_gorgo`.** Same name, very different formulation. main:
   `latency + (req−cached)*t_prefill + (queued+used_kv)*w`. rome: full
   cost model with calibration (go-8cm), fabric contention, etc. Worth
   either (a) running main's simpler scoring as a baseline policy in
   rome's simulator, or (b) reconciling whether the simulator could
   produce numbers comparable to live deployment. — refs:
   `utils/lb_aibrix.py:307-336` on main, `src/routing_harness/cost_model.py`
   on rome.

3. **Pick up the WildChat / LMSYS prefix-overlap analyses as
   external validation.** rome's `routing-comparison.md` and
   `dataset_metrics_*.md` series could cite WildChat `B=34%` and LMSYS
   `B=9%` as the empirical ceiling for prefix-cache-aware routing on
   open chat traffic, with the glm-5.1 `B=55%` as an internal-traffic
   data point. The WildChat analysis's distinction between
   token-weighted (trie) and pairwise (paper) savings is directly
   applicable to how rome reports prefix savings. — refs:
   `data_processing/prefix_trie_results/{wildchat,lmsys-chat-1m,glm-5.1-completions}/`.

4. **Borrow main's RadixTrie design for any live-trie work in rome.**
   `array('I')` uint32 edges + path compression + per-node
   `replica_endpoints` list is a tight implementation. If rome's
   simulator KV-cache ever needs a multi-replica prefix-occupancy
   index, this is a near-drop-in. — refs: `utils/radix_trie.py:1-247`.

5. **Treat main's `proxy/modal_proxy.py` as the eventual deployment
   target for rome's policies.** The control-plane shape (POST `/policy`,
   `/replicas`, `/hyperparameters`, `/flush_cache`) is well-defined in
   `proxy/openapi.yaml`. If rome wants to validate any policy on real
   traffic, the cheapest path is "implement rome's policy in a function
   that matches `route(...)`'s signature in `lb_aibrix.py` and drop it
   in." Worth knowing this exists rather than re-deriving the proxy
   shape from scratch. — refs: `proxy/modal_proxy.py:104-260`,
   `proxy/openapi.yaml`.

6. **Don't merge the lineages.** They're solving different problems
   (sim vs serve) on incompatible scaffolding (`src/routing_harness/`
   vs flat top-level dirs, `pyproject.toml` vs none, pytest vs no
   tests). A merge would be all-or-nothing pain. Cherry-picking ideas
   per the items above is the right shape.

## Appendix: methodology

```
git fetch origin main                                      # ec5a9e0
git log --oneline rome..origin/main                        # 20 commits
git diff --stat rome...origin/main                         # 29 files +8610/-99
git diff --stat origin/main...rome                         # 70 files +6554/-1
git ls-tree -r origin/main --name-only                     # 27 files total on main
```

main and rome share only the trivial (README, .gitignore, models/qwen
shell), so the diff figures above are essentially full-tree adds on
each side.
