# Cost-Model Calibration Plan

> **Status:** Phase 1 approved. Phase 2 pipeline landed (go-8cm).
> Target model switched to Llama-3.1-8B-Instruct on 2026-04-25 (go-27g)
> after HF access was approved for that page; sweep rerun is gated on
> scout approval (tracking: go-26c, see §7). Tracking bead: go-8cm.
> Parent gap:
> research/reports/routing-comparison.md §9 item 2.
>
> **Execution environment.** All sweeps run on the
> `arcadia-research` Modal workspace in the `GORGO` environment. The
> launch invocation is always:
>
> ```bash
> MODAL_PROFILE=arcadia-research modal run --env=GORGO \
>     scripts/calibrate.py::main [--seed 0 ...]
> ```
>
> Do **not** run `modal profile activate` — concurrency-unsafe.
> `HF_TOKEN_ROME` is injected by the `hf_token_rome` Modal secret
> attached to the function and mapped to `HF_TOKEN` inside the
> container for vLLM / HuggingFace model download.

## 0. What we are calibrating

`AnalyticCostModel` (src/routing_harness/cost_model.py) exposes five
coefficients that are currently illustrative:

| Field | Where | What it represents |
|-------|-------|--------------------|
| `ComputeParams.prefill_ms_per_token` | compute | Marginal prefill cost per uncached prompt token |
| `ComputeParams.prefill_overhead_ms`  | compute | Per-request fixed prefill cost (kernel launch, bookkeeping) |
| `ComputeParams.decode_ms_per_token`  | compute | Single-request decode cost per output token at batch=1 |
| `ComputeParams.decode_overhead_ms`   | compute | Per-request fixed decode cost |
| `ComputeParams.decode_batch_k`       | compute | Continuous-batching amortization exponent in `1 / (1 + k · log(1 + (batch-1)))` |

Out of scope for this calibration (addressed elsewhere or deliberately deferred):

- `NetworkParams.*` — fabric model fitted separately (go-uy0 landed the
  fluid fair-share shape; RTT / bandwidth come from the target
  deployment spec, not a microbench).
- `SchedulerParams.*` — router-side; calibrated once we instrument a
  real router, out of scope here.
- `_RHO_MAX = 0.99` — pragmatic numerical clamp on the M/M/1 formula
  (go-4lp). Not a physical quantity; no need to calibrate.

## 1. Target model and hardware

**Proposal:** **Llama-3.1-8B-Instruct** (`meta-llama/Llama-3.1-8B-Instruct`)
on a single **A100-80GB** running vLLM (latest stable, currently
`v0.6.x` — `0.6.6` covers Llama-3.1 without further patches) in
single-replica mode.

Rationale:

- **8B / A100** is the most common reference point in public vLLM
  benchmarking; published numbers exist (vLLM release notes, Anyscale,
  Neural Magic) which we can sanity-check our fit against.
- Fits comfortably in 80 GiB with `--max-model-len 4096` and enough KV
  headroom to hit decode batch sizes of 64–128 — necessary to
  discriminate the `decode_batch_k` amortization curve.
- Smaller options (Llama-3.2-1B / 3B) would run on cheaper A10 / L4
  hardware but decode-batch dynamics differ enough (memory-bandwidth-
  bound vs compute-bound crossover moves) that the fitted `k` would
  not transfer. Explicit non-goal: we are not publishing a
  device-portable `k`.

Alternative considered and rejected: **Llama-3.1-70B** on 4×A100. Richer
data, but ~5× the spend for incremental value on coefficients whose
main use in the harness is *relative* policy comparison.

## 2. Platform

**Decision:** **Modal** only. Scout's Phase 2 approval explicitly
removed the prior Lambda Cloud fallback.

Rationale:

- **Reproducibility.** Modal functions are script-defined; the entire
  calibration run is a `scripts/calibrate.py` invocation checked into
  the repo. Re-running on a new model only requires editing the CLI
  flags and re-submitting.
- **Predictable on-demand pricing.** Modal A100-80GB is ~$3.40/hr
  on-demand (as of Q1 2026); for our ≤4-hour sweep this stays well
  under budget.
- **Teardown is automatic on Modal**, removing a whole class of
  "forgot to stop the box" cost-overrun risk.
- **Workspace pre-configured.** The `arcadia-research` workspace
  already has the `hf_token_rome` secret attached (exposing
  `HF_TOKEN_ROME` → `HF_TOKEN` for gated Llama-3.1 weights), and the
  `GORGO` env isolates this job from other arcadia research work.

Invocation pattern (always, every time):

```bash
MODAL_PROFILE=arcadia-research modal run --env=GORGO \
    scripts/calibrate.py [--seed 0] [--function-timeout 14400]
```

`MODAL_PROFILE` is prefixed per-invocation rather than via
`modal profile activate` because profile activation is global process
state and is not safe under concurrent polecat sessions.

## 3. Sweep design

We microbenchmark **against the vLLM OpenAI-compatible server**
(`vllm serve <model> --disable-log-requests --enforce-eager=false`) and
measure wall-clock latency per request from the client side, using
the `/v1/completions` endpoint with `ignore_eos=true` so output length
is exactly `max_tokens`. Metrics we capture per request:

- `ttft_ms` (time-to-first-token) → isolates prefill
- `tpot_ms` (time-per-output-token, excluding TTFT) → isolates decode
- `total_ms` → sanity-check that prefill + decode ≈ total

### 3.1 Prefill sweep

Isolates `prefill_ms_per_token` and `prefill_overhead_ms`. **No KV
cache reuse** — each request uses a fresh random prefix.

| Axis | Values | Count |
|------|--------|-------|
| `prompt_len` (tokens) | 64, 128, 256, 512, 1024, 2048, 4096 | 7 |
| `max_output_tokens`   | 1 (so TTFT dominates) | 1 |
| `batch_size` (concurrent requests) | 1 | 1 |
| `repeats` | 20 per point | 20 |

Total prefill-sweep requests: 7 × 20 = **140**. Expected wall-clock:
<10 min (average TTFT ~15–300 ms).

**Fit:** linear regression `ttft_ms = prefill_overhead_ms + prefill_ms_per_token · prompt_len`.
Report intercept, slope, R², and 95% CI on both coefficients.
Acceptance: R² ≥ 0.97; residual standard error ≤ 10% of mean TTFT.

### 3.2 Decode sweep (batch = 1)

Isolates `decode_ms_per_token` and `decode_overhead_ms`.

| Axis | Values | Count |
|------|--------|-------|
| `prompt_len` | 128 (fixed, small, to minimize prefill confounding) | 1 |
| `max_output_tokens` | 32, 64, 128, 256, 512, 1024 | 6 |
| `batch_size` | 1 | 1 |
| `repeats` | 15 per point | 15 |

Total: 6 × 15 = **90** requests. Expected wall-clock: ~20 min
(1024-token decodes at ~40 ms/token → 40 s each × 15 = 10 min on the
long tail alone; shorter points amortize).

**Fit:** linear regression `tpot_ms · max_output_tokens = decode_overhead_ms + decode_ms_per_token · max_output_tokens`.
Equivalently, `total_decode_ms = decode_overhead_ms + decode_ms_per_token · output_len`.
Acceptance: R² ≥ 0.98 on the decode-only segment.

### 3.3 Decode batching sweep

Isolates `decode_batch_k`. We send `batch_size` requests concurrently
with identical `max_output_tokens`, wait for all to complete, and
measure mean `tpot_ms` across the batch.

| Axis | Values | Count |
|------|--------|-------|
| `prompt_len` | 128 (fixed) | 1 |
| `max_output_tokens` | 128 (fixed, long enough to amortize startup) | 1 |
| `batch_size` | 1, 2, 4, 8, 16, 32, 64, 128 | 8 |
| `repeats` | 5 per point | 5 |

Total: 8 × 5 = **40** batch-invocations = ~**4,000** requests in
aggregate across batches. Expected wall-clock: ~90 min (batch=128 is
KV-memory-bound; smaller batches are fast).

**Fit:** minimize
`Σ (observed_tpot(batch) − decode_ms_per_token / (1 + k · log(1 + (batch − 1))))²`
over `k`, holding `decode_ms_per_token` fixed at the value from §3.2.
Report fitted `k`, residual standard error, and the batch=1 residual
(should be ≤ 2% by construction — sanity check).
Acceptance: residual standard error ≤ 15% of mean tpot; `k` in [0.1, 2.0].
If `k` lands outside that band, the logarithmic functional form is
suspect and we file a bead before publishing.

### 3.4 Seeding and determinism

vLLM decoding is not bit-reproducible across runs (floating-point
reduction order varies), but per-request latency is measured; we rely
on **repeated sampling** (20 / 15 / 5 per point, above) and report
mean ± 1 SD. No seed parameter is set on vLLM; the random prefix
generator uses a harness-side `numpy.default_rng(seed=<configurable>)`
so prompt content is reproducible. Seeds sweep: `[0, 1, 2]` for the
prefix generator only; each seed is an independent replication of the
full sweep, giving us 3× the measurements per point at 3× the cost if
we decide we need it. **Default plan: single seed** (seed=0); escalate
to 3 seeds only if §3.1 or §3.2 R² misses the acceptance threshold.

## 4. Cost estimate

| Phase | Wall-clock | Unit cost (Modal A100-80GB, $3.40/hr) | Subtotal |
|-------|-----------|---------------------------------------|----------|
| vLLM cold start + warmup | 10 min | $3.40/hr | $0.57 |
| §3.1 prefill sweep | 10 min | $3.40/hr | $0.57 |
| §3.2 decode sweep | 20 min | $3.40/hr | $1.13 |
| §3.3 batching sweep | 90 min | $3.40/hr | $5.10 |
| Slack (debug, retries, 50% buffer) | 65 min | $3.40/hr | $3.68 |
| **Total (single seed)** | **~3.25 hr** | | **≈ $11.05** |
| Optional: 3-seed replication | 9.75 hr | $3.40/hr | +$22 (⚠ over budget) |

Budget target was $5–15. **Single-seed plan lands at ~$11**, within
budget with headroom. The Modal function is launched with
`--function-timeout 4h` (14 400 s), so an accidental infinite loop
cannot spend more than ~$14 regardless.

Three-seed replication is over the $15 envelope and will only be run
if the single-seed acceptance criteria fail; it requires re-approval
from scout before launch.

## 5. Outputs (Phase 2 deliverables)

1. `scripts/calibrate.py` — one script that spins up vLLM (Modal
   function on A100-80GB in the `GORGO` environment), runs
   §3.1/§3.2/§3.3, writes raw measurements to a Modal Volume at
   `/vol/calibration/<timestamp>/` as JSONL, downloads them back to
   `research/data/calibration/<timestamp>/`, and emits a fit summary
   JSON alongside.
2. `configs/calibrated_a100.yaml` — a full RunConfig-compatible file
   (same shape as `configs/example_run.yaml`) with the fitted
   `compute:` block replacing the illustrative values. Network /
   scheduler sections copy from the example for now, annotated as
   "not calibrated here."
3. `tests/unit/test_calibrated_coefficients.py` — sanity properties:
   - `prefill_ms_per_token > 0` and `decode_ms_per_token > 0`.
   - `decode_ms_per_token > prefill_ms_per_token` (decode is more
     expensive per token, always true for modern transformers).
   - Monotonic amortization: for the fitted `k`,
     `effective_decode_ms(batch=b) < effective_decode_ms(batch=b−1)`
     for b ∈ {2, 4, 8, 16, 32, 64, 128}.
   - Acceptance-metric gates (R² thresholds from §3) pass on the
     checked-in fit summary.
4. Update to `research/reports/routing-comparison.md` §9 item 2:
   replace "illustrative" language with a reference to
   `configs/calibrated_a100.yaml` and a one-line summary of the fit
   (model, GPU, date, headline numbers, acceptance-metric pass).

## 6. Target fit metrics (summary table)

| Coefficient | Method | Acceptance |
|-------------|--------|-----------|
| `prefill_ms_per_token` | §3.1 linear regression slope | R² ≥ 0.97 |
| `prefill_overhead_ms`  | §3.1 intercept | residual SE ≤ 10% mean TTFT |
| `decode_ms_per_token`  | §3.2 linear regression slope | R² ≥ 0.98 |
| `decode_overhead_ms`   | §3.2 intercept | — |
| `decode_batch_k`       | §3.3 non-linear least squares | residual SE ≤ 15% mean TPOT; `k ∈ [0.1, 2.0]` |

## 7. Risks and mitigations

- **GPU capacity stall** (Modal A100 queue). *Mitigation:* rely on
  Modal's on-demand queue; if capacity is unavailable for >1 h the
  polecat escalates to scout rather than silently falling back to a
  different platform. The Lambda Cloud fallback that existed in the
  original plan was explicitly removed in Phase 2 approval.
- **HF gated-model access not provisioned on the Modal secret.**
  `meta-llama/Llama-3.1-8B-Instruct` is a gated repo; the HF account
  behind `hf_token_rome` must be in its authorized list. The first
  Phase 2 launch (2026-04-25, modal app `ap-AEy6ze2fgYW8fr6Yptrl0H`,
  on the original Llama-3.0 target) failed with `403 Forbidden` on
  the model config fetch; the account has since been approved for
  the Llama family and the target was switched to Llama-3.1 (go-27g).
  Verify access at
  https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct before the
  rerun. Tracking the rerun as go-26c. *Mitigation / resolution:* the
  account holder visits the model card and clicks "Request access"
  (usually auto-approved), or rotates `hf_token_rome` to a
  pre-authorized token. The pipeline (`scripts/calibrate.py`) fails
  loud on 403 without wasting GPU time beyond vLLM's initial
  model-config fetch (~30 s).
- **vLLM version drift between plan-time and run-time.** *Mitigation:*
  pin vLLM to a specific release tag in `scripts/calibrate.py`.
- **Chunked prefill / prefix caching interferes with prefill sweep.**
  *Mitigation:* disable with `--enable-prefix-caching=false --enable-chunked-prefill=false`
  in §3.1; re-enable chunked prefill only for §3.3 where it is the
  whole point.
- **Fit form wrong.** The `1 + k · log(1 + (batch−1))` amortization is
  a modeling assumption, not a physical law. If §3.3 residuals are
  structured (not zero-mean random), we file a bead to revisit the
  functional form rather than forcing a bad fit.
- **Spend overrun.** Modal billing is per-second; we set a hard
  `--function-timeout 4h` on the calibration job so an accidental
  infinite loop cannot exceed ~$14 regardless.

## 8. What this plan does NOT do (intentionally)

- Does not calibrate `NetworkParams.*` (fabric/RTT) — those are
  deployment-specific and fitted elsewhere.
- Does not calibrate `SchedulerParams.*` — router-side, requires
  router instrumentation.
- Does not claim device-portable coefficients. Numbers fitted here
  apply to the specific (model, GPU, vLLM version) triple recorded in
  the output YAML.
- Does not run multiple replicas or multi-pod topologies. Single-pod
  microbench is sufficient for coefficient fitting; cluster-scale
  behavior is simulated by the harness on top of these coefficients.
