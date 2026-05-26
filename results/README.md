# GORGO Experiment Results

## Experiment Index

All experiments use 3 × L40S:2 replicas across Seoul, Frankfurt, and US-East regions, replaying Mooncake traces at original speed via open-loop arrival.

| Short name | Spec file | Trace | Concurrency | Initial hyperparams | Analysis CSV |
|---|---|---|---|---|---|
| **GLM5 W1** | `policy_matrix_abstract_night.json` | Apr 1 00:30–01:00 | 32 | `prefill_weight=0.07, qw=0.06` (manual) | `analysis/glm5_w1.csv` |
| **GLM5 W2** | `policy_matrix_abstract_night_w2.json` | Apr 1 01:00–01:30 | 32 | `prefill_weight=0.983, qw=0.000442` (W1 hillclimb-learned) | `analysis/glm5_w2.csv` |
| **GLM5 Apr2** | `policy_matrix_abstract_night_w2.json` | Apr 2 00:30–01:00 | 32 | `prefill_weight=0.983, qw=0.000442` (W1 hillclimb-learned) | `analysis/glm5_apr2.csv` |
| **GLM5 Stress (night)** | `policy_matrix_abstract_night_stress.json` | Apr 2 00:30–01:00 | 64 | `prefill_weight=0.983, qw=0.000442` | `analysis/glm5_stress_night.csv` |
| **GLM5 Stress (midday)** | `policy_matrix_abstract_night_stress.json` | Apr 1 12:30–13:00 | 64 | `prefill_weight=0.983, qw=0.000442` | `analysis/glm5_stress_midday.csv` |
| **WildChat W1** | `policy_matrix_abstract_night_wildchat.json` | WildChat rows 0–20.5k | 32 | `prefill_weight=0.07, qw=0.06` (manual) | `analysis/wildchat_w1.csv` |

## Directory Layout

```
results/
├── analysis/                          # CSV tables + markdown summaries + plots
│   ├── glm5_w1.csv                    # GLM5 window 1 (Apr 1, 00:30–01:00, c=32)
│   ├── glm5_w2.csv                    # GLM5 window 2 (Apr 1, 01:00–01:30, c=32)
│   ├── glm5_apr2.csv                  # GLM5 Apr 2 (00:30–01:00, c=32)
│   ├── glm5_stress_night.csv          # GLM5 stress, night (Apr 2, 00:30–01:00, c=64)
│   ├── glm5_stress_midday.csv         # GLM5 stress, midday (Apr 1, 12:30–13:00, c=64)
│   ├── wildchat_w1.csv                # WildChat window 1
│   ├── *.md                           # Markdown summary tables + cross-run comparisons
│   ├── *.png                          # Summary bar charts and time-series plots
│   └── seaborn_images/                # Alternate seaborn-rendered plots
├── workload_runs/                     # Per-policy per-request result JSONs (pulled from Modal)
│   └── <run_id>_<policy>.json
└── policy_matrix_sweep/               # Experiment manifests + aggregate results (pulled from Modal)
    └── abstract_night/
        ├── glm5_w1_v1/               # GLM5 W1 experiment
        ├── glm5_w2_v1/               # GLM5 W2 experiment
        ├── glm5_apr2_v1/             # GLM5 Apr 2 experiment
        ├── glm5_stress_v1/           # GLM5 stress (night) experiment
        ├── glm5_midday_stress_v1/    # GLM5 stress (midday) experiment
        └── wildchat_w1_v2/           # WildChat W1 experiment
```

## Modal Volume Locations

All raw results are stored in the `GORGO-bench-results` Modal volume (env: `alessio-dev`).

| Data | Volume path |
|---|---|
| Per-policy workload results | `/workload_runs/<run_id>_<policy>.json` |
| Proxy traces (metrics + routing) | `/proxy_traces/<run_id>_<policy>/` |
| Experiment manifests | `/policy_matrix_sweep/abstract_night/<experiment_id>/` |

## Downloading Results

### Pull experiment manifests

```bash
modal volume get --env=alessio-dev --force GORGO-bench-results \
  /policy_matrix_sweep/abstract_night/<experiment_id> \
  results/policy_matrix_sweep/abstract_night/
```

### Pull per-policy workload results

```bash
# One policy at a time (Modal doesn't support globs):
for policy in random least-request least-load prefix-cache simple-session-affinity gorgo-static gorgo-autotune gorgo-hillclimb; do
  modal volume get --env=alessio-dev --force GORGO-bench-results \
    "/workload_runs/<run_id>_${policy}.json" \
    results/workload_runs/
done
```

### Pull proxy traces (for RTT / routing / tune convergence analysis)

```bash
for policy in gorgo-hillclimb gorgo-autotune; do
  modal volume get --env=alessio-dev --force GORGO-bench-results \
    "/proxy_traces/<run_id>_${policy}" \
    results/proxy_traces/
done
```

## Running Analysis

### Generate CSV + markdown summary from workload results

```bash
python3 scripts/analyze_results.py \
  --prefix <run_id> \
  --label "Human-readable label" \
  --results-dir results/workload_runs \
  --out-dir results/analysis
```

### Compare two runs side-by-side

```bash
python3 scripts/analyze_results.py \
  --prefix <run_id_1> --label "Run 1" \
  --prefix2 <run_id_2> --label2 "Run 2"
```

### Generate summary plots

```bash
python3 scripts/plot_policy_summary.py \
  --input results/analysis/<name>.csv \
  --output results/analysis/<name>.png \
  --title "Plot title"
```

### Plot hyperparameter convergence (requires tune.jsonl from trace)

First pull the proxy trace for the gorgo-hillclimb or gorgo-autotune policy:

```bash
modal volume get --env=alessio-dev --force GORGO-bench-results \
  "/proxy_traces/<run_id>_gorgo-hillclimb" \
  results/proxy_traces/
```

Then plot:

```bash
# Hillclimb convergence (sigma, score, hyperparameter trajectories, acceptance rate)
python3 scripts/plot_tune_convergence.py \
  --tune-jsonl results/proxy_traces/<run_id>_gorgo-hillclimb/tune.jsonl \
  --out-dir results/analysis \
  --title "GLM5 W1 Hillclimb Convergence"

# Autotune fit convergence (per-replica prefill_weight / load_weight)
python3 scripts/plot_tune_convergence.py \
  --tune-jsonl results/proxy_traces/<run_id>_gorgo-autotune/tune.jsonl \
  --out-dir results/analysis \
  --title "GLM5 W1 Autotune Fit"
```

Note: `tune.jsonl` is only produced by runs launched after the tune tracing was added.
Runs prior to this change have request and metrics traces but no tune trace.

## Run IDs

These are the `--prefix` values used with `analyze_results.py`:

| Experiment | Run ID (prefix) |
|---|---|
| GLM5 W1 | `abstract_night_000_glm5_0030_to_0100` |
| GLM5 W2 | `abstract_night_w2_000_glm5_0100_to_0130` |
| GLM5 Apr2 | `abstract_night_w2_000_glm5_apr2_0030_to_0100` |
| GLM5 Stress (night) | `abstract_night_stress_000_glm5_apr2_0030_to_0100` |
| GLM5 Stress (midday) | `abstract_night_stress_000_glm5_apr1_1230_to_1300` |
| WildChat W1 | `abstract_night_wildchat_000_wildchat_window1` |
