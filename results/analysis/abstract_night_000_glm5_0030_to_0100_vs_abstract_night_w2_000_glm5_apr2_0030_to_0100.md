# GLM5 Apr1 W1 (tuning) vs GLM5 Apr2 (cross-day)

| Policy | GLM5 Apr1 W1 (tuning) p95 | GLM5 Apr2 (cross-day) p95 | Δ p95 | GLM5 Apr1 W1 (tuning) p99 | GLM5 Apr2 (cross-day) p99 | Δ p99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gorgo-autotune | 1.658s | 1.576s | -4.9% | 2.428s | 2.306s | -5.0% |
| gorgo-hillclimb | 1.466s | 1.416s | -3.5% | 2.307s | 2.118s | -8.2% |
| gorgo-static | 1.605s | 1.387s | -13.6% | 2.357s | 1.947s | -17.4% |
| least-load | 1.728s | 1.690s | -2.2% | 2.629s | 2.364s | -10.1% |
| least-request | 1.632s | 1.622s | -0.6% | 2.355s | 2.293s | -2.7% |
| prefix-cache | 1.654s | 1.740s | +5.2% | 3.017s | 2.530s | -16.1% |
| random | 1.828s | 1.610s | -11.9% | 3.171s | 2.306s | -27.3% |
| simple-session-affinity | 1.482s | 1.435s | -3.1% | 2.804s | 2.039s | -27.3% |
