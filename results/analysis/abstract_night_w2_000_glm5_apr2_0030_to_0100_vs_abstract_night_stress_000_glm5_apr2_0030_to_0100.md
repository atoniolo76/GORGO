# GLM5 Apr2 (c=32) vs GLM5 Apr2 Stress (c=64)

| Policy | GLM5 Apr2 (c=32) p95 | GLM5 Apr2 Stress (c=64) p95 | Δ p95 | GLM5 Apr2 (c=32) p99 | GLM5 Apr2 Stress (c=64) p99 | Δ p99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gorgo-autotune | 1.576s | 1.690s | +7.2% | 2.306s | 2.298s | -0.3% |
| gorgo-hillclimb | 1.416s | 1.331s | -6.0% | 2.118s | 1.832s | -13.5% |
| gorgo-static | 1.387s | 1.427s | +2.9% | 1.947s | 2.302s | +18.3% |
| least-load | 1.690s | 1.700s | +0.6% | 2.364s | 2.290s | -3.1% |
| least-request | 1.622s | 1.605s | -1.1% | 2.293s | 2.268s | -1.1% |
| prefix-cache | 1.740s | 1.394s | -19.9% | 2.530s | 1.992s | -21.3% |
| random | 1.610s | 1.724s | +7.1% | 2.306s | 2.357s | +2.2% |
