# GLM5 W1 (tuning) vs GLM5 W2 (eval)

| Policy | GLM5 W1 (tuning) p95 | GLM5 W2 (eval) p95 | Δ p95 | GLM5 W1 (tuning) p99 | GLM5 W2 (eval) p99 | Δ p99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gorgo-autotune | 1.658s | 1.888s | +13.9% | 2.428s | 3.001s | +23.6% |
| gorgo-hillclimb | 1.466s | 1.535s | +4.7% | 2.307s | 2.595s | +12.5% |
| gorgo-static | 1.605s | 1.510s | -6.0% | 2.357s | 2.791s | +18.4% |
| least-load | 1.728s | 1.813s | +4.9% | 2.629s | 2.659s | +1.1% |
| least-request | 1.632s | 1.757s | +7.6% | 2.355s | 2.663s | +13.0% |
| prefix-cache | 1.654s | 1.632s | -1.3% | 3.017s | 2.860s | -5.2% |
| random | 1.828s | 1.763s | -3.6% | 3.171s | 2.612s | -17.6% |
| simple-session-affinity | 1.482s | 2.080s | +40.4% | 2.804s | 4.917s | +75.4% |
