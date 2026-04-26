# Synthetic-potent: median p95 (ms) over 3 zipf values, single seed


| Policy                 | qps=  4 | qps=  8 | qps= 16 | qps= 32 | hit_rate | skew  |
|------------------------|--------|--------|--------|--------|----------|-------|
| gorgo                  |   14313 |   14985 |   15481 |   15377 |    0.727 | 0.994 |
| least-kv-cache         |     940 |   12505 |   15337 |   15257 |    0.729 | 0.107 |
| pd-preble              |     944 |    1038 |   15665 |   15697 |    0.669 | 0.053 |
| prefix-cache-preble    |     938 |    1007 |   15041 |   15041 |    0.749 | 0.121 |

# ShareGPT: median p95 (ms), single seed


| Policy                 | qps=  4 | qps=  8 | qps= 16 | qps= 32 | hit_rate | skew  |
|------------------------|--------|--------|--------|--------|----------|-------|
| gorgo                  |    5369 |    5393 |    5425 |    5449 |    0.014 | 1.268 |
| least-kv-cache         |    2617 |    4897 |    5393 |    5449 |    0.017 | 0.381 |
| pd-preble              |    1605 |    5153 |    5497 |    5497 |    0.015 | 0.138 |
| prefix-cache-preble    |    1604 |    5209 |    5497 |    5497 |    0.017 | 0.129 |

# Gorgo hyperparam sensitivity (synthetic, qps=8, zipf=1.1, single seed)

| t_prefill \\ qtw | qtw=0.0001  | qtw=0.001   | qtw=0.01    |
|------------------|-------------|-------------|-------------|
| t_prefill=0.01    |   15137 |   15105 |   14921 |
| t_prefill=0.05    |   15017 |   14985 |   15289 |
| t_prefill=0.2     |   15105 |   15121 |   14985 |

# Leadership matrix (synthetic): gorgo p95 minus baseline p95, ms; negative = gorgo wins


## gorgo - prefix-cache-preble

| zipf \\ qps | qps=  4 | qps=  8 | qps= 16 | qps= 32 |
|-------------|--------|--------|--------|--------|
| zipf=0.7    | +14832 | +14860 |   +216 |   +232 |
| zipf=1.1    | +13374 | +13978 |   +440 |   +336 |
| zipf=1.5    | +10650 | +11254 |   +616 |   +112 |

## gorgo - pd-preble

| zipf \\ qps | qps=  4 | qps=  8 | qps= 16 | qps= 32 |
|-------------|--------|--------|--------|--------|
| zipf=0.7    | +14818 | +14834 |   -224 |   -272 |
| zipf=1.1    | +13369 | +13946 |   -184 |   -320 |
| zipf=1.5    | +10641 | +11243 |   -296 |   -544 |

## gorgo - least-kv-cache

| zipf \\ qps | qps=  4 | qps=  8 | qps= 16 | qps= 32 |
|-------------|--------|--------|--------|--------|
| zipf=0.7    | +14826 | +14842 |   +104 |   +104 |
| zipf=1.1    | +13373 |   -208 |   +144 |   +120 |
| zipf=1.5    | +10633 |   -280 |    +24 |   -248 |
