Yes, this is reasonable for WildChat. The numbers are internally consistent and the qualitative shape matches what the dataset is known to look like.

## Sanity check

`A + C = B`: `5.30% + 29.06% = 34.35%` ✓ — the trie math checks out, no double-counting bug.

`total_sequences == rows_processed`: confirms `--ingest-all-rows` did what it should — every row inserted, no skipping, no winner-per-conv.

## Headline numbers

- **`T = 9.36B` tokens** across **3.2M rows** → mean **2,925 tokens/row**. Matches the conversation distribution (mean 3,207 tokens, p50 832, p90 7,847, p99 36,726) when you account for some rows being short single-turn snippets.
- **A (intra-user) = 5.30%** — same-IP prefix reuse.
- **C (cross-user) = 29.06%** — extra reuse unlocked by pooling across IPs.
- **B (global) = 34.35%** — total prefix savings under perfect global KV sharing.
- **Runtime 5,629 s ≈ 94 min** vs. the prior failed 14,851 s at row 1.6M — the memory bump killed GC thrashing and made it ~6× faster end-to-end.

## Why these magnitudes are expected for WildChat

1. **Intra-user is small (5.3%) because the IP histogram is flat.**
   `user_count = 1,833,730` IPs across `3,199,860` rows → mean **1.74 rows/IP**. With ~mean of <2 rows per group, most IPs have no opportunity for intra-user prefix sharing. The 5.3% you do see comes from the long tail of heavy IPs, not the median.

2. **Cross-user is high (29%) because of structural sharing**, not pairwise similarity:
   - **Shared system prompts / chat templates.** WildChat was collected via a single shared ChatGPT proxy; many sessions begin with the same or near-same system/instruction header. Hundreds of thousands of rows all sharing a 100–500 token preamble account for a large fraction of the cross-user savings on their own.
   - **Viral / copy-pasted prompts.** Standard "translate this", role-play setups, jailbreak templates show up across many IPs.
   - **Staircase conversations across IPs.** With `ingest_all_rows`, the same conversation submitted as turn 1, turn 2, …, turn N is kept as N separate rows. When variants of the same staircase appear under multiple IPs, they're cross-user from the trie's POV.
   
   The trie metric is **token-weighted**: a single 500-token preamble shared by 10,000 IPs contributes `9,999 × 500 ≈ 5M` cross-user tokens. That's exactly the regime where trie-`C` >> pairwise-similarity `C`. So the gap to the paper's `~2.5%` pairwise metric is expected; we discussed this earlier.

3. **Global 34% is the actually-useful number for KV-cache planning** — under perfect prefix caching across all IPs, a server handling traffic that looks like WildChat saves about a third of token storage.

## What the top groups tell you

The "tokens-vs-unique" ratios for the top 10 IPs are striking:

| key prefix | tokens | unique | savings |
|---|---:|---:|---:|
| `fb0a33b9…` | 40.25M | 10.40M | **74.2%** |
| `5fcc51e2…` | 19.80M | 5.20M | 73.7% |
| `4e615ec2…` | 18.97M | 3.14M | **83.4%** |
| `ab8d1faf…` | 17.13M | 4.20M | 75.5% |
| `757a8dd7…` | 17.08M | 16.97M | **0.7%** |
| `40222c3a…` | 16.88M | 4.38M | 74.1% |
| `e426b10b…` | 16.01M | 3.94M | 75.4% |
| `f4cf1640…` | 15.83M | 9.97M | 37.0% |
| `e61c8945…` | 15.51M | 9.73M | 37.3% |
| `c4c451df…` | 15.01M | 5.15M | 65.7% |

Two regimes of "whale":

- **Heavy reuse bots** (~75–83% intra-savings, e.g. `fb0a33…`, `4e615e…`, `40222c…`, `e426b1…`, `ab8d1f…`): same IP submitting many near-duplicate or staircase conversations. These dominate the intra-user savings number; without them, A would be even smaller.
- **Diverse heavy users** (~0.7%, e.g. `757a8dd7…`): same IP, many distinct conversations. Looks like a popular endpoint or NAT-ed IP behind many real users.
- **Mixed** (`f4cf16…`, `e61c89…`, ~37%): real-world long-tailed sessions.

The top 10 IPs alone account for ~192M tokens = **~2.0% of the whole dataset**, and they swing the intra/cross split a lot. Removing the top decile of IPs would meaningfully lower `A`.

## What the conversation stats tell you

- `2.87M distinct conversation hashes` across `3.2M rows` → only **10.3% of rows are non-unique by conv_hash** (extra rows = `330,778` over `39,306` repeated hashes).
- `rows_per_conversation`: median 1, p99 2, **max 24,688**. Most conv hashes appear once. The few that don't, dominate. The `1f204161…` conversation_hash has 24,688 rows — that's the canonical WildChat staircase (or a stress-test bot). Under `--ingest-all-rows` all 24,688 are inserted, so a conversation like that contributes a massive intra-IP and/or cross-IP prefix-sharing payload depending on how its rows are distributed across IPs. This is exactly why the WildChat default mode (`--dedup-content-prefix-sha256`, partitioned by `hashed_ip`) exists — it would collapse this to just the maximal chain.
- `tokens_per_conversation`: median 832, mean 3,207, p99 ~37k, **max 615k**. The mean is ~4× the median because of those whale conversations. p99 ~37k tokens is consistent with very long multi-turn back-and-forths (≈30–40 turns of ~1k tokens each).

## Bottom line

- **Yes, reasonable.** Numbers are consistent with each other and with WildChat's known properties: long-tailed IP usage, staircase artifacts, shared system prompts, mostly-unique conversation hashes with a few extreme repeats.
- **B = 34% is a defensible KV-cache savings ceiling** for serving WildChat-like traffic with global prefix sharing.
- **C = 29% is "honest" under the trie definition**, but should not be reported as "cross-user prefix similarity" in the paper's sense (~2.5%). They measure different things; this number is fine as long as it's labeled "extra savings from pooling across users (token-weighted, KV-cache footprint)".
- **A = 5.3%** mostly comes from the heavy-reuse whales; the median IP contributes nothing.

If you want a number that's directly comparable to the paper's pairwise definition, the next step is the sampled `common_prefix(a,b) / min(|a|,|b|)` pass we discussed earlier — happy to add that as a new entrypoint that reads from `prefix_trie_checkpoints/rows_03199860.pkl` so you don't have to re-tokenize.