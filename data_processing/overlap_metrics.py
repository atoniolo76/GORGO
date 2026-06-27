"""Pure (stdlib-only) overlap-structure metrics for token-id sequences.

This module deliberately imports **nothing** beyond the standard library so it
can be unit-tested locally without Modal, duckdb, numpy, or any GLM-5.1 data.
``analyze_overlap_structure.py`` (the Modal driver) imports these helpers and
feeds them per-session ``prompt_ids`` streamed off the volume.

It implements the three measurements designed in
``prefix_trie_results/glm-5.1-completions/overlap_structure_analysis.md`` to
separate Rome's hypotheses about GLM-5.1 content overlap:

  (i)   block-size sweep of CONTENT-INDEPENDENT (prefix-stripped) block reuse
        vs. the existing PREFIX-CHAINED (vLLM/SGLang-style) block reuse;
  (ii)  alignment-robust positional collision profile of content-hashed
        rolling n-grams, bucketed by position-in-prompt;
  (iii) matched-segment-length histogram, split on-prefix vs off-prefix.

Hashing notes
-------------
* Token ids are serialized little-endian uint32 -- identical to
  ``build_mooncake_trace.py``/``export_metadata_trace.py`` so digests are
  directly comparable to the existing pipeline.
* The prefix-chained variant folds the *full* sha256 of block ``i-1`` into
  block ``i`` (``h.update(prev_digest)``), exactly mirroring
  ``build_mooncake_trace.py:428-444``. Only the stored/counted KEY is truncated
  (default 8 bytes) to bound memory; chaining always uses the full digest.
* 8-byte keys: with ~5e8 distinct blocks/n-grams the birthday-bound expected
  collision count in 2**64 is < 0.01, negligible for these aggregate counts.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Callable, Iterable, Iterator

DEFAULT_KEY_BYTES = 8


def _tok_bytes(token_ids) -> bytes:
    """Little-endian uint32 serialization (matches the existing pipeline)."""
    return b"".join(int(t).to_bytes(4, "little", signed=False) for t in token_ids)


# --------------------------------------------------------------------------- #
# Block digests
# --------------------------------------------------------------------------- #
def content_block_digests(
    token_ids, block_size: int, key_bytes: int = DEFAULT_KEY_BYTES
) -> list[bytes]:
    """Independent per-block content hash (NO prefix chaining).

    A block matches another block anywhere in the corpus iff its ``block_size``
    tokens are identical, regardless of what precedes it. This is the metric
    that can detect non-prefix / middle overlap (when block-aligned).
    """
    out: list[bytes] = []
    n = len(token_ids)
    for i in range(0, n, block_size):
        out.append(hashlib.sha256(_tok_bytes(token_ids[i : i + block_size])).digest()[:key_bytes])
    return out


def prefix_chained_block_digests(
    token_ids, block_size: int, key_bytes: int = DEFAULT_KEY_BYTES
) -> list[bytes]:
    """Prefix-aware chained block hash -- mirrors build_mooncake_trace.py:428-444.

    Block ``i`` folds in the full digest of block ``i-1`` (which folded in
    ``i-2`` ...), so block ``i`` matches another request's block ``i`` ONLY
    when blocks ``0..i`` are byte-identical (i.e. the whole prefix matches).
    """
    out: list[bytes] = []
    prev = b""
    n = len(token_ids)
    for i in range(0, n, block_size):
        h = hashlib.sha256()
        h.update(prev)
        h.update(_tok_bytes(token_ids[i : i + block_size]))
        d = h.digest()
        out.append(d[:key_bytes])
        prev = d  # chain the FULL digest, like the existing pipeline
    return out


class BlockReuseAccumulator:
    """Streaming accumulator for one (block_size, chained?) configuration.

    Tracks the global set of distinct block keys plus enough side info to
    report both occurrence-weighted reuse (matches the existing
    ``block_reuse_pct`` definition) and token-weighted reuse (comparable to the
    radix-trie ``global_savings_pct``). Memory ~ one entry per distinct block.
    """

    __slots__ = (
        "block_size",
        "chained",
        "key_bytes",
        "seen",
        "_partial_len",
        "total_blocks",
        "total_tokens",
    )

    def __init__(self, block_size: int, chained: bool, key_bytes: int = DEFAULT_KEY_BYTES):
        self.block_size = block_size
        self.chained = chained
        self.key_bytes = key_bytes
        self.seen: set[bytes] = set()
        # digest -> token length, ONLY for partial (final, < block_size) blocks.
        self._partial_len: dict[bytes, int] = {}
        self.total_blocks = 0
        self.total_tokens = 0

    def add(self, token_ids) -> None:
        n = len(token_ids)
        if n == 0:
            return
        self.total_tokens += n
        digs = (
            prefix_chained_block_digests(token_ids, self.block_size, self.key_bytes)
            if self.chained
            else content_block_digests(token_ids, self.block_size, self.key_bytes)
        )
        for idx, d in enumerate(digs):
            self.total_blocks += 1
            if d not in self.seen:
                self.seen.add(d)
                start = idx * self.block_size
                blen = min(self.block_size, n - start)
                if blen != self.block_size:
                    self._partial_len[d] = blen

    def result(self) -> dict:
        unique = len(self.seen)
        # Full unique blocks contribute block_size tokens; partials contribute
        # their actual length.
        deficit = sum(self.block_size - pl for pl in self._partial_len.values())
        unique_token_mass = unique * self.block_size - deficit
        return {
            "block_size": self.block_size,
            "chained": self.chained,
            "total_blocks": self.total_blocks,
            "unique_blocks": unique,
            "block_reuse_pct": (
                100.0 * (self.total_blocks - unique) / self.total_blocks
                if self.total_blocks
                else 0.0
            ),
            "total_tokens": self.total_tokens,
            "unique_token_mass": unique_token_mass,
            "token_reuse_pct": (
                100.0 * (self.total_tokens - unique_token_mass) / self.total_tokens
                if self.total_tokens
                else 0.0
            ),
        }


def block_reuse_stats(
    sequences: Iterable, block_size: int, chained: bool, key_bytes: int = DEFAULT_KEY_BYTES
) -> dict:
    """Convenience: full block-reuse stats over an in-memory list of sequences."""
    acc = BlockReuseAccumulator(block_size, chained, key_bytes)
    for s in sequences:
        acc.add(s)
    return acc.result()


# --------------------------------------------------------------------------- #
# Rolling n-grams (alignment-robust)
# --------------------------------------------------------------------------- #
def ngram_hashes(
    token_ids, w: int, stride: int, key_bytes: int = DEFAULT_KEY_BYTES
) -> Iterator[tuple[bytes, int]]:
    """Yield ``(key, start)`` for width-``w`` windows sampled every ``stride``.

    Sliding (alignment-robust): catches a repeated chunk wherever it occurs,
    independent of block alignment. ``stride`` controls cost/memory.
    """
    n = len(token_ids)
    if n < w:
        return
    last = n - w
    for start in range(0, last + 1, stride):
        yield hashlib.sha256(_tok_bytes(token_ids[start : start + w])).digest()[:key_bytes], start


def whole_prompt_key(token_ids, key_bytes: int = 16) -> bytes:
    return hashlib.sha256(_tok_bytes(token_ids)).digest()[:key_bytes]


def dedup_whole_prompts(sequences: Iterable) -> list:
    """Keep only the first occurrence of each distinct full prompt (in order).

    Removes whole-prompt duplicates so the middle/suffix-overlap measure
    reflects overlap AMONG OTHERWISE-DISTINCT prompts (the reuse-whale
    exact-duplicate effect is what we want to factor out).
    """
    seen: set[bytes] = set()
    out = []
    for s in sequences:
        k = whole_prompt_key(s)
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def position_bucket(start: int, length: int, w: int, n_buckets: int) -> int:
    """Map a window start to a position bucket: 0 = prompt head, n-1 = tail."""
    span = length - w
    if span <= 0:
        return 0
    b = int((start / span) * n_buckets)
    return b if b < n_buckets else n_buckets - 1


def positional_collision_profile(
    sequences: Iterable,
    w: int = 64,
    stride: int = 16,
    n_buckets: int = 4,
    drop_whole_prompt_dupes: bool = True,
    key_bytes: int = DEFAULT_KEY_BYTES,
) -> dict:
    """Two-pass: fraction of sampled windows that collide across the corpus,
    bucketed by where in the prompt they sit.

    If the shared fraction is high only in bucket 0 -> pure prefix overlap.
    If late buckets also show a high shared fraction -> genuine middle/suffix
    overlap that prefix metrics miss.
    """
    seqs = dedup_whole_prompts(sequences) if drop_whole_prompt_dupes else list(sequences)

    global_counts: Counter[bytes] = Counter()
    for s in seqs:
        for h, _ in ngram_hashes(s, w, stride, key_bytes):
            global_counts[h] += 1

    shared = [0] * n_buckets
    total = [0] * n_buckets
    for s in seqs:
        L = len(s)
        if L < w:
            continue
        for h, start in ngram_hashes(s, w, stride, key_bytes):
            b = position_bucket(start, L, w, n_buckets)
            total[b] += 1
            if global_counts[h] >= 2:
                shared[b] += 1

    buckets = [
        {
            "bucket": i,
            "range_pct": [round(100.0 * i / n_buckets, 1), round(100.0 * (i + 1) / n_buckets, 1)],
            "total_windows": total[i],
            "shared_windows": shared[i],
            "shared_pct": 100.0 * shared[i] / total[i] if total[i] else 0.0,
        }
        for i in range(n_buckets)
    ]
    tot_all = sum(total)
    sh_all = sum(shared)
    return {
        "w": w,
        "stride": stride,
        "n_buckets": n_buckets,
        "dropped_whole_prompt_dupes": drop_whole_prompt_dupes,
        "num_sequences": len(seqs),
        "distinct_ngrams": len(global_counts),
        "overall_shared_pct": 100.0 * sh_all / tot_all if tot_all else 0.0,
        "buckets": buckets,
    }


def shared_runs(
    token_ids,
    is_shared: Callable[[bytes], bool],
    w: int,
    stride: int,
    key_bytes: int = DEFAULT_KEY_BYTES,
) -> list[tuple[int, int, bool]]:
    """Merge maximal consecutive shared sampled windows into runs.

    Returns ``(first_start, span_tokens, prefix_anchored)`` per run, where
    ``span_tokens = last_start - first_start + w`` and ``prefix_anchored`` is
    True iff the run begins at token 0 (i.e. it is the shared *prefix*). A run
    breaks at the first non-shared sampled window.
    """
    runs: list[tuple[int, int, bool]] = []
    cur_start: int | None = None
    last_start = 0
    for h, start in ngram_hashes(token_ids, w, stride, key_bytes):
        if is_shared(h):
            if cur_start is None:
                cur_start = start
            last_start = start
        elif cur_start is not None:
            runs.append((cur_start, last_start - cur_start + w, cur_start == 0))
            cur_start = None
    if cur_start is not None:
        runs.append((cur_start, last_start - cur_start + w, cur_start == 0))
    return runs


def _hist_bins(w: int) -> list[int]:
    """Upper edges (token lengths) for the run-length histogram."""
    return [w, 2 * w, 4 * w, 8 * w, 16 * w, 64 * w]


def _bin_index(span: int, edges: list[int]) -> int:
    for i, e in enumerate(edges):
        if span <= e:
            return i
    return len(edges)  # overflow bin


def _empty_hist(edges: list[int]) -> dict:
    labels = [f"<= {e}" for e in edges] + [f"> {edges[-1]}"]
    return {
        "edges": edges,
        "labels": labels,
        "counts": [0] * (len(edges) + 1),
        "runs": 0,
        "tokens": 0,
    }


def segment_length_histogram(
    sequences: Iterable,
    w: int = 64,
    stride: int = 16,
    drop_whole_prompt_dupes: bool = True,
    key_bytes: int = DEFAULT_KEY_BYTES,
    edges: list[int] | None = None,
) -> dict:
    """Histogram of matched-segment (shared-run) lengths, split on/off prefix.

    Heavy tail of long OFF-prefix runs -> few big middle chunks.
    Mass at short OFF-prefix runs -> many small repeated segments.
    """
    seqs = dedup_whole_prompts(sequences) if drop_whole_prompt_dupes else list(sequences)
    edges = edges if edges is not None else _hist_bins(w)

    global_counts: Counter[bytes] = Counter()
    for s in seqs:
        for h, _ in ngram_hashes(s, w, stride, key_bytes):
            global_counts[h] += 1

    def _is_shared(h: bytes) -> bool:
        return global_counts[h] >= 2

    on = _empty_hist(edges)
    off = _empty_hist(edges)
    for s in seqs:
        for first_start, span, anchored in shared_runs(s, _is_shared, w, stride, key_bytes):
            tgt = on if anchored else off
            tgt["counts"][_bin_index(span, edges)] += 1
            tgt["runs"] += 1
            tgt["tokens"] += span

    return {
        "w": w,
        "stride": stride,
        "dropped_whole_prompt_dupes": drop_whole_prompt_dupes,
        "num_sequences": len(seqs),
        "on_prefix": on,
        "off_prefix": off,
    }
