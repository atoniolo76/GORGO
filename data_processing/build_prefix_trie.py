"""Build a path-compressed radix trie over prompt token sequences from the
April week-1 dataset, then report intra-user vs cross-user KV-cache prefix
overlap.

Reads directly from the parquet files in /data via ``process_file`` from
``build_eval_dataset`` (with ``include_token_ids=True``). For each session
the longest conversation is tokenized once; its concatenated token-id
sequence is streamed into:

- one global trie (all users pooled)
- one trie per ``token_hash`` (intra-user only)

Overlap is measured as tokens saved vs. a naive no-sharing baseline:
    saved = total_tokens_inserted - unique_tokens_in_trie
where ``unique_tokens_in_trie`` is the sum of all compressed edge lengths.
"""

import itertools
from array import array

import modal

from app import app, completions_volume
from build_eval_dataset import FILE_CUTOFF, FILE_PREFIX, process_file

image = (
    modal.Image.debian_slim()
    .pip_install("duckdb", "tiktoken")
    .add_local_python_source("app", "build_eval_dataset")
)


class RadixNode:
    """One node in a path-compressed radix trie.

    ``edge`` holds the token ids on the edge leading *into* this node.
    ``count`` is the number of inserted sequences that traverse this node.
    ``terminal`` is the number of sequences that end exactly at this node.
    """

    __slots__ = ("edge", "children", "count", "terminal")

    def __init__(self, edge=None):
        self.edge = edge if edge is not None else array("I")
        self.children: dict[int, "RadixNode"] = {}
        self.count = 0
        self.terminal = 0


class RadixTrie:
    """Path-compressed radix trie over token-id sequences.

    Edges use ``array('I')`` (uint32) so each token costs 4 bytes instead of
    ~28 for a Python int object. Insertion is amortized O(len(seq)).
    """

    __slots__ = ("root", "total_tokens_inserted", "num_sequences")

    def __init__(self):
        self.root = RadixNode()
        self.total_tokens_inserted = 0
        self.num_sequences = 0

    def insert(self, seq) -> None:
        n = len(seq)
        self.total_tokens_inserted += n
        self.num_sequences += 1
        node = self.root
        node.count += 1
        i = 0
        while True:
            if i == n:
                node.terminal += 1
                return
            first = seq[i]
            child = node.children.get(first)
            if child is None:
                leaf = RadixNode(edge=array("I", seq[i:]))
                leaf.count = 1
                leaf.terminal = 1
                node.children[first] = leaf
                return
            edge = child.edge
            elen = len(edge)
            j = 1  # seq[i] == edge[0] already (first-char dispatch)
            remaining = n - i
            cap = elen if elen < remaining else remaining
            while j < cap and edge[j] == seq[i + j]:
                j += 1
            if j == elen:
                child.count += 1
                node = child
                i += j
                continue
            split = RadixNode(edge=edge[:j])
            split.count = child.count + 1
            child.edge = edge[j:]
            split.children[child.edge[0]] = child
            node.children[first] = split
            i += j
            if i == n:
                split.terminal += 1
                return
            leaf = RadixNode(edge=array("I", seq[i:]))
            leaf.count = 1
            leaf.terminal = 1
            split.children[seq[i]] = leaf
            return

    def unique_token_count(self) -> int:
        """Total length of all compressed edges = KV-cache footprint after
        perfect prefix sharing."""
        total = 0
        stack = [self.root]
        while stack:
            node = stack.pop()
            total += len(node.edge)
            stack.extend(node.children.values())
        return total

    def node_count(self) -> int:
        count = 0
        stack = [self.root]
        while stack:
            node = stack.pop()
            count += 1
            stack.extend(node.children.values())
        return count


def _fmt_pct(num: float, den: float) -> str:
    if den <= 0:
        return "  n/a"
    return f"{100.0 * num / den:6.2f}%"


@app.function(
    image=image,
    memory=1024 * 32,
    timeout=7200,
    volumes={"/data": completions_volume},
)
def build_tries(batch_size: int = 20, min_sequence_len: int = 1):
    import os
    import time

    parquet_dir = "/data"
    files = sorted(
        f
        for f in os.listdir(parquet_dir)
        if f.endswith(".parquet") and FILE_PREFIX in f and f < FILE_CUTOFF
    )
    batches = list(itertools.batched(files, batch_size))
    print(f"April 1-7: {len(files)} files in {len(batches)} batch(es) of <= {batch_size}")

    global_trie = RadixTrie()
    per_user_tries: dict[str, RadixTrie] = {}
    total_tokens = 0
    total_sequences = 0
    skipped_empty = 0
    t0 = time.time()

    for batch_idx, batch in enumerate(batches):
        batch_args = [(f, True) for f in batch]
        batch_sequences = 0
        batch_tokens = 0
        for file_results in process_file.starmap(batch_args):
            for entry in file_results:
                token_ids = entry.get("prompt_token_ids")
                if not token_ids or len(token_ids) < min_sequence_len:
                    skipped_empty += 1
                    continue
                token_hash = entry["token_hash"]
                seq = array("I", token_ids)

                global_trie.insert(seq)
                user_trie = per_user_tries.get(token_hash)
                if user_trie is None:
                    user_trie = RadixTrie()
                    per_user_tries[token_hash] = user_trie
                user_trie.insert(seq)

                total_tokens += len(seq)
                total_sequences += 1
                batch_sequences += 1
                batch_tokens += len(seq)

        elapsed = time.time() - t0
        print(
            f"  batch {batch_idx + 1}/{len(batches)}: "
            f"+{batch_sequences:,} seqs (+{batch_tokens:,} toks) | "
            f"cumulative {total_sequences:,} seqs, {total_tokens:,} toks, "
            f"{len(per_user_tries):,} users | elapsed {elapsed:,.0f}s"
        )

    if total_sequences == 0:
        print("No sequences ingested; nothing to report.")
        return

    print("\nComputing overlap stats...")
    global_unique = global_trie.unique_token_count()

    sum_intra_unique = 0
    user_count = 0
    top_users = []  # (sum_tokens, unique, user)
    for token_hash, user_trie in per_user_tries.items():
        u = user_trie.unique_token_count()
        sum_intra_unique += u
        user_count += 1
        top_users.append((user_trie.total_tokens_inserted, u, token_hash))

    intra_savings = total_tokens - sum_intra_unique
    global_savings = total_tokens - global_unique
    cross_user_extra = sum_intra_unique - global_unique

    print(f"\n{'=' * 72}")
    print(f"Sequences inserted:            {total_sequences:>14,}")
    print(f"Unique users (token_hash):     {user_count:>14,}")
    print(f"Empty sequences skipped:       {skipped_empty:>14,}")
    print(f"Total tokens T:                {total_tokens:>14,}")
    print(f"{'-' * 72}")
    print("(A) INTRA-USER overlap   -- prefixes shared within a user's own requests")
    print(
        f"      (T - sum_u U(R_u)) / T   "
        f"= {intra_savings:>14,} / {total_tokens:,}  "
        f"= {_fmt_pct(intra_savings, total_tokens)}"
    )
    print()
    print("(C) CROSS-USER overlap   -- extra sharing from pooling across users")
    print(
        f"      (sum_u U(R_u) - U(all)) / T   "
        f"= {cross_user_extra:>14,} / {total_tokens:,}  "
        f"= {_fmt_pct(cross_user_extra, total_tokens)}"
    )
    print(f"{'-' * 72}")
    print(f"Sanity check: A + C = global overlap (B)")
    print(
        f"      (T - U(all)) / T         "
        f"= {global_savings:>14,} / {total_tokens:,}  "
        f"= {_fmt_pct(global_savings, total_tokens)}"
    )
    print(
        f"      A + C                     "
        f"= {_fmt_pct(intra_savings + cross_user_extra, total_tokens)}"
    )
    print(f"{'=' * 72}")

    top_users.sort(reverse=True)
    print("\nTop 10 users by tokens inserted:")
    print(f"  {'tokens':>14}  {'unique':>14}  {'savings':>8}  token_hash")
    for total_t, uniq_t, th in top_users[:10]:
        saved = total_t - uniq_t
        print(f"  {total_t:>14,}  {uniq_t:>14,}  {_fmt_pct(saved, total_t)}  {th}")


@app.local_entrypoint()
def trie_main(batch_size: int = 20, min_sequence_len: int = 1):
    build_tries.remote(batch_size=batch_size, min_sequence_len=min_sequence_len)
