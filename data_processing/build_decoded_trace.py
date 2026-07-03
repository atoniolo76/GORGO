"""Convert metadata traces into runnable Mooncake-format decoded traces.

Reads a metadata trace (from ``export_metadata_trace.py``) and generates
a Mooncake JSONL with synthetic request bodies. For each message, random
token IDs are generated and decoded back to text via ``enc.decode()``,
guaranteeing that SGLang re-tokenizes to the exact token count.

Multi-turn prefix reuse is preserved: for each user, previous turns'
token IDs are reused as the prefix of the next request, and only new
turn content gets fresh random tokens.

Usage::

    modal run --env=alessio-dev data_processing/build_decoded_trace.py::main \\
        --metadata-path /data/mooncake_traces/metadata/prod_metadata_apr2_0030_to_0100.jsonl \\
        --output-path /data/mooncake_traces/decoded/prod_decoded_apr2_0030_to_0100.jsonl
"""

from __future__ import annotations

import json
import os
import random
import time

import modal

from app import app, completions_volume

DEFAULT_MAX_OUTPUT_TOKENS = 128

MODEL_ID = "Qwen/Qwen3.5-35B-A3B-FP8"
MODEL_REVISION = "0b2752837483aa34b3db6e83e151b150c0e00e49"

image = (
    modal.Image.debian_slim().pip_install("transformers", "jinja2").add_local_python_source("app")
)


@app.function(
    image=image,
    memory=1024 * 8,
    timeout=3600,
    volumes={"/data": completions_volume},
)
def build_decoded(
    metadata_path: str,
    output_path: str,
    seed: int = 42,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=False,
    )
    rng = random.Random(seed)
    t0 = time.perf_counter()

    # Build a pool of common English words that are each exactly 1 token
    # under the Qwen tokenizer (same tokenizer the proxy uses).
    # We interleave words with newlines to prevent BPE merging:
    # "word\nword\nword" = 2N-1 tokens for N words.
    candidate_words = [
        "the",
        "of",
        "and",
        "to",
        "in",
        "is",
        "for",
        "that",
        "it",
        "as",
        "was",
        "with",
        "be",
        "by",
        "on",
        "not",
        "he",
        "are",
        "from",
        "or",
        "his",
        "an",
        "at",
        "but",
        "they",
        "have",
        "had",
        "her",
        "she",
        "my",
        "we",
        "all",
        "if",
        "so",
        "no",
        "up",
        "one",
        "its",
        "out",
        "do",
        "who",
        "when",
        "been",
        "can",
        "more",
        "will",
        "has",
        "just",
        "new",
        "than",
        "may",
        "any",
        "our",
        "now",
        "get",
        "use",
        "how",
        "each",
    ]
    single_token_words = [
        w for w in candidate_words if len(tok.encode(w, add_special_tokens=False)) == 1
    ]
    nl_toks = tok.encode("\n", add_special_tokens=False)
    assert len(nl_toks) == 1, f"newline is {len(nl_toks)} tokens under Qwen, expected 1"
    print(f"[decoded] {len(single_token_words)} single-token words available (Qwen tokenizer)")

    # Read metadata trace
    rows_in: list[dict] = []
    with open(metadata_path) as f:
        for line in f:
            if line.strip():
                rows_in.append(json.loads(line))
    print(f"[decoded] read {len(rows_in)} rows from {metadata_path}")

    # Per-user running list of single-token strings. Each token position
    # maps to one word or one newline, so reusing the first K entries
    # produces character-identical text — which is what SGLang's
    # RadixAttention needs for a cache hit.
    user_token_strings: dict[str, list[str]] = {}

    # Shared system prompt text cache: requests with the same
    # system_prompt_hash get identical text, enabling cross-user
    # prefix reuse at both the proxy and engine level.
    system_prompt_text_cache: dict[str, list[str]] = {}

    def _gen_token_strings(n: int) -> list[str]:
        """Generate n single-token strings (alternating word/newline)."""
        strings: list[str] = []
        for i in range(n):
            if i % 2 == 0:
                strings.append(rng.choice(single_token_words))
            else:
                strings.append("\n")
        return strings

    rows_out: list[dict] = []

    for entry in rows_in:
        token_hash = entry["token_hash"]
        messages = entry["messages"]
        hash_ids = entry["hash_ids"]
        input_length = entry["input_length"]
        output_length = entry.get("output_length") or rng.randint(16, max_output_tokens)
        output_length = min(output_length, max_output_tokens)
        sys_hash = entry.get("system_prompt_hash")

        prev_strings = user_token_strings.get(token_hash, [])

        # Build per-token text strings preserving prefix reuse:
        # 1. System prompt: reuse across users via system_prompt_hash
        # 2. Conversation history: reuse within user via user_token_strings
        # 3. New content: generate fresh
        current_strings: list[str] = []
        tok_offset_in_messages = 0

        for msg in messages:
            n = msg["tokens"]
            msg_start = len(current_strings)

            if msg["role"] == "system" and sys_hash:
                # Cross-user reuse: same system prompt → same text
                if sys_hash in system_prompt_text_cache:
                    cached = system_prompt_text_cache[sys_hash]
                    if len(cached) == n:
                        current_strings.extend(cached)
                        continue
                # Generate and cache for this system prompt hash
                sys_strings = _gen_token_strings(n)
                system_prompt_text_cache[sys_hash] = sys_strings
                current_strings.extend(sys_strings)
            else:
                # Intra-user reuse: reuse prefix from previous request
                for tok_idx in range(n):
                    global_idx = msg_start + tok_idx
                    if global_idx < len(prev_strings):
                        current_strings.append(prev_strings[global_idx])
                    else:
                        if global_idx % 2 == 0:
                            current_strings.append(rng.choice(single_token_words))
                        else:
                            current_strings.append("\n")

        user_token_strings[token_hash] = current_strings

        # Split into per-message text
        synthetic_messages: list[dict] = []
        tok_offset = 0
        for msg in messages:
            n = msg["tokens"]
            msg_strings = current_strings[tok_offset : tok_offset + n]
            tok_offset += n
            text = "".join(msg_strings)
            synthetic_messages.append(
                {
                    "role": msg["role"],
                    "content": text,
                }
            )

        row = {
            "timestamp": entry["timestamp"],
            "input_length": input_length,
            "output_length": output_length,
            "unique_input_tokens": input_length,
            "hash_ids": hash_ids,
            "request": {
                "model": "",
                "messages": synthetic_messages,
                "max_tokens": max_output_tokens,
                "stream": True,
            },
            "response": None,
            "request_id": f"decoded_{len(rows_out):06d}",
            "token_hash": token_hash,
        }
        rows_out.append(row)

    # Write decoded trace
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w") as f:
        for row in rows_out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp_path, output_path)
    completions_volume.commit()

    total_input = sum(r["input_length"] for r in rows_out)
    users = len(user_token_strings)
    duration_ms = rows_out[-1]["timestamp"] if rows_out else 0

    print(f"\n[decoded] wrote {output_path}")
    print(f"  requests: {len(rows_out):,}")
    print(f"  users: {users:,}")
    print(f"  total input tokens: {total_input:,}")
    print(f"  duration: {duration_ms / 1000:.0f}s")
    print(f"  elapsed: {time.perf_counter() - t0:.1f}s")

    return {
        "output_path": output_path,
        "requests": len(rows_out),
        "users": users,
        "total_input_tokens": total_input,
        "duration_ms": duration_ms,
    }


@app.local_entrypoint()
def main(
    metadata_path: str = "/data/mooncake_traces/metadata/prod_metadata_apr2_0030_to_0100.jsonl",
    output_path: str = "",
    seed: int = 42,
):
    if not output_path:
        base = os.path.basename(metadata_path).replace("metadata", "decoded")
        output_path = f"/data/mooncake_traces/decoded/{base}"

    result = build_decoded.remote(
        metadata_path=metadata_path,
        output_path=output_path,
        seed=seed,
    )
    print(json.dumps(result, indent=2))


# ---------------------------------------------------------------------------
# Per-file parallel decode (CPU fan-out, deterministic block-keyed generation)
# ---------------------------------------------------------------------------
# Each metadata shard is decoded independently. Synthetic text is a pure
# function of the block ``hash_ids`` (prefix-cumulative, globally consistent),
# so any two requests that share a real prefix get a character-identical
# synthetic prefix with zero shared state -- preserving both intra-user and
# cross-user (system-prompt) reuse across shards. One RNG seed per 256-token
# block (not per token), so it is cheap.

DEFAULT_BLOCK_SIZE = 256

CANDIDATE_WORDS = [
    "the",
    "of",
    "and",
    "to",
    "in",
    "is",
    "for",
    "that",
    "it",
    "as",
    "was",
    "with",
    "be",
    "by",
    "on",
    "not",
    "he",
    "are",
    "from",
    "or",
    "his",
    "an",
    "at",
    "but",
    "they",
    "have",
    "had",
    "her",
    "she",
    "my",
    "we",
    "all",
    "if",
    "so",
    "no",
    "up",
    "one",
    "its",
    "out",
    "do",
    "who",
    "when",
    "been",
    "can",
    "more",
    "will",
    "has",
    "just",
    "new",
    "than",
    "may",
    "any",
    "our",
    "now",
    "get",
    "use",
    "how",
    "each",
]


def _build_word_pool(tok) -> list[str]:
    """Single-token words under the given tokenizer (stable order, so the pool
    is identical across workers -> reuse is consistent fleet-wide)."""
    words = [w for w in CANDIDATE_WORDS if len(tok.encode(w, add_special_tokens=False)) == 1]
    nl = tok.encode("\n", add_special_tokens=False)
    assert len(nl) == 1, f"newline is {len(nl)} tokens, expected 1"
    return words


@app.function(
    image=image,
    memory=1024 * 8,
    timeout=3600,
    retries=2,
    cpu=4.0,
    volumes={"/data": completions_volume},
)
def decode_file(
    meta_filename: str,
    input_dir: str,
    output_dir: str,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> dict:
    """Decode ONE metadata shard into a Mooncake JSONL shard. Idempotent."""
    import json
    import os
    import random

    from transformers import AutoTokenizer

    stem = meta_filename.replace(".jsonl", "")
    out_path = os.path.join(output_dir, f"{stem}.jsonl")
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(out_path):
        n = sum(1 for line in open(out_path) if line.strip())
        return {"meta_filename": meta_filename, "skipped": True, "requests": n}

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False)
    words = _build_word_pool(tok)

    in_path = os.path.join(input_dir, meta_filename)
    tmp_path = out_path + ".tmp"
    count = 0

    with open(in_path) as fin, open(tmp_path, "w") as fout:
        for idx, line in enumerate(fin):
            if not line.strip():
                continue
            e = json.loads(line)
            hash_ids = e["hash_ids"]
            input_length = e["input_length"]
            messages_meta = e["messages"]
            n_blocks = len(hash_ids)

            # Build the flat per-token string list, one RNG seed per block.
            flat: list[str] = []
            for bi, digest in enumerate(hash_ids):
                blk = (
                    block_size if bi < n_blocks - 1 else input_length - block_size * (n_blocks - 1)
                )
                if blk <= 0:
                    continue
                rng_b = random.Random(int(digest, 16))
                n_words = (blk + 1) // 2  # even local positions are words
                chosen = rng_b.choices(words, k=n_words)
                blk_list = ["\n"] * blk
                blk_list[0::2] = chosen
                flat.extend(blk_list)

            # Split the flat token strings back into per-message content.
            synthetic_messages: list[dict] = []
            off = 0
            for m in messages_meta:
                n = m["tokens"]
                synthetic_messages.append(
                    {"role": m["role"], "content": "".join(flat[off : off + n])}
                )
                off += n

            output_length = e.get("output_length") or 0
            if not (0 < output_length <= max_output_tokens):
                seed = int(hash_ids[0][:8], 16) if hash_ids else 0
                output_length = 16 + (seed % (max_output_tokens - 15))

            row = {
                "abs_timestamp_ms": e["abs_timestamp_ms"],
                "input_length": input_length,
                "output_length": output_length,
                "unique_input_tokens": input_length,
                "hash_ids": hash_ids,
                "request": {
                    "model": "",
                    "messages": synthetic_messages,
                    "max_tokens": max_output_tokens,
                    "stream": True,
                },
                "response": None,
                "request_id": f"decoded_{stem}_{idx:06d}",
                "token_hash": e.get("token_hash", ""),
                "system_prompt_hash": e.get("system_prompt_hash"),
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

    os.replace(tmp_path, out_path)
    completions_volume.commit()
    return {"meta_filename": meta_filename, "skipped": False, "requests": count}


@app.function(image=image, timeout=14400, volumes={"/data": completions_volume})
def decode_week(
    input_dir: str = "/data/mooncake_traces/metadata_week",
    output_dir: str = "/data/mooncake_traces/decoded_week",
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    limit: int = 0,
) -> dict:
    """Fan ``decode_file`` over every metadata shard."""
    import os
    import time

    files = sorted(f for f in os.listdir(input_dir) if f.endswith(".jsonl"))
    if limit:
        files = files[:limit]
    print(f"[decode_week] {len(files)} shards -> {output_dir}")

    args = [(f, input_dir, output_dir, max_output_tokens) for f in files]
    t0 = time.time()
    total = 0
    skipped = 0
    for i, r in enumerate(decode_file.starmap(args), start=1):
        total += r.get("requests", 0)
        if r.get("skipped"):
            skipped += 1
        if i % 20 == 0 or i == len(files):
            print(
                f"  {i}/{len(files)} | {total:,} requests | {skipped} cached | "
                f"{time.time() - t0:.0f}s",
                flush=True,
            )
    completions_volume.commit()
    return {
        "output_dir": output_dir,
        "shards": len(files),
        "requests": total,
        "skipped_files": skipped,
        "elapsed_seconds": time.time() - t0,
    }


@app.local_entrypoint()
def week_decode(
    input_dir: str = "/data/mooncake_traces/metadata_week",
    output_dir: str = "/data/mooncake_traces/decoded_week",
    limit: int = 0,
):
    print(
        json.dumps(
            decode_week.remote(input_dir=input_dir, output_dir=output_dir, limit=limit), indent=2
        )
    )


@app.function(image=image, memory=1024 * 8, timeout=1800, volumes={"/data": completions_volume})
def validate_qwen(
    decoded_dir: str = "/data/mooncake_traces/decoded_week",
    n_per_shard: int = 200,
    n_shards: int = 4,
) -> dict:
    """Re-tokenize decoded message content under the real Qwen tokenizer and
    assert it reproduces ``input_length`` exactly. Also re-checks reuse."""
    import json
    import os

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False)
    shards = sorted(f for f in os.listdir(decoded_dir) if f.endswith(".jsonl"))[:n_shards]
    checked = 0
    mismatches = 0
    worst = 0
    for shard in shards:
        with open(os.path.join(decoded_dir, shard)) as f:
            for i, line in enumerate(f):
                if i >= n_per_shard:
                    break
                if not line.strip():
                    continue
                e = json.loads(line)
                got = sum(
                    len(tok.encode(m["content"], add_special_tokens=False))
                    for m in e["request"]["messages"]
                )
                checked += 1
                if got != e["input_length"]:
                    mismatches += 1
                    worst = max(worst, abs(got - e["input_length"]))
    return {
        "shards": len(shards),
        "checked": checked,
        "exact_match": checked - mismatches,
        "mismatches": mismatches,
        "worst_abs_diff": worst,
    }


@app.local_entrypoint()
def validate():
    print(json.dumps(validate_qwen.remote(), indent=2))
