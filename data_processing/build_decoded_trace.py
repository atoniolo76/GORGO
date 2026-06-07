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
        --metadata-path /data/mooncake_traces/metadata/glm5_metadata_apr2_0030_to_0100.jsonl \\
        --output-path /data/mooncake_traces/decoded/glm5_decoded_apr2_0030_to_0100.jsonl
"""

from __future__ import annotations

import json
import os
import random
import time

import modal

from app import app, completions_volume

DEFAULT_VOCAB_SIZE = 151643  # gpt-4o / Qwen tokenizer vocab
DEFAULT_MAX_OUTPUT_TOKENS = 128

image = modal.Image.debian_slim().pip_install("tiktoken").add_local_python_source("app")


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
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
):
    import tiktoken

    enc = tiktoken.encoding_for_model("gpt-4o")
    rng = random.Random(seed)
    t0 = time.perf_counter()

    # Build a pool of common English words that are each exactly 1 token.
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
        w for w in candidate_words if len(enc.encode(w, disallowed_special=())) == 1
    ]
    print(f"[decoded] {len(single_token_words)} single-token words available")

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
    metadata_path: str = "/data/mooncake_traces/metadata/glm5_metadata_apr2_0030_to_0100.jsonl",
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
