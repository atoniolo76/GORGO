"""Quick measurement: how much cross-user reuse comes from shared system prompts."""

import json, os, hashlib
from collections import Counter, defaultdict

import modal
from app import app, completions_volume

image = modal.Image.debian_slim().pip_install("duckdb", "tiktoken").add_local_python_source("app")


@app.function(image=image, memory=1024 * 8, timeout=1800, volumes={"/data": completions_volume})
def measure():
    import duckdb, tiktoken

    enc = tiktoken.encoding_for_model("gpt-4o")

    FILE_PREFIX = "llm_responses_202604"
    FILE_CUTOFF = "llm_responses_20260408"
    files = sorted(
        f
        for f in os.listdir("/data")
        if f.endswith(".parquet") and f.startswith(FILE_PREFIX) and f < FILE_CUTOFF + ".parquet"
    )

    sys_prompt_hashes = Counter()  # hash -> count of requests
    sys_prompt_tokens = {}  # hash -> token count
    sys_prompt_users = defaultdict(set)  # hash -> set of token_hashes (users)
    total_requests = 0
    total_tokens = 0
    no_system = 0

    con = duckdb.connect()
    for filename in files:
        path = os.path.join("/data", filename)
        cursor = con.execute(
            "SELECT request_metadata.token_hash, request FROM read_parquet(?) WHERE request NOT LIKE '%keep-alive%'",
            [path],
        )
        while True:
            chunk = cursor.fetchmany(4096)
            if not chunk:
                break
            for token_hash, request_raw in chunk:
                total_requests += 1
                try:
                    req = json.loads(request_raw) if isinstance(request_raw, str) else request_raw
                except:
                    continue
                msgs = req.get("messages", [])
                if not msgs:
                    continue

                # Find system prompt
                sys_msg = None
                for m in msgs:
                    if isinstance(m, dict) and m.get("role") == "system":
                        sys_msg = m
                        break

                if sys_msg is None:
                    no_system += 1
                    continue

                content = sys_msg.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        b.get("text", "") if isinstance(b, dict) else str(b) for b in content
                    )

                h = hashlib.sha256(content.encode()).hexdigest()[:16]
                sys_prompt_hashes[h] += 1
                sys_prompt_users[h].add(token_hash or "")

                if h not in sys_prompt_tokens:
                    toks = len(enc.encode(content, disallowed_special=()))
                    sys_prompt_tokens[h] = toks
                    total_tokens += toks  # count once for measurement

        print(
            f"  {filename}: {total_requests} total, {len(sys_prompt_hashes)} unique system prompts",
            flush=True,
        )
    con.close()

    # Analysis
    print(f"\n=== System Prompt Analysis ===")
    print(f"Total requests: {total_requests:,}")
    print(
        f"Requests with system prompt: {total_requests - no_system:,} ({100 * (total_requests - no_system) / total_requests:.1f}%)"
    )
    print(f"Requests without system prompt: {no_system:,}")
    print(f"Unique system prompts: {len(sys_prompt_hashes):,}")

    # Cross-user sharing: system prompts used by >1 user
    multi_user_prompts = {
        h: c for h, c in sys_prompt_hashes.items() if len(sys_prompt_users[h]) > 1
    }
    print(f"\nSystem prompts shared across users: {len(multi_user_prompts):,}")

    # Token savings from shared system prompts
    total_sys_tokens_if_shared = 0
    total_sys_savings = 0
    for h, count in sys_prompt_hashes.items():
        tok_count = sys_prompt_tokens.get(h, 0)
        total_sys_tokens_if_shared += tok_count * count
        if len(sys_prompt_users[h]) > 1:
            # First occurrence per user is unique, rest is cross-user reuse
            n_users = len(sys_prompt_users[h])
            # savings = tokens * (total_requests_with_this_prompt - n_unique_users)
            # but for KV cache across users, it's tokens * (count - 1) if same replica
            total_sys_savings += tok_count * (count - 1)

    print(f"\nTotal system prompt tokens (with repetition): {total_sys_tokens_if_shared:,}")
    print(f"Savings if all system prompts cached: {total_sys_savings:,}")
    print(f"As % of total dataset tokens (8.65B): {100 * total_sys_savings / 8_652_547_293:.3f}%")

    # Top shared system prompts
    print(f"\nTop 10 shared system prompts:")
    for h, count in sys_prompt_hashes.most_common(10):
        n_users = len(sys_prompt_users[h])
        toks = sys_prompt_tokens.get(h, 0)
        print(f"  {h}: {count:,} requests, {n_users} users, {toks} tokens")


@app.local_entrypoint()
def main():
    measure.remote()
