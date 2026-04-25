"""Synthetic workload generator.

Poisson arrivals with Zipf-distributed prefix popularity. Useful as a
fixture-friendly stand-in for the lmsys adapter when running tests
without a dataset on disk.

Two prefix-structure modes:

- **Opaque-key mode** (``shared_prefix_tokens = 0``, the default): each
  family is identified by a single opaque ``prefix_key`` string. The
  engine treats this as a one-block prefix regardless of prompt length,
  which is fine for fixture-style tests but produces degenerate metrics
  under a policy sweep — every pod trivially caches every family's one
  block, so ``hit_rate`` and ``capture_rate`` both pin to ~1.0 across
  all policies and the prefix-length signal is absent.

- **Shared-head mode** (``shared_prefix_tokens > 0``): each family gets
  a deterministic shared-head token sequence of that length prepended
  to every request in the family. ``prefix_key`` is cleared so the
  engine enumerates real block-level hashes over ``prompt_tokens``;
  policies see distinct prefix *lengths*, and KV capacity pressure can
  be tuned by ``n_prefix_families × shared_prefix_tokens`` relative to
  per-pod cache size. Use this mode whenever a metric needs to
  differentiate prefix-aware policies from cache-oblivious ones.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass

from ..core import Request
from .trace import InMemoryTrace


@dataclass(frozen=True)
class SyntheticParams:
    n_requests: int
    arrival_rate_qps: float
    n_prefix_families: int
    zipf_s: float
    prompt_len_min: int
    prompt_len_max: int
    max_output_tokens: int
    n_sessions: int
    seed: int
    shared_prefix_tokens: int = 0


def _zipf_index(rng: random.Random, n: int, s: float) -> int:
    # Simple inverse-transform for truncated Zipf with N items. O(n) per draw;
    # fine for synthetic sizes but we cache the CDF for reuse.
    weights = [1.0 / ((k + 1) ** s) for k in range(n)]
    total = sum(weights)
    r = rng.random() * total
    cum = 0.0
    for k, w in enumerate(weights):
        cum += w
        if r <= cum:
            return k
    return n - 1


def _family_shared_head(fam_idx: int, seed: int, n_tokens: int) -> tuple[int, ...]:
    """Deterministic per-family shared token sequence.

    Seeded off (fam_idx, trace seed) so the same family reproduces the
    same head across requests in a trace, and across repeat runs at the
    same seed — but different families get unrelated heads.
    """
    head_rng = random.Random(f"family-head-{fam_idx}-{seed}")
    return tuple(head_rng.randrange(32_000) for _ in range(n_tokens))


def generate(params: SyntheticParams) -> InMemoryTrace:
    rng = random.Random(params.seed)
    # Precompute prefix "identities" so the same family yields matching prefix_key.
    families = [
        hashlib.blake2b(f"fam-{i}-{params.seed}".encode(), digest_size=8).hexdigest()
        for i in range(params.n_prefix_families)
    ]
    if params.shared_prefix_tokens > 0:
        family_heads = [
            _family_shared_head(i, params.seed, params.shared_prefix_tokens)
            for i in range(params.n_prefix_families)
        ]
    else:
        family_heads = None
    session_ids = [f"s{i:06d}" for i in range(params.n_sessions)]
    t = 0.0
    reqs: list[Request] = []
    for i in range(params.n_requests):
        # Exponential inter-arrival -> Poisson process.
        t += rng.expovariate(max(1e-9, params.arrival_rate_qps))
        fam_idx = _zipf_index(rng, params.n_prefix_families, params.zipf_s)
        prompt_len = rng.randint(params.prompt_len_min, params.prompt_len_max)
        if family_heads is not None:
            head = family_heads[fam_idx]
            # Clamp the head to prompt_len so we never exceed the drawn
            # prompt length; the suffix fills the remainder.
            head_len = min(len(head), prompt_len)
            suffix_len = prompt_len - head_len
            tokens = head[:head_len] + tuple(
                rng.randrange(32_000) for _ in range(suffix_len)
            )
            # Clear prefix_key so the engine enumerates real block-level
            # hashes over prompt_tokens, not a single opaque block.
            prefix_key = None
        else:
            tokens = tuple(rng.randrange(32_000) for _ in range(prompt_len))
            prefix_key = families[fam_idx]
        req = Request(
            request_id=f"r{i:08d}",
            session_id=rng.choice(session_ids),
            arrival_ts=t,
            prompt_tokens=tokens,
            max_output_tokens=params.max_output_tokens,
            prefix_key=prefix_key,
            metadata={"family": fam_idx},
        )
        reqs.append(req)
    return InMemoryTrace(
        requests=reqs,
        source=f"synthetic(seed={params.seed})",
        params=asdict(params),
    )
