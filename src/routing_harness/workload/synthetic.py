"""Synthetic workload generator.

Poisson arrivals with Zipf-distributed prefix popularity. Useful as a
fixture-friendly stand-in for the lmsys adapter when running tests
without a dataset on disk.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

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


def generate(params: SyntheticParams) -> InMemoryTrace:
    rng = random.Random(params.seed)
    # Precompute prefix "identities" so the same family yields matching prefix_key.
    families = [
        hashlib.blake2b(f"fam-{i}-{params.seed}".encode(), digest_size=8).hexdigest()
        for i in range(params.n_prefix_families)
    ]
    session_ids = [f"s{i:06d}" for i in range(params.n_sessions)]
    t = 0.0
    reqs: list[Request] = []
    for i in range(params.n_requests):
        # Exponential inter-arrival -> Poisson process.
        t += rng.expovariate(max(1e-9, params.arrival_rate_qps))
        fam_idx = _zipf_index(rng, params.n_prefix_families, params.zipf_s)
        prefix_key = families[fam_idx]
        prompt_len = rng.randint(params.prompt_len_min, params.prompt_len_max)
        tokens = tuple(rng.randrange(32_000) for _ in range(prompt_len))
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
    return InMemoryTrace(requests=reqs, source=f"synthetic(seed={params.seed})")
