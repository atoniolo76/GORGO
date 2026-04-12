"""Random routing: uniform over prefill-capable pods.

Baseline. Used to establish a floor for KV-reuse metrics (what you get
with zero cache awareness) and to sanity-check fairness metrics.
"""

from __future__ import annotations

import random as _random
from dataclasses import dataclass

from ..cluster import ClusterState
from ..core import Decision, Request
from ..kv_cache import KVCacheState
from ..policy import register_policy


@register_policy("random")
@dataclass
class RandomPolicy:
    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = _random.Random(self.seed)

    def decide(
        self,
        request: Request,
        cluster: ClusterState,
        kv_cache: KVCacheState,
    ) -> Decision:
        candidates = cluster.prefill_capable()
        if not candidates:
            return Decision("__none__", "__none__", "no-prefill-capable-pod")
        pick = self._rng.choice(candidates)
        return Decision(
            prefill_pod_id=pick.spec.pod_id,
            decode_pod_id=pick.spec.pod_id,
            rationale="uniform-random",
        )
