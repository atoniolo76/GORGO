"""ClusterState: the set of pods + helpers policies use to read state.

The simulator owns mutation of PodRuntime. Policies receive a
ClusterState and must treat it as read-only. Defensive copies are not
made for performance; contract tests enforce non-mutation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .core import Phase, PodRuntime, PodSpec


@dataclass
class ClusterState:
    pods: dict[str, PodRuntime] = field(default_factory=dict)

    @classmethod
    def from_specs(cls, specs: Iterable[PodSpec]) -> "ClusterState":
        return cls(pods={s.pod_id: PodRuntime(spec=s) for s in specs})

    def prefill_capable(self) -> list[PodRuntime]:
        return [
            p for p in self.pods.values()
            if p.spec.role in (Phase.PREFILL, Phase.BOTH)
        ]

    def decode_capable(self) -> list[PodRuntime]:
        return [
            p for p in self.pods.values()
            if p.spec.role in (Phase.DECODE, Phase.BOTH)
        ]

    def get(self, pod_id: str) -> PodRuntime:
        return self.pods[pod_id]

    def __len__(self) -> int:
        return len(self.pods)
