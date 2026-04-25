"""WorkloadTrace: iterable of Requests with monotonic arrival times.

A trace is the interface the simulator consumes. Adapters (lmsys,
synthetic, replay) produce traces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol

from ..core import Request


class WorkloadTrace(Protocol):
    """A trace is iterable of Request in non-decreasing arrival_ts order."""

    def __iter__(self) -> Iterator[Request]: ...

    def describe(self) -> dict: ...


@dataclass
class InMemoryTrace:
    requests: list[Request]
    source: str = "in-memory"
    # Generator inputs (e.g. SyntheticParams as dict, or adapter cfg+params).
    # Surfaced in describe() so the runner's content-addressed run_id
    # discriminates on full workload config — without this, a sweep over
    # workload params (zipf_s, n_prefix_families, ...) collides on run_id
    # because describe() is otherwise insensitive to those axes.
    params: dict | None = None

    def __post_init__(self) -> None:
        # Assert non-decreasing arrivals; this is a correctness invariant
        # for the simulator's event loop.
        for a, b in zip(self.requests, self.requests[1:]):
            if b.arrival_ts < a.arrival_ts:
                raise ValueError(
                    f"trace not sorted: {a.arrival_ts} -> {b.arrival_ts}"
                )

    def __iter__(self) -> Iterator[Request]:
        return iter(self.requests)

    def describe(self) -> dict:
        n = len(self.requests)
        base: dict = {"source": self.source, "n": n}
        if n > 0:
            base["t_start"] = self.requests[0].arrival_ts
            base["t_end"] = self.requests[-1].arrival_ts
            base["unique_sessions"] = len({r.session_id for r in self.requests})
        if self.params is not None:
            base["params"] = self.params
        return base
