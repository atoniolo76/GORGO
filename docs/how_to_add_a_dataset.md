# Adding a Dataset / Workload Adapter

An adapter is a module under `src/routing_harness/workload/` that builds
a `WorkloadTrace` (an iterable of `Request`) from some source.

## Interface

A `WorkloadTrace` is any object with:

- `__iter__(self) -> Iterator[Request]` yielding requests in
  non-decreasing `arrival_ts`.
- `describe(self) -> dict` returning a JSON-friendly summary (used in
  the results snapshot for reproducibility).

`InMemoryTrace` is the most common container; you can also stream.

## Minimum viable adapter

```python
# src/routing_harness/workload/my_dataset.py
from dataclasses import dataclass
from ..core import Request
from .trace import InMemoryTrace


@dataclass(frozen=True)
class MyDatasetConfig:
    local_path: str            # never auto-download; user provides path
    max_records: int | None = None
    seed: int = 0


def build_trace(cfg: MyDatasetConfig, arrival_rate_qps: float) -> InMemoryTrace:
    reqs: list[Request] = []
    # ... parse local file, produce Request objects with monotonic arrival_ts ...
    return InMemoryTrace(requests=reqs, source=f"my_dataset:{cfg.local_path}")
```

## Wire it into the CLI

Extend `src/routing_harness/cli.py::_build_trace` with a new `kind`:

```python
if kind == "my-dataset":
    cfg = MyDatasetConfig(**params)
    return build_trace(cfg, params["arrival_rate_qps"])
```

## Tests

Add a StubLoader or small fixture under `tests/fixtures/` so tests run
without hitting disk. Add:

- a determinism test (same seed → same trace),
- a schema/parsing test against a 2-3 record fixture,
- an invariant test (arrivals monotonic).

## Don'ts

- **Don't auto-download.** Require a local path. Document where the
  user should get it from.
- **Don't leak secrets.** If the dataset requires auth, document how
  to bring credentials, don't embed any.
- **Don't silently truncate.** If `max_records` is hit, describe it in
  `InMemoryTrace.describe()`.
