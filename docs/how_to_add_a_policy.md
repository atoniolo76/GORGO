# Adding a Routing Policy

A new policy is three things:

1. A class that satisfies the `RoutingPolicy` protocol.
2. A registration line.
3. A contract-test pass.

## 1. The class

Create `src/routing_harness/policies/my_policy.py`:

```python
from dataclasses import dataclass

from ..cluster import ClusterState
from ..core import Decision, Request
from ..kv_cache import KVCacheState
from ..policy import register_policy


@register_policy("my-policy")
@dataclass
class MyPolicy:
    # Policy-specific config goes here; YAML `policy.params` maps to
    # these fields by name.
    alpha: float = 1.0

    def decide(
        self,
        request: Request,
        cluster: ClusterState,
        kv_cache: KVCacheState,
    ) -> Decision:
        cands = cluster.prefill_capable()
        if not cands:
            return Decision("__none__", "__none__", "no-prefill-capable-pod")
        # ... your logic ...
        pick = cands[0]
        return Decision(pick.spec.pod_id, pick.spec.pod_id, rationale="my-rule")
```

### Rules

- **Do not mutate** `cluster` or `kv_cache`. Contract tests enforce it.
- **Return the `__none__` sentinel** on empty clusters; never raise.
- **Use the injected RNG** (constructor argument) for randomness, so
  seeding works.
- **Keep it pure.** Side effects belong in the simulator.

## 2. Register it

Import your module in `src/routing_harness/policies/__init__.py` so the
registry is populated on package import.

## 3. Make it configurable

Policy kwargs are passed through from YAML unchanged:

```yaml
policy:
  policy_id: "my-policy"
  params:
    alpha: 0.7
```

## 4. Test it

Contract tests in `tests/contract/test_policy_contract.py` are
parameterized over `list_policies()` and will run automatically.

Add a targeted behavior test in `tests/unit/test_policies_individual.py`
covering what makes your policy unique, e.g. "routes to warm prefix
owner", "deprioritizes tenants over quota", etc.

## 5. Run it (once execution is permitted)

```bash
routing-harness run --config configs/my_policy_run.yaml
```
