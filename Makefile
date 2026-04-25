.PHONY: smoke-policies

# One command to confirm all 12 routing policies are wired up and
# don't crash: liveness/contract/per-policy unit suites in one go.
# Catches registry, config-wiring, and end-to-end runtime regressions
# without running the full test suite. See tests/integration/test_all_policies_smoke.py.
smoke-policies:
	pytest \
		tests/integration/test_all_policies_smoke.py \
		tests/unit/test_policies_individual.py \
		tests/contract/test_policy_contract.py
