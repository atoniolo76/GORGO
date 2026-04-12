"""Routing strategy comparison harness for LLM inference serving.

This package provides a deterministic, test-first experiment harness for
comparing KV-cache-aware routing policies against a configurable cluster
and workload model. It does not run real inference; it simulates.

See docs/harness_overview.md for architecture and design rationale.
"""

from __future__ import annotations

__version__ = "0.1.0"
