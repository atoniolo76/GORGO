"""Hand-crafted deterministic tiny traces for tests.

Keeping these in Python (not JSONL on disk) means tests run without any
file I/O and can be golden-checked precisely.
"""

from __future__ import annotations

from routing_harness.core import Request


def shared_prefix_trace() -> list[Request]:
    """Two conversations sharing the first 32 tokens (2 blocks of 16)."""
    shared = tuple(range(32))
    a_tail = tuple(range(100, 148))
    b_tail = tuple(range(200, 260))
    return [
        Request("r0", "sA", 0.0, shared + a_tail, max_output_tokens=8, prefix_key=None),
        Request("r1", "sB", 0.05, shared + b_tail, max_output_tokens=8, prefix_key=None),
        Request("r2", "sA", 0.10, shared + a_tail + (300, 301, 302, 303, 304, 305, 306, 307), max_output_tokens=8),
        Request("r3", "sC", 0.15, tuple(range(500, 580)), max_output_tokens=8),
    ]


def sticky_session_trace() -> list[Request]:
    return [
        Request(f"r{i}", "sA" if i % 2 == 0 else "sB", 0.01 * i, tuple(range(i, i + 48)), 8)
        for i in range(8)
    ]
