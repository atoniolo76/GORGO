"""Metrics collector.

Accumulates per-request cost breakdowns and derives aggregate metrics
required by the report: latency percentiles, goodput, hit rate, migration
count, pod utilization skew, hotspot precision/recall (if ground truth
supplied).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Any

from ..core import CostBreakdown, Decision, Request


@dataclass
class RequestRecord:
    request: Request
    decision: Decision
    cost: CostBreakdown
    cached_prefix_tokens: int
    reuse_available_blocks: int
    reuse_captured_blocks: int
    kv_transport_bytes: int
    migrated: bool = False


@dataclass
class MetricsCollector:
    records: list[RequestRecord] = field(default_factory=list)
    per_pod_busy_ms: dict[str, float] = field(default_factory=dict)

    def observe(self, rec: RequestRecord) -> None:
        self.records.append(rec)
        pod = rec.decision.prefill_pod_id
        self.per_pod_busy_ms[pod] = self.per_pod_busy_ms.get(pod, 0.0) + (
            rec.cost.compute_prefill_ms + rec.cost.compute_decode_ms
        )

    def _percentile(self, xs: list[float], p: float) -> float:
        if not xs:
            return 0.0
        s = sorted(xs)
        k = max(0, min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1)))))
        return s[k]

    def summary(self) -> dict[str, Any]:
        if not self.records:
            return {"n": 0}
        totals = [r.cost.total_ms for r in self.records]
        # TTFT: time to first token. Everything before decode, with
        # KV transport overlapped against prefill (max, not sum) to
        # match total_ms's prefill/transport bottleneck model.
        ttfts = [
            r.cost.routing_ms
            + r.cost.queueing_ms
            + r.cost.prefill_block_ms
            + r.cost.network_ms
            for r in self.records
        ]
        kv_bytes = sum(r.kv_transport_bytes for r in self.records)
        reuse_avail = sum(r.reuse_available_blocks for r in self.records)
        reuse_cap = sum(r.reuse_captured_blocks for r in self.records)
        reuse_denom = max(1, reuse_avail)
        # Macro (per-request) capture rate: avoids one huge request
        # drowning out many small ones. Reported alongside the
        # aggregate-weighted "micro" figure.
        per_req_rates = [
            (r.reuse_captured_blocks / r.reuse_available_blocks)
            for r in self.records
            if r.reuse_available_blocks > 0
        ]
        per_pod = self.per_pod_busy_ms
        pod_vals = list(per_pod.values()) or [0.0]
        mean_busy = mean(pod_vals)
        skew = (max(pod_vals) - min(pod_vals)) / max(1e-9, mean_busy)
        return {
            "n": len(self.records),
            "latency_ms": {
                "p50": self._percentile(totals, 50),
                "p95": self._percentile(totals, 95),
                "p99": self._percentile(totals, 99),
                "mean": mean(totals),
            },
            "ttft_ms": {
                "p50": self._percentile(ttfts, 50),
                "p95": self._percentile(ttfts, 95),
                "p99": self._percentile(ttfts, 99),
                "mean": mean(ttfts),
            },
            "decomposition_ms_mean": {
                "routing": mean(r.cost.routing_ms for r in self.records),
                "queueing": mean(r.cost.queueing_ms for r in self.records),
                "compute_prefill": mean(r.cost.compute_prefill_ms for r in self.records),
                "compute_decode": mean(r.cost.compute_decode_ms for r in self.records),
                "network": mean(r.cost.network_ms for r in self.records),
                "kv_transport": mean(r.cost.kv_transport_ms for r in self.records),
            },
            "kv": {
                "transport_bytes_total": kv_bytes,
                "reuse_available_blocks": reuse_avail,
                "reuse_captured_blocks": reuse_cap,
                "capture_rate_micro": reuse_cap / reuse_denom,
                "capture_rate_macro": mean(per_req_rates) if per_req_rates else 0.0,
                # Retain the legacy key so existing consumers keep working;
                # prefer the explicit micro/macro pair above.
                "capture_rate": reuse_cap / reuse_denom,
                "hit_rate": sum(1 for r in self.records if r.cached_prefix_tokens > 0)
                / len(self.records),
            },
            "load": {
                "per_pod_busy_ms": dict(per_pod),
                "skew": skew,
            },
            "migrations": sum(1 for r in self.records if r.migrated),
        }
