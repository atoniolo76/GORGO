"""Config schema + sweep expansion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from routing_harness.config.schema import ConfigError, load_run, load_sweep
from routing_harness.config.sweep import expand


MIN_BASE = {
    "name": "t",
    "policy": {"policy_id": "random", "params": {"seed": 0}},
    "topology": {
        "pods": [
            {
                "pod_id": "p0", "role": "both", "gpu_count": 1,
                "kv_cache_bytes": 1024, "max_concurrent_prefill": 1,
                "max_concurrent_decode": 1,
            }
        ]
    },
    "compute": {
        "prefill_ms_per_token": 0.1, "decode_ms_per_token": 1.0,
        "prefill_overhead_ms": 1.0, "decode_overhead_ms": 0.5,
    },
    "network": {
        "client_rtt_ms": 1.0, "inter_pod_rtt_ms": 0.1,
        "inter_pod_bandwidth_gbps": 100.0, "kv_bytes_per_token": 1024,
        "serialization_overhead_ms": 0.1,
    },
    "scheduler": {"base_routing_ms": 0.1, "per_pod_consideration_us": 1.0},
    "engine": {},
    "workload": {
        "kind": "synthetic",
        "params": {
            "n_requests": 10, "arrival_rate_qps": 5.0,
            "n_prefix_families": 2, "zipf_s": 1.0,
            "prompt_len_min": 8, "prompt_len_max": 16,
            "max_output_tokens": 4, "n_sessions": 2,
        }
    },
    "seeds": [0],
    "output_dir": "/tmp/unused",
}


def test_load_run_json(tmp_path: Path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps(MIN_BASE))
    rc = load_run(p)
    assert rc.policy.policy_id == "random"
    assert len(rc.topology.pods) == 1


def test_load_run_missing_key(tmp_path: Path):
    raw = dict(MIN_BASE)
    raw.pop("scheduler")
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(raw))
    with pytest.raises(ConfigError):
        load_run(p)


def test_sweep_expansion(tmp_path: Path):
    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps(MIN_BASE))
    sweep_spec = {
        "name": "s",
        "base": "base.json",
        "grid": {
            "policy.policy_id": ["random", "least-request"],
            "workload.params.arrival_rate_qps": [5.0, 10.0],
        },
    }
    sp = tmp_path / "sweep.json"
    sp.write_text(json.dumps(sweep_spec))
    sweep = load_sweep(sp)
    axes = list(expand(sweep))
    assert len(axes) == 4
    pids = {rc.policy.policy_id for _, rc in axes}
    qps = {rc.workload.params["arrival_rate_qps"] for _, rc in axes}
    assert pids == {"random", "least-request"}
    assert qps == {5.0, 10.0}
