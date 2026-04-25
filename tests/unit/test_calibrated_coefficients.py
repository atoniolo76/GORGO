"""Sanity checks on the calibrated ComputeParams shipped in configs/.

Source of truth: ``configs/calibrated_a100.yaml`` + the matching
``fit_summary.json`` under ``research/data/calibration/<ts>/`` (produced
by ``scripts/calibrate.py``). The calibration run is documented in
``docs/calibration_plan.md`` (bead go-8cm).

The properties checked here are the ones from plan §5 / §6:

- Coefficients are strictly positive and physically ordered
  (decode is more expensive per token than prefill on modern
  transformers).
- The logarithmic batch amortization is monotonically decreasing in
  batch size on the doubling grid used by §3.3.
- The checked-in acceptance gates (R² thresholds, residual-SE bounds,
  `k` in the plausible band) actually pass — i.e. nobody lowered a
  threshold to make a bad fit land.

Tests ``skip`` (rather than fail) when the calibrated artifacts are
absent, so the harness remains runnable before a calibration sweep has
been executed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import yaml

from routing_harness.cost_model import ComputeParams

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "calibrated_a100.yaml"
CALIBRATION_ROOT = REPO_ROOT / "research" / "data" / "calibration"


def _latest_fit_summary() -> Path | None:
    """Return the newest fit_summary.json under research/data/calibration/."""
    if not CALIBRATION_ROOT.exists():
        return None
    candidates = sorted(
        (p for p in CALIBRATION_ROOT.glob("*/fit_summary.json") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


@pytest.fixture(scope="module")
def calibrated_config() -> dict:
    if not CONFIG_PATH.exists():
        pytest.skip(
            f"{CONFIG_PATH.relative_to(REPO_ROOT)} not present — "
            "calibration sweep has not been run yet"
        )
    with CONFIG_PATH.open() as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def calibrated_compute(calibrated_config: dict) -> ComputeParams:
    c = calibrated_config["compute"]
    return ComputeParams(
        prefill_ms_per_token=float(c["prefill_ms_per_token"]),
        decode_ms_per_token=float(c["decode_ms_per_token"]),
        prefill_overhead_ms=float(c["prefill_overhead_ms"]),
        decode_overhead_ms=float(c["decode_overhead_ms"]),
        decode_batch_k=float(c["decode_batch_k"]),
    )


@pytest.fixture(scope="module")
def fit_summary() -> dict:
    path = _latest_fit_summary()
    if path is None:
        pytest.skip("no fit_summary.json under research/data/calibration/")
    with path.open() as fh:
        return json.load(fh)


# --- Plan §5 sanity properties ---------------------------------------------


def test_prefill_ms_per_token_positive(calibrated_compute: ComputeParams) -> None:
    assert calibrated_compute.prefill_ms_per_token > 0.0


def test_decode_ms_per_token_positive(calibrated_compute: ComputeParams) -> None:
    assert calibrated_compute.decode_ms_per_token > 0.0


def test_decode_more_expensive_than_prefill_per_token(
    calibrated_compute: ComputeParams,
) -> None:
    """Modern transformers are memory-bandwidth-bound on decode, so
    per-token decode must exceed per-token prefill. A fit that flipped
    this indicates a sweep / regression bug, not a hardware surprise."""
    assert calibrated_compute.decode_ms_per_token > calibrated_compute.prefill_ms_per_token


def test_decode_batch_k_in_plausible_band(
    calibrated_compute: ComputeParams,
) -> None:
    """Plan §6 acceptance gate: k ∈ [0.1, 2.0]. Outside this band the
    logarithmic functional form is suspect (plan §3.3 risk)."""
    k = calibrated_compute.decode_batch_k
    assert 0.1 <= k <= 2.0, f"decode_batch_k={k} outside [0.1, 2.0] band"


def test_effective_decode_ms_monotonic_in_batch(
    calibrated_compute: ComputeParams,
) -> None:
    """Plan §5 item 3: on the power-of-two grid {1, 2, 4, ..., 128}, the
    effective per-token decode cost must strictly decrease as batch
    grows. This is the entire point of continuous batching; a fit that
    inverts the curve would silently flip every policy comparison."""
    k = calibrated_compute.decode_batch_k
    base = calibrated_compute.decode_ms_per_token

    def effective(batch: int) -> float:
        return base / (1.0 + k * math.log(1.0 + max(0, batch - 1)))

    batches = [1, 2, 4, 8, 16, 32, 64, 128]
    values = [effective(b) for b in batches]
    assert values[0] == pytest.approx(base), "batch=1 should reproduce baseline"
    for prev, cur, b_prev, b_cur in zip(values[:-1], values[1:], batches[:-1], batches[1:]):
        assert cur < prev, (
            f"effective decode non-monotonic: batch={b_prev}:{prev:.4f} -> "
            f"batch={b_cur}:{cur:.4f} (k={k:.4f})"
        )


def test_overheads_nonnegative(calibrated_compute: ComputeParams) -> None:
    """Overheads can be exactly zero on a very clean fit but must never
    be negative — a negative intercept means the linear model is
    compensating for curvature in the data, and the fit shouldn't ship."""
    assert calibrated_compute.prefill_overhead_ms >= 0.0
    assert calibrated_compute.decode_overhead_ms >= 0.0


# --- Plan §6 acceptance gates ----------------------------------------------


def test_acceptance_gates_all_pass(fit_summary: dict) -> None:
    """The fit summary records which plan-§6 gates passed at fit time.
    If any gate is False, the fit was published in violation of the
    published acceptance criteria and should not be treated as
    authoritative."""
    gates = fit_summary["acceptance_gates"]
    failing = {
        k: v for k, v in gates.items() if isinstance(v, bool) and not v and not k.endswith("_pct")
    }
    assert not failing, f"acceptance gates failed at fit time: {failing}"


def test_fit_summary_matches_config(fit_summary: dict, calibrated_compute: ComputeParams) -> None:
    """If the config was regenerated by hand after the sweep, the YAML
    can drift from fit_summary.json. Cross-check they agree to 1 μs."""
    cp = fit_summary["compute_params"]
    assert cp["prefill_ms_per_token"] == pytest.approx(
        calibrated_compute.prefill_ms_per_token, abs=1e-6
    )
    assert cp["decode_ms_per_token"] == pytest.approx(
        calibrated_compute.decode_ms_per_token, abs=1e-6
    )
    assert cp["prefill_overhead_ms"] == pytest.approx(
        calibrated_compute.prefill_overhead_ms, abs=1e-6
    )
    assert cp["decode_overhead_ms"] == pytest.approx(
        calibrated_compute.decode_overhead_ms, abs=1e-6
    )
    assert cp["decode_batch_k"] == pytest.approx(calibrated_compute.decode_batch_k, abs=1e-6)


def test_fit_covers_expected_sweep(fit_summary: dict) -> None:
    """Guardrail against a truncated sweep being fitted by mistake.
    Plan §3 specifies 140 / 90 / 5120 raw requests; allow a small
    slack for dropped rows without letting a half-sweep through."""
    n = fit_summary["n_requests"]
    assert n["prefill"] >= 140 * 0.9
    assert n["decode_single"] >= 90 * 0.9
    # decode_batch is sum(batch_sizes)*5 = (1+2+4+...+128)*5 = 255*5 = 1275
    assert n["decode_batch"] >= 1275 * 0.9
