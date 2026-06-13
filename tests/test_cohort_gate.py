"""Offline unit tests for the cohort generalizability gate.

Pure-function tests — no npz / no scratch access. Verifies the scaling-law
math and the GO / RECALIBRATE / NO-GO verdict thresholds are internally
consistent with the baked-in coefficients.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "cohort_gate", REPO / "scripts" / "cohort_generalizability_gate.py"
)
gate = importlib.util.module_from_spec(_spec)
# Avoid importing heavy deps at module import — scipy is needed, so guard.
_spec.loader.exec_module(gate)  # type: ignore[union-attr]


def test_law_constants_sane():
    assert gate.LAW_SLOPE < 0, "coverage must decrease with KS distance"
    assert 0 < gate.LAW_INTERCEPT, "intercept (coverage at KS=0) must be positive"
    assert 0.0 <= gate.LAW_R2 <= 1.0
    assert gate.LAW_RESID_STD > 0


def test_predict_coverage_monotone_decreasing():
    a, b = gate.LAW_INTERCEPT, gate.LAW_SLOPE
    c_low = gate.predict_coverage(0.10, b, a)
    c_high = gate.predict_coverage(0.50, b, a)
    assert c_low > c_high, "higher KS -> lower predicted coverage"


def test_predict_coverage_clipped_to_unit_interval():
    a, b = gate.LAW_INTERCEPT, gate.LAW_SLOPE
    assert gate.predict_coverage(0.0, b, a) <= 1.0      # intercept 1.73 -> clip to 1.0
    assert gate.predict_coverage(0.0, b, a) >= 0.0
    assert gate.predict_coverage(2.0, b, a) == 0.0      # huge KS -> clip to 0.0


@pytest.mark.parametrize(
    "cov,expected",
    [
        (0.95, "GO"),
        (0.85, "GO"),          # boundary inclusive
        (0.8499, "RECALIBRATE"),
        (0.70, "RECALIBRATE"),  # boundary inclusive
        (0.6999, "NO-GO"),
        (0.10, "NO-GO"),
    ],
)
def test_verdict_thresholds(cov, expected):
    assert gate.verdict_for(cov) == expected


def test_ks_thresholds_consistent_with_verdicts():
    """The KS thresholds derived from the law must round-trip through the
    verdict function."""
    a, b = gate.LAW_INTERCEPT, gate.LAW_SLOPE
    ks_go = (gate.GO_COVERAGE - a) / b
    ks_recal = (gate.RECAL_COVERAGE - a) / b
    # Just below the GO KS threshold -> GO; just above -> RECALIBRATE.
    assert gate.verdict_for(gate.predict_coverage(ks_go - 0.01, b, a)) == "GO"
    assert gate.verdict_for(gate.predict_coverage(ks_go + 0.01, b, a)) == "RECALIBRATE"
    # Just above the RECAL KS threshold -> NO-GO.
    assert gate.verdict_for(gate.predict_coverage(ks_recal + 0.01, b, a)) == "NO-GO"
    # Ordering sanity: GO threshold is a smaller KS than RECAL threshold.
    assert ks_go < ks_recal


def test_compute_mean_ks_empty_shared_is_nan():
    import numpy as np
    x = np.zeros((10, 5))
    assert np.isnan(gate.compute_mean_ks(x, x, []))


def test_compute_mean_ks_identical_distributions_near_zero():
    import numpy as np
    rng = np.random.default_rng(0)
    x = rng.random((2000, 4))
    ks = gate.compute_mean_ks(x, x, [0, 1, 2, 3])
    assert ks == 0.0, "KS of a distribution with itself is exactly 0"
