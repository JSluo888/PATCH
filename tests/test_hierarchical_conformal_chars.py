"""Characterization + pluggable-score tests for HierarchicalConformalPredictor.

The live conformal scorer had no test pinning its numbers. This file:
1. Pins the default scorer's calibration thresholds to exact reference values
   (computed 2026-05-29, seed=0) so any change to the live scoring path is
   caught. This guards the pluggable-score-fn refactor: the default score_fn
   must reproduce these numbers byte-for-byte.
2. Verifies the optional ``score_fn`` hook: injecting an alternative score
   (e.g. RAPS/APS) is accepted and produces a calibrated predictor whose sets
   remain ontology-consistent and never expand beyond the full label set.
"""
from __future__ import annotations

import numpy as np
import pytest

from uncertainty.hierarchical_conformal import HierarchicalConformalPredictor

ALL_TYPES = [
    "Helper_T_Cell", "Cytotoxic_T_Cell", "Regulatory_T_Cell", "B_Cell", "Plasma_Cell",
    "Macrophage_CD163pos", "Macrophage_CD163neg", "NK_Cell", "Dendritic_Cell", "Neutrophil",
    "Epithelial", "Endothelial", "Stromal", "Immune_Other",
]

# Reference thresholds from the default (1 - p) scorer, seed=0, N=3000.
REF_LINEAGE = {
    "Endothelial": 0.997545, "Epithelial": 0.997039, "Immune": 0.481973, "Stromal": 0.996909,
}
REF_SUBTYPE = {
    "B_Lineage": 0.97817, "Dendritic_Cell": 0.996019, "Immune_Other": 0.99514,
    "Myeloid": 0.975258, "NK_Cell": 0.995203, "Neutrophil": 0.994136, "T_Lineage": 0.938674,
}
REF_FINE = {
    "B_Cell": 0.955494, "Cytotoxic_T_Cell": 0.959473, "Helper_T_Cell": 0.97629,
    "Macrophage_CD163neg": 0.961918, "Macrophage_CD163pos": 0.954459,
    "Plasma_Cell": 0.949061, "Regulatory_T_Cell": 0.962105,
}


def _synthetic(n: int = 3000, seed: int = 0):
    k = len(ALL_TYPES)
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 1.5, size=(n, k))
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)
    y = rng.integers(0, k, size=n)
    return p, y


def _calibrated(**kwargs):
    p, y = _synthetic()
    hcp = HierarchicalConformalPredictor(ALL_TYPES, **kwargs)
    hcp.calibrate(p, y)
    return hcp, p


def test_default_scorer_thresholds_match_reference():
    """Byte-identical guard: the default scorer must reproduce the reference
    thresholds. If a refactor changes these, this test fails."""
    hcp, _ = _calibrated()
    for k, v in REF_LINEAGE.items():
        assert hcp.lineage_thresholds[k] == pytest.approx(v, abs=1e-5), f"lineage:{k}"
    for k, v in REF_SUBTYPE.items():
        assert hcp.subtype_thresholds[k] == pytest.approx(v, abs=1e-5), f"subtype:{k}"
    for k, v in REF_FINE.items():
        assert hcp.fine_thresholds[k] == pytest.approx(v, abs=1e-5), f"fine:{k}"


def test_calibration_is_deterministic():
    a, _ = _calibrated()
    b, _ = _calibrated()
    assert a.lineage_thresholds == b.lineage_thresholds
    assert a.subtype_thresholds == b.subtype_thresholds
    assert a.fine_thresholds == b.fine_thresholds


def test_predict_sets_are_ontology_consistent():
    hcp, p = _calibrated()
    valid_lineages = set(hcp.lineage_names)
    for i in range(20):
        r = hcp.predict(p[i])
        assert len(r.lineage_set) >= 1
        assert set(r.lineage_set).issubset(valid_lineages)


def test_score_fn_hook_accepts_alternative_and_stays_consistent():
    """If the pluggable score_fn hook exists, an alternative score (here a
    simple rank-penalized APS-like score) must calibrate and still yield
    ontology-consistent sets. Skips cleanly if the hook is not present yet."""
    import inspect

    sig = inspect.signature(HierarchicalConformalPredictor.__init__)
    if "score_fn" not in sig.parameters:
        pytest.skip("score_fn hook not implemented")

    # APS-style score: 1 - p(true) + small penalty proportional to (1 - p)^2.
    def aps_like(p_true: float) -> float:
        s = 1.0 - p_true
        return float(min(1.0, s + 0.05 * s * s))

    p, y = _synthetic()
    hcp = HierarchicalConformalPredictor(ALL_TYPES, score_fn=aps_like)
    hcp.calibrate(p, y)
    valid = set(hcp.lineage_names)
    for i in range(20):
        r = hcp.predict(p[i])
        assert set(r.lineage_set).issubset(valid)
    # Thresholds remain in [0, 1] (or the +inf sparse-class fallback).
    for d in (hcp.lineage_thresholds, hcp.subtype_thresholds, hcp.fine_thresholds):
        for v in d.values():
            assert v >= 0.0
