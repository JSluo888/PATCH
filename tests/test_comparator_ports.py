"""Tests for HCC and scConform comparator ports.

Run with:  pytest tests/test_comparator_ports.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from uncertainty.comparators import HCCPort, scConformPort
from uncertainty.comparators.hcc_port import _quantile_threshold


# Tiny synthetic ontology for tests.
# Hierarchy:
#   root
#   ├── Immune
#   │   ├── T_Lineage
#   │   │   ├── CD4_T
#   │   │   └── CD8_T
#   │   └── B_Cell
#   └── Stromal
TOY_LEAVES = ["CD4_T", "CD8_T", "B_Cell", "Stromal"]
TOY_ANCESTORS = {
    "CD4_T":   ["CD4_T", "T_Lineage", "Immune", "root"],
    "CD8_T":   ["CD8_T", "T_Lineage", "Immune", "root"],
    "B_Cell":  ["B_Cell", "Immune", "root"],
    "Stromal": ["Stromal", "root"],
}


@pytest.fixture
def synthetic_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate well-separated per-class probabilities for cal + test."""
    rng = np.random.default_rng(42)
    K = len(TOY_LEAVES)
    n_per_class = 100

    def gen_block(true_class: int) -> tuple[np.ndarray, np.ndarray]:
        # Probs centered on the true class with noise
        P = rng.dirichlet(np.ones(K) * 0.3, size=n_per_class)
        # Boost true class
        P[:, true_class] += 1.5
        P = P / P.sum(axis=1, keepdims=True)
        y = np.full(n_per_class, true_class, dtype=np.int64)
        return P, y

    cal_blocks = [gen_block(c) for c in range(K)]
    P_cal = np.vstack([b[0] for b in cal_blocks]).astype(np.float32)
    y_cal = np.concatenate([b[1] for b in cal_blocks])

    test_blocks = [gen_block(c) for c in range(K)]
    P_test = np.vstack([b[0] for b in test_blocks]).astype(np.float32)
    y_test = np.concatenate([b[1] for b in test_blocks])
    return P_cal, y_cal, P_test, y_test


def test_quantile_threshold_finite_sample_correction():
    # n=100, alpha=0.10 -> level = ceil(101*0.9)/100 = 0.91
    # scores = linspace(0.1, 0.9, 100) -> the 91st-percentile is at index 90 ≈ 0.827
    # method='higher' rounds up to 0.836
    scores = np.linspace(0.1, 0.9, 100)
    tau = _quantile_threshold(scores, alpha=0.10)
    assert 0.82 <= tau <= 0.92, f"expected within finite-sample-corrected bounds, got {tau}"


def test_quantile_threshold_too_small_returns_inf():
    tau = _quantile_threshold(np.array([0.1, 0.2]), alpha=0.10)
    assert tau == float("inf")


def test_hcc_port_basic_fit_predict(synthetic_data):
    P_cal, y_cal, P_test, y_test = synthetic_data
    hcc = HCCPort(
        ontology_parents=TOY_ANCESTORS,
        target_types=TOY_LEAVES,
        alpha=0.10,
    ).fit(P_cal, y_cal)

    assert hcc.is_fitted
    assert "Immune" in hcc.nodes
    assert "root" in hcc.nodes
    assert "T_Lineage" in hcc.nodes

    sets = hcc.predict(P_test)
    assert sets.shape == P_test.shape
    assert sets.dtype == bool
    # Marginal coverage in well-separated case should be near-perfect
    coverage = sets[np.arange(len(y_test)), y_test].mean()
    assert coverage >= 0.85, f"HCC coverage {coverage:.3f} < 0.85"


def test_hcc_port_subtree_descendants_correct():
    """Verify the descendant map matches the ontology structure."""
    hcc = HCCPort(
        ontology_parents=TOY_ANCESTORS,
        target_types=TOY_LEAVES,
        alpha=0.10,
    )
    hcc._enumerate_nodes()

    type_to_idx = {t: i for i, t in enumerate(TOY_LEAVES)}
    # Immune subtree should include CD4, CD8, B_Cell but NOT Stromal
    immune_desc = hcc.node_to_descendants["Immune"]
    assert type_to_idx["CD4_T"] in immune_desc
    assert type_to_idx["CD8_T"] in immune_desc
    assert type_to_idx["B_Cell"] in immune_desc
    assert type_to_idx["Stromal"] not in immune_desc
    # T_Lineage subtree: only CD4, CD8
    t_desc = hcc.node_to_descendants["T_Lineage"]
    assert t_desc == {type_to_idx["CD4_T"], type_to_idx["CD8_T"]}


def test_scconform_port_basic_fit_predict(synthetic_data):
    P_cal, y_cal, P_test, y_test = synthetic_data
    sccp = scConformPort(
        ontology_parents=TOY_ANCESTORS,
        target_types=TOY_LEAVES,
        alpha=0.10,
    ).fit(P_cal, y_cal)

    assert sccp.is_fitted
    assert sccp.thresholds.shape == (len(TOY_LEAVES),)

    sets = sccp.predict(P_test)
    assert sets.shape == P_test.shape
    assert sets.dtype == bool
    # Every set must be non-empty (parent-closure backstop)
    assert (sets.sum(axis=1) >= 1).all()


def test_scconform_parent_closed_property():
    """When the initial set contains multiple leaves, parent-closure should
    expand to all sibling leaves under their lowest common ancestor.

    Construct a hand-tuned case so we can deterministically test the closure
    semantics independent of fitted thresholds.
    """
    sccp = scConformPort(
        ontology_parents=TOY_ANCESTORS,
        target_types=TOY_LEAVES,
        alpha=0.10,
    )
    sccp.is_fitted = True
    # Threshold tuning: CD4 + CD8 enter initial set (score ≤ 0.6),
    # B_Cell stays out (score 0.95 > 0.5), Stromal stays out (0.95 > 0.1).
    sccp.thresholds = np.array([0.6, 0.6, 0.5, 0.1])

    P = np.array([[0.45, 0.45, 0.05, 0.05]], dtype=np.float32)
    sets = sccp.predict(P)
    # Initial set = {CD4, CD8} → LCA = T_Lineage → closure adds T_Lineage's
    # leaves (which is exactly {CD4, CD8}). No siblings beyond T_Lineage.
    assert sets[0, 0], "CD4_T should be in set"
    assert sets[0, 1], "CD8_T should be in set"
    assert not sets[0, 2], "B_Cell should not be added by T-only closure"
    assert not sets[0, 3], "Stromal should not be in set"


def test_scconform_laplacian_smoothing_changes_thresholds(synthetic_data):
    P_cal, y_cal, _, _ = synthetic_data
    sccp_no_smooth = scConformPort(
        ontology_parents=TOY_ANCESTORS, target_types=TOY_LEAVES, alpha=0.10,
        laplacian_alpha=0.0,
    ).fit(P_cal, y_cal)
    sccp_smooth = scConformPort(
        ontology_parents=TOY_ANCESTORS, target_types=TOY_LEAVES, alpha=0.10,
        laplacian_alpha=0.5,
    ).fit(P_cal, y_cal)

    # CD4_T and CD8_T are neighbors via T_Lineage parent — smoothing should
    # pull their thresholds toward each other.
    cd4 = TOY_LEAVES.index("CD4_T")
    cd8 = TOY_LEAVES.index("CD8_T")
    diff_no = abs(sccp_no_smooth.thresholds[cd4] - sccp_no_smooth.thresholds[cd8])
    diff_yes = abs(sccp_smooth.thresholds[cd4] - sccp_smooth.thresholds[cd8])
    assert diff_yes <= diff_no + 1e-9, "Laplacian smoothing should not increase neighbor distance"


def test_both_ports_provide_marginal_coverage(synthetic_data):
    """Both HCC and scConform ports should hit ≥ 1-α on synthetic data
    (modulo finite-sample noise in our small fixture)."""
    P_cal, y_cal, P_test, y_test = synthetic_data

    hcc = HCCPort(TOY_ANCESTORS, TOY_LEAVES, alpha=0.20).fit(P_cal, y_cal)
    sccp = scConformPort(TOY_ANCESTORS, TOY_LEAVES, alpha=0.20).fit(P_cal, y_cal)

    sets_hcc = hcc.predict(P_test)
    sets_sccp = sccp.predict(P_test)

    cov_hcc = sets_hcc[np.arange(len(y_test)), y_test].mean()
    cov_sccp = sets_sccp[np.arange(len(y_test)), y_test].mean()
    assert cov_hcc >= 0.75, f"HCC coverage {cov_hcc:.3f} too low"
    assert cov_sccp >= 0.75, f"scConform coverage {cov_sccp:.3f} too low"
