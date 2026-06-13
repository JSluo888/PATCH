"""HCC port — Hierarchical Conformal Classification (Principato et al., 2025).

Reference paper:
    Principato, F. et al. "Hierarchical Conformal Classification."
    arXiv:2508.13288 (August 2025).

The original paper formulates hierarchical CP as a constrained optimization:
prediction sets at internal nodes with a coverage guarantee, ascending the
hierarchy when leaves cannot be confidently distinguished. Importantly, HCC
does **not** model panel-conditional identifiability — it operates on a fixed
ontology and assumes all leaves are observable from the input features.

This port is a faithful approximation of the HCC algorithm at split-CP scale:

    1. For each internal node v in the ontology, compute per-cell aggregated
       probability p̂(v|x) = Σ_{leaf c ∈ subtree(v)} p(c|x).
    2. Calibration: per-node nonconformity score s_v(x) = 1 - p̂(v|x)
       computed only over calibration cells whose true class is a descendant
       of v. Per-node threshold τ_v = Quantile_{1-α}({s_v(x_i)}_{i: y_i ∈ subtree(v)})
       with finite-sample correction ⌈(n_v+1)(1-α)⌉/n_v.
    3. Test-time ascent: starting from the leaf with highest p̂, climb to the
       lowest ancestor v* such that 1 - p̂(v*|x) ≤ τ_{v*}. Return v*.

The constrained-rep-complexity (set-size budget) of HCC reduces, in the
single-output ascent setting, to "smallest covering ancestor" — which is what
this port computes. Multi-output / set-valued HCC is a strict generalization
not covered here; we benchmark the canonical single-ascent variant.

Differences vs PACE (this codebase):
    - HCC has NO panel filter. If a panel lacks CD68, HCC may still place
      a Myeloid leaf in the prediction set despite no marker support.
    - HCC's coverage guarantee is per-internal-node and assumes exchangeability.
      PACE additionally provides per-panel adaptive resolution.
    - HCC ascends to the smallest covering ancestor; PACE climbs only when
      the panel cannot resolve below.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

import numpy as np

# Finite-sample-corrected quantile: ⌈(n+1)(1-α)⌉/n
MIN_CAL_PER_NODE = 10


def _quantile_threshold(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample-corrected (1-α) quantile.

    Returns +inf when calibration set is too small.
    """
    n = len(scores)
    if n < MIN_CAL_PER_NODE:
        return float("inf")
    level = min(np.ceil((1.0 - alpha) * (n + 1)) / n, 1.0)
    return float(np.quantile(scores, level, method="higher"))


@dataclass
class HCCPort:
    """Port of arXiv:2508.13288 hierarchical conformal classification.

    Args:
        ontology_parents: dict mapping each leaf type → its lineage of ancestors,
            ordered from leaf to root. e.g.
                {'Helper_T_Cell': ['Helper_T_Cell','T_Lineage','Immune','root'], ...}
        target_types:     ordered list of leaf class names (matches columns of P).
        alpha:            miscoverage level (default 0.10 → 90% target).
    """

    ontology_parents: Dict[str, List[str]]
    target_types: Sequence[str]
    alpha: float = 0.10

    # State after fit()
    nodes: List[str] = field(default_factory=list)
    node_to_descendants: Dict[str, Set[int]] = field(default_factory=dict)
    thresholds: Dict[str, float] = field(default_factory=dict)
    is_fitted: bool = False

    def _enumerate_nodes(self) -> None:
        """Collect all unique nodes (leaves + internal) from the ontology."""
        node_set: Set[str] = set()
        for ancestors in self.ontology_parents.values():
            node_set.update(ancestors)
        self.nodes = sorted(node_set)

        type_to_idx = {t: i for i, t in enumerate(self.target_types)}
        for node in self.nodes:
            descendants = {
                type_to_idx[leaf]
                for leaf, ancestors in self.ontology_parents.items()
                if node in ancestors and leaf in type_to_idx
            }
            self.node_to_descendants[node] = descendants

    def fit(self, P_cal: np.ndarray, y_cal: np.ndarray) -> "HCCPort":
        """Calibrate per-node thresholds.

        Args:
            P_cal: (N, K) classifier probabilities, rows sum to 1.
            y_cal: (N,) integer class labels in [0, K).
        """
        if not self.nodes:
            self._enumerate_nodes()

        for node in self.nodes:
            descendants = self.node_to_descendants[node]
            if not descendants:
                self.thresholds[node] = float("inf")
                continue
            # Aggregate probability of node v: sum of leaf probs in subtree(v)
            mass_v = P_cal[:, list(descendants)].sum(axis=1)
            # Cells whose true class is a descendant of v
            in_subtree = np.isin(y_cal, list(descendants))
            scores = 1.0 - mass_v[in_subtree]
            self.thresholds[node] = _quantile_threshold(scores, self.alpha)

        self.is_fitted = True
        return self

    def predict(self, P_test: np.ndarray) -> np.ndarray:
        """Return boolean prediction-set matrix at the LEAF level.

        For HCC's single-output ascent: climb to the smallest covering ancestor;
        the returned set marks all leaves under that ancestor as 'in set' for
        the purposes of ontology-coverage scoring.

        Args:
            P_test: (n, K) classifier probabilities.

        Returns:
            sets: (n, K) bool, where sets[i, c] is True iff leaf c is under the
                  ancestor that HCC ascends to for cell i.
        """
        if not self.is_fitted:
            raise RuntimeError("HCCPort: call fit() first.")

        n, K = P_test.shape
        sets = np.zeros((n, K), dtype=bool)
        type_to_idx = {t: i for i, t in enumerate(self.target_types)}

        for i in range(n):
            top_leaf_idx = int(np.argmax(P_test[i]))
            top_leaf_name = self.target_types[top_leaf_idx]
            ancestors = self.ontology_parents.get(top_leaf_name, [top_leaf_name])
            chosen_node = ancestors[-1]  # default to root if no ancestor passes
            for v in ancestors:
                desc = self.node_to_descendants.get(v, set())
                if not desc:
                    continue
                mass_v = P_test[i, list(desc)].sum()
                if (1.0 - mass_v) <= self.thresholds.get(v, float("inf")):
                    chosen_node = v
                    break
            # Mark all leaves in subtree(chosen_node) as in-set
            for leaf_idx in self.node_to_descendants.get(chosen_node, set()):
                sets[i, leaf_idx] = True
        return sets


def fit_predict(
    P_cal: np.ndarray,
    y_cal: np.ndarray,
    P_test: np.ndarray,
    target_types: Sequence[str],
    ontology_parents: Dict[str, List[str]],
    alpha: float = 0.10,
) -> Dict[str, object]:
    """Convenience wrapper matching the variants-benchmark interface.

    Returns:
        dict with keys: coverage, avg_set_size, leaf_frac, sets (n, K) bool
    """
    hcc = HCCPort(
        ontology_parents=ontology_parents, target_types=target_types, alpha=alpha
    ).fit(P_cal, y_cal)
    sets = hcc.predict(P_test)
    return {
        "sets": sets,
        "avg_set_size": float(sets.sum(axis=1).mean()),
        "leaf_frac": float((sets.sum(axis=1) == 1).mean()),
    }
