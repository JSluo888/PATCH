"""scConform port — conformal inference for cell type prediction with cell ontology.

Reference paper:
    Wojnowska, K. et al. "Conformal inference for cell type prediction
    leveraging cell ontology." arXiv:2410.23786 (2024 / Bioconductor 2025).

Original method targets scRNA-seq with the OBO Foundry cell ontology. Three
core ideas, ported here for spatial proteomics:

    1. **Class-conditional split-CP** — per-leaf-class nonconformity threshold,
       finite-sample-corrected quantile (same as Mondrian CP).
    2. **Ontology-graph constraint (parent-closed sets)** — if a leaf c is
       included in the prediction set, all of c's ancestors are forced in too.
       This produces *connected* subgraphs in the ontology DAG, matching the
       biological semantics: "I might be a CD4 T cell or a Treg, but I'm
       definitely an immune cell."
    3. **Soft graph regularization** — score smoothing across ontology
       neighbors via a graph-Laplacian penalty (optional in the original paper;
       we omit by default and expose it via `laplacian_alpha`).

The original paper also addresses non-exchangeability via covariate weights;
we omit that here because the variants benchmark already runs Tibshirani
weighted CP separately. This port focuses on the ontology-respecting set
construction, which is scConform's distinctive contribution.

Differences vs PACE (this codebase):
    - scConform has NO panel-awareness. Identifiability of an internal node
      from the available markers is not modeled.
    - scConform's ontology constraint forces parent-closed sets at test time;
      PACE allows narrow leaf-only sets when the panel resolves them.
    - scConform smoothes scores along ontology edges; PACE uses crisp
      per-level thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

MIN_CAL_PER_CLASS = 10


def _quantile_threshold(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    if n < MIN_CAL_PER_CLASS:
        return float("inf")
    level = min(np.ceil((1.0 - alpha) * (n + 1)) / n, 1.0)
    return float(np.quantile(scores, level, method="higher"))


@dataclass
class scConformPort:
    """Port of arXiv:2410.23786 — class-conditional CP with parent-closed sets.

    Args:
        ontology_parents: dict mapping each leaf type → list of ancestors
            (leaf to root). Used to enforce parent-closed prediction sets.
        target_types:     ordered list of leaf class names (matches P columns).
        alpha:            miscoverage level (default 0.10).
        laplacian_alpha:  optional Laplacian-smoothing weight for scores. 0
            disables. Higher values average each class's threshold with its
            ontology neighbors. Default 0 (faithful to the simpler variant).
    """

    ontology_parents: Dict[str, List[str]]
    target_types: Sequence[str]
    alpha: float = 0.10
    laplacian_alpha: float = 0.0

    # Calibrated state
    thresholds: np.ndarray = field(default_factory=lambda: np.array([]))
    is_fitted: bool = False

    def _build_neighbor_map(self) -> Dict[int, List[int]]:
        """For each leaf class index, list neighbor class indices via shared
        immediate parent. Used for Laplacian smoothing."""
        type_to_idx = {t: i for i, t in enumerate(self.target_types)}
        immediate_parent: Dict[str, str] = {}
        for leaf, ancestors in self.ontology_parents.items():
            if len(ancestors) >= 2:
                immediate_parent[leaf] = ancestors[1]  # ancestors[0] is self

        neighbors: Dict[int, List[int]] = {}
        for leaf, parent in immediate_parent.items():
            if leaf not in type_to_idx:
                continue
            same_parent = [
                type_to_idx[other]
                for other, op in immediate_parent.items()
                if op == parent and other != leaf and other in type_to_idx
            ]
            neighbors[type_to_idx[leaf]] = same_parent
        return neighbors

    def fit(self, P_cal: np.ndarray, y_cal: np.ndarray) -> "scConformPort":
        K = len(self.target_types)
        # Class-conditional nonconformity scores
        s_cal = 1.0 - P_cal[np.arange(len(y_cal)), y_cal]
        thresholds = np.full(K, np.inf, dtype=np.float64)
        for c in range(K):
            mask = y_cal == c
            if mask.sum() < MIN_CAL_PER_CLASS:
                continue
            thresholds[c] = _quantile_threshold(s_cal[mask], self.alpha)

        # Optional Laplacian smoothing across ontology neighbors
        if self.laplacian_alpha > 0:
            neighbors = self._build_neighbor_map()
            smoothed = thresholds.copy()
            for c in range(K):
                nb = neighbors.get(c, [])
                if not nb:
                    continue
                neighbor_thresh = [
                    thresholds[n] for n in nb if np.isfinite(thresholds[n])
                ]
                if not neighbor_thresh:
                    continue
                smoothed[c] = (
                    (1.0 - self.laplacian_alpha) * thresholds[c]
                    + self.laplacian_alpha * np.mean(neighbor_thresh)
                )
            thresholds = smoothed

        self.thresholds = thresholds
        self.is_fitted = True
        return self

    def predict(self, P_test: np.ndarray) -> np.ndarray:
        """Return parent-closed prediction-set matrix at the LEAF level.

        At test time:
            1. Form initial set: S_init = {c : 1 - P_test[i, c] ≤ τ_c}.
            2. **Parent-closure**: for any leaf c in S_init, force all c's
               ancestors-as-leaves into S? — In our ontology, internal nodes
               (Immune, T_Lineage, etc.) are NOT leaves, so 'parent closure'
               operationally means: also include any other leaves that share
               an ancestor with cells already in S, which is interpreted as
               'climb to the lowest common ancestor'. We implement this by
               adding all leaves whose lowest common ancestor with a member
               is already a member-ancestor.

        For ontology-coverage scoring, the parent-closed set effectively
        marks the lowest covering ancestor in the tree, mirroring the
        original scConform connected-subgraph semantics.
        """
        if not self.is_fitted:
            raise RuntimeError("scConformPort: call fit() first.")

        scores_test = 1.0 - P_test
        sets_init = scores_test <= self.thresholds[None, :]
        n, K = sets_init.shape

        # Parent-closure: for each cell, find the lowest common ancestor of
        # all leaves currently in the set. Then add all leaves that share
        # that ancestor as one of their ancestors.
        sets_closed = sets_init.copy()
        type_to_idx = {t: i for i, t in enumerate(self.target_types)}
        for i in range(n):
            members = np.where(sets_init[i])[0]
            if len(members) == 0:
                # Backstop: include the top-1 leaf
                top = int(np.argmax(P_test[i]))
                sets_closed[i, top] = True
                continue
            # Compute LCA (lowest common ancestor): intersect ancestor lists
            anc_lists = [
                self.ontology_parents.get(self.target_types[c], [self.target_types[c]])
                for c in members
            ]
            common: set = set(anc_lists[0])
            for al in anc_lists[1:]:
                common &= set(al)
            # Pick the deepest (= earliest in any leaf-to-root list) common ancestor
            lca = None
            for node in anc_lists[0]:
                if node in common:
                    lca = node
                    break
            if lca is None:
                lca = anc_lists[0][-1]  # root fallback
            # Add all leaves whose ancestor list includes lca
            for leaf, anc in self.ontology_parents.items():
                if lca in anc and leaf in type_to_idx:
                    sets_closed[i, type_to_idx[leaf]] = True
        return sets_closed


def fit_predict(
    P_cal: np.ndarray,
    y_cal: np.ndarray,
    P_test: np.ndarray,
    target_types: Sequence[str],
    ontology_parents: Dict[str, List[str]],
    alpha: float = 0.10,
    laplacian_alpha: float = 0.0,
) -> Dict[str, object]:
    """Convenience wrapper matching the variants-benchmark interface."""
    sccp = scConformPort(
        ontology_parents=ontology_parents,
        target_types=target_types,
        alpha=alpha,
        laplacian_alpha=laplacian_alpha,
    ).fit(P_cal, y_cal)
    sets = sccp.predict(P_test)
    return {
        "sets": sets,
        "avg_set_size": float(sets.sum(axis=1).mean()),
        "leaf_frac": float((sets.sum(axis=1) == 1).mean()),
    }
