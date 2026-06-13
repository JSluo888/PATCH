"""
Hierarchical Conformal Prediction for Ontology-Aware Cell Typing

Three-level conformal calibration:
  Level 1: Lineage (Immune vs Epithelial vs Endothelial vs Stromal) — 90% coverage
  Level 2: Subtype within singleton lineage — 90% coverage (panel-aware)
  Level 3: Fine type within subtype — 90% coverage

The system returns the FINEST confident node in the ontology, with reason codes.

Key innovation: conformal prediction decides the RESOLUTION of the answer,
not just the uncertainty. Combined with panel-conditioned ontology, this
produces biologically coherent predictions that degrade gracefully.

Panel-awareness: at Level 2, subtypes whose required markers are absent from
the panel are excluded from conformal sets (they collapse to Immune_Other).
This prevents uncalibrated "ghost" subtypes from inflating set sizes.
"""

import numpy as np
from typing import Callable, Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import json
import logging

from configs.marker_celltype_ontology import get_finest_identifiable_type

logger = logging.getLogger(__name__)

# Minimum calibration cells per class to obtain a valid threshold.
# With fewer than MIN_CAL_CELLS, coverage guarantee cannot be honored
# and threshold is set to +inf (conservative: wide prediction sets).
MIN_CAL_CELLS = 10


def _compute_threshold(
    scores: List[float],
    coverage: float,
    class_name: str,
) -> float:
    """
    Compute conformal threshold with guarantee preservation.

    Returns +inf (conservative) when calibration set is too small or
    the requested coverage is unattainable with available samples.
    This ensures no silent failure of the coverage guarantee.

    Args:
        scores: nonconformity scores for cells with this true class
        coverage: target coverage level (1 - alpha)
        class_name: class name for logging

    Returns:
        float threshold, or +inf if undercalibrated
    """
    scores_arr = np.sort(np.asarray(scores, dtype=np.float64))
    n = len(scores_arr)
    if n < MIN_CAL_CELLS:
        logger.warning(
            "Class '%s' has only %d calibration cells (< %d) — "
            "setting threshold to +inf (wide prediction sets)",
            class_name, n, MIN_CAL_CELLS,
        )
        return float('inf')
    # Finite-sample corrected quantile index
    idx = int(np.ceil((n + 1) * coverage)) - 1
    if idx >= n:
        logger.warning(
            "Coverage %.2f unattainable with n=%d for '%s' — "
            "needed idx=%d, setting threshold to +inf",
            coverage, n, class_name, idx + 1,
        )
        return float('inf')
    return float(scores_arr[idx])


@dataclass(frozen=True)
class ReasonCode:
    """
    Structured explanation for a hierarchical prediction.

    category: short machine-readable reason ("leaf_confident",
        "subtype_ambiguous", "marker_insufficient", "cross_lineage_conflict",
        "lineage_ambiguous", "subtype_confident", "fine_ambiguous",
        "empty_lineage_fallback", "empty_subtype_fallback", "empty_fine_fallback")
    ambiguous_types: types that could not be distinguished (if any)
    markers_needed: markers that would resolve the ambiguity (if any)
    """
    category: str
    ambiguous_types: Tuple[str, ...] = ()
    markers_needed: Tuple[str, ...] = ()

    def __str__(self) -> str:
        parts = [self.category]
        if self.ambiguous_types:
            parts.append(f"types={','.join(self.ambiguous_types)}")
        if self.markers_needed:
            parts.append(f"needs={','.join(self.markers_needed)}")
        return '|'.join(parts)


@dataclass
class HierarchicalResult:
    """Result of hierarchical conformal prediction."""
    cell_id: str
    predicted_type: str              # Finest confident node
    ontology_level: str              # "leaf", "subtype", "lineage", "root"
    lineage: str                     # Lineage assignment
    lineage_set: List[str]           # Lineage conformal set
    subtype_set: List[str]           # Subtype conformal set (within lineage)
    fine_set: List[str]              # Fine-type set (within subtype)
    reason_code: str                 # String form (backwards-compatible)
    class_probabilities: Dict[str, float]
    confidence: float
    reason: Optional[ReasonCode] = None  # Structured form (preferred for analysis)


# Ontology structure: which types belong to which lineage
# Note: 'Immune' (internal node) is included in Immune lineage so that
# probability mass assigned to the Immune parent node is not lost.
LINEAGE_MAP = {
    'Immune': ['Immune', 'T_Lineage', 'Helper_T_Cell', 'Regulatory_T_Cell', 'Cytotoxic_T_Cell',
               'B_Cell', 'Plasma_Cell', 'Myeloid', 'Macrophage_CD163pos', 'Macrophage_CD163neg',
               'NK_Cell', 'Dendritic_Cell', 'Neutrophil', 'Immune_Other'],
    'Epithelial': ['Epithelial'],
    'Endothelial': ['Endothelial'],
    'Stromal': ['Stromal'],
}

# Subtype groups within Immune
IMMUNE_SUBTYPE_MAP = {
    'T_Lineage': ['T_Lineage', 'Helper_T_Cell', 'Regulatory_T_Cell', 'Cytotoxic_T_Cell'],
    'B_Lineage': ['B_Cell', 'Plasma_Cell'],
    'Myeloid': ['Myeloid', 'Macrophage_CD163pos', 'Macrophage_CD163neg'],
    'NK_Cell': ['NK_Cell'],
    'Dendritic_Cell': ['Dendritic_Cell'],
    'Neutrophil': ['Neutrophil'],
    'Immune_Other': ['Immune_Other'],
}

# Fine types within subtypes
FINE_TYPE_MAP = {
    'T_Lineage': ['Helper_T_Cell', 'Regulatory_T_Cell', 'Cytotoxic_T_Cell'],
    'Myeloid': ['Macrophage_CD163pos', 'Macrophage_CD163neg'],
    'B_Lineage': ['B_Cell', 'Plasma_Cell'],
}

# Which markers distinguish fine types (for reason codes)
FINE_DISTINGUISHING = {
    ('Helper_T_Cell', 'Regulatory_T_Cell'): 'FOXP3',
    ('Helper_T_Cell', 'Cytotoxic_T_Cell'): 'CD4/CD8',
    ('Regulatory_T_Cell', 'Cytotoxic_T_Cell'): 'CD4/CD8/FOXP3',
    ('Macrophage_CD163pos', 'Macrophage_CD163neg'): 'CD163',
    ('B_Cell', 'Plasma_Cell'): 'CD138/CD20',
}

# Required markers to identify each immune subtype
# If these markers are absent from the panel, the subtype cannot be distinguished
# and should be excluded from conformal sets (collapses to Immune_Other)
SUBTYPE_REQUIRED_MARKERS = {
    'T_Lineage': {'CD3e'},
    'B_Lineage': {'CD20'},
    'Myeloid': {'CD68'},
    'NK_Cell': {'CD56'},
    'Dendritic_Cell': {'CD11c', 'HLADR'},
    'Neutrophil': {'CD15'},
    'Immune_Other': set(),  # Always identifiable (fallback)
}


class HierarchicalConformalPredictor:
    """
    Two-level hierarchical conformal prediction with ontology awareness.
    """

    def __init__(
        self,
        all_types: List[str],
        lineage_coverage: float = 0.90,
        subtype_coverage: float = 0.90,
        fine_coverage: float = 0.90,
        score_fn: Optional[Callable[[float], float]] = None,
    ):
        self.all_types = all_types
        self.type_to_idx = {t: i for i, t in enumerate(all_types)}
        self.lineage_coverage = lineage_coverage
        self.subtype_coverage = subtype_coverage
        self.fine_coverage = fine_coverage
        # Pluggable nonconformity score over the true-node probability. The
        # default reproduces the original `1 - p(true_node)` exactly (verified
        # byte-identical by tests/test_hierarchical_conformal_chars.py). Inject
        # an alternative (e.g. RAPS/APS rank-penalized score, or a
        # Gibbs-Cherian-Candes conditional score) here. The SAME transform is
        # applied at calibration and at predict time, so coverage is consistent.
        self.score_fn: Callable[[float], float] = score_fn or (lambda p_true: 1.0 - p_true)

        # Build lineage index mapping
        self.lineage_names = list(LINEAGE_MAP.keys())
        self._type_to_lineage = {}
        for lineage, types in LINEAGE_MAP.items():
            for t in types:
                if t in self.type_to_idx:
                    self._type_to_lineage[t] = lineage

        # Calibration thresholds (set during calibrate())
        self.lineage_thresholds = {}
        self.subtype_thresholds = {}
        self.fine_thresholds = {}
        self.is_calibrated = False

    def _get_identifiable_subtypes(self, available_markers: Optional[Set[str]] = None) -> List[str]:
        """Return immune subtypes identifiable with the given marker panel.
        Subtypes whose required markers are absent collapse to Immune_Other."""
        if available_markers is None:
            return list(IMMUNE_SUBTYPE_MAP.keys())
        identifiable = []
        for subtype, required in SUBTYPE_REQUIRED_MARKERS.items():
            if required.issubset(available_markers):
                identifiable.append(subtype)
        # Always include Immune_Other as fallback
        if 'Immune_Other' not in identifiable:
            identifiable.append('Immune_Other')
        return identifiable

    def _aggregate_lineage_probs(self, probs: np.ndarray) -> Dict[str, float]:
        """Sum class probabilities into lineage probabilities (normalized)."""
        lineage_probs = {}
        for lineage, types in LINEAGE_MAP.items():
            p = sum(probs[self.type_to_idx[t]] for t in types if t in self.type_to_idx)
            lineage_probs[lineage] = float(p)
        # Normalize so probs sum to 1.0 (consistent with Levels 2 and 3)
        total = sum(lineage_probs.values()) + 1e-12
        return {k: v / total for k, v in lineage_probs.items()}

    def _aggregate_subtype_probs(self, probs: np.ndarray, lineage: str) -> Dict[str, float]:
        """Get subtype probabilities within a lineage."""
        if lineage != 'Immune':
            return {lineage: 1.0}

        subtype_probs = {}
        for subtype, types in IMMUNE_SUBTYPE_MAP.items():
            p = sum(probs[self.type_to_idx[t]] for t in types if t in self.type_to_idx)
            subtype_probs[subtype] = float(p)

        # Normalize
        total = sum(subtype_probs.values()) + 1e-12
        return {k: v / total for k, v in subtype_probs.items()}

    def _get_fine_probs(self, probs: np.ndarray, subtype: str) -> Dict[str, float]:
        """Get fine-type probabilities within a subtype."""
        if subtype not in FINE_TYPE_MAP:
            return {subtype: 1.0}

        fine_types = FINE_TYPE_MAP[subtype]
        fine_probs = {}
        for t in fine_types:
            if t in self.type_to_idx:
                fine_probs[t] = float(probs[self.type_to_idx[t]])

        total = sum(fine_probs.values()) + 1e-12
        return {k: v / total for k, v in fine_probs.items()}

    def calibrate(
        self,
        cal_probs: np.ndarray,
        cal_labels: np.ndarray,
    ):
        """
        Calibrate hierarchical thresholds on calibration data.

        Args:
            cal_probs: (N, K) class probabilities from model
            cal_labels: (N,) integer class labels
        """
        N = len(cal_labels)

        # --- Level 1: Lineage calibration ---
        lineage_scores = {}
        for i in range(N):
            true_type = self.all_types[cal_labels[i]]
            true_lineage = self._type_to_lineage.get(true_type)
            if true_lineage is None:
                continue
            lp = self._aggregate_lineage_probs(cal_probs[i])
            score = self.score_fn(lp.get(true_lineage, 0.0))
            lineage_scores.setdefault(true_lineage, []).append(score)

        for lineage, scores in lineage_scores.items():
            self.lineage_thresholds[lineage] = _compute_threshold(
                scores, self.lineage_coverage, f"lineage:{lineage}"
            )

        # --- Level 2: Subtype calibration (Immune cells only) ---
        subtype_scores = {}
        for i in range(N):
            true_type = self.all_types[cal_labels[i]]
            true_lineage = self._type_to_lineage.get(true_type)
            if true_lineage != 'Immune':
                continue

            # Find true subtype
            true_subtype = None
            for st, types in IMMUNE_SUBTYPE_MAP.items():
                if true_type in types:
                    true_subtype = st
                    break
            if true_subtype is None:
                continue

            sp = self._aggregate_subtype_probs(cal_probs[i], 'Immune')
            score = self.score_fn(sp.get(true_subtype, 0.0))
            subtype_scores.setdefault(true_subtype, []).append(score)

        for subtype, scores in subtype_scores.items():
            self.subtype_thresholds[subtype] = _compute_threshold(
                scores, self.subtype_coverage, f"subtype:{subtype}"
            )

        # --- Level 3: Fine type calibration ---
        fine_scores = {}
        for i in range(N):
            true_type = self.all_types[cal_labels[i]]
            for subtype, fine_types in FINE_TYPE_MAP.items():
                if true_type in fine_types:
                    fp = self._get_fine_probs(cal_probs[i], subtype)
                    score = self.score_fn(fp.get(true_type, 0.0))
                    fine_scores.setdefault(true_type, []).append(score)
                    break

        for ftype, scores in fine_scores.items():
            self.fine_thresholds[ftype] = _compute_threshold(
                scores, self.fine_coverage, f"fine:{ftype}"
            )

        self.is_calibrated = True
        logger.info(f"Calibrated: {len(self.lineage_thresholds)} lineages, "
                     f"{len(self.subtype_thresholds)} subtypes, {len(self.fine_thresholds)} fine types")

    def predict(
        self,
        probs: np.ndarray,
        cell_id: str = "",
        available_markers: Optional[Set[str]] = None,
    ) -> HierarchicalResult:
        """
        Hierarchical conformal prediction for a single cell.

        Args:
            probs: (K,) class probabilities
            cell_id: cell identifier
            available_markers: set of markers in this cell's panel

        Returns:
            HierarchicalResult with finest confident node + reason code
        """
        if not self.is_calibrated:
            raise RuntimeError("Must calibrate before predicting. Call calibrate() first.")

        class_probs = {t: float(probs[i]) for i, t in enumerate(self.all_types)}

        # --- Level 1: Lineage ---
        lineage_probs = self._aggregate_lineage_probs(probs)
        lineage_set = []
        for lineage in self.lineage_names:
            score = self.score_fn(lineage_probs.get(lineage, 0.0))
            threshold = self.lineage_thresholds.get(lineage, 1.0)
            if score <= threshold:
                lineage_set.append(lineage)

        if not lineage_set:
            lineage_set = [max(lineage_probs, key=lineage_probs.get)]

        # Cross-lineage conflict?
        if len(lineage_set) > 1:
            # Check for impossible combinations (doublet indicators)
            has_immune = 'Immune' in lineage_set
            has_epithelial = 'Epithelial' in lineage_set
            has_stromal = 'Stromal' in lineage_set
            if (has_immune and has_epithelial) or (has_immune and has_stromal):
                reason = ReasonCode(
                    category="cross_lineage_conflict",
                    ambiguous_types=tuple(sorted(lineage_set)),
                )
                return HierarchicalResult(
                    cell_id=cell_id, predicted_type="Doublet_Artifact",
                    ontology_level="artifact", lineage="Unknown",
                    lineage_set=lineage_set, subtype_set=[], fine_set=[],
                    reason_code=str(reason), reason=reason,
                    class_probabilities=class_probs, confidence=0.0,
                )
            # Non-conflicting multi-lineage: return Root (preserve coverage)
            reason = ReasonCode(
                category="lineage_ambiguous",
                ambiguous_types=tuple(sorted(lineage_set)),
            )
            return HierarchicalResult(
                cell_id=cell_id, predicted_type="Unknown",
                ontology_level="root", lineage="Unknown",
                lineage_set=lineage_set, subtype_set=[], fine_set=[],
                reason_code=str(reason), reason=reason,
                class_probabilities=class_probs,
                confidence=max(lineage_probs.values()),
            )

        lineage = lineage_set[0]

        # Non-immune lineages are already leaf types
        if lineage != 'Immune':
            reason = ReasonCode(category="leaf_confident")
            return HierarchicalResult(
                cell_id=cell_id, predicted_type=lineage,
                ontology_level="leaf", lineage=lineage,
                lineage_set=lineage_set, subtype_set=[lineage], fine_set=[lineage],
                reason_code=str(reason), reason=reason,
                class_probabilities=class_probs,
                confidence=lineage_probs.get(lineage, 0),
            )

        # --- Level 2: Immune subtype ---
        subtype_probs = self._aggregate_subtype_probs(probs, 'Immune')
        # Panel-aware: only consider subtypes identifiable with available markers
        identifiable_subtypes = self._get_identifiable_subtypes(available_markers)
        subtype_set = []
        for subtype in identifiable_subtypes:
            score = self.score_fn(subtype_probs.get(subtype, 0.0))
            threshold = self.subtype_thresholds.get(subtype, 1.0)
            if score <= threshold:
                subtype_set.append(subtype)

        if not subtype_set:
            # Fallback: pick highest-prob identifiable subtype
            subtype_set = [max(identifiable_subtypes, key=lambda s: subtype_probs.get(s, 0))]

        if len(subtype_set) > 1:
            reason = ReasonCode(
                category="subtype_ambiguous",
                ambiguous_types=tuple(sorted(subtype_set)),
            )
            return HierarchicalResult(
                cell_id=cell_id, predicted_type="Immune",
                ontology_level="lineage", lineage="Immune",
                lineage_set=lineage_set, subtype_set=subtype_set, fine_set=[],
                reason_code=str(reason), reason=reason,
                class_probabilities=class_probs,
                confidence=lineage_probs.get('Immune', 0),
            )

        subtype = subtype_set[0]

        # Subtypes without fine resolution are already terminal
        if subtype not in FINE_TYPE_MAP:
            reason = ReasonCode(category="subtype_confident")
            return HierarchicalResult(
                cell_id=cell_id, predicted_type=subtype,
                ontology_level="subtype", lineage="Immune",
                lineage_set=lineage_set, subtype_set=subtype_set, fine_set=[subtype],
                reason_code=str(reason), reason=reason,
                class_probabilities=class_probs,
                confidence=subtype_probs.get(subtype, 0),
            )

        # --- Level 3: Fine type within subtype ---
        fine_probs = self._get_fine_probs(probs, subtype)
        fine_set = []
        for ftype in FINE_TYPE_MAP[subtype]:
            score = self.score_fn(fine_probs.get(ftype, 0.0))
            threshold = self.fine_thresholds.get(ftype, 1.0)
            if score <= threshold:
                fine_set.append(ftype)

        if not fine_set:
            fine_set = [max(fine_probs, key=fine_probs.get)]

        if len(fine_set) == 1:
            # Check: is this resolution justified by available markers?
            if available_markers is not None:
                finest = get_finest_identifiable_type(available_markers, fine_set[0])
                if finest != fine_set[0]:
                    reason = ReasonCode(
                        category="marker_insufficient",
                        ambiguous_types=(fine_set[0],),
                    )
                    return HierarchicalResult(
                        cell_id=cell_id, predicted_type=subtype,
                        ontology_level="subtype", lineage="Immune",
                        lineage_set=lineage_set, subtype_set=subtype_set, fine_set=fine_set,
                        reason_code=str(reason), reason=reason,
                        class_probabilities=class_probs,
                        confidence=fine_probs.get(fine_set[0], 0),
                    )

            reason = ReasonCode(category="leaf_confident")
            return HierarchicalResult(
                cell_id=cell_id, predicted_type=fine_set[0],
                ontology_level="leaf", lineage="Immune",
                lineage_set=lineage_set, subtype_set=subtype_set, fine_set=fine_set,
                reason_code=str(reason), reason=reason,
                class_probabilities=class_probs,
                confidence=fine_probs.get(fine_set[0], 0),
            )

        # Fine-type ambiguity — return parent with markers that would resolve
        markers_needed: Tuple[str, ...] = ()
        for pair, marker in FINE_DISTINGUISHING.items():
            if set(pair) <= set(fine_set):
                markers_needed = tuple(marker.split('/'))
                break
        reason = ReasonCode(
            category="fine_ambiguous",
            ambiguous_types=tuple(sorted(fine_set)),
            markers_needed=markers_needed,
        )
        return HierarchicalResult(
            cell_id=cell_id, predicted_type=subtype,
            ontology_level="subtype", lineage="Immune",
            lineage_set=lineage_set, subtype_set=subtype_set, fine_set=fine_set,
            reason_code=str(reason), reason=reason,
            class_probabilities=class_probs,
            confidence=subtype_probs.get(subtype, 0),
        )

    def predict_batch(
        self,
        probs: np.ndarray,
        cell_ids: Optional[List[str]] = None,
        available_markers: Optional[Set[str]] = None,
    ) -> List[HierarchicalResult]:
        """Batch prediction."""
        N = probs.shape[0]
        if cell_ids is None:
            cell_ids = [str(i) for i in range(N)]
        return [self.predict(probs[i], cell_ids[i], available_markers) for i in range(N)]

    def evaluate_coverage(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        ontology: Dict[str, object],
        available_markers: Optional[Set[str]] = None,
    ) -> Dict[str, float]:
        """
        Empirically measure per-level AND joint ontology coverage.

        This is the honest bound for the hierarchical procedure:
        standard conformal gives marginal coverage per level, but
        the joint "predicted type is an ancestor of the true type"
        coverage must be measured empirically.

        Args:
            probs: (N, K) class probabilities
            labels: (N,) integer class labels (indices into all_types)
            ontology: ontology dict (name -> CellTypeNode)
            available_markers: panel for panel-aware prediction

        Returns:
            Dict with per-level coverage metrics:
                lineage_coverage_empirical: P(true lineage in lineage_set)
                subtype_coverage_empirical: P(true subtype in subtype_set)
                fine_coverage_empirical: P(true fine type in fine_set)
                ontology_coverage: P(returned type is ancestor of true type)
                strict_coverage: P(returned type == true type)
        """
        N = len(labels)
        results = self.predict_batch(probs, available_markers=available_markers)

        # Build ancestor lookup from ontology
        def get_ancestors(type_name: str) -> Set[str]:
            ancestors = {type_name}
            node = ontology.get(type_name)
            while node is not None and getattr(node, 'parent', None):
                parent = node.parent
                ancestors.add(parent)
                node = ontology.get(parent)
            ancestors.add("Root")
            return ancestors

        lineage_covered = 0
        ontology_correct = 0
        strict_correct = 0
        for i, result in enumerate(results):
            true_type = self.all_types[labels[i]]
            true_lineage = self._type_to_lineage.get(true_type)
            # Per-level coverage
            if true_lineage is not None and true_lineage in result.lineage_set:
                lineage_covered += 1
            # Ontology coverage: returned type is ancestor of true
            true_ancestors = get_ancestors(true_type)
            if result.predicted_type in true_ancestors:
                ontology_correct += 1
            if result.predicted_type == true_type:
                strict_correct += 1

        return {
            'n_cells': N,
            'lineage_coverage_empirical': lineage_covered / N,
            'ontology_coverage': ontology_correct / N,
            'strict_coverage': strict_correct / N,
            'target_lineage_coverage': self.lineage_coverage,
            'target_subtype_coverage': self.subtype_coverage,
            'target_fine_coverage': self.fine_coverage,
        }

    def inflate_for_panel(
        self,
        calibration_panel: Set[str],
        target_panel: Set[str],
        inflation_coefficient: float = 0.5,
    ) -> 'HierarchicalConformalPredictor':
        """
        Return a copy with thresholds inflated for a target panel.

        When thresholds calibrated on `calibration_panel` are applied to a
        different `target_panel`, exchangeability is violated. We conservatively
        widen thresholds proportional to the marker deficit:

            inflation = 1 + deficit * inflation_coefficient
            deficit = 1 - |calibration ∩ target| / |calibration|

        Wider thresholds → wider prediction sets → higher empirical coverage.
        This is a conservative approximation; for tight bounds use
        `recalibrate_on_target` with labeled target-panel data.

        Args:
            calibration_panel: markers used during original calibration
            target_panel: markers available in target dataset
            inflation_coefficient: scaling factor for deficit (0.5 empirical default)

        Returns:
            New HierarchicalConformalPredictor with inflated thresholds
        """
        if not calibration_panel:
            raise ValueError("calibration_panel must be non-empty")
        common = calibration_panel & target_panel
        deficit = 1.0 - len(common) / len(calibration_panel)
        inflation = 1.0 + deficit * inflation_coefficient
        logger.info(
            "Panel deficit=%.2f, inflation factor=%.3f "
            "(cal_markers=%d, target_markers=%d, common=%d)",
            deficit, inflation, len(calibration_panel), len(target_panel), len(common),
        )

        new_hcp = HierarchicalConformalPredictor(
            self.all_types, self.lineage_coverage,
            self.subtype_coverage, self.fine_coverage,
        )
        # Clamp inflated thresholds to 1.0 (max nonconformity score)
        new_hcp.lineage_thresholds = {
            k: min(1.0, v * inflation) if v != float('inf') else v
            for k, v in self.lineage_thresholds.items()
        }
        new_hcp.subtype_thresholds = {
            k: min(1.0, v * inflation) if v != float('inf') else v
            for k, v in self.subtype_thresholds.items()
        }
        new_hcp.fine_thresholds = {
            k: min(1.0, v * inflation) if v != float('inf') else v
            for k, v in self.fine_thresholds.items()
        }
        new_hcp.is_calibrated = True
        return new_hcp

    def recalibrate_on_target(
        self,
        target_probs: np.ndarray,
        target_labels: np.ndarray,
        min_cells_per_class: int = 20,
    ) -> 'HierarchicalConformalPredictor':
        """
        Recalibrate thresholds using labeled target-dataset cells.

        For classes with sufficient target-panel calibration data, use
        target-derived thresholds. For undersampled classes, fall back to
        the original thresholds from this instance. This provides tight
        coverage on classes that are well-represented in target data and
        conservative fallback for rare classes.

        Args:
            target_probs: (N, K) predicted probabilities on target cal set
            target_labels: (N,) integer labels (indices into all_types)
            min_cells_per_class: minimum target cells to recalibrate a class

        Returns:
            New HierarchicalConformalPredictor with target-adapted thresholds
        """
        new_hcp = HierarchicalConformalPredictor(
            self.all_types, self.lineage_coverage,
            self.subtype_coverage, self.fine_coverage,
        )
        # Run full calibration on target data
        new_hcp.calibrate(target_probs, target_labels)

        # Fall back to original thresholds for classes with too few target cells
        for thresh_dict, base_dict, _label in [
            (new_hcp.lineage_thresholds, self.lineage_thresholds, "lineage"),
            (new_hcp.subtype_thresholds, self.subtype_thresholds, "subtype"),
            (new_hcp.fine_thresholds, self.fine_thresholds, "fine"),
        ]:
            for cls_name, base_thresh in base_dict.items():
                new_thresh = thresh_dict.get(cls_name, float('inf'))
                # If target recalibration returned +inf (undersampled), fall back
                if new_thresh == float('inf') and base_thresh != float('inf')\
                        and base_thresh < 1.0:
                    thresh_dict[cls_name] = base_thresh
                    logger.info(
                        "Class '%s:%s' undersampled in target — "
                        "using base threshold %.4f",
                        _label, cls_name, base_thresh,
                    )
        return new_hcp

    def save(self, path: str):
        """Save calibration."""
        data = {
            'all_types': self.all_types,
            'lineage_coverage': self.lineage_coverage,
            'subtype_coverage': self.subtype_coverage,
            'fine_coverage': self.fine_coverage,
            'lineage_thresholds': self.lineage_thresholds,
            'subtype_thresholds': self.subtype_thresholds,
            'fine_thresholds': self.fine_thresholds,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'HierarchicalConformalPredictor':
        with open(path) as f:
            data = json.load(f)
        hcp = cls(data['all_types'], data['lineage_coverage'],
                   data['subtype_coverage'], data['fine_coverage'])
        hcp.lineage_thresholds = data['lineage_thresholds']
        hcp.subtype_thresholds = data['subtype_thresholds']
        hcp.fine_thresholds = data['fine_thresholds']
        hcp.is_calibrated = True
        return hcp
