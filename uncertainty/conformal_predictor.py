"""
Conformal Prediction for Cell Type Classification

Provides calibrated uncertainty quantification with coverage guarantees.

Key concepts:
- Nonconformity score: How "unusual" is this prediction? Higher = more unusual
- Prediction set: Set of types that are plausible given the markers
- Coverage guarantee: If calibrated for 90%, then 90% of true labels will be in prediction set

Reference: Vovk et al., "Algorithmic Learning in a Random World"
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.cell_type_rules import (
    VALID_CELL_TYPES,
    CLASSIFICATION_RULES,
    filter_rules_for_dataset,
    get_dataset_type,
)


@dataclass
class ConformalResult:
    """Result of conformal prediction for a single cell."""
    cell_id: str
    predicted_type: str           # Most likely type (highest conformity)
    prediction_set: List[str]     # Set of plausible types at target coverage
    set_size: int                 # Size of prediction set (1 = confident)
    p_values: Dict[str, float]    # Per-type p-values (higher = more plausible)
    conformity_scores: Dict[str, float]  # Per-type conformity scores
    is_uncertain: bool            # True if set_size > 1
    coverage_level: float         # Target coverage (e.g., 0.90)
    confidence: str = "medium"    # "high", "medium", "low" based on score gaps
    confidence_score: float = 0.5  # Numeric confidence (0-1)


@dataclass
class CalibrationData:
    """Stored calibration data for conformal prediction."""
    nonconformity_scores: List[float]  # Scores from calibration set
    n_calibration: int
    threshold_90: float           # Threshold for 90% coverage
    threshold_95: float           # Threshold for 95% coverage
    per_class_scores: Dict[str, List[float]]  # Scores per class


class ConformalPredictor:
    """
    Conformal predictor for cell type classification.

    Uses rule-based classification with marker posteriors to compute
    conformity scores, then calibrates thresholds on held-out data.
    """

    def __init__(self, coverage: float = 0.90):
        """
        Args:
            coverage: Target coverage level (e.g., 0.90 for 90% coverage)
        """
        self.coverage = coverage
        self.calibration_data = None
        self.is_calibrated = False

    def compute_conformity_score(
        self,
        marker_posteriors: Dict[str, float],
        candidate_type: str,
        sample_name: str,
        _cached_rules: List = None,
    ) -> float:
        """
        Compute conformity score for a candidate cell type.

        Higher score = more conforming (more plausible).
        Score is based on how well markers match the rule requirements.

        Uses a more discriminative scoring approach:
        - Required positive markers: want high posteriors (use min for hard AND)
        - Required negative markers: penalize if posterior is high
        - Missing markers contribute uncertainty penalty

        Args:
            marker_posteriors: Dict of marker -> P(positive)
            candidate_type: Cell type to evaluate
            sample_name: Sample name for rule filtering
            _cached_rules: Pre-filtered rules (optimization to avoid re-filtering)

        Returns:
            Conformity score in [0, 1]
        """
        rules = _cached_rules if _cached_rules is not None else filter_rules_for_dataset(sample_name)

        # Find the rule for this cell type
        matching_rule = None
        for priority, cell_type, req_pos, req_neg, notes in rules:
            if cell_type == candidate_type:
                matching_rule = (req_pos, req_neg)
                break

        if matching_rule is None:
            # No rule for this type - low conformity
            return 0.0

        req_pos, req_neg = matching_rule

        if not req_pos and not req_neg:
            # Unassigned rule - compute based on NOT matching specific types
            # High conformity if no other type has high markers
            # Check if any major marker is strongly positive
            major_markers = ['PanCK', 'CD45', 'CD31', 'aSMA', 'CD3e', 'CD20', 'CD68']
            max_marker_posterior = 0.0
            for m in major_markers:
                if m in marker_posteriors:
                    max_marker_posterior = max(max_marker_posterior, marker_posteriors[m])

            # If no strong markers, Unassigned is plausible
            if max_marker_posterior < 0.3:
                return 0.8
            elif max_marker_posterior < 0.5:
                return 0.5
            else:
                return 0.3  # Some marker is positive, so Unassigned is less likely

        # Compute conformity using discriminative approach
        pos_scores = []
        neg_penalties = []
        missing_markers = 0

        # Score for required positive markers
        for marker in req_pos:
            if marker in marker_posteriors:
                p = marker_posteriors[marker]
                # Transform to make more discriminative: sigmoid-like stretching
                # p < 0.3 -> very low score, p > 0.7 -> very high score
                stretched = 1.0 / (1.0 + np.exp(-10 * (p - 0.5)))
                pos_scores.append(stretched)
            else:
                missing_markers += 1
                pos_scores.append(0.3)  # Penalty for missing marker

        # Penalty for required negative markers that are positive
        for marker in req_neg:
            if marker in marker_posteriors:
                p = marker_posteriors[marker]
                if p > 0.5:  # Marker should be negative but is positive
                    # Strong penalty proportional to how positive it is
                    penalty = (p - 0.5) * 2  # 0 to 1 scale
                    neg_penalties.append(penalty)

        # Compute base conformity from positive markers
        if pos_scores:
            # Use minimum to enforce AND logic - all markers must match
            min_pos = min(pos_scores)
            mean_pos = np.mean(pos_scores)
            # Weighted combination: emphasize minimum but consider mean
            conformity = 0.6 * min_pos + 0.4 * mean_pos
        else:
            conformity = 0.5

        # Apply penalties for negative marker violations
        if neg_penalties:
            # Strong penalty - any violation significantly reduces conformity
            max_penalty = max(neg_penalties)
            total_penalty = min(sum(neg_penalties), 1.0)
            conformity *= (1.0 - 0.5 * max_penalty - 0.3 * total_penalty)

        # Penalty for many missing markers
        if missing_markers > 0 and len(req_pos) > 0:
            missing_ratio = missing_markers / len(req_pos)
            conformity *= (1.0 - 0.3 * missing_ratio)

        return float(np.clip(conformity, 0.0, 1.0))

    def compute_nonconformity_score(
        self,
        marker_posteriors: Dict[str, float],
        true_type: str,
        sample_name: str,
    ) -> float:
        """
        Compute nonconformity score for calibration.

        Uses 1 - conformity of the true type.
        Higher = more unusual/harder to classify.

        Args:
            marker_posteriors: Dict of marker -> P(positive)
            true_type: Ground truth cell type
            sample_name: Sample name

        Returns:
            Nonconformity score in [0, 1]
        """
        conformity = self.compute_conformity_score(
            marker_posteriors, true_type, sample_name
        )
        return 1.0 - conformity

    def calibrate(
        self,
        calibration_cells: List[Dict],
        verbose: bool = True,
    ):
        """
        Calibrate the conformal predictor on held-out data.

        Args:
            calibration_cells: List of dicts with keys:
                - marker_posteriors: Dict[str, float]
                - true_type: str
                - sample_name: str
            verbose: Print calibration stats
        """
        if verbose:
            print(f"[ConformalPredictor] Calibrating on {len(calibration_cells)} cells...")

        nonconformity_scores = []
        per_class_scores = defaultdict(list)

        for cell in calibration_cells:
            score = self.compute_nonconformity_score(
                cell['marker_posteriors'],
                cell['true_type'],
                cell['sample_name'],
            )
            nonconformity_scores.append(score)
            per_class_scores[cell['true_type']].append(score)

        nonconformity_scores = np.array(nonconformity_scores)

        # Compute thresholds for different coverage levels
        # For coverage α, threshold = αth quantile of scores (with finite sample correction)
        # Include type y if its nonconformity <= threshold
        n = len(nonconformity_scores)

        # Finite sample correction: use ceil((n+1)*α)/n quantile
        # This ensures at least α fraction of true types have nonconformity <= threshold
        def get_threshold(alpha):
            q = np.ceil((n + 1) * alpha) / n
            return float(np.quantile(nonconformity_scores, min(q, 1.0)))

        self.calibration_data = CalibrationData(
            nonconformity_scores=nonconformity_scores.tolist(),
            n_calibration=n,
            threshold_90=get_threshold(0.90),
            threshold_95=get_threshold(0.95),
            per_class_scores={k: v for k, v in per_class_scores.items()},
        )

        self.is_calibrated = True

        # Pre-compute sorted calibration scores for fast p-value lookup
        self._cal_scores_sorted = np.sort(nonconformity_scores)
        self._cal_n = len(self._cal_scores_sorted)

        if verbose:
            print(f"[ConformalPredictor] Calibration complete:")
            print(f"  N cells: {n}")
            print(f"  Mean nonconformity: {np.mean(nonconformity_scores):.3f}")
            print(f"  90% threshold: {self.calibration_data.threshold_90:.3f}")
            print(f"  95% threshold: {self.calibration_data.threshold_95:.3f}")

    def predict(
        self,
        cell_id: str,
        marker_posteriors: Dict[str, float],
        sample_name: str,
        coverage: float = None,
    ) -> ConformalResult:
        """
        Make a conformal prediction for a single cell.

        Args:
            cell_id: Cell identifier
            marker_posteriors: Dict of marker -> P(positive)
            sample_name: Sample name for rule filtering
            coverage: Coverage level (default: self.coverage)

        Returns:
            ConformalResult with prediction set and p-values
        """
        if coverage is None:
            coverage = self.coverage

        # Get threshold for desired coverage
        if self.is_calibrated:
            if coverage >= 0.95:
                threshold = self.calibration_data.threshold_95
            elif coverage >= 0.90:
                threshold = self.calibration_data.threshold_90
            else:
                warnings.warn(
                    f"[ConformalPredictor] Requested coverage {coverage} is below "
                    f"precomputed levels (0.90, 0.95). Falling back to 0.90 threshold.",
                    stacklevel=2,
                )
                threshold = self.calibration_data.threshold_90
        else:
            # Not calibrated - use default threshold
            threshold = 0.5

        # Cache filtered rules for this sample (avoids re-filtering per type)
        cached_rules = filter_rules_for_dataset(sample_name)

        # Compute conformity scores for all candidate types
        conformity_scores = {}
        for cell_type in VALID_CELL_TYPES:
            score = self.compute_conformity_score(
                marker_posteriors, cell_type, sample_name,
                _cached_rules=cached_rules,
            )
            conformity_scores[cell_type] = score

        # Convert to nonconformity and compute p-values
        p_values = {}
        for cell_type, conformity in conformity_scores.items():
            nonconformity = 1 - conformity
            if self.is_calibrated and hasattr(self, '_cal_scores_sorted'):
                # Fast p-value using binary search on sorted calibration scores
                idx = np.searchsorted(self._cal_scores_sorted, nonconformity, side='left')
                n_geq = self._cal_n - idx
                p_val = (n_geq + 1) / (self._cal_n + 1)
            elif self.is_calibrated:
                # Fallback: full comparison (slower)
                cal_scores = np.array(self.calibration_data.nonconformity_scores)
                p_val = (np.sum(cal_scores >= nonconformity) + 1) / (len(cal_scores) + 1)
            else:
                # Approximation without calibration
                p_val = conformity
            p_values[cell_type] = float(p_val)

        # Build prediction set using calibrated threshold when available
        if self.is_calibrated:
            # Use calibrated conformal prediction: include type y if its
            # nonconformity score <= calibrated threshold (coverage guarantee)
            prediction_set = []
            for cell_type, conformity in conformity_scores.items():
                nonconformity = 1.0 - conformity
                if nonconformity <= threshold:
                    prediction_set.append(cell_type)

            # Sort by conformity score (highest first)
            prediction_set = sorted(
                prediction_set,
                key=lambda t: conformity_scores[t],
                reverse=True,
            )
        else:
            # Fallback heuristic when no calibration data exists:
            # Sort types by conformity (highest first) and include until
            # cumulative softmax probability exceeds threshold or a gap is found
            sorted_types = sorted(
                conformity_scores.items(),
                key=lambda x: x[1],
                reverse=True,
            )

            # Compute normalized scores (softmax-like)
            scores_array = np.array([s for _, s in sorted_types])
            temp = 0.5
            exp_scores = np.exp(scores_array / temp)
            normalized = exp_scores / (exp_scores.sum() + 1e-10)

            prediction_set = []
            cumulative = 0.0
            gap_threshold = 0.20

            for i, (cell_type, conf) in enumerate(sorted_types):
                prediction_set.append(cell_type)
                cumulative += normalized[i]

                if cumulative >= 0.8 and i < len(sorted_types) - 1:
                    next_conf = sorted_types[i + 1][1]
                    if conf - next_conf > gap_threshold:
                        break
                if cumulative >= 0.98:
                    break
                if len(prediction_set) >= 8 and cumulative >= 0.6:
                    break

            # Sort by conformity score
            prediction_set = sorted(
                prediction_set,
                key=lambda t: conformity_scores[t],
                reverse=True,
            )

        # Ensure at least one prediction
        if not prediction_set:
            best_type = max(conformity_scores, key=conformity_scores.get)
            prediction_set = [best_type]

        predicted_type = prediction_set[0]

        # Compute confidence based on gap between top-1 and top-2
        sorted_types_for_conf = sorted(
            conformity_scores.items(), key=lambda x: x[1], reverse=True
        )
        top1_conf = sorted_types_for_conf[0][1] if sorted_types_for_conf else 0.5
        top2_conf = sorted_types_for_conf[1][1] if len(sorted_types_for_conf) > 1 else 0.0
        gap = top1_conf - top2_conf

        # Normalized probability of top type (softmax)
        scores_array_conf = np.array([s for _, s in sorted_types_for_conf])
        exp_conf = np.exp(scores_array_conf / 0.5)
        norm_conf = exp_conf / (exp_conf.sum() + 1e-10)
        top1_prob = norm_conf[0] if len(norm_conf) > 0 else 0.5

        # Confidence score combines gap and probability
        confidence_score = min(1.0, gap * 2 + top1_prob * 0.5)

        # Discretize to high/medium/low
        if confidence_score >= 0.7 and gap >= 0.2:
            confidence = "high"
        elif confidence_score >= 0.4 or gap >= 0.1:
            confidence = "medium"
        else:
            confidence = "low"

        return ConformalResult(
            cell_id=cell_id,
            predicted_type=predicted_type,
            prediction_set=prediction_set,
            set_size=len(prediction_set),
            p_values=p_values,
            conformity_scores=conformity_scores,
            is_uncertain=(len(prediction_set) > 1),
            coverage_level=coverage,
            confidence=confidence,
            confidence_score=confidence_score,
        )

    def predict_batch(
        self,
        cells: List[Dict],
        coverage: float = None,
        verbose: bool = True,
    ) -> List[ConformalResult]:
        """
        Make conformal predictions for a batch of cells.

        Args:
            cells: List of dicts with keys:
                - cell_id: str
                - marker_posteriors: Dict[str, float]
                - sample_name: str
            coverage: Coverage level
            verbose: Print progress

        Returns:
            List of ConformalResult
        """
        if not cells:
            return []

        results = []
        n_uncertain = 0
        set_sizes = []
        import time as _time
        _t0 = _time.time()

        for i, cell in enumerate(cells):
            if verbose and (i + 1) % 5000 == 0:
                elapsed = _time.time() - _t0
                rate = (i + 1) / elapsed
                eta = (len(cells) - i - 1) / rate
                print(f"[ConformalPredictor] Processing {i+1}/{len(cells)} ({rate:.0f} cells/s, ETA {eta:.0f}s)...")

            result = self.predict(
                cell['cell_id'],
                cell['marker_posteriors'],
                cell['sample_name'],
                coverage=coverage,
            )
            results.append(result)

            if result.is_uncertain:
                n_uncertain += 1
            set_sizes.append(result.set_size)

        if verbose:
            print(f"[ConformalPredictor] Prediction complete:")
            print(f"  Total cells: {len(cells)}")
            print(f"  Uncertain (set_size > 1): {n_uncertain} ({100*n_uncertain/len(cells):.1f}%)")
            print(f"  Mean set size: {np.mean(set_sizes):.2f}")
            print(f"  Set size distribution: 1={sum(s==1 for s in set_sizes)}, 2={sum(s==2 for s in set_sizes)}, 3+={sum(s>=3 for s in set_sizes)}")

        return results

    def save_calibration(self, path: str):
        """Save calibration data to file."""
        if not self.is_calibrated:
            raise ValueError("Not calibrated yet!")

        data = {
            'coverage': self.coverage,
            'n_calibration': self.calibration_data.n_calibration,
            'threshold_90': self.calibration_data.threshold_90,
            'threshold_95': self.calibration_data.threshold_95,
            'nonconformity_scores': self.calibration_data.nonconformity_scores,
            'per_class_scores': self.calibration_data.per_class_scores,
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"[ConformalPredictor] Saved calibration to {path}")

    def load_calibration(self, path: str):
        """Load calibration data from file."""
        with open(path, 'r') as f:
            data = json.load(f)

        self.coverage = data['coverage']
        self.calibration_data = CalibrationData(
            nonconformity_scores=data['nonconformity_scores'],
            n_calibration=data['n_calibration'],
            threshold_90=data['threshold_90'],
            threshold_95=data['threshold_95'],
            per_class_scores=data['per_class_scores'],
        )
        self.is_calibrated = True

        # Pre-compute sorted calibration scores for fast p-value lookup
        self._cal_scores_sorted = np.sort(np.array(data['nonconformity_scores']))
        self._cal_n = len(self._cal_scores_sorted)

        print(f"[ConformalPredictor] Loaded calibration from {path}")


def evaluate_coverage(
    results: List[ConformalResult],
    true_types: List[str],
) -> Dict:
    """
    Evaluate the actual coverage of conformal predictions.

    Args:
        results: List of ConformalResult
        true_types: List of ground truth types

    Returns:
        Dict with coverage metrics
    """
    n = len(results)

    # Check if true type is in prediction set
    covered = [
        true_type in result.prediction_set
        for result, true_type in zip(results, true_types)
    ]

    actual_coverage = sum(covered) / n

    # Coverage by set size
    coverage_by_size = defaultdict(list)
    for result, true_type, is_covered in zip(results, true_types, covered):
        coverage_by_size[result.set_size].append(is_covered)

    coverage_by_size = {
        size: sum(cov) / len(cov)
        for size, cov in coverage_by_size.items()
    }

    # Average set size
    avg_set_size = np.mean([r.set_size for r in results])

    return {
        'actual_coverage': actual_coverage,
        'target_coverage': results[0].coverage_level if results else 0.90,
        'n_samples': n,
        'avg_set_size': avg_set_size,
        'coverage_by_set_size': coverage_by_size,
        'n_uncertain': sum(1 for r in results if r.is_uncertain),
    }
