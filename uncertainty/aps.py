"""APS / RAPS — adaptive and regularised adaptive prediction sets.

References:
    Romano, Sesia, Candes (2020).
    "Classification with Valid and Adaptive Coverage." NeurIPS 2020.
    arXiv:2006.02544.

    Angelopoulos, Bates, Jordan, Malik (2021).
    "Uncertainty Sets for Image Classifiers using Conformal Prediction."
    ICLR 2021. arXiv:2009.14193.

APS produces *adaptive* sets — the size grows with class-conditional
uncertainty rather than being uniform. RAPS adds a regularisation term to
discourage overly large sets on hard examples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def aps_score(probs_row: np.ndarray, label: int, randomise: bool = True) -> float:
    """APS non-conformity score for a single (probs, label) pair.

    Sort class probs in descending order; the score is the cumulative sum of
    probs from rank 1 down to (and partly including) the rank of `label`.
    Randomisation breaks ties at the boundary.
    """
    order = np.argsort(-probs_row)  # descending
    cumsum = np.cumsum(probs_row[order])
    rank = int(np.where(order == label)[0][0])
    score = float(cumsum[rank])
    if randomise:
        u = np.random.uniform()
        score -= u * float(probs_row[order[rank]])
    return score


def raps_score(
    probs_row: np.ndarray,
    label: int,
    *,
    k_reg: int = 5,
    lam: float = 0.01,
    randomise: bool = True,
) -> float:
    """RAPS non-conformity score with regularisation.

    Adds `lam * max(rank - k_reg, 0)` penalty for labels deep in the
    sorted prob list — discourages large sets on hard examples.
    """
    order = np.argsort(-probs_row)
    cumsum = np.cumsum(probs_row[order])
    rank = int(np.where(order == label)[0][0])
    score = float(cumsum[rank])
    score += lam * max(rank + 1 - k_reg, 0)
    if randomise:
        u = np.random.uniform()
        score -= u * float(probs_row[order[rank]])
    return score


@dataclass
class APSPredictor:
    """Adaptive Prediction Sets (APS / RAPS).

    Args:
        alpha: target miscoverage (coverage = 1 - alpha).
        regularised: True for RAPS, False for plain APS.
        k_reg: rank threshold above which RAPS penalises extra labels.
        lam: RAPS penalty weight.
        randomise: True to break ties at the set boundary (yields exact coverage
            in expectation; required for validity in RAPS paper).
    """

    alpha: float = 0.10
    regularised: bool = False
    k_reg: int = 5
    lam: float = 0.01
    randomise: bool = True

    def __post_init__(self) -> None:
        self._threshold: Optional[float] = None

    # -------------------------------------------------------------------

    def calibrate(self, cal_probs: np.ndarray, cal_labels: np.ndarray) -> None:
        """Compute the alpha-quantile of cal non-conformity scores."""
        scores = np.array(
            [
                self._score(cal_probs[i], int(cal_labels[i]))
                for i in range(cal_probs.shape[0])
            ]
        )
        n = len(scores)
        # the (1 - alpha)*(n+1)/n quantile (finite-sample correction)
        q_level = np.ceil((n + 1) * (1.0 - self.alpha)) / n
        q_level = float(np.clip(q_level, 0.0, 1.0))
        self._threshold = float(np.quantile(scores, q_level, method="higher"))

    def predict_set(
        self, test_probs: np.ndarray, class_ids: list[str]
    ) -> list[list[str]]:
        if self._threshold is None:
            raise RuntimeError("calibrate(...) must be called first")
        sets: list[list[str]] = []
        for i in range(test_probs.shape[0]):
            kept = []
            for k in range(test_probs.shape[1]):
                if self._score(test_probs[i], k) <= self._threshold:
                    kept.append(class_ids[k])
            sets.append(kept if kept else [class_ids[int(np.argmax(test_probs[i]))]])
        return sets

    def _score(self, probs_row: np.ndarray, label: int) -> float:
        if self.regularised:
            return raps_score(
                probs_row, label, k_reg=self.k_reg, lam=self.lam, randomise=self.randomise
            )
        return aps_score(probs_row, label, randomise=self.randomise)


# Convenience factories
def make_aps(alpha: float = 0.10, randomise: bool = True) -> APSPredictor:
    return APSPredictor(alpha=alpha, regularised=False, randomise=randomise)


def make_raps(
    alpha: float = 0.10, k_reg: int = 5, lam: float = 0.01, randomise: bool = True
) -> APSPredictor:
    return APSPredictor(alpha=alpha, regularised=True, k_reg=k_reg, lam=lam, randomise=randomise)
