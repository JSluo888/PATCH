"""PATCH uncertainty-quantification modules.

Importing this package is intentionally side-effect-free: submodules are
imported lazily by the caller (``from uncertainty.hierarchical_conformal
import ...``) so that pulling in one component does not drag in optional
dependencies (xgboost, scipy, ...) used by sibling modules.

Public components
-----------------
- ``hierarchical_conformal.HierarchicalConformalPredictor`` — panel-aware,
  tree-hierarchical conformal predictor (the core of PATCH).
- ``aps`` — APS / RAPS adaptive prediction-set scorers.
- ``conformal_predictor.ConformalPredictor`` — flat split-conformal baseline.
- ``vlm_set_refiner.refine_conformal_set`` — coverage-preserving VLM set refiner.
- ``comparators`` — faithful ports of HCC (arXiv:2508.13288) and scConform
  (arXiv:2410.23786) used in the conformal-variant benchmark.
"""

__all__ = [
    "hierarchical_conformal",
    "aps",
    "conformal_predictor",
    "vlm_set_refiner",
    "comparators",
]
