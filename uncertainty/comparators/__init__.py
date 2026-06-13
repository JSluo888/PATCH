"""Conformal-prediction competitor ports for the PACE comparison (Fig 4b).

Modules:
    hcc_port:        Hierarchical Conformal Classification (Principato et al.,
                     arXiv:2508.13288). Per-internal-node conformal thresholds
                     with constrained-rep-complexity (set-size-budget) ascent.
    scconform_port:  scConform (arXiv:2410.23786, Bioconductor 2025). Class-
                     conditional split-CP with parent-closed (ontology-graph-
                     respecting) prediction sets.

Both expose the same interface as the variants in
`scripts/conformal_variants_benchmark.py`:

    fit(P_cal, y_cal, target_types, ontology) -> calibrated state
    predict(P_test) -> bool array (n_test, n_classes)

so they slot directly into the benchmark grid as variants (g) and (h).

Reference for our PACE method (this codebase, NOT a port):
    `uncertainty.hierarchical_conformal.HierarchicalConformalPredictor`
"""

from .hcc_port import HCCPort
from .scconform_port import scConformPort

__all__ = ["HCCPort", "scConformPort"]
