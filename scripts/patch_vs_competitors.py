#!/usr/bin/env python3
"""PATCH vs competitors — cross-cohort benchmark with HCC + scConform ports.

Iterates over the benchmark cohorts (a hardcoded default list; optionally read
from a Hydra dataset-groups YAML if one is present). For each cohort it loads
the matching cal / test NPZ and the holdout XGBoost model, then computes the HCC
and scConform comparators alongside an optional scConform-with-Laplacian variant.

Both comparators run panel-BLIND exactly as published. PATCH's panel-aware edge
is reported as the gap in downstream analysis — this script does not add panel
filtering to either port.

Output JSON per cohort:
    results/comparators/patch_vs_competitors_{cohort}.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import xgboost as xgb

try:
    import yaml  # optional: only needed if a dataset-groups YAML is provided
except ImportError:  # pragma: no cover
    yaml = None

# --- Release path resolution (see configs/paths.py) ---------------------------
# Artifact root is $PATCH_DATA_DIR or <repo>/data; override via the CLI flags
# below. NO absolute /n/scratch or /home paths are baked in.
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from configs.marker_celltype_ontology import CELL_TYPE_ONTOLOGY  # noqa: E402
from configs.paths import models_dir, results_dir, uq_ablation_dir  # noqa: E402
from uncertainty.comparators import HCCPort, scConformPort  # noqa: E402

# ---------------------------------------------------------------------------
# Per-cohort path resolution
# ---------------------------------------------------------------------------

_UQ = str(uq_ablation_dir())
MODEL_ROOT = str(models_dir())
# Optional Hydra dataset-groups YAML defining the exact cohort order; when it is
# absent (the default in this release) the built-in COHORT_SLUG list is used.
DATASET_GROUPS_YAML = str(REPO / "conf" / "dataset_groups" / "all.yaml")
DEFAULT_OUT_DIR = str(results_dir() / "comparators")

# Hydra-config dataset name → short cohort slug used in NPZ / model paths.
# Mirrors the LODO_NPZ keys in ``scripts/lodo_evaluation_v6.py``.
COHORT_SLUG: Dict[str, str] = {
    "crc": "crc",
    "hoch_2022_melanoma_imc": "hoch",
    "liu_2022_multitissue_mibi": "liu",
    "immuncan_2024": "immucan",
    "schurch_2020_crc_codex": "schurch",
    "tietscher_2022_breast_imc": "tietscher",
    "hartmann_2020_crc_mibi": "hartmann",
    "sorin_2023_nsclc_imc": "sorin",
    "karimi_2023_braintumor_imc": "karimi",
    "maps_hodgkin": "maps",
    "hickey_2023_intestine_codex": "hickey",
    "cords_2023_pancancer_caf_imc": "cords",
    "risom_2022_dcis_mibi": "risom",
}

# v6 holdout model dirs use a mix of plain Title-case and shouty all-caps for
# historical reasons (matches what train_v6_holdout.py wrote to disk).
MODEL_SLUG_OVERRIDES: Dict[str, str] = {
    "immucan": "IMMUcan",
    "maps": "MAPS",
}

# Cohorts that exceed 1M cells benefit from a larger memory tier when batched.
LARGE_COHORTS = frozenset({"sorin", "maps"})


def load_cohort_list() -> List[str]:
    """Return the cohort slug list.

    If the optional Hydra dataset-groups YAML (``conf/dataset_groups/all.yaml``)
    is present it defines the exact order (training_pool + lodo_default +
    lodo_extended + external_eval); otherwise we fall back to the built-in
    ``COHORT_SLUG`` mapping shipped in this release.
    """
    if yaml is None or not Path(DATASET_GROUPS_YAML).exists():
        # Self-contained fallback: deduplicated slug list from COHORT_SLUG.
        seen: List[str] = []
        for slug in COHORT_SLUG.values():
            if slug not in seen:
                seen.append(slug)
        return seen
    with open(DATASET_GROUPS_YAML) as f:
        cfg = yaml.safe_load(f)
    groups = cfg["groups"]
    wanted_keys = ("training_pool", "lodo_default", "lodo_extended", "external_eval")
    slugs: List[str] = []
    for key in wanted_keys:
        for dataset_name in groups.get(key, []):
            slug = COHORT_SLUG.get(dataset_name)
            if slug is None:
                print(f"[warn] no slug mapping for dataset '{dataset_name}', skipping")
                continue
            if slug not in slugs:
                slugs.append(slug)
    return slugs


def paths_for_cohort(slug: str) -> Tuple[str, str, str]:
    """Return ``(cal_npz, test_npz, model_path)`` for ``slug``.

    For ``crc`` the v5.3 nimbus model is used (no holdout — CRC is the
    training base); for every other cohort the matching ``v6.0_holdout_<Slug>``
    XGBoost model is used.
    """
    cal_npz = f"{_UQ}/{slug}_cal.npz"
    test_npz = f"{_UQ}/{slug}_test.npz"
    if slug == "crc":
        model_path = f"{MODEL_ROOT}/v5.3_nimbus/xgboost/model.ubj"
    else:
        model_suffix = MODEL_SLUG_OVERRIDES.get(slug, slug.capitalize())
        model_path = f"{MODEL_ROOT}/v6.0_holdout_{model_suffix}/xgboost/model.ubj"
    return cal_npz, test_npz, model_path


# ---------------------------------------------------------------------------
# Data + ontology helpers
# ---------------------------------------------------------------------------


def load_data(path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load (X_gmm, y, target_types) from a v5.3 ablation NPZ."""
    d = np.load(path)
    target_types = d["target_types"].tolist()
    return d["X_gmm"].astype(np.float32), d["y"].astype(np.int64), target_types


def get_ontology_parents(target_types: List[str]) -> Dict[str, List[str]]:
    """Build leaf → [self, ..., root] ancestor lists from CELL_TYPE_ONTOLOGY.

    Required by HCCPort and scConformPort.
    """

    def ancestors_of(name: str) -> List[str]:
        chain = [name]
        cur = name
        while True:
            node = CELL_TYPE_ONTOLOGY.get(cur)
            parent = getattr(node, "parent", None) if node is not None else None
            if parent is None or parent == cur:
                break
            chain.append(parent)
            cur = parent
        # Ensure 'root' is the top
        if chain[-1] != "root":
            chain.append("root")
        return chain

    return {leaf: ancestors_of(leaf) for leaf in target_types}


def hier_coverage(
    sets_bool: np.ndarray,
    y: np.ndarray,
    target_types: List[str],
    ancestors: Dict[str, List[str]],
) -> float:
    """Ontology coverage: cell is covered if SOME ancestor of y is in set.

    Vectorised reformulation of the original O(n_test * K) double loop:

        A[c, d] = True iff target_types[c] is in ancestors[target_types[d]]
        cell i covered iff any c with sets_bool[i, c] and A[c, y[i]]

    Verified equal to the legacy loop on synthetic data and the CRC test set.
    """
    K = len(target_types)
    if sets_bool.shape[1] != K:
        raise ValueError(f"sets_bool last dim {sets_bool.shape[1]} != K={K}")
    # K x K ancestor-of matrix
    A = np.zeros((K, K), dtype=bool)
    for c, ancestor_name in enumerate(target_types):
        for d, leaf_name in enumerate(target_types):
            if ancestor_name in ancestors[leaf_name]:
                A[c, d] = True
    # For each test row i, broadcast A[:, y[i]] and AND with sets_bool[i].
    ancestor_mask_per_row = A[:, y]  # shape (K, n_test)
    hits = (sets_bool & ancestor_mask_per_row.T).any(axis=1)
    return float(hits.mean())


# ---------------------------------------------------------------------------
# Per-cohort benchmark
# ---------------------------------------------------------------------------


def benchmark_cohort(slug: str, alpha: float, out_path: Path) -> bool:
    """Run all three comparator variants for one cohort and write JSON.

    Returns True on success, False if any required input is missing.
    """
    cal_npz, test_npz, model_path = paths_for_cohort(slug)
    for required in (cal_npz, test_npz, model_path):
        if not Path(required).exists():
            print(f"[skip] {slug}: missing {required}")
            return False

    X_cal, y_cal, target_types = load_data(cal_npz)
    X_test, y_test, _ = load_data(test_npz)
    n_classes = len(target_types)
    print(f"[load] {slug}: cal n={len(y_cal)} test n={len(y_test)} classes={n_classes}")

    booster = xgb.Booster()
    booster.load_model(model_path)
    P_cal = booster.predict(xgb.DMatrix(X_cal))
    P_test = booster.predict(xgb.DMatrix(X_test))
    print(f"[infer] {slug}: cal P {P_cal.shape}  test P {P_test.shape}")

    ontology_parents = get_ontology_parents(target_types)

    results: Dict = {
        "cohort": slug,
        "alpha": alpha,
        "n_cal": int(len(y_cal)),
        "n_test": int(len(y_test)),
        "n_classes": n_classes,
        "target_types": target_types,
        "cal_npz": cal_npz,
        "test_npz": test_npz,
        "model_path": model_path,
        "variants": {},
    }

    # ----- (g) HCC port -----
    print(f"[variant g] {slug}: HCC port (arXiv:2508.13288)")
    hcc = HCCPort(
        ontology_parents=ontology_parents,
        target_types=target_types,
        alpha=alpha,
    ).fit(P_cal, y_cal)
    sets_g = hcc.predict(P_test)
    cov_g = hier_coverage(sets_g, y_test, target_types, ontology_parents)
    size_g = float(sets_g.sum(axis=1).mean())
    leaf_g = float((sets_g.sum(axis=1) == 1).mean())
    results["variants"]["g_hcc_port"] = {
        "method": "HCC (Principato 2025, arXiv:2508.13288)",
        "ontology_coverage": cov_g,
        "avg_set_size": size_g,
        "leaf_frac": leaf_g,
        "n_internal_nodes": len(hcc.nodes),
    }
    print(f"  ontology-cov={cov_g:.3f}  |S|={size_g:.2f}  leaf%={leaf_g:.3f}")

    # ----- (h) scConform port -----
    print(f"[variant h] {slug}: scConform port (arXiv:2410.23786)")
    sccp = scConformPort(
        ontology_parents=ontology_parents,
        target_types=target_types,
        alpha=alpha,
        laplacian_alpha=0.0,
    ).fit(P_cal, y_cal)
    sets_h = sccp.predict(P_test)
    cov_h = hier_coverage(sets_h, y_test, target_types, ontology_parents)
    size_h = float(sets_h.sum(axis=1).mean())
    leaf_h = float((sets_h.sum(axis=1) == 1).mean())
    results["variants"]["h_scconform_port"] = {
        "method": "scConform (Wojnowska 2024, arXiv:2410.23786)",
        "ontology_coverage": cov_h,
        "avg_set_size": size_h,
        "leaf_frac": leaf_h,
        "laplacian_alpha": 0.0,
    }
    print(f"  ontology-cov={cov_h:.3f}  |S|={size_h:.2f}  leaf%={leaf_h:.3f}")

    # ----- Bonus: scConform with Laplacian smoothing -----
    print(f"[variant h2] {slug}: scConform port (Laplacian alpha=0.5)")
    sccp_smooth = scConformPort(
        ontology_parents=ontology_parents,
        target_types=target_types,
        alpha=alpha,
        laplacian_alpha=0.5,
    ).fit(P_cal, y_cal)
    sets_h2 = sccp_smooth.predict(P_test)
    cov_h2 = hier_coverage(sets_h2, y_test, target_types, ontology_parents)
    size_h2 = float(sets_h2.sum(axis=1).mean())
    results["variants"]["h2_scconform_smoothed"] = {
        "method": "scConform + Laplacian (alpha=0.5)",
        "ontology_coverage": cov_h2,
        "avg_set_size": size_h2,
        "leaf_frac": float((sets_h2.sum(axis=1) == 1).mean()),
    }
    print(f"  ontology-cov={cov_h2:.3f}  |S|={size_h2:.2f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[done] wrote {out_path}")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument(
        "--cohorts",
        nargs="+",
        default=None,
        help="Subset of cohort slugs to process (default: all 14).",
    )
    parser.add_argument(
        "--out_dir",
        default=DEFAULT_OUT_DIR,
        help="Directory for per-cohort JSON outputs.",
    )
    args = parser.parse_args()

    all_cohorts = load_cohort_list()
    if args.cohorts:
        unknown = sorted(set(args.cohorts) - set(all_cohorts))
        if unknown:
            print(f"[warn] unknown cohort slugs (not in dataset_groups/all.yaml): {unknown}")
        cohorts = [c for c in args.cohorts if c in all_cohorts]
    else:
        cohorts = all_cohorts
    print(f"[plan] {len(cohorts)} cohort(s): {cohorts}")

    out_dir = Path(args.out_dir)
    n_ok = 0
    for slug in cohorts:
        print(f"\n========== {slug} ==========")
        out_path = out_dir / f"patch_vs_competitors_{slug}.json"
        ok = benchmark_cohort(slug, args.alpha, out_path)
        if ok:
            n_ok += 1

    print(f"\n[summary] {n_ok}/{len(cohorts)} cohorts produced results")
    return 0


if __name__ == "__main__":
    sys.exit(main())
