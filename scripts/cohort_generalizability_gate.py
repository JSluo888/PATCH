#!/usr/bin/env python3
"""Cohort generalizability gate — decide if a cohort is safe to predict on
*before* any ground-truth labels are available.

Motivation
----------
PACE's panel-aware hierarchical conformal predictor only carries its coverage
guarantee under exchangeability with the calibration platform (Orion CRC). When
a new cohort is far from the training distribution, empirical coverage degrades.
We previously *measured* that degradation post hoc (the KS<->coverage scaling
law, r approx -0.95). This script turns that law into a *predictive* gate: it
computes the label-free Kolmogorov-Smirnov distance between a cohort's GMM
marker posteriors and the CRC calibration set, then uses the fitted law to
predict the cohort's hierarchical ontology coverage and emit a deployment
verdict — GO / RECALIBRATE-FIRST / NO-GO — without ever touching labels.

Method (faithful to scripts/ks_coverage_sample_level.py)
--------------------------------------------------------
For each marker present in BOTH the cohort panel and the CRC panel, compute the
two-sample KS statistic between the cohort's GMM posterior column and the CRC
calibration GMM posterior column. Average over shared markers -> ``mean_ks``.
Then::

    predicted_coverage = INTERCEPT + SLOPE * mean_ks

The coefficients are fitted **self-consistently** (2026-05-29): mean_ks is
computed by *this gate's* method (KS over markers_present & CRC panel) for the 5
LODO cohorts {Hoch, Liu, IMMUcan, Schurch, Hartmann}, paired with their observed
hierarchical ontology coverage from ``ks_coverage_with_hartmann.json``, and
regressed: INTERCEPT=1.7339, SLOPE=-2.7879, R^2=0.596, residual std=0.1166
(cohort-level n=5, pearson r=-0.77). The slope agrees with the published
batch-level fit (slope -2.80, r=-0.85 to -0.95 over 50 batch points computed on a
curated per-cohort panel); the intercept is ~0.13 lower here because the gate
averages KS over the fuller ``markers_present`` panel rather than the curated
LODO panel. We use the self-consistent fit so the gate's prediction matches the
gate's own KS computation. Pass ``--refit`` to recompute self-consistently.

This gate is a deployment TRIAGE heuristic, not a coverage guarantee: it predicts
which cohorts are safe to predict on before labels exist. Treat predictions near a
threshold (within the +/- residual band) as RECALIBRATE rather than GO.

Verdict thresholds (on predicted coverage at the 0.90 conformal target):
    >= 0.85  -> GO              (predicted within ~5 pp of target)
    0.70-0.85 -> RECALIBRATE     (usable after a small in-platform recal set)
    < 0.70   -> NO-GO           (predict only after retraining / panel fix)

Usage
-----
    python scripts/cohort_generalizability_gate.py                 # score all cached cohorts
    python scripts/cohort_generalizability_gate.py --cohorts hoch_40d sorin_40d
    python scripts/cohort_generalizability_gate.py --refit         # re-fit the law first
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import ks_2samp

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# --- Release path resolution (see configs/paths.py) ---------------------------
# Artifact root is $PATCH_DATA_DIR or <repo>/data; override via the CLI flags
# below. NO absolute /n/scratch or /home paths are baked in.
from configs.paths import results_dir, uq_ablation_dir  # noqa: E402

UQ_DIR = uq_ablation_dir()
CAL_NPZ = UQ_DIR / "crc_cal.npz"
FIT_JSON = UQ_DIR / "ks_coverage_with_hartmann.json"
DEFAULT_OUT = results_dir() / "cohort_generalizability_gate.json"
DOC_OUT = results_dir() / "cohort_generalizability_gate.md"

# Fitted KS->coverage scaling law (2026-05-29, self-consistent; see module docstring).
LAW_INTERCEPT = 1.7339
LAW_SLOPE = -2.7879
LAW_R2 = 0.5960
LAW_RESID_STD = 0.1166

# The 5 LODO cohorts with both a feature cache and observed coverage, used to
# (re)fit the law self-consistently. Maps display name -> cache stem.
FIT_COHORTS = {
    "Hoch": "hoch_40d", "Liu": "liu_40d", "IMMUcan": "immucan_40d",
    "Schurch": "schurch_40d", "Hartmann": "hartmann_40d",
}

# Verdict thresholds on predicted coverage (0.90 conformal target).
GO_COVERAGE = 0.85
RECAL_COVERAGE = 0.70

# Number of GMM-posterior columns at the front of the 160D CRC feature vector.
N_GMM = 40


def crc_panel() -> set:
    """The Orion-CRC marker panel the scaling law was calibrated against."""
    from configs import ontology_v6 as o  # local import: heavy module

    return set(o.PANEL_MANIFEST["crc_orion"])


def load_reference(cal_npz: Path) -> Tuple[np.ndarray, List[str]]:
    """Load the CRC calibration GMM posteriors and the universal marker order.

    Returns (X_cal_gmm [N, 40], universal_markers [40])."""
    d = np.load(cal_npz, allow_pickle=True)
    x = np.asarray(d["X_gmm"])[:, :N_GMM]
    universal = [str(m) for m in d["universal_markers"]]
    return x, universal


def compute_mean_ks(
    x_cohort_gmm: np.ndarray,
    x_cal_gmm: np.ndarray,
    shared_idx: List[int],
) -> float:
    """Mean per-marker KS over shared markers (matches ks_coverage_sample_level)."""
    if not shared_idx:
        return float("nan")
    kss = [ks_2samp(x_cohort_gmm[:, i], x_cal_gmm[:, i]).statistic for i in shared_idx]
    return float(np.mean(kss))


def predict_coverage(mean_ks: float, slope: float, intercept: float) -> float:
    return float(np.clip(intercept + slope * mean_ks, 0.0, 1.0))


def verdict_for(pred_cov: float) -> str:
    if pred_cov >= GO_COVERAGE:
        return "GO"
    if pred_cov >= RECAL_COVERAGE:
        return "RECALIBRATE"
    return "NO-GO"


def refit_law(
    fit_json: Path,
    uq_dir: Path,
    x_cal_gmm: np.ndarray,
    crc_markers: set,
) -> Tuple[float, float, float, float]:
    """Self-consistently re-fit coverage = intercept + slope*mean_ks.

    mean_ks is recomputed by *this gate's* method for each of the FIT_COHORTS
    and paired with that cohort's observed hierarchical ontology coverage
    (cohort-mean of the batch rows in ``fit_json``). This keeps the law on the
    same KS scale the gate uses at inference time.
    """
    payload = json.loads(fit_json.read_text())
    obs: Dict[str, List[float]] = {}
    for r in payload["rows"]:
        obs.setdefault(r["cohort"], []).append(r["ontology_coverage"])
    obs_cov = {k: float(np.mean(v)) for k, v in obs.items()}

    ks_list, cov_list = [], []
    for name, stem in FIT_COHORTS.items():
        if name not in obs_cov:
            continue
        p = uq_dir / f"{stem}.npz"
        if not p.exists():
            continue
        d = np.load(p, allow_pickle=True)
        x = np.asarray(d["X_gmm"])[:, :N_GMM]
        universal = [str(m) for m in d["universal_markers"]]
        present = np.asarray(d["markers_present"]).astype(bool) if "markers_present" in d \
            else np.ones(len(universal), dtype=bool)
        idx = [i for i in range(min(N_GMM, len(universal)))
               if present[i] and universal[i] in crc_markers]
        ks_list.append(compute_mean_ks(x, x_cal_gmm, idx))
        cov_list.append(obs_cov[name])

    ks = np.array(ks_list, dtype=float)
    cov = np.array(cov_list, dtype=float)
    slope, intercept = np.polyfit(ks, cov, 1)
    resid = cov - (intercept + slope * ks)
    r2 = 1.0 - resid.var() / cov.var()
    return float(intercept), float(slope), float(r2), float(resid.std())


def score_cohort(
    npz_path: Path,
    x_cal_gmm: np.ndarray,
    crc_markers: set,
    slope: float,
    intercept: float,
    resid_std: float,
) -> Optional[Dict]:
    """Score one cohort cache; None if it lacks the needed arrays."""
    d = np.load(npz_path, allow_pickle=True)
    if "X_gmm" not in d:
        return None
    x = np.asarray(d["X_gmm"])
    if x.shape[1] >= N_GMM:
        x = x[:, :N_GMM]
    universal = [str(m) for m in d["universal_markers"]]
    if "markers_present" in d:
        present_mask = np.asarray(d["markers_present"]).astype(bool)
    else:
        present_mask = np.ones(len(universal), dtype=bool)

    shared_idx = [
        i for i in range(min(N_GMM, len(universal)))
        if present_mask[i] and universal[i] in crc_markers
    ]
    shared_markers = [universal[i] for i in shared_idx]
    mean_ks = compute_mean_ks(x, x_cal_gmm, shared_idx)
    pred_cov = predict_coverage(mean_ks, slope, intercept)
    lo = float(np.clip(pred_cov - 1.96 * resid_std, 0.0, 1.0))
    hi = float(np.clip(pred_cov + 1.96 * resid_std, 0.0, 1.0))
    return {
        "cohort": npz_path.stem,
        "n_cells": int(x.shape[0]),
        "n_shared_markers": len(shared_idx),
        "shared_markers": shared_markers,
        "mean_ks": round(mean_ks, 4),
        "predicted_coverage": round(pred_cov, 4),
        "predicted_coverage_ci95": [round(lo, 4), round(hi, 4)],
        "verdict": verdict_for(pred_cov),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cal", type=Path, default=CAL_NPZ, help="CRC calibration npz (reference distribution)")
    ap.add_argument("--cohorts", nargs="*", default=None,
                    help="cohort npz stems to score (default: all *_40d.npz in the uq_ablation dir)")
    ap.add_argument("--uq-dir", type=Path, default=UQ_DIR)
    ap.add_argument("--refit", action="store_true", help="re-fit the scaling law from ks_coverage_with_hartmann.json")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--doc-out", type=Path, default=DOC_OUT)
    args = ap.parse_args()

    crc_markers = crc_panel()
    x_cal_gmm, _ = load_reference(args.cal)

    intercept, slope, r2, resid_std = LAW_INTERCEPT, LAW_SLOPE, LAW_R2, LAW_RESID_STD
    if args.refit and FIT_JSON.exists():
        intercept, slope, r2, resid_std = refit_law(FIT_JSON, args.uq_dir, x_cal_gmm, crc_markers)

    ks_go = (GO_COVERAGE - intercept) / slope
    ks_recal = (RECAL_COVERAGE - intercept) / slope

    if args.cohorts:
        paths = [args.uq_dir / (c if c.endswith(".npz") else f"{c}.npz") for c in args.cohorts]
    else:
        paths = sorted(p for p in args.uq_dir.glob("*_40d.npz"))

    results: List[Dict] = []
    for p in paths:
        if not p.exists():
            print(f"[skip] {p.name}: not found")
            continue
        try:
            row = score_cohort(p, x_cal_gmm, crc_markers, slope, intercept, resid_std)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"[skip] {p.name}: {type(exc).__name__}: {exc}")
            continue
        if row is None:
            print(f"[skip] {p.name}: no X_gmm")
            continue
        results.append(row)

    results.sort(key=lambda r: r["mean_ks"])

    payload = {
        "law": {
            "intercept": round(intercept, 4), "slope": round(slope, 4),
            "r2": round(r2, 4), "resid_std": round(resid_std, 4),
            "refit": bool(args.refit),
            "source": "ks_coverage_with_hartmann.json (5 cohorts x 10 batches)",
        },
        "thresholds": {
            "go_coverage": GO_COVERAGE, "recal_coverage": RECAL_COVERAGE,
            "ks_go_max": round(ks_go, 4), "ks_recal_max": round(ks_recal, 4),
        },
        "reference": str(args.cal),
        "crc_panel_size": len(crc_markers),
        "cohorts": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))

    # Human-readable table.
    print()
    print(f"KS->coverage law: coverage = {intercept:.4f} + ({slope:.4f}) * mean_ks   "
          f"(R^2={r2:.3f}, resid_std={resid_std:.3f}{', refit' if args.refit else ''})")
    print(f"Gate: GO if mean_ks <= {ks_go:.3f} | RECALIBRATE if <= {ks_recal:.3f} | else NO-GO")
    print()
    hdr = f"{'cohort':<18}{'n_cells':>9}{'shared':>7}{'mean_ks':>9}{'pred_cov':>9}  verdict"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['cohort']:<18}{r['n_cells']:>9}{r['n_shared_markers']:>7}"
              f"{r['mean_ks']:>9.3f}{r['predicted_coverage']:>9.3f}  {r['verdict']}")

    # Markdown summary (committable).
    _write_doc(args.doc_out, payload)
    print(f"\nWrote {args.out}\nWrote {args.doc_out}")


def _write_doc(doc_out: Path, payload: Dict) -> None:
    law = payload["law"]
    th = payload["thresholds"]
    lines = [
        "# Cohort generalizability gate\n",
        "Label-free deployment gate: predict a cohort's hierarchical ontology "
        "coverage from its marker-distribution KS distance to CRC, **before** any "
        "ground-truth labels exist. Generated by "
        "`scripts/cohort_generalizability_gate.py`.\n",
        f"**Scaling law:** `coverage = {law['intercept']} + ({law['slope']}) * mean_ks`  "
        f"(R^2={law['r2']}, residual std={law['resid_std']}; source: {law['source']}).\n",
        f"**Gate:** GO if `mean_ks <= {th['ks_go_max']}` (pred cov >= {th['go_coverage']}); "
        f"RECALIBRATE if `mean_ks <= {th['ks_recal_max']}` (pred cov >= {th['recal_coverage']}); "
        "else NO-GO.\n",
        "| Cohort | n cells | shared markers | mean KS | predicted coverage | verdict |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for r in payload["cohorts"]:
        ci = r["predicted_coverage_ci95"]
        lines.append(
            f"| {r['cohort']} | {r['n_cells']:,} | {r['n_shared_markers']} | "
            f"{r['mean_ks']:.3f} | {r['predicted_coverage']:.3f} "
            f"[{ci[0]:.2f}, {ci[1]:.2f}] | **{r['verdict']}** |"
        )
    lines.append("")
    lines.append(
        "Interpretation: a NO-GO cohort should not receive leaf-level predictions "
        "from the CRC-calibrated model without in-platform recalibration "
        "(a small labeled audit set), which the recalibration experiment showed "
        "restores coverage to ~0.90.\n"
    )
    doc_out.parent.mkdir(parents=True, exist_ok=True)
    doc_out.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
