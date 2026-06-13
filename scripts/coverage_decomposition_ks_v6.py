#!/usr/bin/env python3
"""v6 coverage-decomposition KS analysis (§6 workshop paper).

For each LODO dataset, compute the Kolmogorov-Smirnov distance between CRC
calibration and deployment per-marker GMM posteriors on shared markers, and
correlate with the observed LODO hierarchical coverage.

v6 variant: reads coverage from v6.0_holdout_<X> results (produced by task
#33 eval). Panel and marker handling is unchanged vs v5 — the NPZ feature
layout (40D GMM posteriors) is shared across v5 and v6 because v6 only
changes the label space, not the feature construction.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp, pearsonr

# --- Release path resolution (see configs/paths.py) ---------------------------
# Artifact root is $PATCH_DATA_DIR or <repo>/data; override via the CLI flags
# below. NO absolute /n/scratch or /home paths are baked in.
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from configs.paths import results_dir, uq_ablation_dir  # noqa: E402

_UQ = uq_ablation_dir()
DEFAULT_CAL = str(_UQ / "crc_cal.npz")
LODO_FILES = {
    "Hoch":    str(_UQ / "hoch_40d.npz"),
    "Liu":     str(_UQ / "liu_40d.npz"),
    "IMMUcan": str(_UQ / "immucan_40d.npz"),
    "Schurch": str(_UQ / "schurch_40d.npz"),
}

DEFAULT_OUT = str(results_dir() / "coverage_decomp_ks_v6.json")


def per_marker_posteriors(npz_path: str):
    d = np.load(npz_path)
    markers = d["universal_markers"].tolist()
    X = d["X_gmm"]
    if X.shape[1] == 160:
        X = X[:, :40]
    elif X.shape[1] != 40:
        raise ValueError(f"{npz_path}: unexpected X_gmm shape {X.shape}")
    present = d["markers_present"].astype(bool) if "markers_present" in d.files else np.ones(40, dtype=bool)
    return markers, X, present


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverage_json", required=True,
                    help="JSON with per-LODO coverage at 90%% ontology target. "
                         "Produced by task #33 eval on the v6 holdout models.")
    ap.add_argument("--cal", default=DEFAULT_CAL)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    with open(args.coverage_json) as fp:
        coverage_bundle = json.load(fp)
    lodo_coverage = coverage_bundle["lodo_ontology_coverage"]  # dict: dataset -> coverage
    target = float(coverage_bundle.get("target_coverage", 0.90))

    cal_markers, cal_X, _ = per_marker_posteriors(args.cal)
    results = {
        "target_coverage": target, "ontology": "v6",
        "method": "Kolmogorov-Smirnov on per-marker GMM posteriors",
    }
    ds_entries = []
    for ds, path in LODO_FILES.items():
        markers, X, present = per_marker_posteriors(path)
        assert markers == cal_markers, "marker vocabulary mismatch"
        shared_idx = np.where(present)[0]
        ks_per_marker = {}
        ks_vals = []
        for i in shared_idx:
            ks = float(ks_2samp(cal_X[:, i], X[:, i]).statistic)
            ks_per_marker[cal_markers[i]] = ks
            ks_vals.append(ks)
        ks_mean = float(np.mean(ks_vals))
        ks_med = float(np.median(ks_vals))
        ks_max = float(np.max(ks_vals))
        if ds not in lodo_coverage:
            print(f"[warn] no coverage for {ds}; skipping correlation row")
            continue
        coverage = float(lodo_coverage[ds])
        ds_entries.append({
            "dataset": ds,
            "n_shared_markers": int(len(shared_idx)),
            "ks_mean": ks_mean, "ks_median": ks_med, "ks_max": ks_max,
            "ontology_coverage": coverage,
            "per_marker_ks": ks_per_marker,
        })
        print(f"{ds:10s}  shared={len(shared_idx):2d}  KS mean={ks_mean:.3f}  "
              f"med={ks_med:.3f}  max={ks_max:.3f}  coverage={coverage:.3f}")

    if len(ds_entries) >= 3:
        ks_means = np.asarray([e["ks_mean"] for e in ds_entries])
        ks_meds = np.asarray([e["ks_median"] for e in ds_entries])
        covs = np.asarray([e["ontology_coverage"] for e in ds_entries])
        r_mean, p_mean = pearsonr(ks_means, covs)
        r_med, p_med = pearsonr(ks_meds, covs)
        print()
        print(f"Pearson r(ks_mean,   coverage) = {r_mean:+.3f}  (p={p_mean:.3f})")
        print(f"Pearson r(ks_median, coverage) = {r_med:+.3f}  (p={p_med:.3f})")
        results["correlation"] = {
            "r_ks_mean_vs_coverage": float(r_mean),
            "p_ks_mean_vs_coverage": float(p_mean),
            "r_ks_median_vs_coverage": float(r_med),
            "p_ks_median_vs_coverage": float(p_med),
        }

    results["datasets"] = ds_entries
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\n[done] wrote {args.out}")


if __name__ == "__main__":
    main()
