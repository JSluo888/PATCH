"""RAPS / APS vs baseline split-conformal ablation (roadmap #3).

Does the adaptive score (APS / RAPS, uncertainty/aps.py) produce smaller
prediction sets than the default `1 - p(true)` split-conformal score at matched
coverage? Run on CRC test using the v5.3 model's class probabilities (flat
17-class — the aps.py predictors are flat; hierarchical integration is a
follow-up, see roadmap risk #4).

For each scorer we report empirical marginal coverage, mean / median |S|, and
singleton rate at alpha=0.10. A win = same-or-better coverage at smaller |S|.

CPU-only. Usage:
    python scripts/raps_ablation_v6.py
    python scripts/raps_ablation_v6.py --test-subsample 50000 --alpha 0.10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# --- Release path resolution (see configs/paths.py) ---------------------------
# Artifact root is $PATCH_DATA_DIR or <repo>/data; override individual paths via
# the CLI flags below. NO absolute /n/scratch or /home paths are baked in.
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from configs.paths import models_dir, results_dir, uq_ablation_dir  # noqa: E402
from uncertainty.aps import make_aps, make_raps  # noqa: E402

_UQ = uq_ablation_dir()
CAL_NPZ = _UQ / "crc_cal.npz"
TEST_NPZ = _UQ / "crc_test.npz"
MODEL_PATH = str(models_dir() / "v5.3_nimbus" / "xgboost" / "model.ubj")
OUT_JSON = results_dir() / "raps_ablation_v6.json"
DOC_OUT = results_dir() / "raps_ablation_v6.md"


def load(path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    d = np.load(path, allow_pickle=True)
    tt = [str(t) for t in d["target_types"]]
    return d["X_gmm"].astype(np.float32), d["y"].astype(np.int64), tt


def split_cp_baseline(
    p_cal: np.ndarray, y_cal: np.ndarray, p_test: np.ndarray, alpha: float,
) -> List[List[int]]:
    """Marginal split conformal with the default 1 - p(true) score."""
    scores = 1.0 - p_cal[np.arange(len(y_cal)), y_cal]
    n = len(scores)
    q_level = np.ceil((n + 1) * (1.0 - alpha)) / n
    q = float(np.quantile(scores, min(q_level, 1.0), method="higher"))
    keep = (1.0 - p_test) <= q  # (n_test, K) boolean
    sets = []
    for i in range(p_test.shape[0]):
        s = list(np.where(keep[i])[0])
        sets.append(s if s else [int(np.argmax(p_test[i]))])
    return sets


def metrics(sets: List[List[int]], y_true: np.ndarray, as_str: bool, class_ids: List[str]) -> Dict:
    """Coverage + set-size stats. `sets` may be lists of ints or class strings."""
    sizes = np.array([len(s) for s in sets])
    if as_str:
        truth = [class_ids[int(y)] for y in y_true]
        covered = np.array([truth[i] in sets[i] for i in range(len(sets))])
    else:
        covered = np.array([int(y_true[i]) in sets[i] for i in range(len(sets))])
    return {
        "coverage": float(covered.mean()),
        "mean_set_size": float(sizes.mean()),
        "median_set_size": float(np.median(sizes)),
        "singleton_rate": float((sizes == 1).mean()),
        "max_set_size": int(sizes.max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--test-subsample", type=int, default=50000,
                    help="subsample test cells for the O(nK^2) APS scoring (0 = all)")
    ap.add_argument("--k-reg", type=int, nargs="*", default=[3, 5, 8])
    ap.add_argument("--lam", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cal", type=Path, default=CAL_NPZ, help="CRC calibration npz")
    ap.add_argument("--test", type=Path, default=TEST_NPZ, help="CRC test npz")
    ap.add_argument("--model", type=str, default=MODEL_PATH, help="XGBoost .ubj booster")
    args = ap.parse_args()

    import xgboost as xgb

    X_cal, y_cal, tt = load(args.cal)
    X_test, y_test, tt_test = load(args.test)
    assert tt == tt_test, "cal/test target_types mismatch"
    K = len(tt)

    booster = xgb.Booster()
    booster.load_model(args.model)
    P_cal = booster.predict(xgb.DMatrix(X_cal))
    P_test = booster.predict(xgb.DMatrix(X_test))

    # Subsample test for the per-(cell,class) APS scoring loop.
    if args.test_subsample and args.test_subsample < len(y_test):
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(y_test), args.test_subsample, replace=False)
        P_test_s, y_test_s = P_test[idx], y_test[idx]
    else:
        P_test_s, y_test_s = P_test, y_test

    print(f"[raps] cal={len(y_cal)} test={len(y_test_s)} (of {len(y_test)}) K={K} alpha={args.alpha}")
    results: Dict[str, Dict] = {}

    # Baseline split-CP (1 - p).
    base_sets = split_cp_baseline(P_cal, y_cal, P_test_s, args.alpha)
    results["split_cp_1minusp"] = metrics(base_sets, y_test_s, as_str=False, class_ids=tt)

    # APS.
    aps = make_aps(alpha=args.alpha)
    aps.calibrate(P_cal, y_cal)
    aps_sets = aps.predict_set(P_test_s, tt)
    results["aps"] = metrics(aps_sets, y_test_s, as_str=True, class_ids=tt)

    # RAPS at several k_reg.
    for k in args.k_reg:
        raps = make_raps(alpha=args.alpha, k_reg=k, lam=args.lam)
        raps.calibrate(P_cal, y_cal)
        raps_sets = raps.predict_set(P_test_s, tt)
        results[f"raps_k{k}_lam{args.lam}"] = metrics(raps_sets, y_test_s, as_str=True, class_ids=tt)

    payload = {
        "alpha": args.alpha, "n_cal": int(len(y_cal)), "n_test": int(len(y_test_s)),
        "n_classes": K, "model": str(args.model), "note": "flat 17-class conformal",
        "variants": results,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))

    # Report.
    hdr = f"{'variant':<24}{'coverage':>10}{'mean|S|':>10}{'med|S|':>8}{'singl':>8}{'max':>6}"
    print("\n=== RAPS/APS vs split-CP (CRC test, flat 17-class) ===")
    print(hdr); print("-" * len(hdr))
    lines = ["# RAPS/APS vs baseline split-conformal (CRC test, flat 17-class)\n",
             f"alpha={args.alpha}, target coverage {1-args.alpha:.2f}, n_test={len(y_test_s)}, "
             f"model=v5.3_nimbus. A win = matched coverage at smaller mean |S|.\n",
             "| variant | coverage | mean \\|S\\| | median \\|S\\| | singleton rate | max \\|S\\| |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, m in results.items():
        print(f"{name:<24}{m['coverage']:>10.4f}{m['mean_set_size']:>10.3f}"
              f"{m['median_set_size']:>8.1f}{m['singleton_rate']:>8.3f}{m['max_set_size']:>6d}")
        lines.append(f"| {name} | {m['coverage']:.4f} | {m['mean_set_size']:.3f} | "
                     f"{m['median_set_size']:.1f} | {m['singleton_rate']:.3f} | {m['max_set_size']} |")
    lines.append("")
    DOC_OUT.write_text("\n".join(lines))
    print(f"\n[raps] wrote {OUT_JSON}\n[raps] wrote {DOC_OUT}")


if __name__ == "__main__":
    main()
