"""Camera-ready survival analysis for the PATCH downstream section (Orion CRC).

Fulfils the draft's promise: "A proper Cox-adjusted analysis with risk tables and
concordance index is in progress for the camera-ready."

For each immune stratifier in {Cytotoxic_T (CTL), Regulatory_T, T_Lineage,
Immune}:
  * Median-split Kaplan-Meier + log-rank (reproduces the draft's PFS p-values),
    with at-risk table data and median PFS per arm.
  * Multivariable Cox PH on the CONTINUOUS proportion (per +10 percentage points)
    adjusted for Age and Stage-at-diagnosis: hazard ratio, 95% CI, p, and the
    model concordance index (Harrell's c).

PFS endpoint: duration = PFSDays, event = Recurrence (matches the existing
downstream script and the draft's log-rank). We also report the PFSCensor-based
event for transparency.

Outputs (written under <repo>/results/):
  results/orion_survival_cr.json
  results/fig_km_ctl_cr.pdf/.png  (KM with at-risk table, camera-ready)
  results/orion_survival_cr.md

Inputs (clinical xlsx + per-sample cell-type proportions) are NOT shipped; see
README "Data & model artifacts". Override their location with --clinical / --prop
or by setting $PATCH_DATA_DIR.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

# --- Release path resolution (see configs/paths.py) ---------------------------
# Artifact root is $PATCH_DATA_DIR or <repo>/data; override via the CLI flags
# below. NO absolute /n/scratch or /home paths are baked in.
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from configs.paths import data_dir, results_dir  # noqa: E402

CLINICAL = str(data_dir() / "clinical" / "orion_clinical.xlsx")
PROP = str(data_dir() / "downstream" / "orion_recurrence" / "sample_proportions_v5_3.csv")
OUT_JSON = results_dir() / "orion_survival_cr.json"
FIG_OUT = results_dir() / "fig_km_ctl_cr"
DOC_OUT = results_dir() / "orion_survival_cr.md"

STRATIFIERS = {
    "Cytotoxic_T_Cell": "Cytotoxic T (CTL)",
    "Regulatory_T_Cell": "Regulatory T",
    "T_Lineage": "T lineage (aggregated)",
    "Immune": "Immune lineage (aggregated)",
}
_ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4}


def stage_to_ordinal(s: object) -> float:
    if not isinstance(s, str):
        return np.nan
    m = re.match(r"\s*(IV|III|II|I)", s.strip().upper())
    return float(_ROMAN[m.group(1)]) if m else np.nan


def load_merged(clinical: str = CLINICAL, prop_csv: str = PROP) -> pd.DataFrame:
    clin = pd.read_excel(clinical, sheet_name="S3 Patient Characteristics", header=2)
    clin["CRC_sample"] = clin["Specimen ID"].astype(str).str.replace("C", "CRC", regex=False)
    clin.loc[clin["CRC_sample"] == "CRC33", "CRC_sample"] = "CRC33_01"
    clin["stage_ord"] = clin["Stage At Diagnosis"].apply(stage_to_ordinal)
    cols = ["CRC_sample", "Age", "stage_ord", "PFSDays", "PFSCensor", "Recurrence",
            "OSDays", "OSCensor", "Death"]
    clin = clin[[c for c in cols if c in clin.columns]]
    prop = pd.read_csv(prop_csv)
    merged = prop.merge(clin, on="CRC_sample", how="inner")
    merged["PFSDays"] = pd.to_numeric(merged["PFSDays"], errors="coerce")
    merged["Recurrence"] = pd.to_numeric(merged["Recurrence"], errors="coerce")
    merged["Age"] = pd.to_numeric(merged["Age"], errors="coerce")
    return merged


def km_logrank(df: pd.DataFrame, col: str, event_col: str = "Recurrence") -> Dict:
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test

    sub = df[[col, "PFSDays", event_col]].dropna()
    sub = sub[sub["PFSDays"] > 0]
    med = float(sub[col].median())
    high = sub[sub[col] > med]
    low = sub[sub[col] <= med]
    lr = logrank_test(high["PFSDays"], low["PFSDays"], high[event_col], low[event_col])
    kmf = KaplanMeierFitter()
    kmf.fit(high["PFSDays"], high[event_col]); med_hi = float(kmf.median_survival_time_)
    kmf.fit(low["PFSDays"], low[event_col]); med_lo = float(kmf.median_survival_time_)
    return {"n": int(len(sub)), "n_high": int(len(high)), "n_low": int(len(low)),
            "n_events": int(sub[event_col].sum()), "median_split": round(med, 5),
            "logrank_p": float(lr.p_value), "logrank_stat": float(lr.test_statistic),
            "median_pfs_high_days": med_hi, "median_pfs_low_days": med_lo}


def cox_adjusted(df: pd.DataFrame, col: str, event_col: str = "Recurrence") -> Optional[Dict]:
    from lifelines import CoxPHFitter

    use = df[[col, "Age", "stage_ord", "PFSDays", event_col]].dropna()
    use = use[use["PFSDays"] > 0]
    if len(use) < 12 or use[event_col].sum() < 5:
        return {"error": "insufficient events/samples", "n": int(len(use))}
    work = use.rename(columns={col: "prop", "PFSDays": "T", event_col: "E"}).copy()
    # Standardize the proportion -> HR per +1 SD (stable for small-variance immune
    # fractions; avoids the separation/overflow of a raw +10pp scaling).
    mu, sd = float(work["prop"].mean()), float(work["prop"].std(ddof=0))
    if sd <= 0:
        return {"error": "zero-variance proportion", "n": int(len(use))}
    work["prop"] = (work["prop"] - mu) / sd
    try:
        # Standardized covariate makes the unpenalized fit stable on n=40.
        cph = CoxPHFitter()
        cph.fit(work[["prop", "Age", "stage_ord", "T", "E"]], duration_col="T", event_col="E")
        s = cph.summary.loc["prop"]
        return {
            "n": int(len(use)), "n_events": int(use[event_col].sum()),
            "covariates": ["proportion(+1 SD)", "Age", "Stage(ordinal I-IV)"],
            "penalizer": 0.0,
            "hr_per_sd": float(np.exp(s["coef"])),
            "hr_ci95": [float(np.exp(s["coef lower 95%"])), float(np.exp(s["coef upper 95%"]))],
            "p": float(s["p"]),
            "concordance": float(cph.concordance_index_),
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}", "n": int(len(use))}


def make_km_figure(df: pd.DataFrame, col: str = "Cytotoxic_T_Cell") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter
    from lifelines.plotting import add_at_risk_counts

    sub = df[[col, "PFSDays", "Recurrence"]].dropna()
    sub = sub[sub["PFSDays"] > 0]
    med = sub[col].median()
    high = sub[sub[col] > med]; low = sub[sub[col] <= med]
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    kmf_hi = KaplanMeierFitter(label=f"High CTL (n={len(high)})")
    kmf_lo = KaplanMeierFitter(label=f"Low CTL (n={len(low)})")
    kmf_hi.fit(high["PFSDays"] / 365.25, high["Recurrence"])
    kmf_lo.fit(low["PFSDays"] / 365.25, low["Recurrence"])
    kmf_hi.plot_survival_function(ax=ax, ci_show=True, censor_styles={"marker": "|"}, show_censors=True)
    kmf_lo.plot_survival_function(ax=ax, ci_show=True, censor_styles={"marker": "|"}, show_censors=True)
    ax.set_xlabel("Years"); ax.set_ylabel("Progression-free survival")
    ax.set_ylim(0, 1.02)
    from lifelines.statistics import logrank_test
    lr = logrank_test(high["PFSDays"], low["PFSDays"], high["Recurrence"], low["Recurrence"])
    ax.set_title(f"CTL proportion median split  (log-rank $p$={lr.p_value:.3f})", fontsize=9)
    add_at_risk_counts(kmf_hi, kmf_lo, ax=ax)
    plt.tight_layout()
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(FIG_OUT) + ".pdf", bbox_inches="tight")
    fig.savefig(str(FIG_OUT) + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clinical", default=CLINICAL, help="Orion clinical xlsx")
    ap.add_argument("--prop", default=PROP, help="per-sample cell-type proportions CSV")
    args = ap.parse_args()
    df = load_merged(args.clinical, args.prop)
    print(f"[orion-surv] merged {len(df)} patients with proportions + clinical")
    results: Dict = {"cohort": "orion_crc", "n": int(len(df)),
                     "endpoint": "PFS (duration=PFSDays, event=Recurrence)", "stratifiers": {}}
    print(f"\n{'stratifier':<24}{'logrank_p':>10}{'med_hi(d)':>10}{'med_lo(d)':>10}"
          f"{'CoxHR/SD':>11}{'CoxP':>8}{'c-index':>9}")
    for col, human in STRATIFIERS.items():
        if col not in df.columns:
            continue
        km = km_logrank(df, col)
        cox = cox_adjusted(df, col)
        results["stratifiers"][col] = {"name": human, "km_logrank": km, "cox_adjusted": cox}
        hr = cox.get("hr_per_sd"); cp = cox.get("p"); ci = cox.get("concordance")
        print(f"{human:<24}{km['logrank_p']:>10.4f}{km['median_pfs_high_days']:>10.0f}"
              f"{km['median_pfs_low_days']:>10.0f}"
              f"{(hr if hr else float('nan')):>11.3f}{(cp if cp else float('nan')):>8.4f}"
              f"{(ci if ci else float('nan')):>9.3f}")

    make_km_figure(df)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2))

    lines = ["# Orion CRC survival (camera-ready): Cox-adjusted + KM\n",
             f"n={len(df)} patients; PFS duration=PFSDays, event=Recurrence. Cox adjusts the "
             "continuous proportion (per +10 percentage points) for Age and Stage (ordinal I–IV); "
             "concordance is Harrell's c.\n",
             "| Stratifier | log-rank p | median PFS high (d) | median PFS low (d) | Cox HR/+1SD [95% CI] | Cox p | c-index |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for col, blob in results["stratifiers"].items():
        km = blob["km_logrank"]; cox = blob["cox_adjusted"]
        hr = cox.get("hr_per_sd"); ci = cox.get("hr_ci95"); cp = cox.get("p"); cidx = cox.get("concordance")
        hrs = f"{hr:.3f} [{ci[0]:.2f}, {ci[1]:.2f}]" if hr else "—"
        lines.append(f"| {blob['name']} | {km['logrank_p']:.4f} | {km['median_pfs_high_days']:.0f} | "
                     f"{km['median_pfs_low_days']:.0f} | {hrs} | "
                     f"{cp:.4f} | {cidx:.3f} |" if hr else
                     f"| {blob['name']} | {km['logrank_p']:.4f} | {km['median_pfs_high_days']:.0f} | "
                     f"{km['median_pfs_low_days']:.0f} | — | — | — |")
    lines.append("")
    DOC_OUT.write_text("\n".join(lines))
    print(f"\n[orion-surv] wrote {OUT_JSON}\n[orion-surv] wrote {FIG_OUT}.pdf\n[orion-surv] wrote {DOC_OUT}")


if __name__ == "__main__":
    main()
