# PATCH — Panel-Aware Hierarchical Conformal Cell Typing

> Code release for the paper *“PATCH: Panel-Aware Hierarchical Conformal Cell
> Typing for Spatial Proteomics under Marker-Panel Shift.”*

**PATCH** is a deployment-aware uncertainty-quantification stack for single-cell
spatial-proteomics cell typing. It combines (i) a *panel-aware,
tree-hierarchical conformal predictor* that lets conformal calibration decide
the **resolution** of each call — returning the finest ontology node the marker
panel can confidently support and climbing the lineage→subtype→fine-type tree
when it cannot; (ii) a *label-free Kolmogorov–Smirnov shift diagnostic* that
predicts a new cohort's coverage from its marker-distribution distance to the
calibration platform and emits a GO / RECALIBRATE / NO-GO deployment verdict
before any ground-truth labels exist; and (iii) a *constrained VLM verifier*
that may only drop a cell (artefact) or shrink a conformal set to a committed
leaf already inside it — never expand it — so the `1-α` coverage guarantee is
preserved by construction.

This repository is a curated, self-contained subset of the research codebase
that reproduces the core method and the headline results. It is **not** the full
monorepo.

---

## Repository map

```
.                                    # repo root
├── uncertainty/                     # The conformal method (core contribution)
│   ├── hierarchical_conformal.py    #   panel-aware tree-hierarchical conformal predictor
│   ├── aps.py                       #   APS / RAPS adaptive-prediction-set scorers
│   ├── conformal_predictor.py       #   flat split-conformal baseline
│   ├── vlm_set_refiner.py           #   coverage-preserving VLM set refiner (DROP/REFINE/FALLBACK)
│   └── comparators/                 #   faithful ports of competing CP methods
│       ├── hcc_port.py              #     HCC (Principato et al., arXiv:2508.13288)
│       └── scconform_port.py        #     scConform (Wojnowska et al., arXiv:2410.23786)
├── configs/                         # Pure-Python config (no Hydra runtime needed here)
│   ├── ontology_v6.py               #   22-leaf standard-immunology ontology + panel manifests
│   ├── marker_celltype_ontology.py  #   17-class ontology + panel→finest-type resolution
│   ├── cell_type_rules.py           #   deterministic marker→type rules (baseline)
│   └── paths.py                     #   overridable artifact-path resolution ($PATCH_DATA_DIR)
├── scripts/
│   ├── cohort_generalizability_gate.py   # label-free KS→coverage deployment gate
│   ├── coverage_decomposition_ks_v6.py   # KS-distance ↔ LODO coverage decomposition
│   ├── raps_ablation_v6.py               # APS/RAPS vs split-CP set-size ablation
│   ├── downstream_orion_survival_cr.py   # Cox-adjusted + KM survival (Orion CRC)
│   ├── patch_vs_competitors.py          # PATCH vs HCC / scConform benchmark
│   └── vlm_audit/                        # constrained VLM verifier audit harness
│       ├── run_audit.py                  #   driver (Gemini / Claude / GPT / OSS vLLM)
│       ├── score_audit.py                #   compute the 4 audit numbers the paper reports
│       ├── reanalyze_protocols.py        #   selective-conformal re-analysis of audit results
│       └── README.md                     #   protocol details
├── modal_apps/
│   ├── qwen3vl_vlm.py                # OSS VLM server (Qwen3-VL-32B-FP8) on Modal, OpenAI-compatible
│   ├── qwen3vl_audit.py              # in-container 100-cell audit (localhost; bypasses Modal cold-start edge)
│   └── README.md
├── results/
│   └── vlm_eval/
│       └── vlm_audit_results_qwen3vl32b.json  # scored Qwen3-VL-32B-FP8 audit (shipped)
├── tests/                           # offline unit tests for the released code
├── requirements.txt
├── LICENSE                          # MIT
└── README.md
```

---

## Install

```bash
# conda (recommended)
conda create -n patch python=3.11 -y
conda activate patch
pip install -r requirements.txt

# or plain pip into an existing environment
pip install -r requirements.txt
```

The **core** method (`uncertainty/`) needs only `numpy` + `scipy`. The other
dependency groups (xgboost, lifelines/pandas/matplotlib, pyyaml, openai/modal)
are required only for specific reproduce steps — see `requirements.txt`.

Quick check that the install works:

```bash
python -c "import sys; sys.path.insert(0,'.'); \
  import uncertainty.hierarchical_conformal, uncertainty.vlm_set_refiner, uncertainty.aps; \
  print('PATCH import OK')"

python -m pytest tests/ -q          # 45 offline tests, no data needed
```

---

## Data & model artifacts

The reproduce steps consume pre-computed caches that are **too large to ship in
git** and are **not** included here:

| Artifact | Path under the artifact root | Used by |
|---|---|---|
| CRC calibration / test feature caches | `uq_ablation/crc_cal.npz`, `uq_ablation/crc_test.npz` | RAPS ablation, gate |
| Per-cohort 40-D GMM-posterior caches | `uq_ablation/<cohort>_40d.npz` | gate, KS decomposition |
| Per-cohort cal/test caches | `uq_ablation/<cohort>_{cal,test}.npz` | PATCH-vs-competitors |
| Trained XGBoost boosters | `models/v5.3_nimbus/...`, `models/v6.0_holdout_<Cohort>/...` | RAPS, competitors |
| KS↔coverage fit table | `uq_ablation/ks_coverage_with_hartmann.json` | gate `--refit` |
| Orion clinical table | `clinical/orion_clinical.xlsx` | survival |
| Per-sample cell-type proportions | `downstream/orion_recurrence/sample_proportions_v5_3.csv` | survival |
| 100-cell VLM input bundle | `vlm_100cells/` | VLM audit |

**These are available on request and will be deposited on Zenodo
(DOI: `10.5281/zenodo.XXXXXXX` — placeholder, to be minted for the
camera-ready).** Point the scripts at your local copy by either:

```bash
export PATCH_DATA_DIR=/path/to/patch_artifacts     # resolves <root>/uq_ablation, <root>/models, ...
```

or by passing the explicit `--cal`, `--test`, `--model`, `--uq-dir`,
`--clinical`, `--prop` flags each script exposes. Outputs are written under
`<repo>/results/`. No absolute lab paths are baked into the code.

---

## Reproduce

All commands run from inside this `release/` directory.

### 1. Cohort generalizability gate (label-free deployment triage)

Predicts each cohort's hierarchical ontology coverage from its marker-posterior
KS distance to the CRC calibration set, then emits GO / RECALIBRATE / NO-GO —
**without using any labels**.

```bash
python scripts/cohort_generalizability_gate.py                 # score all cached cohorts
python scripts/cohort_generalizability_gate.py --cohorts hoch_40d liu_40d
python scripts/cohort_generalizability_gate.py --refit         # re-fit the KS→coverage law first
```
Needs: `uq_ablation/crc_cal.npz` + `uq_ablation/<cohort>_40d.npz`
(+ `ks_coverage_with_hartmann.json` for `--refit`). Writes
`results/cohort_generalizability_gate.{json,md}`.

The complementary KS-decomposition analysis (Pearson r of KS distance vs.
observed LODO coverage) is:

```bash
python scripts/coverage_decomposition_ks_v6.py --coverage_json <lodo_coverage.json>
```

### 2. RAPS ablation (adaptive vs. split-conformal set sizes)

Does APS / RAPS produce smaller prediction sets than the default `1 - p(true)`
split-conformal score at matched coverage, on CRC test?

```bash
python scripts/raps_ablation_v6.py                                  # defaults: alpha=0.10, 50k subsample
python scripts/raps_ablation_v6.py --alpha 0.10 --k-reg 3 5 8 --test-subsample 0
```
Needs: `uq_ablation/crc_{cal,test}.npz` + `models/v5.3_nimbus/xgboost/model.ubj`.
Writes `results/raps_ablation_v6.{json,md}`. CPU-only.

### 3. Orion CRC survival analysis

Median-split Kaplan–Meier + log-rank and a multivariable Cox PH (adjusted for
age and stage, with Harrell's c) for each immune stratifier (CTL, Treg,
T-lineage, Immune).

```bash
python scripts/downstream_orion_survival_cr.py
python scripts/downstream_orion_survival_cr.py --clinical <clin.xlsx> --prop <proportions.csv>
```
Needs: `clinical/orion_clinical.xlsx` + `downstream/.../sample_proportions_v5_3.csv`.
Writes `results/orion_survival_cr.{json,md}` and `results/fig_km_ctl_cr.{pdf,png}`.

### 4. Conformal-variant benchmark (PATCH vs. competitors)

Runs the HCC and scConform ports (and a Laplacian-smoothed scConform variant)
per cohort, reporting ontology coverage, mean |S|, and leaf fraction. PATCH's
own panel-aware numbers come from the hierarchical predictor in
`uncertainty/hierarchical_conformal.py`.

```bash
python scripts/patch_vs_competitors.py                      # all cohorts
python scripts/patch_vs_competitors.py --cohorts crc hoch hartmann --alpha 0.10
```
Needs: `uq_ablation/<cohort>_{cal,test}.npz` + the matching XGBoost boosters.
Writes `results/comparators/patch_vs_competitors_<cohort>.json`.

### 5. VLM selective-conformal re-analysis

The constrained verifier (`uncertainty/vlm_set_refiner.py`) only DROPs artefacts
or REFINEs a set to a committed in-set leaf; it never expands. The audit harness
drives a VLM over the 100-cell bundle, scores it, and re-analyzes the
selective-conformal coverage on the retained sub-population.

```bash
# (a) Run the audit. Choose a model; the SDK is imported lazily per --model.
export OPENAI_API_KEY=...        # or ANTHROPIC_API_KEY / GOOGLE_API_KEY for those models
python scripts/vlm_audit/run_audit.py --model gpt-5 \
    --limit 100 --out vlm_responses --cells-dir vlm_100cells

# (a') Fully open-source path — no proprietary key, no public endpoint needed.
#      Runs the strongest single-A100 Qwen3-VL (Qwen3-VL-32B-Instruct-FP8) and
#      the 100-cell loop INSIDE one Modal container against localhost:8000, so it
#      never touches Modal's public web edge (whose ~6-min FP8 cold-start 303
#      redirect otherwise hangs an external client). One cold start, 100 calls.
modal run modal_apps/qwen3vl_audit.py --limit 100   # -> vlm_responses_qwen3vl32b/

# (b) Score the responses (the 4 numbers the paper reports).
python scripts/vlm_audit/score_audit.py --responses vlm_responses/ \
    --out results/vlm_audit_results.json

# (c) Selective-conformal re-analysis (coverage on retained sub-population).
python scripts/vlm_audit/reanalyze_protocols.py --audit results/vlm_audit_results.json
```
Needs: the `vlm_100cells/` input bundle and a VLM endpoint/key. See
`scripts/vlm_audit/README.md` and `modal_apps/README.md`. The pure-logic refiner
is fully covered by `tests/test_vlm_set_refiner.py` and runs offline.

#### VLM verifier audit — measured numbers (100-cell stratified stress test)

Constrained QC verdict in `S ∪ {abstain, artefact}`, scored ontology-aware. The
**open-weights Qwen3-VL-32B-Instruct-FP8** result is shipped at
`results/vlm_eval/vlm_audit_results_qwen3vl32b.json`.

| Verifier | Hit | Wrong | Artefact | Abstain | Artefact recall | Abstention |
|---|---:|---:|---:|---:|---:|---:|
| Claude Opus 4.7 | 28 | 23 | 48 | 1 | 0.867 | 0.01 |
| GPT-5.5 (xHigh) | 35 | 25 | 24 | 16 | 0.333 | 0.16 |
| **Qwen3-VL-32B-FP8** (open) | **33** | **54** | **12** | **1** | **0.133** | ~0 |

Pairwise exact-verdict agreement: Claude–GPT 0.55, Claude–Qwen3 0.42, GPT–Qwen3 0.46.

**Takeaway.** The open-weights verifier **matches the proprietary models on raw
ontology accuracy (33% vs GPT 35%)** — so the QC layer is not gated on a closed
model — but it is **less calibrated as a verifier** (≈0% deliberate abstention,
fewest artefact flags → highest wrong-label rate). That calibration gap is
precisely what the constrained conformal refiner (`uncertainty/vlm_set_refiner.py`)
enforces: it can only DROP an artefact or REFINE to an in-set leaf, never expand,
so the coverage guarantee holds for any verifier.

---

## Tests

```bash
python -m pytest tests/ -q
```

- `test_vlm_set_refiner.py` — the four structural branches of the verifier and a
  synthetic coverage-preservation experiment (offline).
- `test_cohort_gate.py` — the KS→coverage scaling-law math and GO/RECALIBRATE/
  NO-GO thresholds (offline).
- `test_hierarchical_conformal_chars.py` — characterization (byte-identical
  threshold reference) + pluggable-score-fn hook.
- `test_comparator_ports.py` — HCC / scConform port correctness on a toy ontology.

All four are offline; none touch the artifact bundle.

---

## Citing

If you use PATCH, please cite the paper (camera-ready citation forthcoming) and
this code release. The competitor ports credit their original authors in each
module's docstring (HCC: arXiv:2508.13288; scConform: arXiv:2410.23786; APS/RAPS:
arXiv:2006.02544, arXiv:2009.14193).

Licensed under the MIT License — see `LICENSE`.
