"""Centralised, overridable data-artifact paths for the PATCH release.

The experiment scripts in ``scripts/`` consume pre-computed caches (``.npz``
feature dumps, trained XGBoost ``.ubj`` boosters, clinical tables). In the
research monorepo these lived under absolute cluster paths; for the public
release every such path is resolved through this module so a reviewer can
point the scripts at a local copy of the artifact bundle.

Resolution order for the artifact root:
    1. ``$PATCH_DATA_DIR`` environment variable, if set.
    2. ``<repo_root>/data`` (the default; create it and drop the Zenodo
       bundle here, or symlink it to your artifact directory).

Override examples
-----------------
    export PATCH_DATA_DIR=/path/to/patch_artifacts
    python scripts/raps_ablation_v6.py            # picks up $PATCH_DATA_DIR

or per-script via the CLI flags each script exposes (``--cal``, ``--model``,
``--uq-dir``, ...). The artifact bundle layout expected under the root is::

    <root>/uq_ablation/crc_cal.npz
    <root>/uq_ablation/crc_test.npz
    <root>/uq_ablation/<cohort>_40d.npz
    <root>/models/v5.3_nimbus/xgboost/model.ubj
    <root>/models/v6.0_holdout_<Cohort>/xgboost/model.ubj
    <root>/clinical/orion_clinical.xlsx
    <root>/downstream/orion_recurrence/sample_proportions_v5_3.csv

The caches/models are NOT shipped in this repo (they are large); see the
README "Data & model artifacts" section for how to obtain them.
"""
from __future__ import annotations

import os
from pathlib import Path

# Repository root = parent of this ``configs/`` directory.
REPO_ROOT = Path(__file__).resolve().parents[1]


def data_dir() -> Path:
    """Return the artifact root (``$PATCH_DATA_DIR`` or ``<repo>/data``)."""
    env = os.environ.get("PATCH_DATA_DIR")
    return Path(env) if env else (REPO_ROOT / "data")


# Convenience sub-roots used across scripts.
def uq_ablation_dir() -> Path:
    return data_dir() / "uq_ablation"


def models_dir() -> Path:
    return data_dir() / "models"


def downstream_dir() -> Path:
    return data_dir() / "downstream"


def results_dir() -> Path:
    """Writable directory for script outputs (created on demand)."""
    out = REPO_ROOT / "results"
    out.mkdir(parents=True, exist_ok=True)
    return out
