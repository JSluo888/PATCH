"""
Cell Type Rules - Single Source of Truth

Tier 1 lineage classification rules for Pheno (CODEX) and CRC (Orion) datasets.
Rules auto-skip when required markers are absent from dataset.
"""

from typing import List, Dict, Tuple, Optional
import re

# ============================================================================
# Valid Tier 1 Cell Types (17 + Unassigned + Artifact_Doublet)
# ============================================================================

VALID_CELL_TYPES = [
    # Dendritic / Myeloid
    "Dendritic_Cell",
    "Neutrophil",
    # T cell lineage
    "NKT_Cell",
    "Regulatory_T_Cell",
    "Helper_T_Cell",
    "Cytotoxic_T_Cell",
    "T_Cell_DP",       # CD4+CD8+ double-positive
    "T_Cell",          # Generic CD3+
    # Other lymphoid
    "B_Cell",
    "NK_Cell",
    # Macrophages
    "Macrophage_M2",   # CD163+ first (more specific)
    "Macrophage_M1",   # CD68+CD163-
    # Generic immune
    "Immune_Other",
    # Non-immune
    "Epithelial",
    "Endothelial",
    "Stromal",
    # Fallback categories (hierarchical)
    "Unassigned",
    # Low-confidence fallback types (highest posterior assignment)
    "Epithelial_Low",  # Weak PanCK/Ecadherin signal
    "Endothelial_Low", # Weak CD31 signal
    "Stromal_Low",     # Weak aSMA signal
    "T_Cell_Low",      # Weak CD3e signal
    "B_Cell_Low",      # Weak CD20 signal
    "Macrophage_Low",  # Weak CD68/CD163 signal
    "Immune_Low",      # CD45+ but no specific lineage
    "Unknown",         # All markers very low
    # QC
    "Artifact_Doublet",
]

# ============================================================================
# Core Markers per Dataset
# ============================================================================

# Pheno (box* samples) - 40 markers, metastatic breast cancer
PHENO_MARKERS = [
    "DAPI", "PanCK", "Ki67", "aSMA", "CAIX", "CD11c", "CD163", "CD20", "CD25",
    "CD28", "CD31", "CD3e", "CD4", "CD44", "CD45", "CD45RO", "CD56", "CD66b",
    "CD68", "CD8", "CK18", "CK5", "DCLAMP", "ER", "FAP", "FOXP3", "GZMB",
    "HER2", "HLAABC", "HLADR", "MCM2", "P21", "P27", "PD1", "PDL1", "PR",
    "Podoplanin", "RB1", "TCF7",
]

# CRC (CRC* samples) - 19 markers, colorectal cancer
CRC_MARKERS = [
    "Hoechst", "AF1", "CD31", "CD45", "CD68", "Argo550", "CD4", "FOXP3",
    "CD8", "CD45RO", "CD20", "PDL1", "CD3e", "CD163", "Ecadherin", "PD1",
    "Ki67", "PanCK", "aSMA",
]

# ============================================================================
# Marker Name Aliases (raw name -> canonical name)
# ============================================================================

MARKER_ALIASES = {
    # CRC aliases
    "CD3": "CD3e",
    "CD8a": "CD8",
    "E-cadherin": "Ecadherin",
    "Pan-CK": "PanCK",
    "SMA": "aSMA",
    # Pheno aliases
    "CD20-H1": "CD20",
    "CD45-RO": "CD45RO",
    "DC-LAMP": "DCLAMP",
    "HLA-ABC": "HLAABC",
    "HLA-DR": "HLADR",
    "PD-1": "PD1",
    "PD-L1": "PDL1",
    # Common variations
    "DAPI-01": "DAPI",
    "DAPI-03": "DAPI",
}

def resolve_marker(name: str) -> str:
    """Resolve marker alias to canonical name."""
    return MARKER_ALIASES.get(name, name)

# ============================================================================
# Tier 1 Classification Rules
#
# Format: (priority, cell_type, required_positive, required_negative, notes)
# Rules are evaluated in priority order (1 first). First match wins.
# Rules auto-skip if ANY required marker is absent from the dataset.
# ============================================================================

CLASSIFICATION_RULES: List[Tuple[int, str, List[str], List[str], str]] = [
    # --- Dendritic cells (Pheno only - requires CD11c, HLADR, DCLAMP) ---
    (1,  "Dendritic_Cell",    ["CD45", "CD11c", "HLADR"],           ["CD68", "CD163", "CD3e"],
         "CD11c alone insufficient - HLA-DR required for DC specificity"),
    (2,  "Dendritic_Cell",    ["CD45", "DCLAMP"],                   [],
         "DC-LAMP marks mature DCs"),

    # --- Neutrophils (Pheno only - requires CD66b) ---
    (3,  "Neutrophil",        ["CD45", "CD66b"],                    [],
         ""),

    # --- NKT cells (Pheno only - requires CD56) ---
    (4,  "NKT_Cell",          ["CD45", "CD56", "CD3e"],             [],
         "CD56+CD3e+ = NKT"),

    # --- Regulatory T cells ---
    (5,  "Regulatory_T_Cell", ["CD45", "CD3e", "CD4", "FOXP3", "CD25"], [],
         "CD25 improves specificity; Pheno only for full confidence"),
    (6,  "Regulatory_T_Cell", ["CD45", "CD3e", "CD4", "FOXP3"],     [],
         "CRC fallback - FOXP3 alone, lower confidence"),

    # --- Helper T cells ---
    (7,  "Helper_T_Cell",     ["CD45", "CD3e", "CD4"],              ["FOXP3"],
         "CD4+ T cells excluding Tregs"),

    # --- Cytotoxic T cells ---
    (8,  "Cytotoxic_T_Cell",  ["CD45", "CD3e", "CD8"],              [],
         ""),

    # --- Double-positive T cells (CD4+CD8+) ---
    (9,  "T_Cell_DP",         ["CD45", "CD3e", "CD4", "CD8"],       [],
         "Real population: 1-3% of tissue T cells, up to 30% TILs"),

    # --- Generic T cells ---
    (10, "T_Cell",            ["CD45", "CD3e"],                     [],
         "Catch-all for CD3+ cells"),

    # --- B cells ---
    (11, "B_Cell",            ["CD45", "CD20"],                     [],
         "Note: CD20 lost on plasma cells"),

    # --- NK cells (Pheno only - requires CD56) ---
    (12, "NK_Cell",           ["CD45", "CD56"],                     ["CD3e"],
         "CD56+CD3e- = NK"),

    # --- Macrophages ---
    (13, "Macrophage_M2",     ["CD45", "CD163"],                    [],
         "CD163 highly specific for M2/TAM; classify first"),
    (14, "Macrophage_M1",     ["CD45", "CD68"],                     ["CD163"],
         "CD68+CD163- = M1-like"),

    # --- Generic immune ---
    (15, "Immune_Other",      ["CD45"],                             [],
         "CD45+ but no specific lineage markers"),

    # --- Non-immune lineages ---
    (16, "Epithelial",        ["PanCK"],                            ["CD45"],
         ""),
    (17, "Endothelial",       ["CD31"],                             [],
         "Caveat: CD31 low on platelets/monocytes - use intensity threshold"),
    (18, "Stromal",           ["aSMA"],                             [],
         "aSMA captures myofibroblasts; misses quiescent fibroblasts"),

    # --- Fallback ---
    (99, "Unassigned",        [],                                   [],
         "No clear marker pattern"),
]

# ============================================================================
# Conflict Pairs for Artifact/Doublet Detection
#
# Format: (marker1, marker2, threshold, action)
# - threshold: minimum conflict score (product of posteriors) to trigger
# - action: "artifact" = flag as Artifact_Doublet
#           "review" = flag for LLM review (may be real biology)
#           "type:X" = assign as specific type (e.g., T_Cell_DP)
# ============================================================================

CONFLICT_PAIRS: List[Tuple[str, str, float, str]] = [
    # Hard artifacts - biologically impossible co-expression
    # Thresholds raised to 0.85 to reduce false positives with likelihood-ratio posteriors
    ("PanCK", "CD45",  0.85, "artifact"),  # Epithelial vs Immune
    ("CD3e",  "CD20",  0.85, "artifact"),  # T cell vs B cell - always exclusive
    ("CD68",  "CD3e",  0.85, "artifact"),  # Macrophage vs T cell

    # Context-dependent - may be real EMT
    ("PanCK", "aSMA",  0.85, "review"),    # Epithelial vs Stromal - EMT possible

    # Biologically real - NOT artifacts
    # ("CD4", "CD8") is handled by T_Cell_DP rule, not as conflict
]

# ============================================================================
# Dataset Configurations
# ============================================================================

DATASET_CONFIG = {
    "pheno": {
        "markers": PHENO_MARKERS,
        "available_rules": [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 99],
        "notes": "Full panel - all Tier 1 types available",
    },
    "crc": {
        "markers": CRC_MARKERS,
        "available_rules": [6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 99],
        "notes": "Limited panel - no DC, Neutrophil, NKT, NK; Treg uses FOXP3 only",
    },
}

def get_dataset_type(sample_name: str) -> str:
    """Determine dataset type from sample name."""
    if sample_name.lower().startswith("crc"):
        return "crc"
    elif sample_name.lower().startswith("box"):
        return "pheno"
    else:
        # Default to pheno (more markers)
        return "pheno"

def get_available_markers(sample_name: str) -> List[str]:
    """Get list of available markers for a sample."""
    dataset = get_dataset_type(sample_name)
    return DATASET_CONFIG[dataset]["markers"]

def filter_rules_for_dataset(sample_name: str) -> List[Tuple[int, str, List[str], List[str], str]]:
    """Filter classification rules to those applicable for a dataset."""
    dataset = get_dataset_type(sample_name)
    available_markers = set(DATASET_CONFIG[dataset]["markers"])

    filtered = []
    for priority, cell_type, req_pos, req_neg, notes in CLASSIFICATION_RULES:
        # Check if all required markers are available
        all_markers = set(req_pos) | set(req_neg)
        if all_markers <= available_markers or len(all_markers) == 0:
            filtered.append((priority, cell_type, req_pos, req_neg, notes))

    return filtered

# ============================================================================
# Cell Type Normalization (LLM output -> canonical name)
# ============================================================================

CELL_TYPE_NORMALIZATION = {
    # Dendritic
    "dendritic cell": "Dendritic_Cell",
    "dendritic_cell": "Dendritic_Cell",
    "dendritic cells": "Dendritic_Cell",
    "dc": "Dendritic_Cell",
    "mature dc": "Dendritic_Cell",
    "mature_dc": "Dendritic_Cell",
    "mature dendritic cell": "Dendritic_Cell",
    "mature_dendritic_cells": "Dendritic_Cell",

    # Neutrophil
    "neutrophil": "Neutrophil",
    "neutrophils": "Neutrophil",
    "granulocyte": "Neutrophil",
    "granulocytes": "Neutrophil",

    # NKT
    "nkt cell": "NKT_Cell",
    "nkt_cell": "NKT_Cell",
    "nkt cells": "NKT_Cell",

    # Regulatory T
    "regulatory t cell": "Regulatory_T_Cell",
    "regulatory_t_cell": "Regulatory_T_Cell",
    "regulatory_t_cells": "Regulatory_T_Cell",
    "treg": "Regulatory_T_Cell",
    "tregs": "Regulatory_T_Cell",
    "cd25_regulatory_t_cells": "Regulatory_T_Cell",

    # Helper T
    "helper t cell": "Helper_T_Cell",
    "helper_t_cell": "Helper_T_Cell",
    "helper_t_cells": "Helper_T_Cell",
    "cd4 t cell": "Helper_T_Cell",
    "cd4+ t cell": "Helper_T_Cell",
    "th cell": "Helper_T_Cell",

    # Cytotoxic T
    "cytotoxic t cell": "Cytotoxic_T_Cell",
    "cytotoxic_t_cell": "Cytotoxic_T_Cell",
    "cytotoxic_t_cells": "Cytotoxic_T_Cell",
    "cd8 t cell": "Cytotoxic_T_Cell",
    "cd8+ t cell": "Cytotoxic_T_Cell",
    "ctl": "Cytotoxic_T_Cell",
    "ctls": "Cytotoxic_T_Cell",

    # T Cell DP
    "t cell dp": "T_Cell_DP",
    "t_cell_dp": "T_Cell_DP",
    "double positive t cell": "T_Cell_DP",
    "cd4+cd8+ t cell": "T_Cell_DP",

    # Generic T
    "t cell": "T_Cell",
    "t_cell": "T_Cell",
    "t cells": "T_Cell",
    "t_cells": "T_Cell",
    "t lymphocyte": "T_Cell",
    "memory t cell": "T_Cell",
    "memory_t_cells": "T_Cell",
    "exhausted t cell": "T_Cell",
    "exhausted_t_cells": "T_Cell",

    # B Cell
    "b cell": "B_Cell",
    "b_cell": "B_Cell",
    "b cells": "B_Cell",
    "b_cells": "B_Cell",
    "b lymphocyte": "B_Cell",

    # NK Cell
    "nk cell": "NK_Cell",
    "nk_cell": "NK_Cell",
    "nk cells": "NK_Cell",
    "nk_cells": "NK_Cell",
    "natural killer": "NK_Cell",
    "natural killer cell": "NK_Cell",

    # Macrophages
    "macrophage m2": "Macrophage_M2",
    "macrophage_m2": "Macrophage_M2",
    "macrophages_m2": "Macrophage_M2",
    "m2 macrophage": "Macrophage_M2",
    "m2-like macrophage": "Macrophage_M2",
    "tam": "Macrophage_M2",

    "macrophage m1": "Macrophage_M1",
    "macrophage_m1": "Macrophage_M1",
    "macrophages_m1": "Macrophage_M1",
    "m1 macrophage": "Macrophage_M1",
    "m1-like macrophage": "Macrophage_M1",
    "macrophage": "Macrophage_M1",  # Default to M1 if unspecified
    "macrophages": "Macrophage_M1",

    # Immune Other
    "immune cell": "Immune_Other",
    "immune_cell": "Immune_Other",
    "immune cells": "Immune_Other",
    "immune_cells": "Immune_Other",
    "immune_other": "Immune_Other",
    "leukocyte": "Immune_Other",

    # Epithelial
    "epithelial": "Epithelial",
    "epithelial cell": "Epithelial",
    "epithelial_cell": "Epithelial",
    "epithelial cells": "Epithelial",
    "epithelial_cells": "Epithelial",
    "tumor cell": "Epithelial",
    "tumor cells": "Epithelial",
    "carcinoma": "Epithelial",
    "proliferating_epithelial": "Epithelial",
    "ck18_epithelial": "Epithelial",
    "ck5_epithelial": "Epithelial",
    "tumor_aggresive_epithelial": "Epithelial",
    "her2_positive_epithelial": "Epithelial",
    "er_positive_epithelial": "Epithelial",
    "pr_positive_epithelial": "Epithelial",

    # Endothelial
    "endothelial": "Endothelial",
    "endothelial cell": "Endothelial",
    "endothelial_cell": "Endothelial",
    "endothelial cells": "Endothelial",
    "endothelial_cells": "Endothelial",

    # Stromal
    "stromal": "Stromal",
    "stromal cell": "Stromal",
    "stromal_cell": "Stromal",
    "stromal cells": "Stromal",
    "stromal_cells": "Stromal",
    "fibroblast": "Stromal",
    "fibroblasts": "Stromal",
    "caf": "Stromal",
    "caf_activated": "Stromal",
    "caf_pod": "Stromal",
    "myofibroblast": "Stromal",

    # Low-confidence fallback types
    "epithelial_low": "Epithelial_Low",
    "epithelial low": "Epithelial_Low",
    "low epithelial": "Epithelial_Low",
    "weak epithelial": "Epithelial_Low",
    "endothelial_low": "Endothelial_Low",
    "endothelial low": "Endothelial_Low",
    "low endothelial": "Endothelial_Low",
    "weak endothelial": "Endothelial_Low",
    "stromal_low": "Stromal_Low",
    "stromal low": "Stromal_Low",
    "low stromal": "Stromal_Low",
    "weak stromal": "Stromal_Low",
    "t_cell_low": "T_Cell_Low",
    "t cell low": "T_Cell_Low",
    "low t cell": "T_Cell_Low",
    "weak t cell": "T_Cell_Low",
    "b_cell_low": "B_Cell_Low",
    "b cell low": "B_Cell_Low",
    "low b cell": "B_Cell_Low",
    "weak b cell": "B_Cell_Low",
    "macrophage_low": "Macrophage_Low",
    "macrophage low": "Macrophage_Low",
    "low macrophage": "Macrophage_Low",
    "weak macrophage": "Macrophage_Low",
    "immune_low": "Immune_Low",
    "immune low": "Immune_Low",
    "low immune": "Immune_Low",
    "weak immune": "Immune_Low",

    # Unassigned / Unknown
    "unassigned": "Unassigned",
    "unknown": "Unknown",
    "unclassified": "Unassigned",

    # Artifact
    "artifact": "Artifact_Doublet",
    "artifact_doublet": "Artifact_Doublet",
    "doublet": "Artifact_Doublet",
}

def normalize_cell_type(raw_label: str) -> str:
    """
    Normalize a raw cell type label to one of the valid Tier 1 types.
    """
    if not raw_label:
        return "Unassigned"

    cleaned = raw_label.strip().strip('*').strip('"').strip("'")

    # Exact match
    if cleaned in VALID_CELL_TYPES:
        return cleaned

    # Normalization map (case-insensitive)
    lower = cleaned.lower().strip()
    if lower in CELL_TYPE_NORMALIZATION:
        return CELL_TYPE_NORMALIZATION[lower]

    # Underscore/space normalization
    lower_norm = lower.replace(' ', '_').replace('-', '_')
    for valid in VALID_CELL_TYPES:
        if lower_norm == valid.lower():
            return valid

    # Keyword fallback with word boundary awareness
    def _has(pattern):
        return bool(re.search(r'(?<![a-z0-9])' + pattern + r'(?![a-z0-9])', lower))

    if 'regulatory' in lower or 'treg' in lower:
        return "Regulatory_T_Cell"
    if 'cytotoxic' in lower or (_has('cd8') and 't' in lower):
        return "Cytotoxic_T_Cell"
    if 'helper' in lower or (_has('cd4') and 't' in lower and 'foxp3' not in lower):
        return "Helper_T_Cell"
    if 'nkt' in lower or ('nk' in lower and 't cell' in lower):
        return "NKT_Cell"
    if 't cell' in lower or 't_cell' in lower:
        return "T_Cell"
    if 'b cell' in lower or 'b_cell' in lower or _has('cd20'):
        return "B_Cell"
    if 'nk' in lower or 'natural killer' in lower:
        return "NK_Cell"
    if 'm2' in lower or _has('cd163'):
        return "Macrophage_M2"
    if 'macrophage' in lower or _has('cd68'):
        return "Macrophage_M1"
    if 'dendritic' in lower or 'dc' in lower:
        return "Dendritic_Cell"
    if 'neutrophil' in lower or 'granulocyte' in lower:
        return "Neutrophil"
    if 'epithelial' in lower or 'tumor' in lower or 'panck' in lower or 'carcinoma' in lower:
        return "Epithelial"
    if 'endothelial' in lower or _has('cd31'):
        return "Endothelial"
    if 'stromal' in lower or 'fibroblast' in lower or 'caf' in lower or 'sma' in lower:
        return "Stromal"
    if 'immune' in lower or 'leukocyte' in lower or _has('cd45'):
        return "Immune_Other"
    if 'doublet' in lower or 'artifact' in lower:
        return "Artifact_Doublet"

    return "Unassigned"

# ============================================================================
# Helper Functions
# ============================================================================

def rules_as_text(sample_name: str = None) -> str:
    """
    Return classification rules formatted for LLM system prompt.
    If sample_name provided, filter to applicable rules.
    """
    if sample_name:
        rules = filter_rules_for_dataset(sample_name)
    else:
        rules = CLASSIFICATION_RULES

    lines = ["CLASSIFICATION RULES (apply in priority order, first match wins):"]
    for priority, cell_type, pos, neg, notes in rules:
        if priority == 99:
            lines.append(f"  {priority}. {cell_type}: no clear marker pattern")
            continue
        pos_str = " AND ".join(f"{m}+" for m in pos) if pos else ""
        neg_str = " AND ".join(f"{m}-" for m in neg) if neg else ""
        rule_str = f"  {priority}. {cell_type}: {pos_str}"
        if neg_str:
            rule_str += f", {neg_str}"
        if notes:
            rule_str += f"  ({notes})"
        lines.append(rule_str)
    return "\n".join(lines)

def get_conflict_pairs() -> List[Tuple[str, str, float, str]]:
    """Return conflict pairs for artifact detection."""
    return CONFLICT_PAIRS

def validate_rules():
    """Validate that rules are properly ordered and have unique priorities."""
    priorities = [r[0] for r in CLASSIFICATION_RULES]
    if len(priorities) != len(set(priorities)):
        raise ValueError("Duplicate priorities in CLASSIFICATION_RULES")
    if priorities != sorted(priorities):
        raise ValueError("CLASSIFICATION_RULES not sorted by priority")
    print(f"[cell_type_rules] Validated {len(CLASSIFICATION_RULES)} rules")

# Run validation on import
try:
    validate_rules()
except Exception as e:
    print(f"[cell_type_rules] WARNING: {e}")
