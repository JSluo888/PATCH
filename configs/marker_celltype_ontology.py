"""
Universal Marker-Cell Type Ontology for Multiplexed Tissue Imaging

Defines the cell type hierarchy, marker signatures, and panel distinguishability
rules. The system returns the FINEST biologically defensible label under the
observed marker panel.

Sources: Cell Ontology (CL), CellMarker 2.0, PanglaoDB, published multiplex panels
(Schürch 2020, Keren 2018, Jackson 2020, Hoch 2022, IMMUcan 2024)

Reviewed by: Codex (GPT-5.4), Gemini (gemini-3-pro), Claude Opus 4.6
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class CellTypeNode:
    """A node in the cell type ontology tree."""
    name: str
    parent: Optional[str] = None
    required_positive: List[str] = field(default_factory=list)
    required_negative: List[str] = field(default_factory=list)
    optional_markers: List[str] = field(default_factory=list)
    level: str = "L1"  # L1=lineage, L2=subtype, L3=state
    cl_id: Optional[str] = None  # Cell Ontology ID
    description: str = ""
    is_leaf: bool = True
    children: List[str] = field(default_factory=list)


# ============================================================================
# Cell Type Ontology
# ============================================================================

CELL_TYPE_ONTOLOGY: Dict[str, CellTypeNode] = {
    # --- Root ---
    "Root": CellTypeNode(
        name="Root", parent=None, level="L0", is_leaf=False,
        children=["Immune", "Epithelial", "Endothelial", "Stromal", "Unknown", "Doublet_Artifact"],
    ),

    # --- Major Lineages (L1) ---
    "Immune": CellTypeNode(
        name="Immune", parent="Root", level="L1", is_leaf=False,
        required_positive=["CD45"],
        required_negative=["PanCK"],
        cl_id="CL:0000738",  # leukocyte
        children=["T_Lineage", "B_Cell", "Myeloid", "NK_Cell", "Dendritic_Cell", "Plasma_Cell", "Neutrophil", "Immune_Other"],
        description="All hematopoietic/immune cells (CD45+)",
    ),
    "Epithelial": CellTypeNode(
        name="Epithelial", parent="Root", level="L1", is_leaf=True,
        required_positive=["PanCK"],
        required_negative=["CD45"],
        optional_markers=["Ecadherin", "Ki67"],
        cl_id="CL:0000066",
        description="Epithelial/tumor cells (PanCK+ or Ecadherin+, CD45-)",
    ),
    "Endothelial": CellTypeNode(
        name="Endothelial", parent="Root", level="L1", is_leaf=True,
        required_positive=["CD31"],
        required_negative=["PanCK", "CD45"],
        cl_id="CL:0000115",
        description="Blood vessel endothelial cells (CD31+)",
    ),
    "Stromal": CellTypeNode(
        name="Stromal", parent="Root", level="L1", is_leaf=True,
        required_positive=["aSMA"],
        required_negative=["CD45", "CD31", "PanCK"],
        optional_markers=["Vimentin"],
        cl_id="CL:0000499",  # stromal cell
        description="Myofibroblast-like stromal cells (aSMA+). Note: aSMA marks myofibroblasts, not all fibroblasts",
    ),
    "Unknown": CellTypeNode(
        name="Unknown", parent="Root", level="L1", is_leaf=True,
        description="Cells with no clear marker pattern",
    ),
    "Doublet_Artifact": CellTypeNode(
        name="Doublet_Artifact", parent="Root", level="L1", is_leaf=True,
        description="Segmentation artifacts with biologically impossible marker co-expression",
    ),

    # --- Immune Subtypes ---
    "T_Lineage": CellTypeNode(
        name="T_Lineage", parent="Immune", level="L1.1", is_leaf=False,
        required_positive=["CD45", "CD3e"],
        required_negative=["CD20", "CD68"],
        cl_id="CL:0000084",  # T cell
        children=["Helper_T_Cell", "Regulatory_T_Cell", "Cytotoxic_T_Cell"],
        description="T cell lineage (CD45+ CD3e+). Returned when CD4/CD8/FOXP3 markers are absent or ambiguous.",
    ),
    "Helper_T_Cell": CellTypeNode(
        name="Helper_T_Cell", parent="T_Lineage", level="L2",
        required_positive=["CD45", "CD3e", "CD4"],
        required_negative=["FOXP3", "CD8"],
        optional_markers=["PD1", "CD45RO", "Ki67"],
        cl_id="CL:0000492",  # CD4-positive helper T cell
        description="CD4+ conventional T helper cell (CD3e+ CD4+ FOXP3-)",
    ),
    "Regulatory_T_Cell": CellTypeNode(
        name="Regulatory_T_Cell", parent="T_Lineage", level="L2",
        required_positive=["CD45", "CD3e", "CD4", "FOXP3"],
        required_negative=["CD8"],
        optional_markers=["PD1", "CD45RO", "Ki67"],
        cl_id="CL:0000815",  # regulatory T cell
        description="Regulatory T cell (CD3e+ CD4+ FOXP3+)",
    ),
    "Cytotoxic_T_Cell": CellTypeNode(
        name="Cytotoxic_T_Cell", parent="T_Lineage", level="L2",
        required_positive=["CD45", "CD3e", "CD8"],
        required_negative=["CD4"],
        optional_markers=["PD1", "CD45RO", "Ki67", "PDL1"],
        cl_id="CL:0000794",  # CD8-positive cytotoxic T cell
        description="CD8+ cytotoxic T cell (CD3e+ CD8+)",
    ),

    "B_Cell": CellTypeNode(
        name="B_Cell", parent="Immune", level="L1.1", is_leaf=True,
        required_positive=["CD45", "CD20"],
        required_negative=["CD3e", "CD68"],
        optional_markers=["Ki67"],
        cl_id="CL:0000236",  # B cell
        description="B cell (CD45+ CD20+)",
    ),

    "Myeloid": CellTypeNode(
        name="Myeloid", parent="Immune", level="L1.1", is_leaf=False,
        required_positive=["CD45", "CD68"],
        required_negative=["CD3e", "CD20"],
        cl_id="CL:0000763",  # myeloid cell
        children=["Macrophage_CD163pos", "Macrophage_CD163neg"],
        description="Myeloid/macrophage lineage (CD45+ CD68+). Returned when CD163 is absent.",
    ),
    "Macrophage_CD163pos": CellTypeNode(
        name="Macrophage_CD163pos", parent="Myeloid", level="L2",
        required_positive=["CD45", "CD68", "CD163"],
        required_negative=["CD3e", "CD20"],
        optional_markers=["PDL1", "Ki67"],
        cl_id="CL:0000235",  # macrophage
        description="CD163+ macrophage / TAM-like (CD68+ CD163+). Often called M2 but represents a spectrum.",
    ),
    "Macrophage_CD163neg": CellTypeNode(
        name="Macrophage_CD163neg", parent="Myeloid", level="L2",
        required_positive=["CD45", "CD68"],
        required_negative=["CD163", "CD3e", "CD20"],
        optional_markers=["PDL1", "Ki67"],
        description="CD163-negative macrophage (CD68+ CD163-). Often called M1 but represents a spectrum.",
    ),

    "NK_Cell": CellTypeNode(
        name="NK_Cell", parent="Immune", level="L1.1", is_leaf=True,
        required_positive=["CD45", "CD56"],
        required_negative=["CD3e", "CD20", "CD68"],
        optional_markers=["CD16", "GrzB"],
        cl_id="CL:0000623",  # natural killer cell
        description="Natural killer cell (CD45+ CD56+ CD3e-). Collapses to Immune_Other when CD56 absent.",
    ),

    "Dendritic_Cell": CellTypeNode(
        name="Dendritic_Cell", parent="Immune", level="L1.1", is_leaf=True,
        required_positive=["CD45", "CD11c", "HLADR"],
        required_negative=["CD3e", "CD20", "CD68"],
        optional_markers=["CD303", "CD1c"],
        cl_id="CL:0000451",  # dendritic cell
        description="Dendritic cell (CD45+ CD11c+ HLA-DR+ CD68-). Reported in 5/8 papers. Collapses to Immune_Other when CD11c or HLA-DR absent.",
    ),

    "Plasma_Cell": CellTypeNode(
        name="Plasma_Cell", parent="Immune", level="L1.1", is_leaf=True,
        required_positive=["CD45", "CD138"],
        required_negative=["CD3e", "CD20"],
        optional_markers=["CD38"],
        cl_id="CL:0000786",  # plasma cell
        description="Plasma cell (CD45+ CD138+ CD20-). Terminally differentiated B-lineage. Collapses to B_Cell when CD138 absent.",
    ),

    "Neutrophil": CellTypeNode(
        name="Neutrophil", parent="Immune", level="L1.1", is_leaf=True,
        required_positive=["CD45", "CD15"],  # CD66b is alternative (checked separately)
        required_negative=["CD3e", "CD20"],
        optional_markers=["CD66b", "MPO"],
        cl_id="CL:0000775",  # neutrophil
        description="Neutrophil (CD45+ CD15+ or CD66b+). Collapses to Immune_Other when CD15/CD66b absent.",
    ),

    "Immune_Other": CellTypeNode(
        name="Immune_Other", parent="Immune", level="L1.1", is_leaf=True,
        required_positive=["CD45"],
        required_negative=[],
        cl_id="CL:0000738",
        description="CD45+ immune cell without specific lineage markers. Fallback when DC/NK/Neutrophil markers unavailable.",
    ),
}


# ============================================================================
# Panel Distinguishability
# ============================================================================

# For each pair of cell types, which marker(s) distinguish them
DISTINGUISHING_MARKERS: Dict[Tuple[str, str], List[str]] = {
    ("Helper_T_Cell", "Regulatory_T_Cell"): ["FOXP3"],
    ("Helper_T_Cell", "Cytotoxic_T_Cell"): ["CD4", "CD8"],
    ("Regulatory_T_Cell", "Cytotoxic_T_Cell"): ["CD4", "CD8", "FOXP3"],
    ("Macrophage_CD163pos", "Macrophage_CD163neg"): ["CD163"],
    ("T_Lineage", "B_Cell"): ["CD3e", "CD20"],
    ("T_Lineage", "Myeloid"): ["CD3e", "CD68"],
    ("B_Cell", "Myeloid"): ["CD20", "CD68"],
    ("Immune", "Epithelial"): ["CD45", "PanCK"],
    ("Immune", "Endothelial"): ["CD45", "CD31"],
    ("Immune", "Stromal"): ["CD45", "aSMA"],
    ("Epithelial", "Stromal"): ["PanCK", "aSMA"],
    ("Epithelial", "Endothelial"): ["PanCK", "CD31"],
    ("NK_Cell", "T_Lineage"): ["CD56", "CD3e"],
    ("Dendritic_Cell", "Myeloid"): ["CD11c", "HLADR", "CD68"],
    ("Plasma_Cell", "B_Cell"): ["CD138", "CD20"],
    ("Neutrophil", "Myeloid"): ["CD15", "CD68"],
}

# Conflict pairs: biologically impossible co-expression
IMPOSSIBLE_COEXPRESSION = [
    ("PanCK", "CD45", "Epithelial vs Immune — likely doublet"),
    ("CD3e", "CD20", "T cell vs B cell — mutually exclusive"),
    ("CD68", "CD3e", "Macrophage vs T cell — mutually exclusive"),
    ("PanCK", "CD31", "Epithelial vs Endothelial — likely doublet"),
]


def get_finest_identifiable_type(
    available_markers: Set[str],
    cell_type: str = None,
) -> str:
    """
    Given an available marker panel, return the finest cell type
    that can be identified (may be a parent node if distinguishing
    markers are missing).

    Only collapses to parent when POSITIVE markers are missing (can't confirm
    the type). Missing NEGATIVE markers mean ambiguity between siblings
    (can't exclude), but the type is still identifiable if positives are present.

    Args:
        available_markers: set of marker names in the panel
        cell_type: optional specific leaf type to check

    Returns:
        The finest identifiable type name (str). If cell_type is None,
        returns "Unknown".
    """
    if cell_type is None:
        return "Unknown"

    if cell_type not in CELL_TYPE_ONTOLOGY:
        return "Unknown"

    node = CELL_TYPE_ONTOLOGY[cell_type]
    # Check if all required positive markers are available
    missing_pos = [m for m in node.required_positive if m not in available_markers]
    if missing_pos:
        # Can't confirm this type — collapse to parent
        if node.parent and node.parent in CELL_TYPE_ONTOLOGY:
            return get_finest_identifiable_type(available_markers, node.parent)
        return "Unknown"
    # Missing negative markers = ambiguity between siblings, but type is
    # still identifiable (e.g., CD3e+CD4+ without FOXP3 → still Helper_T_Cell,
    # just can't distinguish from Treg). Don't collapse for missing negatives.
    return cell_type


def get_identifiable_types(available_markers: Set[str]) -> List[str]:
    """
    For a given marker panel, return all cell types that can be identified.

    Checks both required_positive and required_negative marker availability.

    Args:
        available_markers: set of marker names in the panel

    Returns:
        List of identifiable type names
    """
    identifiable = []
    for name, node in CELL_TYPE_ONTOLOGY.items():
        if not node.required_positive:
            continue
        has_pos = all(m in available_markers for m in node.required_positive)
        has_neg = all(m in available_markers for m in node.required_negative)
        if has_pos and has_neg:
            identifiable.append(name)
    return identifiable


def get_panel_resolution(available_markers: Set[str]) -> Dict[str, str]:
    """
    For each leaf type, determine if it can be identified or must collapse
    to a parent given the available marker panel.

    Returns:
        Dict mapping leaf_type → finest_identifiable_type
    """
    resolution = {}
    for name, node in CELL_TYPE_ONTOLOGY.items():
        if node.is_leaf or not node.children:
            finest = get_finest_identifiable_type(available_markers, name)
            resolution[name] = finest
    return resolution
