"""Standard-immunology cell type ontology (v6, April 2026).

Supersedes ``marker_celltype_ontology.py``. Key changes vs v5:

- Round-1 lineages are explicit internal nodes: ``Immune``, ``Epithelial``,
  ``Myoepithelial``, ``Endothelial``, ``Fibroblast``, ``Unclassified``,
  ``Artifact``.
- ``Immune`` splits into ``Lymphoid`` / ``Myeloid`` internal nodes plus the
  two catch-alls ``APC_generic`` and ``ImmOther``. ``T_cell`` sits under
  ``Lymphoid`` and has ``CD4_T`` / ``CD8_T`` / ``Treg`` / ``DN_T`` leaves.
- ``Epithelial`` and ``Fibroblast`` are internal nodes with flat children.
- ``NK``, ``Plasma``, ``Myoepithelial``, DC, Mast, Monocyte, CAF, Luminal/Basal,
  Treg, DN_T are panel-gated: present only when their gating markers are in the
  panel. Pruning collapses them into the nearest catch-all (``ImmOther`` for
  CD45+ cells, ``Unclassified`` for CD45-).
- Internal nodes are legitimate prediction targets. Hierarchy-aware conformal
  must derive an internal-node score from its descendants (see
  ``aggregate_internal_scores``).

Source: April 2026 team meeting + Panel C two-round gating scheme.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional, Set, Tuple


@dataclass(frozen=True)
class CellTypeNode:
    """A node in the cell type ontology tree.

    ``gate_markers`` are the markers whose *presence* in a panel is required
    for this node to stay in the panel-pruned ontology. Defaults to
    ``required_positive``; overridden when a softer gate is appropriate (e.g.
    Myoepithelial keeps if ``P63`` OR ``CK5`` is present).
    """

    name: str
    parent: Optional[str] = None
    required_positive: Tuple[str, ...] = ()
    required_negative: Tuple[str, ...] = ()
    optional_markers: Tuple[str, ...] = ()
    gate_markers: Tuple[FrozenSet[str], ...] = ()
    level: str = "L1"
    cl_id: Optional[str] = None
    description: str = ""
    children: Tuple[str, ...] = ()
    catch_all_parent: Optional[str] = None

    @property
    def is_leaf(self) -> bool:
        return not self.children


_F = frozenset
_GATE_ANY_OF = lambda *groups: tuple(_F(g) if not isinstance(g, frozenset) else g for g in groups)


# ---------------------------------------------------------------------------
# Ontology definition
# ---------------------------------------------------------------------------

CELL_TYPE_ONTOLOGY: Dict[str, CellTypeNode] = {
    "Root": CellTypeNode(
        name="Root", parent=None, level="L0",
        children=(
            "Immune", "Epithelial", "Myoepithelial", "Endothelial",
            "Fibroblast", "Unclassified", "Artifact",
        ),
        description="Top of the tree.",
    ),

    # ------------------------------------------------------------------
    # Immune (CD45+)
    # ------------------------------------------------------------------
    "Immune": CellTypeNode(
        name="Immune", parent="Root", level="L1",
        required_positive=("CD45",),
        required_negative=("PanCK",),
        cl_id="CL:0000738",
        children=("Lymphoid", "Myeloid", "APC_generic", "ImmOther"),
        description="Hematopoietic/immune cells (CD45+).",
    ),

    # --- Lymphoid branch ---------------------------------------------
    "Lymphoid": CellTypeNode(
        name="Lymphoid", parent="Immune", level="L2",
        required_positive=("CD45",),
        required_negative=("CD68", "PanCK"),
        children=("T_cell", "B_cell", "NK", "Plasma"),
        catch_all_parent="ImmOther",
        description="Lymphoid lineage (CD45+, CD68-).",
    ),
    "T_cell": CellTypeNode(
        name="T_cell", parent="Lymphoid", level="L3",
        required_positive=("CD45", "CD3e"),
        required_negative=("CD20", "CD68"),
        cl_id="CL:0000084",
        children=("CD4_T", "CD8_T", "Treg", "DN_T"),
        catch_all_parent="ImmOther",
        description="T cells (CD3e+, CD45+). Covers CD4 / CD8 / Treg / DN-T.",
    ),
    "CD4_T": CellTypeNode(
        name="CD4_T", parent="T_cell", level="L4",
        required_positive=("CD45", "CD3e", "CD4"),
        required_negative=("CD8", "FOXP3"),
        optional_markers=("PD1", "CD45RO", "Ki67"),
        cl_id="CL:0000492",
        description="CD4+ conventional helper T cell (CD3e+ CD4+ FOXP3-).",
    ),
    "CD8_T": CellTypeNode(
        name="CD8_T", parent="T_cell", level="L4",
        required_positive=("CD45", "CD3e", "CD8"),
        required_negative=("CD4",),
        optional_markers=("PD1", "CD45RO", "Ki67", "PDL1", "GZMB"),
        cl_id="CL:0000794",
        description="CD8+ cytotoxic T cell (CD3e+ CD8+).",
    ),
    "Treg": CellTypeNode(
        name="Treg", parent="T_cell", level="L4",
        required_positive=("CD45", "CD3e", "CD4", "FOXP3"),
        required_negative=("CD8",),
        optional_markers=("CD25", "PD1", "Ki67"),
        gate_markers=_GATE_ANY_OF({"FOXP3"}),
        cl_id="CL:0000815",
        catch_all_parent="CD4_T",
        description="Regulatory T cell (CD3e+ CD4+ FOXP3+). Panel-gated on FOXP3.",
    ),
    "DN_T": CellTypeNode(
        name="DN_T", parent="T_cell", level="L4",
        required_positive=("CD45", "CD3e"),
        required_negative=("CD4", "CD8"),
        gate_markers=_GATE_ANY_OF({"CD4"}, {"CD8"}),
        catch_all_parent="T_cell",
        description="CD4-CD8- double-negative T cell (CD3e+, CD4-, CD8-). Rare but real.",
    ),
    "B_cell": CellTypeNode(
        name="B_cell", parent="Lymphoid", level="L3",
        required_positive=("CD45", "CD20"),
        required_negative=("CD3e", "CD68"),
        optional_markers=("HLADR", "Ki67"),
        cl_id="CL:0000236",
        description="B cell (CD45+ CD20+). Plasma cells lose CD20 and have their own leaf.",
    ),
    "NK": CellTypeNode(
        name="NK", parent="Lymphoid", level="L3",
        required_positive=("CD45", "CD56"),
        required_negative=("CD3e", "CD20", "CD68"),
        optional_markers=("CD16", "NKp46", "GZMB"),
        gate_markers=_GATE_ANY_OF({"CD56"}, {"NKp46"}),
        cl_id="CL:0000623",
        catch_all_parent="ImmOther",
        description="Natural killer cell (CD56+ or NKp46+, CD3e-). CD3- guard prevents CD8+CD56+ T-cell bleed-through.",
    ),
    "Plasma": CellTypeNode(
        name="Plasma", parent="Lymphoid", level="L3",
        required_positive=("CD45", "CD138"),
        required_negative=("CD3e",),
        optional_markers=("IRF4", "CD38"),
        gate_markers=_GATE_ANY_OF({"CD138"}),
        cl_id="CL:0000786",
        catch_all_parent="ImmOther",
        description="Plasma cell (CD138+). B cells downregulate CD20 on plasma-cell differentiation.",
    ),

    # --- Myeloid branch ----------------------------------------------
    "Myeloid": CellTypeNode(
        name="Myeloid", parent="Immune", level="L2",
        required_positive=("CD45",),
        required_negative=("CD3e", "CD20"),
        cl_id="CL:0000763",
        children=("Monocyte", "MonoDC", "DC", "Macrophage", "Mast", "Neutrophil"),
        catch_all_parent="ImmOther",
        description="Myeloid lineage (CD45+, CD3e-, CD20-).",
    ),
    "Monocyte": CellTypeNode(
        name="Monocyte", parent="Myeloid", level="L3",
        required_positive=("CD45", "CD14"),
        required_negative=("CD68",),
        optional_markers=("HLADR", "CD11c"),
        gate_markers=_GATE_ANY_OF({"CD14"}),
        cl_id="CL:0000576",
        catch_all_parent="ImmOther",
        description="Monocyte (CD14+, CD68-). Panel-gated on CD14.",
    ),
    "MonoDC": CellTypeNode(
        name="MonoDC", parent="Myeloid", level="L3",
        required_positive=("CD45", "CD14", "CD11c", "HLADR"),
        required_negative=("CD68",),
        gate_markers=_GATE_ANY_OF({"CD14", "CD11c", "HLADR"}),
        catch_all_parent="ImmOther",
        description="Monocyte-derived DC (CD14+ CD11c+ HLADR+). Transitional state.",
    ),
    "DC": CellTypeNode(
        name="DC", parent="Myeloid", level="L3",
        required_positive=("CD45", "CD11c", "HLADR"),
        required_negative=("CD3e", "CD20", "CD68"),
        optional_markers=("DCLAMP",),
        gate_markers=_GATE_ANY_OF({"CD11c", "HLADR"}, {"DCLAMP"}),
        cl_id="CL:0000451",
        catch_all_parent="ImmOther",
        description="Conventional dendritic cell (CD11c+ HLADR+, CD68-).",
    ),
    "Macrophage": CellTypeNode(
        name="Macrophage", parent="Myeloid", level="L3",
        required_positive=("CD45", "CD68"),
        required_negative=("CD3e", "CD20"),
        cl_id="CL:0000235",
        children=("Macrophage_CD163pos", "Macrophage_CD163neg"),
        description="Macrophage (CD68+, CD45+). Splits on CD163.",
    ),
    "Macrophage_CD163pos": CellTypeNode(
        name="Macrophage_CD163pos", parent="Macrophage", level="L4",
        required_positive=("CD45", "CD68", "CD163"),
        required_negative=("CD3e", "CD20"),
        optional_markers=("HLADR", "PDL1", "Ki67"),
        description="CD163+ macrophage / TAM-like (CD68+ CD163+). Often labelled M2 but spans a continuum.",
    ),
    "Macrophage_CD163neg": CellTypeNode(
        name="Macrophage_CD163neg", parent="Macrophage", level="L4",
        required_positive=("CD45", "CD68"),
        required_negative=("CD163", "CD3e", "CD20"),
        optional_markers=("HLADR", "PDL1", "Ki67"),
        description="CD163- macrophage (CD68+ CD163-). Often labelled M1 but spans a continuum.",
    ),
    "Mast": CellTypeNode(
        name="Mast", parent="Myeloid", level="L3",
        required_positive=("CD45", "Tryptase"),
        gate_markers=_GATE_ANY_OF({"Tryptase"}),
        cl_id="CL:0000097",
        catch_all_parent="ImmOther",
        description="Mast cell (Tryptase+). No tryptase in panel means no mast leaf.",
    ),
    "Neutrophil": CellTypeNode(
        name="Neutrophil", parent="Myeloid", level="L3",
        required_positive=("CD45",),
        required_negative=("CD3e", "CD20"),
        optional_markers=("CD15", "CD66b", "MPO"),
        gate_markers=_GATE_ANY_OF({"CD15"}, {"CD66b"}, {"MPO"}),
        cl_id="CL:0000775",
        catch_all_parent="ImmOther",
        description="Neutrophil (CD15+ OR CD66b+ OR MPO+, CD3e-, CD20-).",
    ),

    # --- Immune catch-alls -------------------------------------------
    "APC_generic": CellTypeNode(
        name="APC_generic", parent="Immune", level="L2",
        required_positive=("CD45", "HLADR"),
        required_negative=("CD68", "CD11c", "CD20", "CD3e"),
        gate_markers=_GATE_ANY_OF({"HLADR"}),
        catch_all_parent="ImmOther",
        description="HLADR+ antigen-presenting cell that fails the stricter DC / Mac / B gates.",
    ),
    "ImmOther": CellTypeNode(
        name="ImmOther", parent="Immune", level="L2",
        required_positive=("CD45",),
        required_negative=(),
        description="CD45+ immune cell without specific lineage markers. Absorbs NK/plasma/baso/eos on panels without their markers.",
    ),

    # ------------------------------------------------------------------
    # Epithelial (PanCK+ or Ecadherin+, CD45-)
    # ------------------------------------------------------------------
    "Epithelial": CellTypeNode(
        name="Epithelial", parent="Root", level="L1",
        required_positive=("PanCK",),
        required_negative=("CD45", "CD31"),
        optional_markers=("Ecadherin", "Ki67"),
        cl_id="CL:0000066",
        children=("Luminal", "Basal", "EMT", "CK5_7_low"),
        description="Epithelial / tumor cells (PanCK+ or Ecadherin+, CD45-).",
    ),
    "Luminal": CellTypeNode(
        name="Luminal", parent="Epithelial", level="L2",
        required_positive=("PanCK", "CK7"),
        required_negative=("CD45",),
        optional_markers=("Ecadherin", "ER", "PR"),
        gate_markers=_GATE_ANY_OF({"CK7"}),
        catch_all_parent="Epithelial",
        description="Luminal epithelial (CK7+, PanCK+).",
    ),
    "Basal": CellTypeNode(
        name="Basal", parent="Epithelial", level="L2",
        required_positive=("PanCK", "CK5"),
        required_negative=("CD45",),
        optional_markers=("Ecadherin",),
        gate_markers=_GATE_ANY_OF({"CK5"}),
        catch_all_parent="Epithelial",
        description="Basal epithelial (CK5+, PanCK+).",
    ),
    "EMT": CellTypeNode(
        name="EMT", parent="Epithelial", level="L2",
        required_positive=("Vimentin",),
        required_negative=("CD45",),
        optional_markers=("PanCK", "Ecadherin"),
        gate_markers=_GATE_ANY_OF({"Vimentin"}),
        catch_all_parent="Epithelial",
        description="Epithelial-mesenchymal transition (VIM+, ECAD+/-).",
    ),
    "CK5_7_low": CellTypeNode(
        name="CK5_7_low", parent="Epithelial", level="L2",
        required_positive=("PanCK",),
        required_negative=("CD45", "CK5", "CK7"),
        optional_markers=("Ecadherin",),
        gate_markers=_GATE_ANY_OF({"CK5", "CK7"}),
        catch_all_parent="Epithelial",
        description="Epithelial cell with neither CK5 nor CK7 strongly expressed.",
    ),

    # ------------------------------------------------------------------
    # Myoepithelial (P63-gated, CK5 fallback)
    # ------------------------------------------------------------------
    "Myoepithelial": CellTypeNode(
        name="Myoepithelial", parent="Root", level="L1",
        required_positive=("aSMA",),
        required_negative=("CD45",),
        optional_markers=("P63", "CK5", "PanCK", "Ecadherin"),
        gate_markers=_GATE_ANY_OF({"P63"}, {"CK5"}),
        cl_id="CL:0002327",
        catch_all_parent="Unclassified",
        description="Myoepithelial cell (SMA+, P63+ or CK5+, CD45-). Diagnostic for DCIS/IDC in breast.",
    ),

    # ------------------------------------------------------------------
    # Endothelial
    # ------------------------------------------------------------------
    "Endothelial": CellTypeNode(
        name="Endothelial", parent="Root", level="L1",
        required_positive=("CD31",),
        required_negative=("PanCK", "CD45"),
        optional_markers=("Vimentin",),
        cl_id="CL:0000115",
        description="Blood-vessel endothelial cell (CD31+).",
    ),

    # ------------------------------------------------------------------
    # Fibroblast (internal node, flat children)
    # ------------------------------------------------------------------
    "Fibroblast": CellTypeNode(
        name="Fibroblast", parent="Root", level="L1",
        required_positive=(),
        required_negative=("CD45", "PanCK", "Ecadherin", "CD31"),
        optional_markers=("Vimentin", "aSMA", "FAP", "CD36"),
        gate_markers=_GATE_ANY_OF({"Vimentin"}, {"aSMA"}),
        cl_id="CL:0000057",
        children=("Normal_Fibroblast", "Myofibroblast", "CAF", "Resting_Fibroblast"),
        catch_all_parent="Unclassified",
        description="Fibroblast lineage (VIM+ or aSMA+, CD45-, PanCK-, ECAD-, CD31-). Flat children.",
    ),
    "Normal_Fibroblast": CellTypeNode(
        name="Normal_Fibroblast", parent="Fibroblast", level="L2",
        required_positive=("Vimentin", "CD36"),
        required_negative=("CD45", "PanCK", "Ecadherin"),
        gate_markers=_GATE_ANY_OF({"CD36"}),
        catch_all_parent="Fibroblast",
        description="Normal (CD36+) fibroblast.",
    ),
    "Myofibroblast": CellTypeNode(
        name="Myofibroblast", parent="Fibroblast", level="L2",
        required_positive=("aSMA",),
        required_negative=("CD45", "PanCK", "Ecadherin", "FAP"),
        optional_markers=("Vimentin",),
        gate_markers=_GATE_ANY_OF({"aSMA"}),
        catch_all_parent="Fibroblast",
        description="Myofibroblast (SMA+, FAP-). Identifiable on aSMA-only panels.",
    ),
    "CAF": CellTypeNode(
        name="CAF", parent="Fibroblast", level="L2",
        required_positive=("Vimentin", "FAP"),
        required_negative=("CD45", "PanCK", "Ecadherin"),
        optional_markers=("aSMA",),
        gate_markers=_GATE_ANY_OF({"FAP"}),
        catch_all_parent="Fibroblast",
        description="Cancer-associated fibroblast (FAP+, SMA+/-, VIM+). CAFs are frequently SMA-negative.",
    ),
    "Resting_Fibroblast": CellTypeNode(
        name="Resting_Fibroblast", parent="Fibroblast", level="L2",
        required_positive=("Vimentin",),
        required_negative=("CD45", "PanCK", "Ecadherin", "aSMA", "FAP", "CD36"),
        catch_all_parent="Fibroblast",
        description="Resting/quiescent fibroblast (VIM+ only).",
    ),

    # ------------------------------------------------------------------
    # Catch-all terminals
    # ------------------------------------------------------------------
    "Unclassified": CellTypeNode(
        name="Unclassified", parent="Root", level="L1",
        required_negative=("CD45", "PanCK", "Ecadherin", "aSMA", "CD31", "Vimentin"),
        description="Lineage markers all negative. Often adipocytes, rare stromal, or poorly-stained cells.",
    ),
    "Artifact": CellTypeNode(
        name="Artifact", parent="Root", level="L1",
        description="Pan-positive cells with multiple incompatible lineage markers (likely merged-cell segmentation error).",
    ),
}


# ---------------------------------------------------------------------------
# Panel manifest (per-dataset marker availability)
# ---------------------------------------------------------------------------

PANEL_MANIFEST: Dict[str, FrozenSet[str]] = {
    "crc_orion": frozenset({
        "Hoechst", "AF1", "CD31", "CD45", "CD68", "Argo550", "CD4", "FOXP3",
        "CD8", "CD45RO", "CD20", "PDL1", "CD3e", "CD163", "Ecadherin", "PD1",
        "Ki67", "PanCK", "aSMA",
    }),
    "pheno_codex": frozenset({
        "DAPI", "PanCK", "Ki67", "aSMA", "CAIX", "CD11c", "CD163", "CD20",
        "CD25", "CD28", "CD31", "CD3e", "CD4", "CD44", "CD45", "CD45RO",
        "CD56", "CD66b", "CD68", "CD8", "CK18", "CK5", "DCLAMP", "ER", "FAP",
        "FOXP3", "GZMB", "HER2", "HLAABC", "HLADR", "MCM2", "P21", "P27",
        "PD1", "PDL1", "PR", "Podoplanin", "RB1", "TCF7",
    }),
    # External datasets — populated when label remap is run (task #31).
    "schurch_codex": frozenset(),
    "hoch_imc": frozenset(),
    "immucan_imc": frozenset(),
    "tietscher_imc": frozenset(),
    "liu_mibi": frozenset(),
    "maps_codex": frozenset(),
    "hartmann_mibi": frozenset(),
}


# ---------------------------------------------------------------------------
# Marker aliases (raw → canonical)
# ---------------------------------------------------------------------------

MARKER_ALIASES: Dict[str, str] = {
    "CD3": "CD3e",
    "CD8a": "CD8",
    "E-cadherin": "Ecadherin",
    "ECAD": "Ecadherin",
    "Pan-CK": "PanCK",
    "PanKRT": "PanCK",
    "PanKeratin": "PanCK",
    "PanCytokeratin": "PanCK",
    "SMA": "aSMA",
    "alphaSMA": "aSMA",
    "VIM": "Vimentin",
    "vim": "Vimentin",
    "CD20-H1": "CD20",
    "CD45-RO": "CD45RO",
    "DC-LAMP": "DCLAMP",
    "HLA-ABC": "HLAABC",
    "HLA-DR": "HLADR",
    "PD-1": "PD1",
    "PD-L1": "PDL1",
    "DAPI-01": "DAPI",
    "DAPI-03": "DAPI",
    "CD56-NCAM": "CD56",
    "NCAM1": "CD56",
    "Syndecan-1": "CD138",
    "SDC1": "CD138",
    "CK7+": "CK7",
    "CK5+": "CK5",
}


def resolve_marker(name: str) -> str:
    """Return the canonical marker name."""
    return MARKER_ALIASES.get(name, name)


def canonicalise_panel(panel: Iterable[str]) -> FrozenSet[str]:
    """Resolve aliases across a panel and return a canonical marker set."""
    return frozenset(resolve_marker(m) for m in panel)


# ---------------------------------------------------------------------------
# Panel-aware pruning
# ---------------------------------------------------------------------------

def _gate_satisfied(node: CellTypeNode, panel: FrozenSet[str]) -> bool:
    """Check if at least one of the node's gate-marker groups is satisfied.

    ``gate_markers`` is a tuple of frozensets: the panel must contain *all*
    markers in at least one frozenset. If ``gate_markers`` is empty, the gate
    is considered satisfied (the node has no extra gating requirement beyond
    its ``required_positive`` markers).
    """
    if not node.gate_markers:
        return all(m in panel for m in node.required_positive) or not node.required_positive
    return any(all(m in panel for m in group) for group in node.gate_markers)


def prune_ontology(panel: Iterable[str]) -> Dict[str, CellTypeNode]:
    """Return a panel-pruned copy of the ontology.

    A node is kept iff:
      - its ``required_positive`` markers are all in ``panel`` (without a
        specific gate, this also gates the node), OR
      - its ``gate_markers`` are satisfied.

    Pruned nodes have their children reassigned to the nearest surviving
    ancestor or ``catch_all_parent``.
    """
    canon = canonicalise_panel(panel)
    kept: Dict[str, CellTypeNode] = {}
    # First pass: decide kept/pruned.
    for name, node in CELL_TYPE_ONTOLOGY.items():
        if name == "Root":
            kept[name] = node
            continue
        if _gate_satisfied(node, canon):
            kept[name] = node

    # Second pass: rewrite children lists to only reference kept names,
    # reassigning pruned leaves' parents to their catch-all where needed.
    rewritten: Dict[str, CellTypeNode] = {}
    for name, node in kept.items():
        new_children = tuple(c for c in node.children if c in kept)
        rewritten[name] = CellTypeNode(
            name=node.name, parent=node.parent,
            required_positive=node.required_positive,
            required_negative=node.required_negative,
            optional_markers=node.optional_markers,
            gate_markers=node.gate_markers,
            level=node.level, cl_id=node.cl_id,
            description=node.description,
            children=new_children,
            catch_all_parent=node.catch_all_parent,
        )
    return rewritten


def identifiable_leaves(panel: Iterable[str]) -> List[str]:
    """Leaves that survive panel-aware pruning."""
    pruned = prune_ontology(panel)
    return sorted(n for n, node in pruned.items() if node.is_leaf and node.parent is not None)


def identifiable_internal_nodes(panel: Iterable[str]) -> List[str]:
    """Internal nodes (non-leaves) that survive pruning. These are legitimate
    conformal prediction targets; scores are aggregated from descendants."""
    pruned = prune_ontology(panel)
    return sorted(
        n for n, node in pruned.items()
        if not node.is_leaf and node.parent is not None and n != "Root"
    )


# ---------------------------------------------------------------------------
# Tree utilities
# ---------------------------------------------------------------------------

def ancestors(name: str, include_self: bool = False) -> List[str]:
    """Return ancestors of ``name`` from parent up to Root."""
    out: List[str] = [name] if include_self else []
    cur = CELL_TYPE_ONTOLOGY.get(name)
    while cur is not None and cur.parent is not None:
        out.append(cur.parent)
        cur = CELL_TYPE_ONTOLOGY.get(cur.parent)
    return out


def descendants(name: str, include_self: bool = False) -> List[str]:
    """Return all descendants of ``name`` (DFS order)."""
    out: List[str] = [name] if include_self else []
    stack = list(CELL_TYPE_ONTOLOGY.get(name).children) if name in CELL_TYPE_ONTOLOGY else []
    while stack:
        n = stack.pop()
        out.append(n)
        node = CELL_TYPE_ONTOLOGY.get(n)
        if node is not None:
            stack.extend(node.children)
    return out


def leaves_under(name: str) -> List[str]:
    """Leaf descendants of ``name``. If ``name`` is itself a leaf, returns [name]."""
    node = CELL_TYPE_ONTOLOGY.get(name)
    if node is None:
        return []
    if node.is_leaf:
        return [name]
    return [d for d in descendants(name) if CELL_TYPE_ONTOLOGY[d].is_leaf]


def lowest_common_ancestor(a: str, b: str) -> str:
    """LCA of two nodes. Returns 'Root' as fallback."""
    a_anc = ancestors(a, include_self=True)
    b_anc_set = set(ancestors(b, include_self=True))
    for x in a_anc:
        if x in b_anc_set:
            return x
    return "Root"


def tree_distance(a: str, b: str) -> int:
    """Number of edges between ``a`` and ``b`` in the tree.

    Used to weight hierarchy-aware conformal error costs. CD4_T↔CD8_T (sibling
    under T_cell) is distance 2; CD4_T↔Macrophage (cross Lymphoid/Myeloid) is
    distance 6. Root is at depth 0.
    """
    if a == b:
        return 0
    lca = lowest_common_ancestor(a, b)
    d_a = len(ancestors(a, include_self=True)) - len(ancestors(lca, include_self=True))
    d_b = len(ancestors(b, include_self=True)) - len(ancestors(lca, include_self=True))
    return d_a + d_b


def collapse_to_identifiable(label: str, panel: Iterable[str]) -> str:
    """Collapse a ground-truth label to the finest identifiable node under a panel.

    Walks from the label upward (preferring ``catch_all_parent``, falling back
    to the structural parent) until it finds a node that survived pruning.
    ``Root`` is never returned — cells that would collapse to Root fall into
    ``Unclassified`` instead.
    """
    pruned = prune_ontology(panel)
    if label in pruned and label != "Root":
        return label

    node = CELL_TYPE_ONTOLOGY.get(label)
    while node is not None:
        fallback = node.catch_all_parent or node.parent
        if fallback is None or fallback == "Root":
            return "Unclassified"
        if fallback in pruned:
            return fallback
        node = CELL_TYPE_ONTOLOGY.get(fallback)
    return "Unclassified"


# ---------------------------------------------------------------------------
# Conformal score aggregation (internal nodes)
# ---------------------------------------------------------------------------

def aggregate_internal_scores(
    leaf_scores: Dict[str, float],
    pruned_ontology: Optional[Dict[str, CellTypeNode]] = None,
) -> Dict[str, float]:
    """Sum leaf scores onto every internal node in the (pruned) ontology.

    Conformal implementations that score only leaves can call this to obtain
    first-class scores for the internal nodes ``T_cell``, ``Lymphoid``,
    ``Myeloid``, ``Macrophage``, ``Epithelial``, ``Fibroblast``, ``Immune``.
    The resulting map contains both leaves and internals and can be fed to
    RAPS / SAPS / class-conditional split conformal directly.
    """
    ont = pruned_ontology or CELL_TYPE_ONTOLOGY
    scores: Dict[str, float] = dict(leaf_scores)
    for name, node in ont.items():
        if node.is_leaf:
            continue
        scores[name] = sum(
            leaf_scores.get(d, 0.0)
            for d in descendants(name)
            if d in ont and ont[d].is_leaf
        )
    return scores


# ---------------------------------------------------------------------------
# Conflict pairs for artifact detection
# ---------------------------------------------------------------------------

IMPOSSIBLE_COEXPRESSION: List[Tuple[str, str, float, str]] = [
    ("PanCK",     "CD45",  0.85, "Artifact"),
    ("CD3e",      "CD20",  0.85, "Artifact"),
    ("CD68",      "CD3e",  0.85, "Artifact"),
    ("PanCK",     "CD31",  0.85, "Artifact"),
    ("Ecadherin", "CD45",  0.85, "Artifact"),
    ("CD4",       "CD8",   0.85, "review"),  # real DP T cells exist; let DN_T / T_cell reasoning handle
    ("PanCK",     "aSMA",  0.85, "review"),  # may be EMT
]
