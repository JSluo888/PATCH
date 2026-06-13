#!/usr/bin/env python3
"""Score VLM audit responses against ground truth.

Reads YAML responses from vlm_responses/ (produced by run_audit.py) and the
ground-truth prompts from vlm_100cells/cell_NNN_prompt.txt. Emits the four
numbers the paper needs.

Usage:
    python scripts/vlm_audit/score_audit.py --responses vlm_responses/ --out vlm_audit_results.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import yaml

REPO = Path(__file__).resolve().parents[2]
CELLS_DIR = REPO / "vlm_100cells"  # default; overridable via --cells-dir

sys.path.insert(0, str(REPO))
from configs import ontology_v6 as o  # noqa: E402


# v5 -> v6 label mapping for GT (evaluation prompts use v5 names).
V5_TO_V6 = {
    "B_Cell": "B_cell",
    "Cytotoxic_T_Cell": "CD8_T",
    "Dendritic_Cell": "DC",
    "Endothelial": "Endothelial",
    "Epithelial": "Epithelial",
    "Helper_T_Cell": "CD4_T",
    "Immune": "Immune",
    "Immune_Other": "ImmOther",
    "Macrophage_CD163neg": "Macrophage_CD163neg",
    "Macrophage_CD163pos": "Macrophage_CD163pos",
    "Myeloid": "Myeloid",
    "NK_Cell": "NK",
    "Neutrophil": "Neutrophil",
    "Plasma_Cell": "Plasma",
    "Regulatory_T_Cell": "Treg",
    "Stromal": "Fibroblast",
    "T_Lineage": "T_cell",
}

# VLM outputs may use slightly different casing.
VLM_NORMALIZE = {
    "cd4_t": "CD4_T", "cd4t": "CD4_T", "helper_t_cell": "CD4_T",
    "cd8_t": "CD8_T", "cd8t": "CD8_T", "cytotoxic_t_cell": "CD8_T",
    "treg": "Treg", "regulatory_t_cell": "Treg",
    "b_cell": "B_cell", "b_lineage": "B_lineage",
    "nk": "NK", "nk_cell": "NK",
    "dc": "DC", "dendritic_cell": "DC",
    "neutrophil": "Neutrophil",
    "macrophage_cd163pos": "Macrophage_CD163pos", "macrophage_m2": "Macrophage_CD163pos",
    "macrophage_cd163neg": "Macrophage_CD163neg", "macrophage_m1": "Macrophage_CD163neg",
    "macrophage": "Macrophage",
    "endothelial": "Endothelial",
    "epithelial": "Epithelial",
    "fibroblast": "Fibroblast", "stromal": "Fibroblast",
    "plasma": "Plasma", "plasma_cell": "Plasma",
    "t_cell": "T_cell",
    "lymphoid": "Lymphoid",
    "myeloid": "Myeloid",
    "immune": "Immune",
    "immune_other": "ImmOther",
    "abstain": "abstain",
    "artefact": "artefact", "artifact": "artefact",
}


def parse_response(text: str) -> Optional[Dict]:
    """Extract the YAML block from VLM output."""
    # VLMs may wrap YAML in ```yaml ... ```.
    m = re.search(r"```(?:yaml|yml)?\s*\n(.+?)```", text, re.S)
    body = m.group(1) if m else text
    try:
        data = yaml.safe_load(body)
        if isinstance(data, dict) and "verdict" in data:
            return data
    except Exception:
        pass
    # Fallback: try first YAML-like chunk.
    try:
        data = yaml.safe_load(text)
        if isinstance(data, dict) and "verdict" in data:
            return data
    except Exception:
        pass
    return None


def load_gt(i: int, cells_dir: Path, manifest: Optional[Dict[str, Dict]] = None) -> Dict:
    """Parse GT (and, when available, the v6 conformal set) for one cell.

    Resolution order:

    1. ``manifest[cell_NNN]`` (preferred — written by ``cell_generator``;
       contains every NEW field including ``conformal_set_v6`` / ``set_size``).
    2. Fallback: parse the ``GT:`` line from ``cell_NNN_prompt.txt`` as
       before.  This keeps the 100-cell pilot working unchanged.
    """
    cell_id = f"cell_{i:03d}"
    if manifest is not None:
        entry = manifest.get(cell_id)
        if entry is not None:
            gt_raw = entry.get("gt", "")
            gt_v6 = entry.get("gt_v6") or V5_TO_V6.get(gt_raw, gt_raw)
            return {
                "cell_id": cell_id,
                "gt_raw": gt_raw,
                "gt_v6": gt_v6,
                "tier": entry.get("tier", "?"),
                "conformal_set_v6": list(entry.get("conformal_set_v6") or []),
                "set_size": int(entry.get("set_size") or 0),
            }
    text = (cells_dir / f"{cell_id}_prompt.txt").read_text()
    gt_line = next((ln for ln in text.splitlines() if ln.strip().startswith("GT:")), "")
    tier = "?"
    if "Tier:" in gt_line:
        tier = gt_line.split("Tier:")[-1].strip().split()[0]
    gt_raw = gt_line.split("GT:")[1].split("|")[0].strip() if "GT:" in gt_line else ""
    gt_v6 = V5_TO_V6.get(gt_raw, gt_raw)
    # Best-effort: read "CANDIDATE SET S(x):" line if present.
    cand_line = next(
        (ln for ln in text.splitlines() if ln.strip().startswith("CANDIDATE SET S(x):")),
        "",
    )
    conformal_set_v6: List[str] = []
    if cand_line:
        bracket = cand_line.split(":", 1)[1].strip()
        bracket = bracket.strip("[]")
        if bracket:
            conformal_set_v6 = [s.strip() for s in bracket.split(",") if s.strip()]
    return {
        "cell_id": cell_id,
        "gt_raw": gt_raw,
        "gt_v6": gt_v6,
        "tier": tier,
        "conformal_set_v6": conformal_set_v6,
        "set_size": len(conformal_set_v6),
    }


def normalize_vlm(verdict: str) -> str:
    if not verdict:
        return ""
    v = verdict.strip().lower().replace("-", "_").replace(" ", "_")
    return VLM_NORMALIZE.get(v, verdict.strip())


def is_hit(vlm_verdict_v6: str, gt_v6: str) -> bool:
    """VLM is correct if the verdict equals GT OR is an ancestor of GT
    (ontology match)."""
    if not gt_v6 or not vlm_verdict_v6:
        return False
    if vlm_verdict_v6 in {"abstain", "artefact"}:
        return False
    anc = set(o.ancestors(gt_v6, include_self=True))
    return vlm_verdict_v6 in anc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", required=True, help="Directory of YAML responses")
    ap.add_argument("--out", default="vlm_audit_results.json")
    ap.add_argument(
        "--cells-dir",
        default=str(CELLS_DIR),
        help=(
            "Directory of cell_NNN_prompt.txt files and manifest.json. "
            "Default: vlm_100cells.  Use vlm_500cells for the E3 audit."
        ),
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of cells to score; defaults to len(manifest) or 100.",
    )
    args = ap.parse_args()

    resp_dir = Path(args.responses)
    cells_dir = Path(args.cells_dir)

    # Preload manifest.json when present — it carries the NEW E3 fields
    # (gt_v6, conformal_set_v6, set_size, ...) that the 100-cell prompts
    # cannot supply.
    manifest_path = cells_dir / "manifest.json"
    manifest: Optional[Dict[str, Dict]] = None
    if manifest_path.exists():
        entries = json.loads(manifest_path.read_text())
        manifest = {e["cell_name"]: e for e in entries}
        n_cells = args.limit or len(entries)
    else:
        n_cells = args.limit or 100

    rows: List[Dict] = []
    for i in range(1, n_cells + 1):
        gt = load_gt(i, cells_dir=cells_dir, manifest=manifest)
        # Find matching response file (any model).
        matches = sorted(resp_dir.glob(f"cell_{i:03d}*.yaml"))
        if not matches:
            rows.append({**gt, "parsed": False, "reason": "no response file"})
            continue
        try:
            raw = matches[0].read_text()
        except Exception as e:
            rows.append({**gt, "parsed": False, "reason": f"read error: {e}"})
            continue
        parsed = parse_response(raw)
        if parsed is None:
            rows.append({**gt, "parsed": False, "reason": "YAML parse failed",
                         "raw_head": raw[:200]})
            continue
        verdict_raw = str(parsed.get("verdict", "")).strip()
        verdict = normalize_vlm(verdict_raw)
        hit = is_hit(verdict, gt["gt_v6"])
        rows.append({
            **gt,
            "parsed": True,
            "verdict_raw": verdict_raw,
            "verdict": verdict,
            "confidence": parsed.get("confidence", ""),
            "segmentation_quality": parsed.get("segmentation_quality", ""),
            "abstained": verdict == "abstain",
            "marked_artefact": verdict == "artefact",
            "hit": bool(hit),
        })

    # --- Aggregate metrics ---
    total = len(rows)
    parsed = sum(1 for r in rows if r.get("parsed"))
    abstain = sum(1 for r in rows if r.get("abstained"))
    artefact = sum(1 for r in rows if r.get("marked_artefact"))
    committed = [r for r in rows if r.get("parsed") and not r.get("abstained") and not r.get("marked_artefact")]
    precision_committed = (sum(1 for r in committed if r["hit"]) / len(committed)) if committed else float("nan")

    # Artefact-flag agreement on tier=artifact cells
    art_tier = [r for r in rows if r.get("tier") == "artifact" and r.get("parsed")]
    artefact_recall = (sum(1 for r in art_tier if r.get("marked_artefact")) / len(art_tier)) if art_tier else float("nan")

    per_tier: Dict[str, Dict] = defaultdict(lambda: {"n": 0, "hit": 0, "abstain": 0, "artefact": 0})
    for r in rows:
        if not r.get("parsed"):
            continue
        t = r.get("tier") or "unknown"
        per_tier[t]["n"] += 1
        if r.get("hit"): per_tier[t]["hit"] += 1
        if r.get("abstained"): per_tier[t]["abstain"] += 1
        if r.get("marked_artefact"): per_tier[t]["artefact"] += 1

    summary = {
        "n_total": total,
        "n_parsed": parsed,
        "parse_rate": parsed / total if total else 0.0,
        "abstention_rate": abstain / parsed if parsed else float("nan"),
        "artefact_flag_rate": artefact / parsed if parsed else float("nan"),
        "singleton_precision_among_committed": precision_committed,
        "artefact_recall_on_artifact_tier": artefact_recall,
        "per_tier": {k: dict(v) for k, v in per_tier.items()},
    }

    Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print("=== VLM audit summary ===")
    for k, v in summary.items():
        if k == "per_tier":
            continue
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")
    print("  per_tier:")
    for t, d in summary["per_tier"].items():
        acc = d["hit"] / d["n"] if d["n"] else float("nan")
        abst = d["abstain"] / d["n"] if d["n"] else float("nan")
        artf = d["artefact"] / d["n"] if d["n"] else float("nan")
        print(f"    {t:<10} n={d['n']:>3}  acc={acc:.3f}  abstain={abst:.3f}  artefact={artf:.3f}")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
