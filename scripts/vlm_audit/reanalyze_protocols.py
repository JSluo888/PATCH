#!/usr/bin/env python3
"""Re-score the 100-cell VLM audit under four selective-conformal protocols.

The original ``score_audit.py`` reports a single number — ``hit`` =
``verdict == gt_v6`` — which gives 28 / 100 and treats artefact-flagging
and ancestor-correct backoffs as failures.  That mis-credits the VLM:

* 11 hard-tier cells are explicit multiplet artefacts (correctly flagged).
* 7 cells have explicit GT errors per the three-way evaluation
  (expert gating contradicts the marker pattern).
* Many ``backoff`` verdicts (``T_cell`` when GT is ``CD4_T``) are
  ontology-correct ancestors — exactly what the panel-aware ontology
  is designed to allow.

This script re-scores the same 100 responses under four protocols and
emits a single JSON + a markdown report so we can pick the rule that
preserves conformal coverage at the smallest expected set size.

Protocols
---------
P0 NAIVE
    hit := verdict == gt_v6           (the original 28/100 number).

P1 ARTEFACT-FILTERED  (selective conformal)
    Cells the VLM flags as ``artefact`` are removed from the evaluation
    pool entirely; the remaining cells are scored as in P0.  This
    corresponds to a selective predictor that abstains on flagged
    multiplets — coverage is preserved on the *retained* sub-population.

P2 CONFIDENCE-GATED ABSTENTION
    Commit only when ``confidence == high``; otherwise return the
    conformal set (== no-commit).  Singleton precision on the committed
    subset is the relevant guarantee.

P3 ONTOLOGY-AWARE
    Credit a verdict that is an ancestor of the GT (or equal): e.g.
    verdict ``T_cell`` matches GT ``CD4_T`` via the v6 tree.  This is
    the metric the panel-aware ontology was designed to support.

P1+P2+P3 COMBINED
    Selective + confidence-gated + ontology-aware credit.  Reports the
    "best honest use" of the VLM.

Also reports per-tier breakdowns and the artefact-detection F1 against
the artifact tier.

Outputs
-------
results/vlm_eval/protocol_reanalysis.json
results/vlm_eval/protocol_reanalysis.md
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parents[2]
THREE_WAY = REPO / "results/vlm_eval/three_way_evaluation.json"
OUT_DIR = REPO / "results/vlm_eval"

# Default audit file is overridable via --audit / --tag for cross-run comparison.

sys.path.insert(0, str(REPO))
from configs import ontology_v6 as o


def ancestors_of(label: str) -> List[str]:
    """Return ancestors INCLUDING self, walking up the v6 tree.  Empty list if unknown."""
    if label not in o.CELL_TYPE_ONTOLOGY:
        return []
    return o.ancestors(label, include_self=True)


def is_ancestor_or_self(verdict: str, gt: str) -> bool:
    if verdict == gt:
        return True
    anc = ancestors_of(gt)
    return verdict in anc


def load_three_way_gt_errors() -> set[str]:
    if not THREE_WAY.exists():
        return set()
    payload = json.loads(THREE_WAY.read_text())
    return set(payload.get("key_findings", {}).get("gt_errors_identified", []))


@dataclass
class ProtocolResult:
    name: str
    description: str
    n_pool: int           # how many cells the protocol scores (after any filtering)
    n_commit: int         # how many commits the VLM made within the pool
    n_correct: int        # how many commits were correct (under the protocol's hit rule)
    n_artefact_flagged: int
    abstention_rate: float
    singleton_precision_among_committed: float
    overall_accuracy: float


def score(rows: List[dict], protocol: str, gt_error_ids: set[str]) -> ProtocolResult:
    """Score one protocol over the 100-cell audit."""
    n_total = len(rows)

    # Selection: which cells does this protocol score?
    if protocol in ("P1", "P1+P2+P3"):
        pool = [r for r in rows if not r["marked_artefact"]]
    else:
        pool = list(rows)

    # Confidence-gating: convert non-high commits to abstain-equivalent
    if protocol in ("P2", "P1+P2+P3"):
        def is_committed(r: dict) -> bool:
            return (
                not r["marked_artefact"]
                and not r["abstained"]
                and (r.get("confidence") == "high")
            )
    else:
        def is_committed(r: dict) -> bool:
            return not r["marked_artefact"] and not r["abstained"]

    # Hit rule
    use_ancestor = protocol in ("P3", "P1+P2+P3")

    def hit(r: dict) -> bool:
        if not is_committed(r):
            return False
        if use_ancestor:
            return is_ancestor_or_self(r["verdict"], r["gt_v6"])
        return r["verdict"] == r["gt_v6"]

    commits = [r for r in pool if is_committed(r)]
    n_commit = len(commits)
    n_correct = sum(1 for r in commits if hit(r))
    n_artefact = sum(1 for r in pool if r["marked_artefact"])
    abstention_rate = 1.0 - (n_commit / max(1, len(pool)))

    return ProtocolResult(
        name=protocol,
        description={
            "P0": "Naive: hit = (verdict == gt_v6)",
            "P1": "Artefact-filtered: drop VLM-flagged artefacts, then P0",
            "P2": "Confidence-gated: only count commits with confidence=high",
            "P3": "Ontology-aware: credit ancestor matches via v6 tree",
            "P1+P2+P3": "Selective + confidence-gated + ontology-aware (combined)",
        }[protocol],
        n_pool=len(pool),
        n_commit=n_commit,
        n_correct=n_correct,
        n_artefact_flagged=n_artefact,
        abstention_rate=abstention_rate,
        singleton_precision_among_committed=n_correct / max(1, n_commit),
        overall_accuracy=n_correct / max(1, len(pool)),
    )


def artefact_detection_f1(rows: List[dict]) -> dict:
    """Treat VLM ``marked_artefact`` as a binary detector for the
    artifact tier.  Reports precision / recall / F1."""
    y_true = [r["tier"] == "artifact" for r in rows]
    y_pred = [r["marked_artefact"] for r in rows]
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if (not t) and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and (not p))
    tn = sum(1 for t, p in zip(y_true, y_pred) if (not t) and (not p))
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "f1": f1}


def per_tier_breakdown(rows: List[dict], protocol: str, gt_error_ids: set[str]) -> dict:
    """P1+P2+P3 metrics split by tier."""
    out = {}
    for tier in ("easy", "medium", "hard", "artifact"):
        tier_rows = [r for r in rows if r["tier"] == tier]
        if not tier_rows:
            continue
        res = score(tier_rows, protocol, gt_error_ids)
        out[tier] = {
            "n_total": len(tier_rows),
            "n_pool": res.n_pool,
            "n_commit": res.n_commit,
            "n_correct": res.n_correct,
            "precision_among_committed": res.singleton_precision_among_committed,
            "overall_accuracy_on_pool": res.overall_accuracy,
            "abstention_rate": res.abstention_rate,
        }
    return out


def gt_error_aware_p1p2p3(rows: List[dict], gt_error_ids: set[str]) -> dict:
    """Re-run P1+P2+P3 excluding cells flagged as GT errors in the
    three-way eval.  Provides an upper bound on what the VLM could
    have achieved if GT were trustworthy."""
    clean = [r for r in rows if r["cell_id"] not in gt_error_ids]
    res = score(clean, "P1+P2+P3", gt_error_ids)
    return {
        "n_excluded_gt_errors": len(rows) - len(clean),
        "result": {
            "name": "P1+P2+P3 (GT-clean)",
            "description": "P1+P2+P3 after removing cells with explicit GT errors per three-way eval",
            "n_pool": res.n_pool,
            "n_commit": res.n_commit,
            "n_correct": res.n_correct,
            "precision_among_committed": res.singleton_precision_among_committed,
            "overall_accuracy_on_pool": res.overall_accuracy,
            "abstention_rate": res.abstention_rate,
        },
    }


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default=str(REPO / "vlm_audit_results.json"),
                    help="Audit JSON to score (default: post-fix vlm_audit_results.json)")
    ap.add_argument("--tag", default="post_fix",
                    help="Short tag used in output filenames (e.g. post_fix, pre_fix, qwen3vl)")
    args = ap.parse_args()

    out_json = OUT_DIR / f"protocol_reanalysis_{args.tag}.json"
    out_md = OUT_DIR / f"protocol_reanalysis_{args.tag}.md"

    rows = json.loads(Path(args.audit).read_text())["rows"]
    gt_errors = load_three_way_gt_errors()

    protocols = ["P0", "P1", "P2", "P3", "P1+P2+P3"]
    results = {p: score(rows, p, gt_errors).__dict__ for p in protocols}

    per_tier = per_tier_breakdown(rows, "P1+P2+P3", gt_errors)
    art_det = artefact_detection_f1(rows)
    gt_clean = gt_error_aware_p1p2p3(rows, gt_errors)

    payload = {
        "n_cells": len(rows),
        "gt_errors_identified": sorted(gt_errors),
        "protocols": results,
        "per_tier_P1P2P3": per_tier,
        "artefact_detection": art_det,
        "gt_error_aware_upper_bound": gt_clean,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))

    def fmt(x: float) -> str:
        return f"{x*100:.1f}%"

    md_lines: List[str] = []
    md_lines.append("# 100-cell VLM audit — protocol re-analysis\n")
    md_lines.append(
        "Re-scoring of `vlm_audit_results.json` under selective-conformal "
        "protocols.  The original report (`P0`) credits only exact-leaf "
        "matches against the v6 GT, which under-counts the VLM by treating "
        "ancestor backoffs and correctly-flagged multiplets as failures.\n"
    )

    md_lines.append("## Headline\n")
    md_lines.append(
        "| Protocol | Pool | Commit | Correct | Precision (committed) | Overall accuracy |\n"
        "|---|---:|---:|---:|---:|---:|\n"
    )
    for p in protocols:
        r = results[p]
        md_lines.append(
            f"| **{p}** — {r['description']} | "
            f"{r['n_pool']} | {r['n_commit']} | {r['n_correct']} | "
            f"{fmt(r['singleton_precision_among_committed'])} | "
            f"{fmt(r['overall_accuracy'])} |\n"
        )

    md_lines.append(
        "\nThe ``commit`` column is the number of cells where the VLM "
        "actually returned a committed cell-type verdict under the "
        "protocol (i.e. not abstained, not flagged as artefact, "
        "and meeting the confidence gate when one applies).\n"
    )

    md_lines.append("\n## P1+P2+P3 per tier\n")
    md_lines.append(
        "| Tier | n | pool | commit | correct | precision (committed) |\n"
        "|---|---:|---:|---:|---:|---:|\n"
    )
    for tier, blob in per_tier.items():
        md_lines.append(
            f"| {tier} | {blob['n_total']} | {blob['n_pool']} | "
            f"{blob['n_commit']} | {blob['n_correct']} | "
            f"{fmt(blob['precision_among_committed'])} |\n"
        )

    md_lines.append("\n## VLM artefact detection (treating tier=artifact as positive)\n")
    md_lines.append(
        f"- TP={art_det['tp']}, FP={art_det['fp']}, "
        f"FN={art_det['fn']}, TN={art_det['tn']}\n"
        f"- Precision={fmt(art_det['precision'])}, "
        f"Recall={fmt(art_det['recall'])}, "
        f"F1={fmt(art_det['f1'])}\n"
        "\nThis is the artefact-gate the selective-conformal protocol (P1) is "
        "built on.  Recall is the key number — every artefact missed by the gate "
        "is a multiplet that downstream conformal will try to type.\n"
    )

    md_lines.append("\n## GT-error-aware upper bound\n")
    md_lines.append(
        f"- Excluded {gt_clean['n_excluded_gt_errors']} cells flagged in "
        "`results/vlm_eval/three_way_evaluation.json` as explicit GT errors.\n"
    )
    r = gt_clean["result"]
    md_lines.append(
        f"- After exclusion, P1+P2+P3 precision among committed = "
        f"{fmt(r['precision_among_committed'])}, "
        f"overall = {fmt(r['overall_accuracy_on_pool'])}.\n"
    )

    md_lines.append("\n## Interpretation\n")
    md_lines.append(
        "1. The original 28/100 number conflates four distinct VLM actions "
        "(commit, abstain, backoff-to-ancestor, artefact-flag).  Once those "
        "are separated, the VLM's behaviour is what selective / hierarchical "
        "conformal predictors are designed to exploit.\n"
        "2. **P1 (artefact filter)** preserves coverage on the retained "
        "sub-population *by construction*: any cell the VLM flags is removed "
        "from the conformal evaluation, just like a selective predictor.  "
        "The empirical precision on retained commits is the relevant number.\n"
        "3. **P3 (ontology credit)** is the right metric for the panel-aware "
        "ontology — backing off from `CD4_T` to `T_cell` when CD4 evidence is "
        "weak is *correct*, not wrong.  P3 picks this up; P0 does not.\n"
        "4. The **GT-error-aware upper bound** quantifies how much of the "
        "remaining gap is the VLM's fault vs. the gating-derived GT being "
        "internally inconsistent.\n"
        "5. Next experiment: scale the audit (500–1000 cells) on Modal with "
        "Qwen3-VL using the post-fix input format, calibrate a confidence "
        "threshold τ on the first half, validate selective conformal coverage "
        "on the second.\n"
    )

    out_md.write_text("".join(md_lines))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    print()
    print("--- Headline (precision among committed | overall) ---")
    for p in protocols:
        r = results[p]
        print(
            f"  {p:<10s}  n_commit={r['n_commit']:>3d}  "
            f"prec={fmt(r['singleton_precision_among_committed']):>6s}  "
            f"overall={fmt(r['overall_accuracy']):>6s}"
        )


if __name__ == "__main__":
    main()
