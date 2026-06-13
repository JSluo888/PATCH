r"""Coverage-preserving VLM set refiner.

This module operationalizes the protocol in ``docs/conformal_vlm_design.md``.
Given a conformal prediction set :math:`S(x)` produced by an upstream
predictor (the XGBoost / hierarchical-conformal head in PATCH), and a VLM
verdict carrying a committed cell-type, a confidence level, and an
artefact flag, the refiner returns one of:

* ``Status.DROP`` — the VLM flagged the cell as an artefact / multiplet
  / mask error.  The selective-conformal protocol removes the cell from
  the labelled pool; coverage on the retained sub-population is
  preserved exactly when the gate is independent of the conformal score
  (Mondrian selectivity).
* ``Status.REFINED`` — the VLM committed to a leaf that is *already in*
  :math:`S(x)` with high confidence.  The refined set
  :math:`S'(x) = \{\text{commit}\}` is returned.  Coverage is preserved
  by the structural argument: if :math:`y \in S(x)` and
  :math:`y = \text{commit}`, then :math:`y \in S'(x)`; if
  :math:`y \in S(x) \setminus \{\text{commit}\}` the refinement is
  *wrong* by definition and we fall back (see below).
* ``Status.FALLBACK`` — the VLM either committed outside :math:`S(x)`,
  abstained, returned low/medium confidence, or did not produce a
  parseable verdict.  The original conformal set is returned unchanged.

Critically, the refiner **never adds classes to** :math:`S(x)`.  By
construction :math:`S'(x) \subseteq S(x)`, so coverage at level
:math:`1-\alpha` is preserved up to the per-cell error rate of the
refinement decision — which the audit-loop measures empirically.

This module has no side effects and no external dependencies beyond the
standard library + ``dataclasses``.  It is deliberately decoupled from
the VLM transport (HTTP / Modal / Gemini API) so it can be unit-tested
without network access.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import FrozenSet, Iterable, Optional, Set, Tuple


class RefinerStatus(Enum):
    """Three structural outcomes of the refiner."""

    DROP = "drop"
    REFINED = "refined"
    FALLBACK = "fallback"


@dataclass(frozen=True)
class VLMVerdict:
    """The parsed VLM response carried by the audit YAML.

    All fields mirror the schema in ``scripts/vlm_audit/system_prompt.md``.
    ``verdict`` may be a cell-type label, or one of the non-commit
    sentinels ``"artefact"`` / ``"abstain"``.
    """

    verdict: str
    confidence: str  # "high" | "medium" | "low"
    parsed: bool = True
    marked_artefact: bool = False
    abstained: bool = False

    def __post_init__(self) -> None:  # pragma: no cover - trivial assertions
        # Validate confidence
        if self.confidence not in {"high", "medium", "low", ""}:
            raise ValueError(
                f"VLMVerdict.confidence must be high/medium/low, got {self.confidence!r}"
            )

    @property
    def is_commit(self) -> bool:
        """True iff the VLM committed to a real cell-type label."""
        return (
            self.parsed
            and not self.marked_artefact
            and not self.abstained
            and self.verdict not in ("", "artefact", "abstain")
        )


def vlm_action_to_verdict(
    action_value: str,
    top1: Optional[str] = None,  # reserved for a future labelled COMMIT action
) -> Optional["VLMVerdict"]:
    """Bridge a symbolic VLM *action* to a refiner-consumable :class:`VLMVerdict`.

    The agentic loop (``orchestrator/agentic_loop_v6.py``) lets the VLM emit only
    symbolic actions (``VLMActionType``); the controller owns the official set.
    This bridge maps the *terminal* action to the verdict the set-refiner needs,
    WITHOUT importing anything from ``orchestrator`` (we take the action's string
    value, not the enum object, so there is no circular import).

    Mapping:

    * ``"artifact"`` -> artefact verdict -> refiner returns ``DROP`` (the cell is
      removed from the labelled pool; selective-conformal coverage on the retained
      sub-population is preserved).
    * everything else (``stop`` / ``acquire_view`` / ``request_reeval`` /
      ``propose_challenge``) -> ``None``: no refinement.

    Why ``stop`` does NOT produce a commit: a symbolic ``VLMAction`` carries no
    cell-type label, so synthesising a singleton commit would inject a label the
    VLM never named and would violate the controller-owns-the-set invariant. The
    singleton-REFINED path is exercised only by the *direct* labelled-verdict API
    (``refine_conformal_set(S, VLMVerdict(verdict=<label>, ...))``) used in the
    standalone VLM audit / E3 pipeline, where the VLM emits an explicit label.
    The ``top1`` parameter is retained (reserved) for a future labelled COMMIT
    action but is intentionally unused here.

    Parameters
    ----------
    action_value
        ``VLMActionType(...).value`` string (e.g. ``"artifact"``, ``"stop"``).
    top1
        Reserved; unused (see above).
    """
    av = (action_value or "").strip().lower()
    if av == "artifact":
        return VLMVerdict(verdict="artefact", confidence="high", marked_artefact=True)
    return None


@dataclass(frozen=True)
class RefinementResult:
    refined_set: FrozenSet[str]
    status: RefinerStatus
    reason: str

    @property
    def shrunk(self) -> bool:
        """True iff the refined set is strictly smaller than the input set."""
        return self.status is RefinerStatus.REFINED


def refine_conformal_set(
    conformal_set: Iterable[str],
    vlm: VLMVerdict,
    *,
    commit_confidence: str = "high",
) -> RefinementResult:
    """Refine :math:`S(x)` using the VLM verdict, preserving coverage.

    Parameters
    ----------
    conformal_set
        The upstream conformal prediction set :math:`S(x)`.  Any
        non-empty iterable of cell-type label strings.
    vlm
        Parsed VLM verdict.  Must be a ``VLMVerdict`` instance.
    commit_confidence
        Minimum VLM-reported confidence required for a commit to
        actually shrink the set.  Default ``"high"`` mirrors the
        protocol P1+P2+P3 in the design doc; passing ``"medium"`` would
        admit medium-confidence commits as well.

    Returns
    -------
    RefinementResult
        Always returns a frozenset.  ``refined_set`` is empty only when
        the input set was empty (which the upstream conformal layer
        guarantees never happens — included here for defensiveness).

    Raises
    ------
    ValueError
        If ``conformal_set`` is empty (caller bug) or
        ``commit_confidence`` is not one of high/medium/low.

    Notes
    -----
    The four observable behaviours (matching the protocol):

    1. VLM artefact → ``DROP``.
    2. VLM commit ∈ S and confidence ≥ threshold → ``REFINED`` with
       singleton set ``{commit}``.
    3. VLM commit ∉ S → ``FALLBACK`` (the upstream set is the safe
       answer; never expand).
    4. VLM did not commit (abstain / low conf / unparsed) → ``FALLBACK``.
    """
    S: FrozenSet[str] = frozenset(conformal_set)
    if not S:
        raise ValueError("conformal_set must be non-empty")
    if commit_confidence not in {"high", "medium", "low"}:
        raise ValueError(
            f"commit_confidence must be high/medium/low, got {commit_confidence!r}"
        )

    # 1. Artefact gate — selective drop.
    if vlm.marked_artefact:
        return RefinementResult(
            refined_set=S,            # surface the original set even on DROP
            status=RefinerStatus.DROP,
            reason="vlm marked cell as artefact / multiplet",
        )

    # 4. No commit (unparsed / abstain).
    if not vlm.is_commit:
        return RefinementResult(
            refined_set=S,
            status=RefinerStatus.FALLBACK,
            reason="vlm did not commit (abstain or unparsed)",
        )

    # Enforce minimum confidence for a commit to count.
    rank = {"high": 2, "medium": 1, "low": 0}
    if rank[vlm.confidence] < rank[commit_confidence]:
        return RefinementResult(
            refined_set=S,
            status=RefinerStatus.FALLBACK,
            reason=f"vlm confidence {vlm.confidence} below threshold {commit_confidence}",
        )

    # 3. Commit outside S — never expand; safe fallback to S.
    if vlm.verdict not in S:
        return RefinementResult(
            refined_set=S,
            status=RefinerStatus.FALLBACK,
            reason=f"vlm commit '{vlm.verdict}' not in conformal set",
        )

    # 2. Refine to singleton.
    return RefinementResult(
        refined_set=frozenset({vlm.verdict}),
        status=RefinerStatus.REFINED,
        reason=f"vlm singleton commit '{vlm.verdict}' inside conformal set",
    )


# ---------------------------------------------------------------------------
# Population-level summary helpers — useful for the synthetic coverage check.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PopulationSummary:
    """Empirical metrics over a population of refinements."""

    n_total: int
    n_drop: int
    n_refined: int
    n_fallback: int
    avg_set_size_input: float
    avg_set_size_refined: float
    coverage_full: float           # fraction y \in S(x) over n_total
    coverage_retained: float       # fraction y \in S'(x) over (n_total - n_drop)
    coverage_refined_only: float   # fraction y \in S'(x) restricted to REFINED

    @property
    def set_size_reduction(self) -> float:
        return self.avg_set_size_input - self.avg_set_size_refined


def summarize_population(
    items: Iterable[Tuple[Iterable[str], VLMVerdict, str]],
    *,
    commit_confidence: str = "high",
) -> PopulationSummary:
    """Roll the refiner over a population and report empirical coverage.

    Each ``items`` element is ``(conformal_set, vlm_verdict, ground_truth)``.
    ``ground_truth`` is the true label for that cell — only used to
    measure coverage; it is *not* fed to the refiner.
    """
    n_total = 0
    n_drop = 0
    n_refined = 0
    n_fallback = 0
    s_in_sum = 0
    s_out_sum = 0
    cov_full_hit = 0
    cov_retained_hit = 0
    cov_refined_hit = 0
    n_retained = 0
    n_refined_only = 0

    for S, vlm, y in items:
        S = frozenset(S)
        res = refine_conformal_set(S, vlm, commit_confidence=commit_confidence)
        n_total += 1
        s_in_sum += len(S)
        if res.status is RefinerStatus.DROP:
            n_drop += 1
            # We do not count dropped cells in retained coverage
            if y in S:
                cov_full_hit += 1
            continue
        # not dropped
        n_retained += 1
        if y in S:
            cov_full_hit += 1
        if y in res.refined_set:
            cov_retained_hit += 1
        if res.status is RefinerStatus.REFINED:
            n_refined += 1
            n_refined_only += 1
            if y in res.refined_set:
                cov_refined_hit += 1
        elif res.status is RefinerStatus.FALLBACK:
            n_fallback += 1
        s_out_sum += len(res.refined_set)

    return PopulationSummary(
        n_total=n_total,
        n_drop=n_drop,
        n_refined=n_refined,
        n_fallback=n_fallback,
        avg_set_size_input=s_in_sum / max(1, n_total),
        avg_set_size_refined=s_out_sum / max(1, n_total - n_drop),
        coverage_full=cov_full_hit / max(1, n_total),
        coverage_retained=cov_retained_hit / max(1, n_retained),
        coverage_refined_only=cov_refined_hit / max(1, n_refined_only),
    )
