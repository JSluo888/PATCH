"""Unit tests for the coverage-preserving VLM set refiner.

These tests pin down the four structural behaviours of
``refine_conformal_set``:

1. VLM artefact → status DROP.
2. VLM commit ∈ S and confidence ≥ threshold → status REFINED with
   ``refined_set == {commit}``.
3. VLM commit ∉ S → status FALLBACK; the conformal set is returned
   unchanged (the refiner NEVER expands the set).
4. VLM did not commit (abstain / low conf / unparsed) → status FALLBACK.

A small synthetic-population test then checks that empirical coverage
on the retained sub-population is at or above target when the refiner
is applied to a population with known coverage.
"""
from __future__ import annotations

import random
from typing import List, Tuple

import pytest

from uncertainty.vlm_set_refiner import (
    PopulationSummary,
    RefinementResult,
    RefinerStatus,
    VLMVerdict,
    refine_conformal_set,
    summarize_population,
)


# ---------------------------------------------------------------------------
# Structural tests — one per protocol branch.
# ---------------------------------------------------------------------------


def test_artefact_flag_drops_cell():
    S = {"CD4_T", "CD8_T", "Treg"}
    vlm = VLMVerdict(verdict="artefact", confidence="high", marked_artefact=True)
    res = refine_conformal_set(S, vlm)
    assert res.status is RefinerStatus.DROP
    # Even on DROP we expose the original set so callers can decide what to log.
    assert res.refined_set == frozenset(S)
    assert "artefact" in res.reason


def test_commit_in_set_high_conf_refines_to_singleton():
    S = {"CD4_T", "CD8_T", "Treg"}
    vlm = VLMVerdict(verdict="CD4_T", confidence="high")
    res = refine_conformal_set(S, vlm)
    assert res.status is RefinerStatus.REFINED
    assert res.refined_set == frozenset({"CD4_T"})
    assert res.shrunk is True


def test_commit_outside_set_falls_back_to_S():
    S = {"CD4_T", "CD8_T"}
    vlm = VLMVerdict(verdict="B_cell", confidence="high")
    res = refine_conformal_set(S, vlm)
    assert res.status is RefinerStatus.FALLBACK
    # CRUCIAL: refiner must NEVER add classes to S.
    assert res.refined_set == frozenset(S)
    assert "not in conformal set" in res.reason


def test_low_confidence_commit_falls_back():
    S = {"CD4_T", "CD8_T", "Treg"}
    vlm = VLMVerdict(verdict="CD4_T", confidence="low")
    res = refine_conformal_set(S, vlm)
    assert res.status is RefinerStatus.FALLBACK
    assert res.refined_set == frozenset(S)


def test_medium_confidence_commit_falls_back_with_default_threshold():
    S = {"CD4_T", "CD8_T", "Treg"}
    vlm = VLMVerdict(verdict="CD4_T", confidence="medium")
    res = refine_conformal_set(S, vlm)
    assert res.status is RefinerStatus.FALLBACK


def test_medium_confidence_commit_refines_when_threshold_relaxed():
    S = {"CD4_T", "CD8_T", "Treg"}
    vlm = VLMVerdict(verdict="CD4_T", confidence="medium")
    res = refine_conformal_set(S, vlm, commit_confidence="medium")
    assert res.status is RefinerStatus.REFINED
    assert res.refined_set == frozenset({"CD4_T"})


def test_abstain_falls_back():
    S = {"CD4_T", "CD8_T"}
    vlm = VLMVerdict(verdict="abstain", confidence="high", abstained=True)
    res = refine_conformal_set(S, vlm)
    assert res.status is RefinerStatus.FALLBACK
    assert res.refined_set == frozenset(S)


def test_unparsed_response_falls_back():
    S = {"CD4_T", "CD8_T"}
    vlm = VLMVerdict(verdict="", confidence="high", parsed=False)
    res = refine_conformal_set(S, vlm)
    assert res.status is RefinerStatus.FALLBACK
    assert res.refined_set == frozenset(S)


def test_empty_set_is_rejected():
    vlm = VLMVerdict(verdict="CD4_T", confidence="high")
    with pytest.raises(ValueError, match="non-empty"):
        refine_conformal_set([], vlm)


def test_singleton_set_passthrough_on_matching_commit():
    """A 1-element S with a matching high-conf commit is still REFINED
    (the refined set equals the input set, but the status is correctly
    REFINED rather than FALLBACK)."""
    S = {"CD4_T"}
    vlm = VLMVerdict(verdict="CD4_T", confidence="high")
    res = refine_conformal_set(S, vlm)
    assert res.status is RefinerStatus.REFINED
    assert res.refined_set == frozenset({"CD4_T"})


def test_singleton_set_with_disagreeing_commit_falls_back():
    S = {"CD4_T"}
    vlm = VLMVerdict(verdict="CD8_T", confidence="high")
    res = refine_conformal_set(S, vlm)
    assert res.status is RefinerStatus.FALLBACK
    assert res.refined_set == frozenset({"CD4_T"})


# ---------------------------------------------------------------------------
# Set-cannot-expand invariant (sweep test).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "S, verdict, conf, artefact, abstained, parsed",
    [
        (["CD4_T", "CD8_T"], "B_cell", "high", False, False, True),
        (["CD4_T"], "CD8_T", "high", False, False, True),
        (["Macrophage_CD163pos"], "Endothelial", "low", False, False, True),
        (["CD4_T", "CD8_T", "Treg"], "CD4_T", "low", False, False, True),
        (["CD4_T", "CD8_T"], "artefact", "high", True, False, True),
        (["CD4_T", "CD8_T"], "abstain", "high", False, True, True),
        (["CD4_T", "CD8_T"], "", "", False, False, False),
    ],
)
def test_refined_set_is_always_subset_of_input(
    S, verdict, conf, artefact, abstained, parsed
):
    """The refiner must NEVER add classes.  Sweep over many inputs."""
    vlm = VLMVerdict(
        verdict=verdict,
        confidence=conf,
        parsed=parsed,
        marked_artefact=artefact,
        abstained=abstained,
    )
    res = refine_conformal_set(S, vlm)
    assert res.refined_set.issubset(frozenset(S)), (
        f"Refiner expanded set! S={S}, refined={res.refined_set}, status={res.status}"
    )


# ---------------------------------------------------------------------------
# Synthetic coverage-preservation experiment.
# ---------------------------------------------------------------------------


def _generate_synthetic_population(
    n: int,
    target_coverage: float,
    seed: int,
) -> List[Tuple[List[str], VLMVerdict, str]]:
    r"""Build a synthetic population.

    Construction:

    * Each cell has a ground-truth label drawn uniformly from
      ``{A, B, C, D}``.
    * The conformal set is constructed so that exactly
      ``target_coverage`` fraction of cells have ``y \in S``; on those
      cells, S contains 2 random labels including y; on the rest, S is
      2 random labels excluding y.  This matches a calibrated CP at
      the target coverage by construction.
    * The VLM verdict is generated independently with:
        - 20 % of cells marked as artefact;
        - of the remaining 80 %, the VLM commits to the true label with
          probability ``p_correct = 0.6`` and a random wrong label
          otherwise; confidence is "high" 60 % / "medium" 30 % / "low" 10 %.
    """
    rng = random.Random(seed)
    labels = ["A", "B", "C", "D"]
    population: List[Tuple[List[str], VLMVerdict, str]] = []

    for _ in range(n):
        y = rng.choice(labels)
        if rng.random() < target_coverage:
            wrong = [c for c in labels if c != y]
            S = [y, rng.choice(wrong)]
            rng.shuffle(S)
        else:
            wrong = [c for c in labels if c != y]
            rng.shuffle(wrong)
            S = wrong[:2]

        if rng.random() < 0.20:
            vlm = VLMVerdict(
                verdict="artefact", confidence="high", marked_artefact=True
            )
        else:
            if rng.random() < 0.60:
                commit = y
            else:
                commit = rng.choice([c for c in labels if c != y])
            conf_r = rng.random()
            conf = "high" if conf_r < 0.6 else ("medium" if conf_r < 0.9 else "low")
            vlm = VLMVerdict(verdict=commit, confidence=conf)

        population.append((S, vlm, y))

    return population


def test_synthetic_population_preserves_input_coverage():
    """The refiner cannot improve coverage past the input level —
    it can only preserve or degrade.  Verify the empirical retained
    coverage is bounded above by the input coverage."""
    pop = _generate_synthetic_population(n=2000, target_coverage=0.90, seed=42)
    summary = summarize_population(pop, commit_confidence="high")

    assert summary.n_total == 2000
    # Input coverage should be near the target (within sampling noise).
    assert summary.coverage_full == pytest.approx(0.90, abs=0.03)

    # Retained coverage cannot exceed the input coverage — refinement
    # only ever shrinks the set, so any cell whose y was already outside
    # S is still uncovered.  Coverage can DECREASE if the VLM picks the
    # wrong class out of a 2-set.
    assert summary.coverage_retained <= summary.coverage_full + 0.01


def test_refined_only_coverage_reflects_vlm_quality():
    """On the REFINED subset (where the VLM committed inside S), the
    empirical coverage equals P(correct | commit in S).

    By Bayes, this is greater than P(correct) = 0.6 because commits
    that land inside S are biased toward correct ones:

        P(L | C) = P(y in S) = 0.90                 (CP target coverage)
        P(L | ~C) = 0.9 * 1/3 + 0.1 * 2/3 ~= 0.367  (4-label uniform wrongs)
        P(C | L) = 0.54 / (0.54 + 0.4*0.367) ~= 0.786

    So expected ~79 % conditional accuracy when (commit in S)."""
    pop = _generate_synthetic_population(n=4000, target_coverage=0.90, seed=7)
    summary = summarize_population(pop, commit_confidence="high")
    assert summary.coverage_refined_only == pytest.approx(0.786, abs=0.05)


def test_refined_set_size_strictly_smaller_than_input_on_refined_subset():
    """On REFINED cells, |S'| = 1 < |S| ≥ 2 always."""
    pop = _generate_synthetic_population(n=1000, target_coverage=0.90, seed=13)
    refined_count = 0
    for S, vlm, _ in pop:
        res = refine_conformal_set(S, vlm)
        if res.status is RefinerStatus.REFINED:
            refined_count += 1
            assert len(res.refined_set) == 1
            assert len(res.refined_set) < len(set(S))
    # Sanity: at least some cells were refined.
    assert refined_count > 0
