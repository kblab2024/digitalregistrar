"""Tests for preann.py — Δκ, anchoring index, convergence-to-preann."""
from __future__ import annotations

import pytest

pytest.importorskip("sklearn", reason="sklearn provides cohen_kappa_score for paired_delta_kappa")

from digital_registrar_research.benchmarks.eval.preann import (  # noqa: E402
    PairedRecord,
    anchoring_index,
    convergence_to_preann,
    paired_delta_kappa,
)


def _record(case, with_v, without_v, preann_v, gold_v):
    return PairedRecord(
        case_id=case, organ="breast", field="grade",
        with_value=with_v, without_value=without_v,
        preann_value=preann_v, gold_value=gold_v,
    )


def test_delta_kappa_positive_when_with_better():
    """Δκ > 0 when with-preann humans agree better with gold than
    without-preann humans on the same paired cohort.

    Cohen's κ requires at least two distinct gold classes, so the
    fixture mixes grade=2 and grade=3 cases. With-preann humans match
    gold; without-preann humans flip half the time.
    """
    records = []
    for i in range(20):
        gold = "2" if i % 2 == 0 else "3"
        records.append(_record(
            f"c{i}",
            with_v=gold,                                 # always matches gold
            without_v=(gold if i % 4 == 0 else "2"),     # flips ~75% of cases
            preann_v=gold,
            gold_v=gold,
        ))
    result = paired_delta_kappa(records, n_boot=200, random_state=0)
    assert result["n_paired_cases"] == 20
    assert result["delta"] > 0
    # Sanity: with-preann humans matched gold perfectly
    assert result["kappa_with"] > 0.9


def test_convergence_to_preann_high_when_human_copies():
    """If human-with always equals preann, convergence rate is 1.0."""
    records = [
        _record(f"c{i}", with_v="2", without_v="3",
                preann_v="2", gold_v="2")
        for i in range(10)
    ]
    result = convergence_to_preann(records)
    assert result["p_human_eq_preann"] == 1.0
    assert result["n_with_preann"] == 10


def test_anchoring_index_zero_when_no_anchoring():
    """Both with and without humans match preann with same probability:
    AI ≈ 0."""
    records = [
        # 10 cases where human matches preann in both modes
        _record(f"c{i}", with_v="2", without_v="2",
                preann_v="2", gold_v="2")
        for i in range(10)
    ]
    result = anchoring_index(records)
    assert abs(result["ai_overall"]) < 1e-9


def test_anchoring_index_positive_when_with_aligns():
    """With humans copy preann; without humans diverge.  AI > 0."""
    records = (
        [_record(f"w{i}", with_v="2", without_v="3",
                 preann_v="2", gold_v="2") for i in range(10)]
    )
    result = anchoring_index(records)
    assert result["ai_overall"] == 1.0  # P(=preann | with)=1 minus P(=preann | without)=0


def test_convergence_split_by_correctness():
    """Stratify by whether preann was correct."""
    records = [
        # preann correct, human matches preann → increments correct_n + correct_matched
        _record("c1", with_v="2", without_v="2", preann_v="2", gold_v="2"),
        _record("c2", with_v="2", without_v="2", preann_v="2", gold_v="2"),
        # preann wrong, human matches preann → bad anchoring
        _record("c3", with_v="3", without_v="2", preann_v="3", gold_v="2"),
    ]
    result = convergence_to_preann(records)
    assert result["n_preann_correct"] == 2
    assert result["n_preann_incorrect"] == 1
    assert result["p_when_preann_correct"] == 1.0
    assert result["p_when_preann_incorrect"] == 1.0  # human matched the wrong preann
