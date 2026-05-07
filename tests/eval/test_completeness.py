"""Tests for src/.../eval/completeness.py — missingness aggregation,
out-of-vocab rate, refusal calibration."""
from __future__ import annotations

import pandas as pd

from digital_registrar_research.benchmarks.eval.completeness import (
    aggregate_missingness,
    out_of_vocab_rate,
    refusal_calibration,
)


def _atomic(rows):
    """Build a small DataFrame with all required outcome flags."""
    df = pd.DataFrame(rows)
    for col in ["gold_present", "parse_error", "field_missing",
                "attempted", "correct", "wrong"]:
        if col not in df.columns:
            df[col] = False
    return df


def test_aggregate_missingness_basic():
    df = _atomic([
        {"method": "A", "field": "f", "organ": "breast",
         "gold_present": True, "attempted": True, "correct": True, "wrong": False,
         "field_missing": False, "parse_error": False},
        {"method": "A", "field": "f", "organ": "breast",
         "gold_present": True, "attempted": True, "correct": False, "wrong": True,
         "field_missing": False, "parse_error": False},
        {"method": "A", "field": "f", "organ": "breast",
         "gold_present": True, "attempted": False, "correct": False, "wrong": False,
         "field_missing": True, "parse_error": False},
        {"method": "A", "field": "f", "organ": "breast",
         "gold_present": True, "attempted": False, "correct": False, "wrong": False,
         "field_missing": False, "parse_error": True},
    ])
    out = aggregate_missingness(df, by=("method", "field", "organ"))
    assert len(out) == 1
    row = out.iloc[0]
    assert row["n_total"] == 4
    assert row["n_correct"] == 1
    assert row["n_wrong"] == 1
    assert row["n_field_missing"] == 1
    assert row["n_parse_error"] == 1
    assert row["attempted_rate"] == 0.5
    assert abs(row["attempted_accuracy"] - 0.5) < 1e-9
    assert abs(row["effective_accuracy"] - 0.25) < 1e-9


def test_out_of_vocab_rate_categorical():
    """Predictions outside the allowed enum are flagged.

    Uses ``nuclear_grade`` whose enum is hardcoded ["1", "2", "3"] in
    scope.py — stable and simple to test against.
    """
    preds = [
        {"cancer_data": {"nuclear_grade": "1"}},
        {"cancer_data": {"nuclear_grade": "2"}},
        # invalid value
        {"cancer_data": {"nuclear_grade": "11"}},
        {"cancer_data": {"nuclear_grade": None}},  # null — skipped
    ]
    out = out_of_vocab_rate(preds, field="nuclear_grade", organ=None)
    # n_attempted excludes the null one
    assert out["n_attempted"] == 3
    assert out["n_oov"] == 1
    assert abs(out["oov_rate"] - 1/3) < 1e-9


def test_refusal_calibration_split():
    """Pred null + gold null = correct_refusal; pred null + gold non-null = lazy."""
    df = _atomic([
        # correct_refusal: pred null, gold null
        {"method": "A", "field": "f", "organ": "breast",
         "gold_present": False, "attempted": False},
        # lazy: pred null, gold present
        {"method": "A", "field": "f", "organ": "breast",
         "gold_present": True, "attempted": False, "field_missing": True},
        # attempted (not in this analysis)
        {"method": "A", "field": "f", "organ": "breast",
         "gold_present": True, "attempted": True, "correct": True},
    ])
    out = refusal_calibration(df, by=("method", "field", "organ"))
    row = out.iloc[0]
    assert row["n_pred_null"] == 2
    assert row["n_correct_refusal"] == 1
    assert row["n_lazy_missing"] == 1
    assert abs(row["correct_refusal_rate"] - 0.5) < 1e-9
    assert abs(row["lazy_missing_rate"] - 0.5) < 1e-9
