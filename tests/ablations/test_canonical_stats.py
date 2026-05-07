"""Unit tests for the canonical statistics suite.

Builds a synthetic long-form atomic table covering every case_status /
field_status branch, runs the full eight-table suite, and asserts
invariants on the outputs.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from digital_registrar_research.ablations.eval.canonical_stats import (
    run_canonical_stats,
    _failure_modes,
    _headline,
    _per_field,
    _per_organ,
    _seed_consistency,
    _modularity_advantage,
    _low_performer_diagnostics,
)
from digital_registrar_research.benchmarks.eval.scope import (
    STATS_EXCLUDED_FIELDS,
)


# ---------------------------------------------------------------------------
# Synthetic atomic builder
# ---------------------------------------------------------------------------

def _row(case_id: str, organ: str, method: str, run: str, field: str,
         correct, attempted: bool, case_status: str = "ok",
         field_status: str = "correct",
         gold_present: bool = True,
         field_error_detail: str = "") -> dict:
    return {
        "case_id": case_id, "organ": organ, "method": method, "run": run,
        "field": field,
        "correct": correct, "attempted": attempted,
        "gold_present": gold_present,
        "case_status": case_status,
        "case_flags": case_status,
        "field_status": field_status,
        "field_error_detail": field_error_detail,
    }


@pytest.fixture
def synthetic_atomic() -> pd.DataFrame:
    """Build a 4-method × 4-field × 5-case × 2-run synthetic atomic.

    Methods: modular (high accuracy), alt_ok (medium), alt_defective
    (mostly defective), alt_low (low accuracy).
    Fields: cancer_category (high accuracy for modular), grade
    (lower-performer for modular), tumor_size (variable), ajcc_version
    (excluded from stats).
    """
    rows: list[dict] = []
    for run in ("run1", "run2"):
        for case_n in range(1, 6):
            cid = f"c{case_n}"
            organ = "breast" if case_n <= 3 else "colorectal"

            # modular: 4/5 correct on cancer_category, 3/5 on grade,
            # 5/5 on tumor_size, 5/5 on ajcc_version (but excluded)
            rows.append(_row(cid, organ, "modular", run, "cancer_category",
                             correct=True if case_n <= 4 else False,
                             attempted=True,
                             field_status="correct" if case_n <= 4
                             else "wrong_value"))
            rows.append(_row(cid, organ, "modular", run, "grade",
                             correct=True if case_n <= 3 else None,
                             attempted=case_n <= 3,
                             field_status="correct" if case_n <= 3
                             else "missing_key"))
            rows.append(_row(cid, organ, "modular", run, "tumor_size",
                             correct=True, attempted=True,
                             field_status="correct"))
            rows.append(_row(cid, organ, "modular", run, "ajcc_version",
                             correct=True, attempted=True,
                             field_status="correct"))

            # alt_ok: 3/5 correct on cancer_category, 2/5 on grade
            rows.append(_row(cid, organ, "alt_ok", run, "cancer_category",
                             correct=True if case_n <= 3 else False,
                             attempted=True,
                             field_status="correct" if case_n <= 3
                             else "wrong_value"))
            rows.append(_row(cid, organ, "alt_ok", run, "grade",
                             correct=True if case_n <= 2 else False,
                             attempted=True,
                             field_status="correct" if case_n <= 2
                             else "wrong_value"))
            rows.append(_row(cid, organ, "alt_ok", run, "tumor_size",
                             correct=True, attempted=True,
                             field_status="correct"))
            rows.append(_row(cid, organ, "alt_ok", run, "ajcc_version",
                             correct=True, attempted=True,
                             field_status="correct"))

            # alt_defective: every other case is parse_error
            is_defective = (case_n % 2 == 0)
            cs = "parse_error" if is_defective else "ok"
            fs = "unscoreable_due_to_case_error" if is_defective else "correct"
            for fld in ("cancer_category", "grade", "tumor_size",
                        "ajcc_version"):
                rows.append(_row(
                    cid, organ, "alt_defective", run, fld,
                    correct=None if is_defective else True,
                    attempted=False if is_defective else True,
                    case_status=cs, field_status=fs))

            # alt_low: 1/5 correct, mostly wrong_value
            rows.append(_row(cid, organ, "alt_low", run, "cancer_category",
                             correct=True if case_n == 1 else False,
                             attempted=True,
                             field_status="correct" if case_n == 1
                             else "wrong_value"))
            rows.append(_row(cid, organ, "alt_low", run, "grade",
                             correct=False, attempted=True,
                             field_status="wrong_value"))
            rows.append(_row(cid, organ, "alt_low", run, "tumor_size",
                             correct=False, attempted=True,
                             field_status="wrong_value"))
            rows.append(_row(cid, organ, "alt_low", run, "ajcc_version",
                             correct=False, attempted=True,
                             field_status="wrong_value"))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-helper smoke
# ---------------------------------------------------------------------------

def test_headline_one_row_per_method(synthetic_atomic):
    h = _headline(synthetic_atomic, modular_method="modular")
    assert sorted(h["method"].tolist()) == [
        "alt_defective", "alt_low", "alt_ok", "modular",
    ]
    # Every row has Wilson CI bounds.
    for col in ("accuracy_eff_lo", "accuracy_eff_hi"):
        assert h[col].notna().all()
    # modular has no delta vs itself (NaN), others have a finite delta.
    modular_row = h[h["method"] == "modular"].iloc[0]
    assert pd.isna(modular_row["delta_vs_modular"])
    other = h[h["method"] == "alt_ok"].iloc[0]
    assert pd.notna(other["delta_vs_modular"])


def test_headline_holm_correction_applied(synthetic_atomic):
    h = _headline(synthetic_atomic, modular_method="modular")
    assert "mcnemar_p_holm" in h.columns
    # Holm-adjusted p must be >= raw p where both are finite.
    sub = h.dropna(subset=["mcnemar_p", "mcnemar_p_holm"])
    assert (sub["mcnemar_p_holm"] >= sub["mcnemar_p"] - 1e-9).all()


def test_failure_modes_counts_consistent(synthetic_atomic):
    fm = _failure_modes(synthetic_atomic)
    # Sum of fractions per method (over field_status) is 1.
    sums = fm.groupby("method")["fraction"].sum()
    for method, total in sums.items():
        assert abs(total - 1.0) < 1e-6, (method, total)


def test_failure_modes_includes_parse_error_for_defective(synthetic_atomic):
    fm = _failure_modes(synthetic_atomic)
    sub = fm[(fm["method"] == "alt_defective")
             & (fm["field_status"] == "unscoreable_due_to_case_error")]
    assert not sub.empty
    assert sub["fraction"].iloc[0] >= 0.4  # ~50% defective


def test_per_field_excludes_stats_excluded(synthetic_atomic):
    pf = _per_field(synthetic_atomic, modular_method="modular")
    # ajcc_version is in STATS_EXCLUDED_FIELDS — must not appear.
    assert "ajcc_version" in STATS_EXCLUDED_FIELDS
    assert (pf["field"] != "ajcc_version").all()


def test_per_field_modular_grade_lower_than_cancer_category(synthetic_atomic):
    pf = _per_field(synthetic_atomic, modular_method="modular")
    modular = pf[pf["method"] == "modular"]
    cc = modular[modular["field"] == "cancer_category"]["accuracy"].iloc[0]
    grade = modular[modular["field"] == "grade"]["accuracy"].iloc[0]
    # cancer_category: 4/5 = 0.8 (per run, both runs); grade: 3/5 = 0.6.
    assert cc > grade


def test_per_organ_includes_breast_and_colorectal(synthetic_atomic):
    po = _per_organ(synthetic_atomic, modular_method="modular")
    organs = sorted(po["organ"].unique().tolist())
    assert organs == ["breast", "colorectal"]


def test_seed_consistency_emits_when_multirun(synthetic_atomic):
    sc = _seed_consistency(synthetic_atomic)
    assert not sc.empty
    assert "fleiss_kappa" in sc.columns
    assert (sc["n_seeds"] == 2).all()


def test_modularity_advantage_chooses_best_alternative(synthetic_atomic):
    ma = _modularity_advantage(synthetic_atomic, modular_method="modular")
    assert not ma.empty
    # ALL row should pick alt_ok (highest accuracy among alternatives).
    all_row = ma[ma["field"] == "ALL"].iloc[0]
    assert all_row["best_alternative_method"] == "alt_ok"
    # Modular advantage is positive.
    assert all_row["modular_advantage"] > 0


def test_low_performer_diagnostics_flags_grade(synthetic_atomic):
    """Modular grade accuracy = 6/10 = 0.6 < 0.90 → must appear."""
    lp = _low_performer_diagnostics(synthetic_atomic,
                                    modular_method="modular")
    # cancer_category at 0.8 also < 0.90.
    fields = lp["field"].tolist()
    assert "grade" in fields
    assert "cancer_category" in fields
    # tumor_size at 1.0 must NOT appear.
    assert "tumor_size" not in fields
    # ajcc_version is excluded from stats — must NOT appear.
    assert "ajcc_version" not in fields


def test_low_performer_decomposition_reasonable(synthetic_atomic):
    lp = _low_performer_diagnostics(synthetic_atomic,
                                    modular_method="modular")
    grade_row = lp[lp["field"] == "grade"].iloc[0]
    # Half of grade rows for modular are missing_key (case_n in {4,5})
    # and 6/10 are correct → field_missing_rate ≈ 0.4.
    assert 0.3 < grade_row["field_missing_rate"] < 0.5


# ---------------------------------------------------------------------------
# End-to-end: write all CSVs + report
# ---------------------------------------------------------------------------

def test_run_canonical_stats_writes_all_outputs(synthetic_atomic, tmp_path):
    out = run_canonical_stats(synthetic_atomic, modular_method="modular",
                              out_dir=tmp_path)
    expected = {
        "headline.csv", "failure_modes.csv", "per_field.csv",
        "per_organ.csv", "seed_consistency.csv",
        "modularity_advantage.csv", "low_performer_diagnostics.csv",
        "canonical_stats_report.md",
    }
    assert set(out.keys()) >= expected
    for name in expected:
        assert (tmp_path / name).is_file(), f"missing {name}"


def test_run_canonical_stats_report_mentions_filters(synthetic_atomic,
                                                     tmp_path):
    run_canonical_stats(synthetic_atomic, modular_method="modular",
                        out_dir=tmp_path)
    report = (tmp_path / "canonical_stats_report.md").read_text(
        encoding="utf-8")
    assert "STATS_EXCLUDED_FIELDS" in report
    assert "ajcc_version" in report
    assert "Defects-as-wrong" in report or "Defects" in report
    assert "Holm" in report
    # Report must NOT contain reviewer-narrative leakage.
    for term in ("reviewer", "Reviewer", "rebuttal", "Major 1",
                 "response letter"):
        assert term.lower() not in report.lower(), (
            f"reviewer-narrative leak: {term!r} found in report")


def test_run_canonical_stats_handles_missing_modular(synthetic_atomic,
                                                     tmp_path):
    out = run_canonical_stats(synthetic_atomic,
                              modular_method="nonexistent_method",
                              out_dir=tmp_path)
    # Headline still emitted; per-method delta columns all NaN, note set.
    headline = pd.read_csv(out["headline.csv"])
    assert headline["delta_vs_modular"].isna().all()
    assert (headline["note"].astype(str).str.contains("not present")).any()


def test_run_canonical_stats_handles_empty_atomic(tmp_path):
    empty = pd.DataFrame(columns=[
        "case_id", "organ", "method", "run", "field", "correct",
        "attempted", "gold_present", "case_status", "case_flags",
        "field_status", "field_error_detail",
    ])
    out = run_canonical_stats(empty, modular_method="modular",
                              out_dir=tmp_path)
    # All CSVs created (even if empty).
    for name in ("headline.csv", "failure_modes.csv",
                 "canonical_stats_report.md"):
        assert (tmp_path / name).is_file()
