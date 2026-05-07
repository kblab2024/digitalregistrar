"""Tests for the pair-focused IAA headline library."""
from __future__ import annotations

from digital_registrar_research.benchmarks.eval.iaa import (
    CaseEntry,
    cohen_kappa,
    extract_pairs,
)
from digital_registrar_research.benchmarks.eval.iaa_headline import (
    HEADLINE_COLUMNS,
    PER_FIELD_COLUMNS,
    agree_disagree_pabak,
    agree_disagree_pairs,
    compute_headline,
    confusion_matrix_for_field,
    disagreement_count_per_field,
    headline_rows_to_df,
    mean_per_field_kappa,
    n_weighted_mean_per_field_kappa,
    per_field_kappas,
    per_organ_rollup,
    per_section_rollup,
    pooled_categorical_kappa,
    pooled_categorical_pairs,
)


def _make_cases(records):
    """records: list of (case_id, organ, ann_a_dict, ann_b_dict, ...)."""
    cases: dict[str, CaseEntry] = {}
    for cid, organ, *anns in records:
        annotations = {}
        for label, payload in anns:
            annotations[label] = payload
        cases[cid] = CaseEntry(organ=organ, annotations=annotations, paths={})
    return cases


def _ann(extras: dict | None = None, **cancer_data) -> dict:
    """Build a tiny annotation dict around `cancer_data`."""
    out = {
        "cancer_excision_report": True,
        "cancer_category": "breast",
        "cancer_data": cancer_data,
    }
    if extras:
        out.update(extras)
    return out


# --- Pooled categorical κ ---------------------------------------------------


def test_pooled_categorical_kappa_perfect_agreement_breast():
    """All categorical fields equal between annotators → κ = 1.0."""
    payload = _ann(
        lymphovascular_invasion=True,
        perineural_invasion=False,
        pt_category="t2",
    )
    cases = _make_cases([
        ("c1", "breast", ("A", payload), ("B", payload)),
        ("c2", "breast", ("A", payload), ("B", payload)),
        ("c3", "breast", ("A", payload), ("B", payload)),
    ])
    pooled = pooled_categorical_pairs(cases, ann_a="A", ann_b="B")
    # Perfect agreement on a single category collapses κ — verify by
    # injecting a second label so the chance baseline is well-defined.
    second = _ann(
        lymphovascular_invasion=False,
        perineural_invasion=True,
        pt_category="t1",
    )
    cases["c4"] = CaseEntry(
        organ="breast",
        annotations={"A": second, "B": second},
        paths={},
    )
    pooled = pooled_categorical_pairs(cases, ann_a="A", ann_b="B")
    assert pooled
    assert pooled_categorical_kappa(pooled) == 1.0


def test_pooled_categorical_missing_token_when_one_side_skips():
    """When one side skipped a field, the missing side is encoded as
    a shared `__missing__` token so the disagreement pollutes the pool."""
    full = _ann(lymphovascular_invasion=True, pt_category="t2")
    sparse = {
        "cancer_excision_report": True,
        "cancer_category": "breast",
        "cancer_data": {"pt_category": "t2"},  # no lymphovascular_invasion key
    }
    cases = _make_cases([
        ("c1", "breast", ("A", full), ("B", sparse)),
    ])
    pooled = pooled_categorical_pairs(cases, ann_a="A", ann_b="B")
    a_vals = [p.a for p in pooled]
    b_vals = [p.b for p in pooled]
    # B never attempted lymphovascular_invasion → at least one __missing__
    # token in B's column.
    assert "__missing__" in b_vals
    # A's lymphovascular_invasion value (True) is paired against
    # __missing__ from B, so they disagree on that observation.
    assert any(
        a is True and b == "__missing__"
        for a, b in zip(a_vals, b_vals, strict=True)
    )


# --- Agree/disagree κ -------------------------------------------------------


def test_agree_disagree_pabak_perfect_agreement_is_one():
    """PABAK = 2·p_o − 1 = +1 when every observation is an agreement."""
    payload = _ann(pt_category="t1", lymphovascular_invasion=True)
    cases = _make_cases([
        ("c1", "breast", ("A", payload), ("B", payload)),
        ("c2", "breast", ("A", payload), ("B", payload)),
    ])
    pairs = agree_disagree_pairs(cases, ann_a="A", ann_b="B")
    assert pairs
    assert agree_disagree_pabak(pairs) == 1.0


def test_agree_disagree_pabak_full_disagreement_is_minus_one():
    """PABAK = −1 when every observation is a disagreement."""
    a = _ann(pt_category="t1", lymphovascular_invasion=True)
    b = _ann(pt_category="t2", lymphovascular_invasion=False)
    cases = _make_cases([
        ("c1", "breast", ("A", a), ("B", b)),
        ("c2", "breast", ("A", a), ("B", b)),
    ])
    pairs = agree_disagree_pairs(cases, ann_a="A", ann_b="B")
    pairs_categorical_only = [p for p in pairs if p.a in (0, 1)]
    # Filter to the two fixture fields whose values are guaranteed to disagree.
    field_names = {p.case_id.split("::", 1)[1] for p in pairs}
    pairs_target = [
        p for p in pairs_categorical_only
        if p.case_id.split("::", 1)[1]
        in {"pt_category", "lymphovascular_invasion"}
    ]
    assert pairs_target  # at least the two fixture fields
    assert all(p.a == 0 for p in pairs_target)
    assert agree_disagree_pabak(pairs_target) == -1.0


def test_agree_disagree_perfect_when_all_fields_match():
    payload = _ann(
        lymphovascular_invasion=True,
        pt_category="t2",
        tumor_size=2.5,
    )
    other = _ann(
        lymphovascular_invasion=False,
        pt_category="t1",
        tumor_size=1.5,
    )
    cases = _make_cases([
        ("c1", "breast", ("A", payload), ("B", payload)),
        ("c2", "breast", ("A", other), ("B", other)),
    ])
    pairs = agree_disagree_pairs(cases, ann_a="A", ann_b="B")
    assert pairs
    # Every observation is "agree" → all p.a == 1; κ degenerates to 1.0
    # via observed_agreement = 1.0.
    assert all(p.a == 1 for p in pairs)


def test_agree_disagree_continuous_within_tolerance():
    """tumor_size differing by ≤ NUMERIC_TOLERANCE_MM (2mm) counts as agreement."""
    a = _ann(tumor_size=10.0, pt_category="t1")
    b = _ann(tumor_size=11.5, pt_category="t1")  # within 2mm → agree
    cases = _make_cases([("c1", "breast", ("A", a), ("B", b))])
    pairs = agree_disagree_pairs(cases, ann_a="A", ann_b="B")
    by_field = {p.case_id.split("::", 1)[1]: p.a for p in pairs}
    assert by_field.get("tumor_size") == 1
    assert by_field.get("pt_category") == 1


def test_agree_disagree_continuous_outside_tolerance():
    a = _ann(tumor_size=10.0, pt_category="t1")
    b = _ann(tumor_size=20.0, pt_category="t1")  # >> 2mm tolerance
    cases = _make_cases([("c1", "breast", ("A", a), ("B", b))])
    pairs = agree_disagree_pairs(cases, ann_a="A", ann_b="B")
    by_field = {p.case_id.split("::", 1)[1]: p.a for p in pairs}
    assert by_field.get("tumor_size") == 0


# --- mean / n-weighted mean -------------------------------------------------


def test_mean_kappa_matches_single_field_kappa():
    """When restricted to a single field, mean_per_field_kappa equals
    the underlying cohen_kappa from iaa.py exactly."""
    cases = _make_cases([
        ("c1", "breast",
         ("A", _ann(pt_category="t1")),
         ("B", _ann(pt_category="t1"))),
        ("c2", "breast",
         ("A", _ann(pt_category="t2")),
         ("B", _ann(pt_category="t1"))),  # disagree
        ("c3", "breast",
         ("A", _ann(pt_category="t2")),
         ("B", _ann(pt_category="t2"))),
        ("c4", "breast",
         ("A", _ann(pt_category="t1")),
         ("B", _ann(pt_category="t2"))),  # disagree
    ])
    expected_pairs = extract_pairs(cases, "pt_category", "A", "B")
    expected_kappa = cohen_kappa(expected_pairs)

    pf_df = per_field_kappas(
        cases, ann_a="A", ann_b="B", fields=["pt_category"],
        n_boot=10, random_state=0,
    )
    mean_k, n_fields = mean_per_field_kappa(pf_df)
    assert n_fields == 1
    assert abs(mean_k - expected_kappa) < 1e-9


def test_n_weighted_mean_reduces_to_plain_mean_with_equal_n():
    """When every field has identical n, n-weighted mean == unweighted mean."""
    cases = _make_cases([
        ("c1", "breast",
         ("A", _ann(pt_category="t1", lymphovascular_invasion=True)),
         ("B", _ann(pt_category="t2", lymphovascular_invasion=True))),
        ("c2", "breast",
         ("A", _ann(pt_category="t1", lymphovascular_invasion=False)),
         ("B", _ann(pt_category="t1", lymphovascular_invasion=False))),
        ("c3", "breast",
         ("A", _ann(pt_category="t2", lymphovascular_invasion=True)),
         ("B", _ann(pt_category="t2", lymphovascular_invasion=False))),
    ])
    pf_df = per_field_kappas(
        cases, ann_a="A", ann_b="B",
        fields=["pt_category", "lymphovascular_invasion"],
        n_boot=10, random_state=0,
    )
    mean_k, _ = mean_per_field_kappa(pf_df)
    wmean_k, _ = n_weighted_mean_per_field_kappa(pf_df)
    assert abs(mean_k - wmean_k) < 1e-9


# --- compute_headline -------------------------------------------------------


def test_compute_headline_returns_expected_stats():
    cases = _make_cases([
        ("c1", "breast",
         ("A", _ann(pt_category="t1", lymphovascular_invasion=True,
                    tumor_size=10.0)),
         ("B", _ann(pt_category="t1", lymphovascular_invasion=True,
                    tumor_size=10.5))),
        ("c2", "breast",
         ("A", _ann(pt_category="t2", lymphovascular_invasion=False,
                    tumor_size=20.0)),
         ("B", _ann(pt_category="t2", lymphovascular_invasion=False,
                    tumor_size=20.0))),
        ("c3", "breast",
         ("A", _ann(pt_category="t1", lymphovascular_invasion=False,
                    tumor_size=15.0)),
         ("B", _ann(pt_category="t2", lymphovascular_invasion=False,
                    tumor_size=15.0))),
    ])
    rows = compute_headline(
        cases, ann_a="A", ann_b="B", n_boot=50, random_state=0,
    )
    stat_names = {r.stat_name for r in rows}
    expected_core = {
        "mean_per_field_kappa",
        "n_weighted_mean_per_field_kappa",
        "pooled_categorical_kappa",
        "agree_disagree_pabak",
    }
    assert expected_core <= stat_names
    # Krippendorff α and case_exact_match_rate come from whole_report_stats
    assert "case_exact_match_rate" in stat_names
    assert any(s.startswith("krippendorff_alpha_") for s in stat_names)


def test_compute_headline_deterministic_under_seed():
    """Same random_state → identical pooled-κ CIs."""
    cases = _make_cases([
        ("c1", "breast",
         ("A", _ann(pt_category="t1", lymphovascular_invasion=True)),
         ("B", _ann(pt_category="t2", lymphovascular_invasion=True))),
        ("c2", "breast",
         ("A", _ann(pt_category="t2", lymphovascular_invasion=False)),
         ("B", _ann(pt_category="t1", lymphovascular_invasion=False))),
    ])
    a = compute_headline(cases, ann_a="A", ann_b="B",
                        n_boot=100, random_state=42)
    b = compute_headline(cases, ann_a="A", ann_b="B",
                        n_boot=100, random_state=42)
    a_pooled = next(r for r in a if r.stat_name == "pooled_categorical_kappa")
    b_pooled = next(r for r in b if r.stat_name == "pooled_categorical_kappa")
    assert a_pooled.estimate == b_pooled.estimate
    assert a_pooled.ci_lo == b_pooled.ci_lo
    assert a_pooled.ci_hi == b_pooled.ci_hi


def test_headline_rows_to_df_has_expected_columns():
    rows = [
        # Minimal one-row instance to verify schema.
    ]
    cases = _make_cases([
        ("c1", "breast",
         ("A", _ann(pt_category="t1")),
         ("B", _ann(pt_category="t2"))),
        ("c2", "breast",
         ("A", _ann(pt_category="t2")),
         ("B", _ann(pt_category="t1"))),
    ])
    rows = compute_headline(cases, ann_a="A", ann_b="B",
                           n_boot=20, random_state=0)
    df = headline_rows_to_df(rows, pair_label="A_vs_B")
    assert list(df.columns) == list(HEADLINE_COLUMNS)
    assert (df["pair"] == "A_vs_B").all()


# --- Confusion matrices -----------------------------------------------------


def test_confusion_matrix_counts_sum_to_attempts():
    cases = _make_cases([
        ("c1", "breast",
         ("A", _ann(pt_category="t1")),
         ("B", _ann(pt_category="t1"))),
        ("c2", "breast",
         ("A", _ann(pt_category="t2")),
         ("B", _ann(pt_category="t1"))),
        ("c3", "breast",
         ("A", _ann(pt_category="t2")),
         ("B", _ann(pt_category="t2"))),
    ])
    cm = confusion_matrix_for_field(cases, "pt_category",
                                    ann_a="A", ann_b="B")
    assert int(cm["count"].sum()) == 3
    # No __missing__ rows when both attempted everywhere.
    assert (cm["value_a"] != "__missing__").all()
    assert (cm["value_b"] != "__missing__").all()


def test_confusion_matrix_includes_missing_token():
    """When one side skipped, a __missing__ row appears."""
    full = _ann(pt_category="t1")
    sparse = {
        "cancer_excision_report": True,
        "cancer_category": "breast",
        "cancer_data": {"lymphovascular_invasion": True},  # no pt_category
    }
    cases = _make_cases([
        ("c1", "breast", ("A", full), ("B", sparse)),
    ])
    cm = confusion_matrix_for_field(cases, "pt_category",
                                    ann_a="A", ann_b="B")
    assert any(cm["value_b"] == "__missing__")


def test_disagreement_count_per_field():
    cases = _make_cases([
        ("c1", "breast",
         ("A", _ann(pt_category="t1", lymphovascular_invasion=True)),
         ("B", _ann(pt_category="t2", lymphovascular_invasion=True))),
        ("c2", "breast",
         ("A", _ann(pt_category="t2", lymphovascular_invasion=True)),
         ("B", _ann(pt_category="t2", lymphovascular_invasion=False))),
    ])
    counts = disagreement_count_per_field(cases, ann_a="A", ann_b="B")
    assert counts.get("pt_category") == 1
    assert counts.get("lymphovascular_invasion") == 1


# --- Edge cases -------------------------------------------------------------


def test_empty_cases_returns_empty_dataframes():
    cases: dict[str, CaseEntry] = {}
    df = per_field_kappas(cases, ann_a="A", ann_b="B", n_boot=10)
    assert list(df.columns) == list(PER_FIELD_COLUMNS)
    assert df.empty


def test_per_section_rollup_assigns_sections():
    cases = _make_cases([
        ("c1", "breast",
         ("A", _ann(pt_category="t1", lymphovascular_invasion=True)),
         ("B", _ann(pt_category="t2", lymphovascular_invasion=False))),
        ("c2", "breast",
         ("A", _ann(pt_category="t2", lymphovascular_invasion=False)),
         ("B", _ann(pt_category="t1", lymphovascular_invasion=True))),
    ])
    pf_df = per_field_kappas(
        cases, ann_a="A", ann_b="B",
        fields=["pt_category", "lymphovascular_invasion"],
        n_boot=10, random_state=0,
    )
    section_df = per_section_rollup(pf_df, cases, ann_a="A", ann_b="B")
    # Both fields are scalar_pathology in the existing taxonomy.
    assert "scalar_pathology" in set(section_df["section"])


def test_per_organ_rollup_one_row_per_organ_per_stat():
    """Single-organ fixture → 4 rows (one per stat)."""
    cases = _make_cases([
        ("c1", "breast",
         ("A", _ann(pt_category="t1", lymphovascular_invasion=True)),
         ("B", _ann(pt_category="t2", lymphovascular_invasion=True))),
        ("c2", "breast",
         ("A", _ann(pt_category="t1", lymphovascular_invasion=False)),
         ("B", _ann(pt_category="t1", lymphovascular_invasion=False))),
    ])
    organ_df = per_organ_rollup(
        cases, ann_a="A", ann_b="B", n_boot=10, random_state=0,
    )
    assert (organ_df["organ"] == "breast").all()
    assert set(organ_df["stat_name"]) == {
        "mean_per_field_kappa",
        "n_weighted_mean_per_field_kappa",
        "pooled_categorical_kappa",
        "agree_disagree_pabak",
    }
