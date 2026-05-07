"""Tests for the cascade redesign.

Covers:
    * Cascade gating in :func:`score_case` (Stage A halts B and C;
      Stage B mismatch halts C; "others" diverts).
    * Biomarker whitelist (colon ``her2`` is invisible; breast
      ``ki67`` is scored).
    * Inner-key exclusions (margins ``description`` not in match
      output; LN ``station_name`` not used in matching).
    * LN category-aggregation scoring.
    * scope.py exclusion structures are well-formed.
    * Stage-C scope spans the FULL per-organ schema (not just FAIR_SCOPE).
    * Nested rows (margins / LN / biomarkers) reach the atomic table
      and the new chapter-3 nested CSVs.
"""
from __future__ import annotations

import pytest

from digital_registrar_research.benchmarks.eval.metrics import (
    field_correct,
    match_nested_list_filtered,
    score_case,
)
from digital_registrar_research.benchmarks.eval.nested_metrics import (
    score_lymph_nodes,
)
from digital_registrar_research.benchmarks.eval.scope import (
    BIOMARKER_WHITELIST,
    BREAST_BIOMARKERS,
    EVAL_EXCLUDED_FIELDS,
    EVAL_EXCLUDED_NESTED_INNER_KEYS,
    EXCLUDED_INNER_KEYS_BY_FIELD,
    biomarkers_for_organ,
    get_organ_scoreable_fields,
)


# --- Scope structures --------------------------------------------------

def test_excluded_fields_contain_ajcc_version_and_treatment_effect():
    assert "ajcc_version" in EVAL_EXCLUDED_FIELDS
    assert "treatment_effect" in EVAL_EXCLUDED_FIELDS


def test_excluded_inner_keys_resolve_correctly():
    assert "description" in EXCLUDED_INNER_KEYS_BY_FIELD["margins"]
    assert "station_name" in EXCLUDED_INNER_KEYS_BY_FIELD["regional_lymph_node"]


def test_excluded_inner_keys_dict_has_human_readable_paths():
    assert "cancer_data.margins[*].description" in EVAL_EXCLUDED_NESTED_INNER_KEYS
    assert ("cancer_data.regional_lymph_node[*].station_name"
            in EVAL_EXCLUDED_NESTED_INNER_KEYS)


def test_biomarker_whitelist_breast():
    assert biomarkers_for_organ("breast") == frozenset({"er", "pr", "her2", "ki67"})
    assert "ki67" in BREAST_BIOMARKERS


def test_biomarker_whitelist_colon():
    assert biomarkers_for_organ("colorectal") == frozenset(
        {"msh2", "msh6", "pms2", "mlh1"}
    )


def test_biomarker_whitelist_empty_for_unrelated_organ():
    assert biomarkers_for_organ("lung") == frozenset()
    assert biomarkers_for_organ("others") == frozenset()
    assert biomarkers_for_organ(None) == frozenset()


# --- Cascade gating ----------------------------------------------------

def _gold_passing():
    return {
        "cancer_excision_report": True,
        "cancer_category": "breast",
        "cancer_data": {
            "pt_category": "t1c",
            "pn_category": "n0",
            "ajcc_version": "8",
            "treatment_effect": "complete_response",
            "biomarkers": [
                {"biomarker_category": "er", "expression": True},
                {"biomarker_category": "pr", "expression": True},
                {"biomarker_category": "her2", "expression": False},
                {"biomarker_category": "ki67", "expression": False},
            ],
            "margins": [
                {"margin_category": "deep", "margin_involved": False,
                 "distance": 5, "description": "negative free-text"},
            ],
            "regional_lymph_node": [],
        },
    }


def test_score_case_passes_all_stages_on_identical_pred():
    g = _gold_passing()
    p = _gold_passing()
    result = score_case(g, p)
    assert result["stage_a"]["correct"] is True
    assert result["stage_b"]["correct"] is True
    assert result["stage_c_eligible"] is True
    assert result["others_disposition"] == "none"


def test_score_case_excludes_ajcc_version_and_treatment_effect():
    g = _gold_passing()
    p = _gold_passing()
    result = score_case(g, p)
    assert "ajcc_version" not in result
    assert "treatment_effect" not in result


def test_score_case_halts_on_stage_a_mismatch():
    g = _gold_passing()
    p = _gold_passing()
    p["cancer_excision_report"] = False
    result = score_case(g, p)
    assert result["stage_a"]["correct"] is False
    assert result["stage_c_eligible"] is False
    assert "stage_b" not in result


def test_score_case_halts_on_stage_b_mismatch():
    g = _gold_passing()
    p = _gold_passing()
    p["cancer_category"] = "lung"
    result = score_case(g, p)
    assert result["stage_a"]["correct"] is True
    assert result["stage_b"]["correct"] is False
    assert result["stage_c_eligible"] is False
    assert "pt_category" not in result


def test_score_case_others_disposition_pred():
    g = _gold_passing()
    p = _gold_passing()
    p["cancer_category"] = "others"
    result = score_case(g, p)
    assert result["others_disposition"] == "pred_others"
    assert result["stage_c_eligible"] is False


def test_score_case_others_disposition_gold():
    g = _gold_passing()
    g["cancer_category"] = "others"
    p = _gold_passing()
    p["cancer_category"] = "others"
    result = score_case(g, p)
    assert result["others_disposition"] == "both_others"
    # Both-others: Stage B matches but Stage C doesn't run.
    assert result["stage_b"]["correct"] is True
    assert result["stage_c_eligible"] is False


def test_score_case_breast_biomarkers_include_ki67():
    g = _gold_passing()
    p = _gold_passing()
    result = score_case(g, p)
    # ki67 should appear in Stage C scoring for breast.
    assert "biomarker_ki67" in result
    assert result["biomarker_ki67"] is True


# --- Biomarker whitelist filtering ------------------------------------

def test_biomarker_whitelist_drops_out_of_scope_categories():
    """Colon biomarkers should only score msh2/msh6/pms2/mlh1.

    Adding a non-whitelisted category to either side must not change
    the F1 (whitelist filters before matching).
    """
    g = {"cancer_data": {"biomarkers": [
        {"biomarker_category": "msh2", "expression": True},
        {"biomarker_category": "her2", "expression": True},  # filtered
    ]}}
    p_clean = {"cancer_data": {"biomarkers": [
        {"biomarker_category": "msh2", "expression": True},
    ]}}
    p_extras = {"cancer_data": {"biomarkers": [
        {"biomarker_category": "msh2", "expression": True},
        {"biomarker_category": "her2", "expression": False},  # filtered
        {"biomarker_category": "p53", "expression": True},    # filtered
    ]}}
    r_clean = match_nested_list_filtered(g, p_clean, "biomarkers", organ="colorectal")
    r_extras = match_nested_list_filtered(g, p_extras, "biomarkers", organ="colorectal")
    assert r_clean["f1"] == r_extras["f1"]
    assert r_clean["fp"] == r_extras["fp"]
    assert r_clean["fn"] == r_extras["fn"]


def test_match_nested_list_filtered_rejects_lymph_node_field():
    with pytest.raises(ValueError, match="regional_lymph_node"):
        match_nested_list_filtered({}, {}, "regional_lymph_node")


# --- Lymph-node category-aggregation scoring --------------------------

def test_ln_aggregates_split_rows_to_single_group():
    """Two gold rows summing to (4 examined, 0 involved) match a single
    pred row of (4, 0). The gold split into multiple textual rows
    should not penalise a model that produced the correct totals."""
    g = {"cancer_data": {"regional_lymph_node": [
        {"lymph_node_side": "right", "lymph_node_category": "nonsentinel",
         "examined": 2, "involved": 0, "station_name": "x #1"},
        {"lymph_node_side": "right", "lymph_node_category": "nonsentinel",
         "examined": 2, "involved": 0, "station_name": "x #2"},
    ]}}
    p = {"cancer_data": {"regional_lymph_node": [
        {"lymph_node_side": "right", "lymph_node_category": "nonsentinel",
         "examined": 4, "involved": 0, "station_name": "any_label"},
    ]}}
    r = score_lymph_nodes(g, p)
    assert r["ln_examined_total_gold"] == 4
    assert r["ln_examined_total_pred"] == 4
    assert r["ln_examined_total_correct_tol"] == 1
    assert r["ln_n_groups_gold"] == 1
    assert r["ln_n_groups_pred"] == 1
    assert r["ln_group_recall"] == 1.0
    assert r["ln_group_precision"] == 1.0


def test_ln_hallucinated_group_drops_precision():
    """Pred adds a hallucinated mesenteric group; precision drops."""
    g = {"cancer_data": {"regional_lymph_node": [
        {"lymph_node_side": "right", "lymph_node_category": "nonsentinel",
         "examined": 4, "involved": 0, "station_name": None},
    ]}}
    p = {"cancer_data": {"regional_lymph_node": [
        {"lymph_node_side": "right", "lymph_node_category": "nonsentinel",
         "examined": 4, "involved": 0, "station_name": None},
        {"lymph_node_side": None, "lymph_node_category": "mesenteric",
         "examined": 5, "involved": 0, "station_name": None},
    ]}}
    r = score_lymph_nodes(g, p)
    assert r["ln_n_groups_gold"] == 1
    assert r["ln_n_groups_pred"] == 2
    assert r["ln_group_precision"] == 0.5
    assert r["ln_group_recall"] == 1.0


# --- Margin description exclusion -------------------------------------

def test_margin_description_excluded_from_filtered_match():
    """Two cases identical in everything except `description` should
    bipartite-match perfectly (description is stripped before matching).
    """
    g = {"cancer_data": {"margins": [
        {"margin_category": "deep", "margin_involved": False,
         "distance": 5, "description": "free-text gold"},
    ]}}
    p = {"cancer_data": {"margins": [
        {"margin_category": "deep", "margin_involved": False,
         "distance": 5, "description": "wholly different free text"},
    ]}}
    r = match_nested_list_filtered(g, p, "margins")
    assert r["fp"] == 0
    assert r["fn"] == 0
    assert r["tp"] >= 3  # margin_category + margin_involved + distance match
    assert r["f1"] == 1.0


# --- Stage-C scope coverage -------------------------------------------

def test_score_case_includes_full_per_organ_scope_breast():
    """Stage C must score every per-organ scoreable field, not just the
    9-element FAIR_SCOPE. Regression check for the bug where Stage C
    was using FAIR_SCOPE and silently dropping most enums and bools.
    """
    g = _gold_passing()
    p = _gold_passing()
    result = score_case(g, p)
    breast_fields = set(get_organ_scoreable_fields("breast").keys())
    # FAIR_SCOPE includes pt/pn/pm category + grade + lvi + pni + tumor_size.
    # Picking a few non-FAIR breast fields that must now be scored.
    for non_fair_field in (
        "histology", "procedure", "tumor_focality", "nuclear_grade",
        "tubule_formation", "mitotic_rate", "total_score",
    ):
        if non_fair_field not in breast_fields:
            continue  # schema may evolve — only assert what's in scope
        assert non_fair_field in result, (
            f"Stage C silently dropped {non_fair_field!r} — "
            "regression of the FAIR_SCOPE bug."
        )


def test_score_case_list_of_literals_uses_set_equality():
    """``tumor_extent`` is a list-of-literals for liver. Order-different
    lists must score equal — string-equality on the rendered list
    representation would falsely flag this as wrong.
    """
    g = {
        "cancer_excision_report": True,
        "cancer_category": "liver",
        "cancer_data": {
            "tumor_extent": ["a", "b"],
        },
    }
    p = {
        "cancer_excision_report": True,
        "cancer_category": "liver",
        "cancer_data": {
            "tumor_extent": ["b", "a"],
        },
    }
    assert field_correct(g, p, "tumor_extent", organ="liver") is True


def test_score_case_excludes_excluded_fields_even_in_full_scope():
    """``ajcc_version`` and ``treatment_effect`` stay excluded after the
    scope expansion."""
    g = _gold_passing()
    p = _gold_passing()
    result = score_case(g, p)
    assert "ajcc_version" not in result
    assert "treatment_effect" not in result


def test_field_correct_span_field_tolerance_generalises():
    """The ±2 mm tolerance applies to all SPAN_FIELDS, not just
    ``tumor_size`` (the legacy hard-coded guard)."""
    g = {"cancer_category": "breast", "cancer_data": {"tumor_size": 12}}
    p = {"cancer_category": "breast", "cancer_data": {"tumor_size": 13}}
    assert field_correct(g, p, "tumor_size") is True
    p2 = {"cancer_category": "breast", "cancer_data": {"tumor_size": 20}}
    assert field_correct(g, p2, "tumor_size") is False


# --- Cascade orchestrator: nested rows reach atomic + sidecar ---------

def test_emit_nested_rows_includes_lnodes_margins_biomarkers():
    """The cascade orchestrator must emit a nested headline row per
    case for each of margins / regional_lymph_node / biomarkers, plus
    sidecar rows with the rich per-case scorer payload.
    """
    from scripts.eval._common.outcome import CaseLoad
    from scripts.eval.cascade.run_cascade import _emit_nested_rows

    base = {
        "run_id": "r0", "method": "llm", "model": "test",
        "annotator": "gold", "case_id": "c1", "organ_idx": 1,
        "organ": "breast", "subgroup": "single_primary",
        "dataset": "cmuh",
    }
    gold = _gold_passing()
    pred = _gold_passing()
    case_load = CaseLoad(ok=True, pred=pred, error_mode=None)
    atomic_rows, sidecar_rows = _emit_nested_rows(
        base=base, gold=gold, pred=pred,
        case_load=case_load, others_disposition="none",
    )
    fields_in_atomic = {r["field"] for r in atomic_rows}
    assert {"margins", "regional_lymph_node", "biomarkers"} <= fields_in_atomic
    for r in atomic_rows:
        assert r["field_kind"] == "nested_list"
        assert r["cascade_stage"] == "C"
    fields_in_sidecar = {r["field"] for r in sidecar_rows}
    # Margins / biomarkers attempted in the fixture; LN list is empty
    # but `regional_lymph_node` key is present, so it counts as attempted.
    assert "margins" in fields_in_sidecar
    assert "biomarkers" in fields_in_sidecar
    assert "regional_lymph_node" in fields_in_sidecar
    margin_row = next(r for r in sidecar_rows if r["field"] == "margins")
    # Rich score_margins payload preserved.
    assert "margin_any_involved_correct" in margin_row
    assert "margin_matched" in margin_row
    ln_row = next(r for r in sidecar_rows if r["field"] == "regional_lymph_node")
    assert "ln_group_recall" in ln_row
    assert "ln_n_groups_gold" in ln_row


def test_chapter3_reducers_consume_nested_rows():
    """The new reducers must handle a sidecar table without crashing
    and must return non-empty frames with the expected columns.
    """
    import pandas as pd
    from scripts.eval._common.outcome import CaseLoad
    from scripts.eval.cascade.reductions import (
        chapter3_biomarker_per_category,
        chapter3_nested_per_attribute_per_organ,
        chapter3_nested_per_field_per_organ,
        chapter3_per_field_overall,
    )
    from scripts.eval.cascade.run_cascade import (
        _emit_nested_rows, _emit_stage_c_rows,
    )

    base = {
        "run_id": "r0", "method": "llm", "model": "test",
        "annotator": "gold", "case_id": "c1", "organ_idx": 1,
        "organ": "breast", "subgroup": "single_primary",
        "dataset": "cmuh",
    }
    gold = _gold_passing()
    pred = _gold_passing()
    case_load = CaseLoad(ok=True, pred=pred, error_mode=None)
    score = score_case(gold, pred)
    scalar_rows = _emit_stage_c_rows(
        base=base, score=score, gold=gold, pred=pred,
        case_load=case_load, others_disposition="none",
    )
    nested_atomic, nested_sidecar = _emit_nested_rows(
        base=base, gold=gold, pred=pred,
        case_load=case_load, others_disposition="none",
    )
    atomic = pd.DataFrame(scalar_rows + nested_atomic)
    nested = pd.DataFrame(nested_sidecar)

    per_field = chapter3_per_field_overall(atomic)
    # Scalar table must NOT contain nested fields (they'd carry F1, not bool).
    assert (per_field["field"] != "margins").all()
    assert (per_field["field"] != "regional_lymph_node").all()
    assert (per_field["field"] != "biomarkers").all()

    nested_per_field = chapter3_nested_per_field_per_organ(nested)
    assert not nested_per_field.empty
    assert {"model", "dataset", "organ", "field", "attempted_f1"} <= set(
        nested_per_field.columns
    )
    assert {"margins", "biomarkers", "regional_lymph_node"} <= set(
        nested_per_field["field"]
    )

    nested_per_attr = chapter3_nested_per_attribute_per_organ(nested)
    # Margins fixture has one matched pair → margin_status_correct = 1.
    if not nested_per_attr.empty:
        assert {"field", "attribute", "accuracy"} <= set(nested_per_attr.columns)

    biomarkers = chapter3_biomarker_per_category(atomic)
    assert not biomarkers.empty
    assert set(biomarkers["category"]) <= {"er", "pr", "her2", "ki67"}


# --- Chapter 4 (margins) + Chapter 5 (lymph nodes) ---------------------

def _ln_breast_gold():
    """Realistic breast LN gold with two (side, category) groups."""
    return {
        "cancer_excision_report": True,
        "cancer_category": "breast",
        "cancer_data": {
            "regional_lymph_node": [
                {"lymph_node_side": "right", "lymph_node_category": "sentinel",
                 "examined": 3, "involved": 1, "station_name": "SLN-1"},
                {"lymph_node_side": "right", "lymph_node_category": "nonsentinel",
                 "examined": 5, "involved": 0, "station_name": "AX-2"},
            ],
            "margins": [
                {"margin_category": "deep", "margin_involved": False,
                 "distance": 5, "description": "negative free-text"},
                {"margin_category": "superficial", "margin_involved": True,
                 "distance": 0, "description": "involved at surface"},
            ],
            "biomarkers": [
                {"biomarker_category": "er", "expression": True},
                {"biomarker_category": "pr", "expression": True},
                {"biomarker_category": "her2", "expression": False},
                {"biomarker_category": "ki67", "expression": False},
            ],
        },
    }


def _build_two_case_nested_fixture():
    """Build (atomic_df, nested_df) for two cases — perfect match + a wrong one."""
    import pandas as pd
    from scripts.eval._common.outcome import CaseLoad
    from scripts.eval.cascade.run_cascade import (
        _emit_nested_rows, _emit_stage_c_rows,
    )

    base_perfect = {
        "run_id": "r0", "method": "llm", "model": "test",
        "annotator": "gold", "case_id": "c1", "organ_idx": 1,
        "organ": "breast", "subgroup": "single_primary",
        "dataset": "cmuh",
    }
    base_wrong = {**base_perfect, "case_id": "c2"}
    gold = _ln_breast_gold()

    pred_perfect = _ln_breast_gold()  # identical
    pred_wrong = _ln_breast_gold()
    # Flip the model's margin status on the second margin and miss one LN group.
    pred_wrong["cancer_data"]["margins"][1]["margin_involved"] = False
    pred_wrong["cancer_data"]["margins"][1]["margin_category"] = "deep"  # confusion!
    pred_wrong["cancer_data"]["regional_lymph_node"] = [
        {"lymph_node_side": "right", "lymph_node_category": "sentinel",
         "examined": 3, "involved": 1, "station_name": "SLN-1"},
        # Hallucinated extra group on a different side.
        {"lymph_node_side": "left", "lymph_node_category": "nonsentinel",
         "examined": 2, "involved": 0, "station_name": "X"},
    ]

    rows = []
    nested_rows = []
    for base, pred in [(base_perfect, pred_perfect), (base_wrong, pred_wrong)]:
        case_load = CaseLoad(ok=True, pred=pred, error_mode=None)
        score = score_case(gold, pred)
        rows.extend(_emit_stage_c_rows(
            base=base, score=score, gold=gold, pred=pred,
            case_load=case_load, others_disposition="none",
        ))
        ar, sr = _emit_nested_rows(
            base=base, gold=gold, pred=pred,
            case_load=case_load, others_disposition="none",
        )
        rows.extend(ar)
        nested_rows.extend(sr)
    return pd.DataFrame(rows), pd.DataFrame(nested_rows)


def test_chapter4_margins_overall_has_clinical_headlines():
    from scripts.eval.cascade.nested_reductions import chapter4_margins_overall
    atomic, nested = _build_two_case_nested_fixture()
    out = chapter4_margins_overall(nested, atomic)
    assert not out.empty
    expected_cols = {
        "n_total", "n_attempted", "coverage",
        "any_involved_acc", "closest_dist_mae_mm", "closest_dist_acc_tol2",
        "attempted_f1", "tp", "fp", "fn",
        "micro_precision", "micro_recall", "micro_f1",
        "hallucination_rate", "miss_rate",
    }
    assert expected_cols <= set(out.columns)


def test_chapter4_margins_per_attribute_returns_three_attributes():
    from scripts.eval.cascade.nested_reductions import (
        chapter4_margins_per_attribute,
    )
    _, nested = _build_two_case_nested_fixture()
    out = chapter4_margins_per_attribute(nested)
    assert not out.empty
    assert set(out["attribute"]) == {
        "margin_category", "margin_involved", "distance_within_2mm",
    }


def test_chapter4_margins_per_category_aggregates_per_case():
    from scripts.eval.cascade.nested_reductions import chapter4_margins_per_category
    _, nested = _build_two_case_nested_fixture()
    out = chapter4_margins_per_category(nested)
    assert not out.empty
    assert {"deep", "superficial"} <= set(out["margin_category"])
    # Perfect-case for "deep" is correct in both cases (the wrong-pred
    # fixture also has deep present); aggregate should not be 0.
    deep_row = out[out["margin_category"] == "deep"].iloc[0]
    assert deep_row["n_cases"] >= 1


def test_chapter4_margins_confusion_emits_long_form():
    from scripts.eval.cascade.nested_reductions import chapter4_margins_confusion
    _, nested = _build_two_case_nested_fixture()
    out = chapter4_margins_confusion(nested)
    assert not out.empty
    assert {"attribute", "gold", "pred", "count"} <= set(out.columns)
    attrs = set(out["attribute"])
    assert attrs == {"margin_category", "margin_involved"}
    # The wrong-fixture has a (superficial → deep) confusion on margin_category.
    cat = out[out["attribute"] == "margin_category"]
    assert ((cat["gold"] == "superficial") & (cat["pred"] == "deep")).any()


def test_chapter4_margins_missingness_four_levels():
    from scripts.eval.cascade.nested_reductions import chapter4_margins_missingness
    atomic, _ = _build_two_case_nested_fixture()
    out = chapter4_margins_missingness(atomic)
    assert not out.empty
    for lvl in ("parse_error", "field_key_absent", "empty_list", "partial_list"):
        assert f"n_{lvl}" in out.columns
        assert f"{lvl}_rate" in out.columns
    # Both cases here have items present → all are partial_list.
    row = out.iloc[0]
    assert row["n_partial_list"] == 2


def test_chapter5_lymph_nodes_overall_has_group_metrics():
    from scripts.eval.cascade.nested_reductions import chapter5_lymph_nodes_overall
    atomic, nested = _build_two_case_nested_fixture()
    out = chapter5_lymph_nodes_overall(nested, atomic)
    assert not out.empty
    expected = {
        "examined_mae", "involved_mae", "any_positive_acc",
        "group_recall", "group_precision",
        "n_groups_gold_mean", "n_groups_pred_mean",
        "tp", "fp", "fn", "micro_f1",
        "hallucination_rate", "miss_rate",
    }
    assert expected <= set(out.columns)


def test_chapter5_lymph_nodes_per_category_split_by_side_category():
    from scripts.eval.cascade.nested_reductions import (
        chapter5_lymph_nodes_per_category,
    )
    _, nested = _build_two_case_nested_fixture()
    out = chapter5_lymph_nodes_per_category(nested)
    assert not out.empty
    assert {"side", "category", "n_gold_present_cases",
            "n_pred_present_cases", "recall", "precision"} <= set(out.columns)
    # Hallucinated (left, nonsentinel) appears on pred-side only in case 2.
    halluc = out[(out["side"] == "left") & (out["category"] == "nonsentinel")]
    assert not halluc.empty
    assert halluc.iloc[0]["n_pred_present_cases"] == 1
    assert halluc.iloc[0]["n_gold_present_cases"] == 0


def test_chapter5_lymph_nodes_per_station_flattens_dict():
    from scripts.eval.cascade.nested_reductions import (
        chapter5_lymph_nodes_per_station,
    )
    _, nested = _build_two_case_nested_fixture()
    out = chapter5_lymph_nodes_per_station(nested)
    assert not out.empty
    assert {"station_name", "n_cases",
            "examined_acc_tol1", "involved_acc_tol1"} <= set(out.columns)
    # ``_per_station_counts`` normalises station_name (lowercase, stripped).
    assert "sln-1" in set(out["station_name"])


def test_chapter5_lymph_nodes_confusion_only_single_group_cases():
    """The wrong-case fixture has 2 groups on each side, perfect-case has
    2 groups too → no single-group cases → confusion is empty here. The
    test guarantees the function returns an empty (or appropriately
    shaped) frame without erroring."""
    from scripts.eval.cascade.nested_reductions import (
        chapter5_lymph_nodes_confusion,
    )
    _, nested = _build_two_case_nested_fixture()
    out = chapter5_lymph_nodes_confusion(nested)
    # Both fixture cases have ≥2 groups → no single-group observations.
    assert out.empty


def test_chapter5_lymph_nodes_missingness_four_levels():
    from scripts.eval.cascade.nested_reductions import (
        chapter5_lymph_nodes_missingness,
    )
    atomic, _ = _build_two_case_nested_fixture()
    out = chapter5_lymph_nodes_missingness(atomic)
    assert not out.empty
    assert "n_partial_list" in out.columns
    assert out.iloc[0]["n_partial_list"] == 2


def test_chapter4_chapter5_multirun_emits_when_two_runs_present():
    """Build a fake two-run atomic with attempted nested rows, confirm
    multirun returns sensible columns."""
    import pandas as pd
    from scripts.eval.cascade.nested_reductions import (
        chapter4_margins_multirun, chapter5_lymph_nodes_multirun,
    )
    base = {
        "method": "llm", "model": "test", "annotator": "gold",
        "organ_idx": 1, "organ": "breast", "subgroup": "single_primary",
        "dataset": "cmuh", "cascade_stage": "C", "field_kind": "nested_list",
        "gate_pass": True, "gold_present": True,
        "field_missing": False, "parse_error": False,
        "error_mode": None, "gold_value": None, "pred_value": None,
        "others_disposition": "none",
        "nested_missingness_level": "partial_list",
    }
    rows = []
    for case_id in ("c1", "c2", "c3"):
        for run_id, f1 in [("r0", 1.0), ("r1", 0.5)]:
            for field in ("margins", "regional_lymph_node"):
                rows.append({
                    **base, "run_id": run_id, "case_id": case_id,
                    "field": field, "attempted": True,
                    "correct": f1, "wrong": 1.0 - f1,
                })
    atomic = pd.DataFrame(rows)
    m_out = chapter4_margins_multirun(atomic)
    ln_out = chapter5_lymph_nodes_multirun(atomic)
    for out in (m_out, ln_out):
        assert not out.empty
        assert {"n_runs", "mean_per_case_f1", "per_case_f1_sd_mean",
                "per_case_f1_sd_p90", "missing_flip_rate"} <= set(out.columns)
        assert int(out.iloc[0]["n_runs"]) == 2
