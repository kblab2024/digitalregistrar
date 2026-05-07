"""Three-way outcome classifier tests.

Hand-crafted gold + pred fixtures covering each outcome type. The
classifier is the foundation for every model-vs-gold scoring metric in
the new pipeline; if these break, every downstream output is wrong.
"""
from __future__ import annotations

from scripts.eval._common.outcome import CaseLoad, classify_outcome

GOLD_BREAST = {
    "cancer_excision_report": True,
    "cancer_category": "breast",
    "cancer_data": {"tumor_size": 18, "grade": 2,
                     "lymphovascular_invasion": False},
}


def test_correct_value_match():
    pred = {"cancer_excision_report": True, "cancer_category": "breast",
            "cancer_data": {"tumor_size": 18}}
    o = classify_outcome(GOLD_BREAST, CaseLoad.ok_load(pred), "tumor_size")
    assert o.kind == "correct"
    assert o.attempted and o.correct
    assert not o.wrong and not o.field_missing and not o.parse_error


def test_correct_within_tolerance():
    """tumor_size has a ±2 mm tolerance — 19 vs 18 is correct."""
    pred = {"cancer_excision_report": True, "cancer_category": "breast",
            "cancer_data": {"tumor_size": 19}}
    o = classify_outcome(GOLD_BREAST, CaseLoad.ok_load(pred), "tumor_size")
    assert o.kind == "correct"


def test_wrong_outside_tolerance():
    pred = {"cancer_excision_report": True, "cancer_category": "breast",
            "cancer_data": {"tumor_size": 25}}
    o = classify_outcome(GOLD_BREAST, CaseLoad.ok_load(pred), "tumor_size")
    assert o.kind == "wrong"
    assert o.attempted and o.wrong and not o.correct


def test_field_missing_key_absent():
    pred = {"cancer_excision_report": True, "cancer_category": "breast",
            "cancer_data": {}}
    o = classify_outcome(GOLD_BREAST, CaseLoad.ok_load(pred), "tumor_size")
    assert o.kind == "field_missing"
    assert o.field_missing and not o.attempted and not o.parse_error


def test_field_missing_key_present_null():
    """Field is attempted (key present) even if value is null — that's
    a deliberate refusal, not a missing field."""
    pred = {"cancer_excision_report": True, "cancer_category": "breast",
            "cancer_data": {"tumor_size": None}}
    o = classify_outcome(GOLD_BREAST, CaseLoad.ok_load(pred), "tumor_size")
    # is_attempted returns True for explicit-null keys; scoring is wrong
    # because gold is 18 but pred is None.
    assert o.attempted
    assert o.kind == "wrong"


def test_parse_error_propagates_to_every_field():
    o = classify_outcome(GOLD_BREAST,
                         CaseLoad.parse_failed("json_parse"), "tumor_size")
    assert o.kind == "parse_error"
    assert o.parse_error and not o.attempted
    assert o.error_mode == "json_parse"


def test_list_of_literals_set_equality():
    """Order-insensitive set equality on list-of-literals fields."""
    gold = {"cancer_excision_report": True, "cancer_category": "liver",
            "cancer_data": {"vascular_invasion":
                            ["small_vessel", "large_portal_vein"]}}
    pred_same = {"cancer_data": {"vascular_invasion":
                                 ["large_portal_vein", "small_vessel"]}}
    pred_subset = {"cancer_data": {"vascular_invasion": ["small_vessel"]}}
    pred_super = {"cancer_data": {"vascular_invasion":
                                  ["small_vessel", "large_portal_vein",
                                   "large_hepatic_vein"]}}

    o_same = classify_outcome(gold, CaseLoad.ok_load(pred_same),
                              "vascular_invasion", organ="liver")
    o_sub = classify_outcome(gold, CaseLoad.ok_load(pred_subset),
                             "vascular_invasion", organ="liver")
    o_sup = classify_outcome(gold, CaseLoad.ok_load(pred_super),
                             "vascular_invasion", organ="liver")
    assert o_same.kind == "correct"
    assert o_sub.kind == "wrong"
    assert o_sup.kind == "wrong"


def test_list_of_literals_organ_aware_dispatch():
    """tumor_extent is list-of-literals for liver but nominal for
    esophagus/stomach. Same field name, different scoring path."""
    # liver: list comparison
    gold_liver = {"cancer_excision_report": True, "cancer_category": "liver",
                  "cancer_data": {"tumor_extent":
                                  ["hepatic_vein", "diaphragm"]}}
    pred_liver = {"cancer_data": {"tumor_extent":
                                  ["diaphragm", "hepatic_vein"]}}
    o = classify_outcome(gold_liver, CaseLoad.ok_load(pred_liver),
                         "tumor_extent", organ="liver")
    assert o.kind == "correct"

    # esophagus: scalar string comparison
    gold_eso = {"cancer_excision_report": True, "cancer_category": "esophagus",
                "cancer_data": {"tumor_extent": "muscularis_propria"}}
    pred_eso = {"cancer_data": {"tumor_extent": "muscularis_propria"}}
    o = classify_outcome(gold_eso, CaseLoad.ok_load(pred_eso),
                         "tumor_extent", organ="esophagus")
    assert o.kind == "correct"


def test_list_of_literals_set_metrics():
    """Set-F1 partial-credit metric for list-of-literals."""
    from scripts.eval._common.outcome import list_of_literals_set_metrics
    g = ["a", "b", "c"]
    p = ["b", "c", "d"]
    m = list_of_literals_set_metrics(g, p)
    # tp = {b, c}, fp = {d}, fn = {a}
    assert m["tp"] == 2
    assert m["fp"] == 1
    assert m["fn"] == 1
    # F1 = 2 * (2/3) * (2/3) / (4/3) = 2/3
    assert abs(m["f1"] - 2/3) < 1e-9
    assert m["exact_match"] is False


def test_invariant_sum_correct_wrong_missing():
    """Across a small fixture cohort, n_correct + n_wrong + n_missing
    + n_parse_error must equal n_total."""
    cases = [
        # correct
        ({"cancer_data": {"tumor_size": 18}}, CaseLoad.ok_load,
         {"cancer_data": {"tumor_size": 18}}),
        # wrong
        ({"cancer_data": {"tumor_size": 18}}, CaseLoad.ok_load,
         {"cancer_data": {"tumor_size": 50}}),
        # field_missing
        ({"cancer_data": {"tumor_size": 18}}, CaseLoad.ok_load,
         {"cancer_data": {}}),
        # parse_error
        ({"cancer_data": {"tumor_size": 18}}, CaseLoad.parse_failed, None),
    ]
    n_correct = n_wrong = n_missing = n_parse = 0
    for gold, loader, pred in cases:
        if loader is CaseLoad.parse_failed:
            cl = CaseLoad.parse_failed("json_parse")
        else:
            cl = loader(pred)
        o = classify_outcome(gold, cl, "tumor_size")
        n_correct += int(o.correct)
        n_wrong += int(o.wrong)
        n_missing += int(o.field_missing)
        n_parse += int(o.parse_error)
    assert n_correct + n_wrong + n_missing + n_parse == len(cases)
    assert (n_correct, n_wrong, n_missing, n_parse) == (1, 1, 1, 1)
