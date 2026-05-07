"""Property tests for the rule extractor's output shape.

These guard the contract that lets rule_based join the BERT scope:
1. Schema conformance: emitted fields are a subset of bert_scope_for_organ.
2. Enum-value validity: every emitted categorical / boolean value is in
   the organ's allowed-values list.
3. Span integer invariant: every emitted span field is a Python int.

Each test runs against a small fixture suite covering all 10 organs.
"""
from __future__ import annotations

import pytest

from digital_registrar_research.benchmarks.baselines.rules import extract_for_organ
from digital_registrar_research.benchmarks.eval.bert_scope import bert_scope_for_organ
from digital_registrar_research.benchmarks.eval.scope import IMPLEMENTED_ORGANS
from digital_registrar_research.benchmarks.eval.scope_organs import (
    ORGAN_BOOL,
    ORGAN_CATEGORICAL,
    ORGAN_SPAN,
)

# Synthetic fixtures sized to surface multiple field types per organ.
ORGAN_FIXTURES: dict[str, list[str]] = {
    "breast": [
        "Modified radical mastectomy. Infiltrating ductal carcinoma. "
        "Nottingham grade G2. Tumor size 2.5 cm. "
        "Stage pT2 N1mi MX. Stage IIB. "
        "Lymphovascular invasion present. Perineural invasion absent. "
        "ER positive. HER2 negative.",
        "Wide local excision. Invasive lobular carcinoma. Grade 1. "
        "Tumor size 8 mm. pT1c N0 MX.",
    ],
    "lung": [
        "Right upper lobe lobectomy. Adenocarcinoma. Grade 2. "
        "Stage pT1mi N0 M0. Visceral pleural invasion absent.",
        "Pneumonectomy. Squamous cell carcinoma. pT3 N1 MX. AJCC 8th edition.",
    ],
    "colorectal": [
        "Right hemicolectomy. Mucinous adenocarcinoma of cecum. "
        "Grade G2. pT3 N1a M0. Stage IIIA. Lymphovascular invasion present.",
        "Low anterior resection. Adenocarcinoma. pT2 N0 MX.",
    ],
    "prostate": [
        "Radical prostatectomy. Acinar adenocarcinoma. "
        "Gleason 4 + 3 = 7. pT2 N0 MX. Extraprostatic extension absent.",
    ],
    "esophagus": [
        "Esophagectomy. Squamous cell carcinoma. Grade 2. pT3 N2 MX. Stage IIIB. "
        "Lymphovascular invasion present.",
    ],
    "stomach": [
        "Total gastrectomy. Tubular adenocarcinoma. Grade 2. "
        "pT3 N1 M0. Stage IIB.",
    ],
    "pancreas": [
        "Whipple procedure. Pancreatic ductal adenocarcinoma. "
        "pT2 N0 M0. AJCC 8.",
    ],
    "thyroid": [
        "Total thyroidectomy. Papillary thyroid carcinoma. "
        "pT1a N0 M0. Tumor size 1.2 cm.",
    ],
    "cervix": [
        "Radical hysterectomy. Squamous cell carcinoma. Grade 2. "
        "pT1a1 N0 M0. Tumor size 3 mm.",
    ],
    "liver": [
        "Partial hepatectomy. Hepatocellular carcinoma. Grade 2. "
        "pT2 N0 MX. Tumor size 4.5 cm.",
    ],
}


@pytest.mark.parametrize("organ", IMPLEMENTED_ORGANS)
def test_schema_conformance_subset(organ: str) -> None:
    """Every emitted field is in bert_scope_for_organ(organ)."""
    allowed = bert_scope_for_organ(organ)
    for fixture in ORGAN_FIXTURES.get(organ, []):
        result = extract_for_organ(fixture, organ)
        # Top-level keys are part of allowed (cancer_category, cancer_excision_report).
        emitted = set(result["cancer_data"].keys())
        leak = emitted - allowed
        assert not leak, (
            f"organ={organ}: rule extractor leaked fields outside bert_scope: {leak}"
        )


@pytest.mark.parametrize("organ", IMPLEMENTED_ORGANS)
def test_categorical_values_in_enum(organ: str) -> None:
    """Every emitted categorical value is in ORGAN_CATEGORICAL[organ][field]."""
    cat = ORGAN_CATEGORICAL.get(organ, {})
    for fixture in ORGAN_FIXTURES.get(organ, []):
        cd = extract_for_organ(fixture, organ)["cancer_data"]
        for field, value in cd.items():
            if field not in cat:
                continue
            allowed = cat[field]
            assert value in allowed, (
                f"organ={organ} field={field}: value {value!r} not in enum {allowed}"
            )


@pytest.mark.parametrize("organ", IMPLEMENTED_ORGANS)
def test_bool_fields_are_bool(organ: str) -> None:
    """Every emitted boolean-field value is a Python bool."""
    bool_fields = ORGAN_BOOL.get(organ, set())
    for fixture in ORGAN_FIXTURES.get(organ, []):
        cd = extract_for_organ(fixture, organ)["cancer_data"]
        for field in bool_fields & set(cd.keys()):
            assert isinstance(cd[field], bool), (
                f"organ={organ} field={field}: value {cd[field]!r} is not bool"
            )


@pytest.mark.parametrize("organ", IMPLEMENTED_ORGANS)
def test_span_fields_are_int(organ: str) -> None:
    """Every emitted span-field value is a Python int."""
    span = ORGAN_SPAN.get(organ, set())
    for fixture in ORGAN_FIXTURES.get(organ, []):
        cd = extract_for_organ(fixture, organ)["cancer_data"]
        for field in span & set(cd.keys()):
            value = cd[field]
            # bool is a subclass of int — exclude it explicitly.
            assert isinstance(value, int) and not isinstance(value, bool), (
                f"organ={organ} field={field}: value {value!r} ({type(value).__name__}) is not int"
            )


def test_top_level_shape() -> None:
    """Output dict shape matches gold-annotation contract for any organ."""
    out = extract_for_organ("Modified radical mastectomy.", "breast")
    assert set(out.keys()) == {
        "cancer_excision_report",
        "cancer_category",
        "cancer_category_others_description",
        "cancer_data",
    }
    assert isinstance(out["cancer_excision_report"], bool)
    assert out["cancer_category"] == "breast"
    assert isinstance(out["cancer_data"], dict)


def test_unknown_organ_emits_empty_data() -> None:
    """Unknown organ → no cancer_data fields (no leakage)."""
    out = extract_for_organ("specimen received", None)
    assert out["cancer_data"] == {}
