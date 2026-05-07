"""Per-organ extraction tests for the rule-based baseline.

Synthetic-fixture tests pin the regex / lexicon contracts:
- TNM enums match longest-suffix correctly per organ.
- Grade is integer for most organs, Gleason group for prostate.
- Boolean fields handle present / absent / abstain phrasings.
- Excision-vs-biopsy detection separates resections from needle biopsies.
"""
from __future__ import annotations

import pytest

from digital_registrar_research.benchmarks.baselines.rules import (
    classify_organ,
    detect_excision_report,
    extract,
    extract_for_organ,
)

# --- Organ classification ----------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("Modified radical mastectomy. Infiltrating ductal carcinoma. Nottingham grade 2.",
     "breast"),
    ("Right upper lobe lobectomy. Pulmonary adenocarcinoma.", "lung"),
    ("Right hemicolectomy specimen. Colonic adenocarcinoma.", "colorectal"),
    ("Radical prostatectomy. Gleason 4 + 3 = 7.", "prostate"),
    ("Esophagectomy. Squamous cell carcinoma of the esophagus.", "esophagus"),
    ("Whipple procedure. Pancreatic ductal adenocarcinoma.", "pancreas"),
    ("Total thyroidectomy. Papillary thyroid carcinoma.", "thyroid"),
    ("Radical hysterectomy. Cervical carcinoma.", "cervix"),
    ("Partial hepatectomy. Hepatocellular carcinoma.", "liver"),
    ("Total gastrectomy. Gastric tubular adenocarcinoma.", "stomach"),
])
def test_classify_organ(text: str, expected: str) -> None:
    assert classify_organ(text) == expected


def test_classify_organ_returns_none_when_no_signal() -> None:
    assert classify_organ("specimen received in formalin") is None


# --- Excision-vs-biopsy detection -------------------------------------------

def test_excision_detects_mastectomy() -> None:
    assert detect_excision_report("Modified radical mastectomy specimen.") is True


def test_biopsy_only_is_not_excision() -> None:
    text = "Core needle biopsy only. No excision performed."
    assert detect_excision_report(text) is False


def test_cytology_is_not_excision() -> None:
    assert detect_excision_report("Fine needle aspiration cytology specimen.") is False


def test_no_signal_is_not_excision() -> None:
    assert detect_excision_report("specimen labeled patient name.") is False


# --- Per-organ TNM extraction ------------------------------------------------

def test_breast_tnm_basic() -> None:
    text = "Stage pT2 N1mi MX. Modified radical mastectomy."
    cd = extract_for_organ(text, "breast")["cancer_data"]
    assert cd["pt_category"] == "t2"
    assert cd["pn_category"] == "n1mi"
    assert cd["pm_category"] == "mx"


def test_cervix_long_suffix_t1a1() -> None:
    """Regression: cervix uses t1a1 / t1b3 — long suffixes the old regex missed."""
    text = "FIGO/AJCC stage pT1a1 N0 M0. Radical hysterectomy."
    cd = extract_for_organ(text, "cervix")["cancer_data"]
    assert cd["pt_category"] == "t1a1"


def test_lung_t1mi() -> None:
    text = "Stage pT1mi N0 M0. Lobectomy specimen."
    cd = extract_for_organ(text, "lung")["cancer_data"]
    assert cd["pt_category"] == "t1mi"


def test_tnm_descriptor_y_prefix() -> None:
    text = "ypT2 N1 M0 after neoadjuvant therapy."
    cd = extract_for_organ(text, "breast")["cancer_data"]
    assert cd.get("tnm_descriptor") == "y"


# --- Grade --------------------------------------------------------------------

def test_breast_nottingham_grade() -> None:
    text = (
        "NOTTINGHAM GRADE: POORLY DIFFERENTIATED (G3). "
        "NOTTINGHAM SCORE: 8/9 (Tubules = 3, Nuclei = 3, Mitoses = 2)."
    )
    cd = extract_for_organ(text, "breast")["cancer_data"]
    assert cd["grade"] == "3"
    assert cd["total_score"] == "8"
    assert cd["tubule_formation"] == "3"
    assert cd["nuclear_grade"] == "3"
    assert cd["mitotic_rate"] == "2"


def test_prostate_gleason_group() -> None:
    text = "Gleason score 4 + 3 = 7. Acinar adenocarcinoma."
    cd = extract_for_organ(text, "prostate")["cancer_data"]
    # Gleason 4+3=7 → ISUP/Grade Group 3 (the higher primary 4 → GG3, not GG2).
    assert cd["grade"] == "group_3_4_3"


def test_prostate_gleason_3_plus_4() -> None:
    text = "Gleason score 3 + 4 = 7."
    cd = extract_for_organ(text, "prostate")["cancer_data"]
    # Gleason 3+4=7 → ISUP/Grade Group 2.
    assert cd["grade"] == "group_2_3_4"


def test_prostate_gleason_5_plus_5() -> None:
    text = "Gleason score 5 + 5 = 10."
    cd = extract_for_organ(text, "prostate")["cancer_data"]
    assert cd["grade"] == "group_5_5_5"


def test_colorectal_grade_is_integer() -> None:
    text = "Tumor grade: G2. Right hemicolectomy."
    cd = extract_for_organ(text, "colorectal")["cancer_data"]
    # In ORGAN_SPAN for colorectal, grade is the integer span head.
    assert cd["grade"] == 2
    assert isinstance(cd["grade"], int)


# --- Boolean fields ----------------------------------------------------------

def test_lvi_present_returns_true() -> None:
    text = "Lymphovascular invasion: present."
    cd = extract_for_organ(text, "breast")["cancer_data"]
    assert cd["lymphovascular_invasion"] is True


def test_lvi_negation_returns_false() -> None:
    text = "Perineural invasion: not identified."
    cd = extract_for_organ(text, "breast")["cancer_data"]
    assert cd["perineural_invasion"] is False


def test_lvi_not_assessed_returns_none() -> None:
    """Abstention phrases should not emit a key (coverage stays 0)."""
    text = "Lymphovascular invasion: unable to assess."
    cd = extract_for_organ(text, "breast")["cancer_data"]
    assert "lymphovascular_invasion" not in cd


def test_distant_metastasis_present() -> None:
    text = "Distant metastasis: present in liver."
    cd = extract_for_organ(text, "lung")["cancer_data"]
    assert cd["distant_metastasis"] is True


# --- Span / numeric fields ---------------------------------------------------

def test_tumor_size_cm_to_mm() -> None:
    text = "TUMOR SIZE (GREATEST DIMENSION): 2.5 CM."
    cd = extract_for_organ(text, "breast")["cancer_data"]
    assert cd["tumor_size"] == 25
    assert isinstance(cd["tumor_size"], int)


def test_tumor_size_max_dimension() -> None:
    text = "Tumor size: 2.5 x 1.8 x 1.0 cm."  # max 25 mm
    cd = extract_for_organ(text, "breast")["cancer_data"]
    assert cd["tumor_size"] == 25


def test_tumor_size_mm_passthrough() -> None:
    text = "Tumor size: 8 mm."
    cd = extract_for_organ(text, "breast")["cancer_data"]
    assert cd["tumor_size"] == 8


def test_ajcc_version_extraction() -> None:
    text = "Staged per AJCC 8th edition."
    cd = extract_for_organ(text, "breast")["cancer_data"]
    assert cd.get("ajcc_version") == 8


def test_maximal_ln_size_cm_to_mm() -> None:
    text = "Largest lymph node measures 1.2 cm."
    cd = extract_for_organ(text, "esophagus")["cancer_data"]
    assert cd.get("maximal_ln_size") == 12


# --- Stage group --------------------------------------------------------------

def test_breast_dual_stage_groups() -> None:
    """An unqualified 'stage IIB' fills BOTH pathologic + anatomic for breast."""
    text = "Final stage: IIB. Modified radical mastectomy."
    cd = extract_for_organ(text, "breast")["cancer_data"]
    assert cd.get("pathologic_stage_group") == "iib"
    assert cd.get("anatomic_stage_group") == "iib"


def test_pathologic_stage_specific_phrasing() -> None:
    text = "Pathologic stage: IIIA. Anatomic stage: IIB. Mastectomy."
    cd = extract_for_organ(text, "breast")["cancer_data"]
    assert cd["pathologic_stage_group"] == "iiia"
    assert cd["anatomic_stage_group"] == "iib"


# --- Histology / procedure ---------------------------------------------------

def test_breast_histology_idc() -> None:
    text = "Tumor type: Infiltrating ductal carcinoma."
    cd = extract_for_organ(text, "breast")["cancer_data"]
    assert cd["histology"] == "invasive_carcinoma_no_special_type"


def test_lung_histology_squamous() -> None:
    text = "Squamous cell carcinoma. Lobectomy."
    cd = extract_for_organ(text, "lung")["cancer_data"]
    assert cd["histology"] == "squamous_cell_carcinoma"


def test_breast_procedure_modified_radical() -> None:
    text = "Modified radical mastectomy specimen."
    cd = extract_for_organ(text, "breast")["cancer_data"]
    assert cd["procedure"] == "modified_radical_mastectomy"


def test_lung_procedure_lobectomy() -> None:
    text = "Right upper lobe lobectomy."
    cd = extract_for_organ(text, "lung")["cancer_data"]
    assert cd["procedure"] == "lobectomy"


# --- Top-level extract() composition ----------------------------------------

def test_extract_breast_full_excision() -> None:
    text = (
        "RIGHT BREAST, MODIFIED RADICAL MASTECTOMY: "
        "Infiltrating ductal carcinoma. "
        "NOTTINGHAM GRADE: G2. "
        "TUMOR SIZE: 2.0 CM. "
        "Lymphovascular invasion: present. "
        "Perineural invasion: not identified. "
        "Stage pT2 N0 MX. "
        "ESTROGEN RECEPTOR: POSITIVE. "
        "PROGESTERONE RECEPTOR: NEGATIVE. "
        "HER2: NEGATIVE."
    )
    out = extract(text)
    assert out["cancer_excision_report"] is True
    assert out["cancer_category"] == "breast"
    cd = out["cancer_data"]
    assert cd["pt_category"] == "t2"
    assert cd["pn_category"] == "n0"
    assert cd["pm_category"] == "mx"
    assert cd["grade"] == "2"
    assert cd["tumor_size"] == 20
    assert cd["lymphovascular_invasion"] is True
    assert cd["perineural_invasion"] is False
    assert cd["procedure"] == "modified_radical_mastectomy"
    # Legacy biomarkers — only emitted by extract(), not extract_for_organ().
    assert "biomarkers" in cd
    biomarkers = {b["biomarker_category"]: b for b in cd["biomarkers"]}
    assert biomarkers["er"]["expression"] is True
    assert biomarkers["pr"]["expression"] is False


def test_extract_biopsy_only_is_not_excision() -> None:
    text = "Core needle biopsy only. Infiltrating ductal carcinoma identified."
    out = extract(text)
    assert out["cancer_excision_report"] is False


def test_extract_unknown_organ_returns_empty() -> None:
    out = extract("Specimen received in formalin. Routine processing.")
    assert out["cancer_category"] is None
    assert out["cancer_data"] == {}
