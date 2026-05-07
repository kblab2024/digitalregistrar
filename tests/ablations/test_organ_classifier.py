"""Tests for the rule-based organ classifier used by no_router and per_section."""
from digital_registrar_research.ablations.utils.organ_classifier import (
    classify_organ_from_text,
)


def test_breast_keywords():
    assert classify_organ_from_text(
        "Modified radical mastectomy specimen. Tumor positive for ER, "
        "PR. HER2 score 2.") == "breast"


def test_thyroid_keywords():
    assert classify_organ_from_text(
        "Total thyroidectomy specimen. Papillary thyroid carcinoma."
    ) == "thyroid"


def test_lung_keywords():
    assert classify_organ_from_text(
        "Right lower lobectomy. Adenocarcinoma of the lung."
    ) == "lung"


def test_colorectal_keywords():
    assert classify_organ_from_text(
        "Sigmoid colectomy specimen. Adenocarcinoma of the colon. MMR "
        "proficient."
    ) == "colorectal"


def test_pancreas_keywords():
    assert classify_organ_from_text(
        "Whipple resection. Adenocarcinoma of the pancreas head."
    ) == "pancreas"


def test_fallback_to_dataset_lookup():
    # Empty text → no keyword hit. With dataset+folder it falls back to
    # configs/organ_code.yaml.
    assert classify_organ_from_text(
        "", dataset="tcga", fallback_organ_n="3") == "esophagus"
    assert classify_organ_from_text(
        "", dataset="cmuh", fallback_organ_n="3") == "cervix"


def test_no_signal_no_fallback_returns_none():
    assert classify_organ_from_text("") is None
    assert classify_organ_from_text("Some unrelated text",
                                    dataset=None) is None


def test_text_overrides_folder():
    # Text says lung; CMUH folder 5 = esophagus. Text wins because at
    # least one keyword matches; the fallback is only used when scores
    # are all zero.
    organ = classify_organ_from_text(
        "Right lower lobectomy. Pulmonary adenocarcinoma.",
        dataset="cmuh", fallback_organ_n="5")
    assert organ == "lung"
