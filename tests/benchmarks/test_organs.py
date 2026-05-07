"""Tests for the dataset-aware organ_n <-> organ_name loader."""
from digital_registrar_research.benchmarks.organs import (
    dataset_organs,
    load_organ_code,
    organ_n_to_name,
    organ_name_to_n,
)


def test_tcga_folder_mapping():
    # Per configs/organ_code.yaml — TCGA's five-organ subset.
    assert organ_n_to_name("tcga", "1") == "breast"
    assert organ_n_to_name("tcga", "2") == "colorectal"
    assert organ_n_to_name("tcga", "3") == "esophagus"
    assert organ_n_to_name("tcga", "4") == "stomach"
    assert organ_n_to_name("tcga", "5") == "liver"


def test_cmuh_folder_mapping():
    # Per configs/organ_code.yaml — CMUH's full ten-organ list. Note
    # CMUH and TCGA have *different* numbering: folder 3 is cervix in
    # CMUH but esophagus in TCGA.
    assert organ_n_to_name("cmuh", "1") == "pancreas"
    assert organ_n_to_name("cmuh", "2") == "breast"
    assert organ_n_to_name("cmuh", "3") == "cervix"
    assert organ_n_to_name("cmuh", "10") == "thyroid"


def test_unknown_dataset_or_organ_n_returns_none():
    assert organ_n_to_name("nope", "1") is None
    assert organ_n_to_name("tcga", "999") is None
    assert organ_n_to_name("tcga", "not-an-int") is None


def test_organ_name_to_n_inverse():
    # TCGA breast = folder 1; CMUH breast = folder 2.
    assert organ_name_to_n("tcga", "breast") == "1"
    assert organ_name_to_n("cmuh", "breast") == "2"
    assert organ_name_to_n("tcga", "lung") is None  # not in TCGA's subset


def test_dataset_organs_ordered_by_folder_number():
    organs = dataset_organs("tcga")
    assert organs == ["breast", "colorectal", "esophagus", "stomach", "liver"]


def test_load_organ_code_matches_yaml():
    code = load_organ_code()
    assert "tcga" in code
    assert "cmuh" in code
    assert code["tcga"][1] == "breast"
    assert code["cmuh"][3] == "cervix"
