"""Round-trip tests for the per-dataset organ-code loader."""
from __future__ import annotations

import pytest

from digital_registrar_research.benchmarks import organs


def test_tcga_mapping_matches_yaml() -> None:
    m = organs.organs_for("tcga")
    assert m == {1: "breast", 2: "colorectal", 3: "esophagus",
                 4: "stomach", 5: "liver"}


def test_cmuh_mapping_matches_yaml() -> None:
    m = organs.organs_for("cmuh")
    assert m == {
        1: "pancreas", 2: "breast", 3: "cervix", 4: "colorectal",
        5: "esophagus", 6: "liver", 7: "lung", 8: "prostate",
        9: "stomach", 10: "thyroid",
    }


def test_organ_name_per_dataset() -> None:
    assert organs.organ_name("tcga", 1) == "breast"
    assert organs.organ_name("cmuh", 1) == "pancreas"
    assert organs.organ_name("tcga", 3) == "esophagus"
    assert organs.organ_name("cmuh", 3) == "cervix"


def test_organ_n_for_reverse_lookup() -> None:
    assert organs.organ_n_for("tcga", "esophagus") == 3
    assert organs.organ_n_for("cmuh", "esophagus") == 5
    assert organs.organ_n_for("cmuh", "thyroid") == 10  # CMUH-only organ
    assert organs.organ_n_for("tcga", "BREAST") == 1  # case-insensitive


def test_organ_name_unknown_raises() -> None:
    with pytest.raises(KeyError):
        organs.organ_name("tcga", 6)
    with pytest.raises(KeyError):
        organs.organ_name("cmuh", 11)
    with pytest.raises(KeyError):
        organs.organ_name("notadataset", 1)


def test_organ_n_for_unknown_raises() -> None:
    with pytest.raises(KeyError):
        organs.organ_n_for("tcga", "pancreas")  # not in TCGA
    with pytest.raises(KeyError):
        organs.organ_n_for("tcga", "thyroid")  # CMUH-only
    with pytest.raises(KeyError):
        organs.organ_n_for("cmuh", "kidney")  # not in either


def test_common_organs_intersection() -> None:
    common = set(organs.common_organs("cmuh", "tcga"))
    assert common == {"breast", "colorectal", "esophagus", "stomach", "liver"}


def test_union_organs_full_coverage() -> None:
    union = set(organs.union_organs("cmuh", "tcga"))
    assert union == {
        "breast", "cervix", "colorectal", "esophagus", "liver",
        "lung", "pancreas", "prostate", "stomach", "thyroid",
    }


def test_common_organs_no_args_uses_all_datasets() -> None:
    assert set(organs.common_organs()) == set(organs.common_organs("cmuh", "tcga"))


def test_parse_case_id_tcga() -> None:
    assert organs.parse_case_id("tcga1_42") == ("tcga", 1, 42)
    assert organs.parse_case_id("tcga5_1") == ("tcga", 5, 1)


def test_parse_case_id_cmuh() -> None:
    assert organs.parse_case_id("cmuh1_42") == ("cmuh", 1, 42)
    assert organs.parse_case_id("cmuh10_5") == ("cmuh", 10, 5)


def test_parse_case_id_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="no organ_n=6"):
        organs.parse_case_id("tcga6_1")
    with pytest.raises(ValueError, match="no organ_n=11"):
        organs.parse_case_id("cmuh11_1")


def test_parse_case_id_rejects_unknown_dataset() -> None:
    with pytest.raises(ValueError, match="unknown dataset"):
        organs.parse_case_id("foo1_1")


def test_parse_case_id_rejects_malformed() -> None:
    with pytest.raises(ValueError, match="malformed"):
        organs.parse_case_id("not_a_case_id")
    with pytest.raises(ValueError, match="malformed"):
        organs.parse_case_id("cmuh1")  # missing _N
    with pytest.raises(ValueError, match="malformed"):
        organs.parse_case_id("cmuhX_1")  # non-numeric organ_n


def test_all_datasets_sorted() -> None:
    assert organs.all_datasets() == ("cmuh", "tcga")


def test_constants_match_organs_for() -> None:
    assert organs.CMUH_ORGANS() == organs.organs_for("cmuh")
    assert organs.TCGA_ORGANS() == organs.organs_for("tcga")
    assert set(organs.COMMON_ORGANS()) == set(organs.common_organs("cmuh", "tcga"))
