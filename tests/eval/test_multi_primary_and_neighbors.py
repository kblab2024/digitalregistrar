"""Tests for multi_primary detection and semantic_neighbors lookups."""
from __future__ import annotations

from digital_registrar_research.benchmarks.eval.multi_primary import (
    detect_multi_primary,
    n_tumors_estimate,
    subgroup_label,
)
from digital_registrar_research.benchmarks.eval.semantic_neighbors import (
    is_neighbor,
    neighbors_for_field,
)

# --- multi_primary -----------------------------------------------------------

def test_bilateral_breast_flagged():
    ann = {"cancer_excision_report": True, "cancer_category": "breast",
           "cancer_data": {"cancer_laterality": "bilateral"}}
    assert detect_multi_primary(ann) is True
    assert subgroup_label(ann) == "multi_primary"
    assert n_tumors_estimate(ann) == 2


def test_multifocal_lung_flagged():
    ann = {"cancer_excision_report": True, "cancer_category": "lung",
           "cancer_data": {"tumor_focality": "multifocal"}}
    assert detect_multi_primary(ann) is True


def test_single_primary_returns_false():
    ann = {"cancer_excision_report": True, "cancer_category": "breast",
           "cancer_data": {"cancer_laterality": "right",
                           "tumor_focality": "unifocal"}}
    assert detect_multi_primary(ann) is False
    assert subgroup_label(ann) == "single_primary"


def test_non_excision_returns_none():
    ann = {"cancer_excision_report": False}
    assert detect_multi_primary(ann) is None
    assert subgroup_label(ann) == "unknown"


def test_multi_clock_string_flagged():
    ann = {"cancer_excision_report": True, "cancer_category": "breast",
           "cancer_data": {"cancer_clock": "12 and 3"}}
    assert detect_multi_primary(ann) is True


# --- semantic_neighbors ------------------------------------------------------

def test_pt_substage_pairs():
    assert is_neighbor("pt_category", "t1", "t1a") is True
    assert is_neighbor("pt_category", "t1a", "t1") is True   # symmetric
    assert is_neighbor("pt_category", "t2", "t2a") is True
    assert is_neighbor("pt_category", "t1", "t3") is False


def test_pm_m0_mx_pair():
    assert is_neighbor("pm_category", "m0", "mx") is True
    assert is_neighbor("pm_category", "mx", "m0") is True


def test_anatomic_vs_pathologic():
    assert is_neighbor("tnm_descriptor", "anatomic", "pathologic") is True


def test_unknown_field_returns_false():
    assert is_neighbor("unknown_field", "x", "y") is False


def test_neighbors_for_field_lookup():
    pairs = neighbors_for_field("pt_category")
    assert len(pairs) >= 4  # at least t1/t1a/t1b/t1c entries
    # All entries should have field == "pt_category"
    assert all(p.field == "pt_category" for p in pairs)
