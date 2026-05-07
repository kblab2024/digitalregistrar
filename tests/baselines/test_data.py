"""Regression tests for the BERT baseline data loader (_data.py).

The loader walks ``<root>/data/<dataset>/annotations/gold/<organ_n>/*.json``
and produces case dicts with the report path derived by convention.
There is no train/test split; every gold annotation is a case.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from digital_registrar_research.benchmarks.baselines._data import (
    _organ_n,
    _walk_dataset,
    load_cases,
    load_predict_cases,
    per_dataset_counts,
)


def _seed_dummy_case(
    root: Path, dataset: str, organ_n: str, case_id: str,
    cancer_category: str | None = "breast",
    cancer_excision_report: bool = True,
) -> None:
    ann_dir = root / "data" / dataset / "annotations" / "gold" / organ_n
    rep_dir = root / "data" / dataset / "reports" / organ_n
    ann_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)
    (ann_dir / f"{case_id}.json").write_text(
        json.dumps({
            "cancer_category": cancer_category,
            "cancer_excision_report": cancer_excision_report,
        }),
        encoding="utf-8",
    )
    (rep_dir / f"{case_id}.txt").write_text("dummy report", encoding="utf-8")


# --- _organ_n ----------------------------------------------------------------


def test_organ_n_parses_string() -> None:
    assert _organ_n("cmuh1_42", "cmuh") == "1"
    assert _organ_n("cmuh10_5", "cmuh") == "10"
    assert _organ_n("tcga3_99", "tcga") == "3"


def test_organ_n_dataset_prefix_required() -> None:
    with pytest.raises(ValueError, match="could not parse organ_n"):
        _organ_n("tcga1_42", "cmuh")


def test_organ_n_rejects_malformed() -> None:
    with pytest.raises(ValueError):
        _organ_n("not_a_case_id", "cmuh")


# --- _walk_dataset -----------------------------------------------------------


def test_walk_dataset_collects_all_gold(tmp_path: Path) -> None:
    _seed_dummy_case(tmp_path, "cmuh", "2", "cmuh2_1", "breast")
    _seed_dummy_case(tmp_path, "cmuh", "9", "cmuh9_1", "stomach")
    cases = _walk_dataset("cmuh", tmp_path)
    ids = sorted(c["id"] for c in cases)
    assert ids == ["cmuh2_1", "cmuh9_1"]
    by_id = {c["id"]: c for c in cases}
    assert by_id["cmuh2_1"]["organ_n"] == "2"
    assert by_id["cmuh2_1"]["cancer_category"] == "breast"
    assert by_id["cmuh9_1"]["organ_n"] == "9"
    assert by_id["cmuh9_1"]["cancer_category"] == "stomach"
    # Report paths derived by convention
    for c in cases:
        assert Path(c["report_path"]).exists()
        assert Path(c["annotation_path"]).exists()


def test_walk_dataset_returns_empty_for_missing_tree(tmp_path: Path) -> None:
    cases = _walk_dataset("cmuh", tmp_path)
    assert cases == []


# --- load_cases (the public entry point) -------------------------------------


def test_load_cases_pools_across_datasets(tmp_path: Path) -> None:
    _seed_dummy_case(tmp_path, "cmuh", "2", "cmuh2_1", "breast")
    _seed_dummy_case(tmp_path, "tcga", "1", "tcga1_1", "breast")
    cases = load_cases(datasets=["cmuh", "tcga"], root=tmp_path)
    assert sorted(c["id"] for c in cases) == ["cmuh2_1", "tcga1_1"]


def test_load_cases_filters_by_organ(tmp_path: Path) -> None:
    _seed_dummy_case(tmp_path, "cmuh", "2", "cmuh2_1", "breast")
    _seed_dummy_case(tmp_path, "cmuh", "9", "cmuh9_1", "stomach")
    _seed_dummy_case(tmp_path, "cmuh", "10", "cmuh10_1", "thyroid")
    cases = load_cases(
        datasets=["cmuh"], root=tmp_path,
        organs={"breast", "stomach"},
    )
    assert sorted(c["id"] for c in cases) == ["cmuh2_1", "cmuh9_1"]


def test_load_cases_included_only_drops_non_excision(tmp_path: Path) -> None:
    _seed_dummy_case(tmp_path, "cmuh", "2", "cmuh2_1", "breast",
                     cancer_excision_report=True)
    _seed_dummy_case(tmp_path, "cmuh", "2", "cmuh2_2", None,
                     cancer_excision_report=False)
    cases = load_cases(datasets=["cmuh"], root=tmp_path, included_only=True)
    assert [c["id"] for c in cases] == ["cmuh2_1"]


def test_per_dataset_counts(tmp_path: Path) -> None:
    _seed_dummy_case(tmp_path, "cmuh", "2", "cmuh2_1", "breast")
    _seed_dummy_case(tmp_path, "tcga", "1", "tcga1_1", "breast")
    _seed_dummy_case(tmp_path, "tcga", "2", "tcga2_1", "colorectal")
    cases = load_cases(datasets=["cmuh", "tcga"], root=tmp_path)
    counts = per_dataset_counts(cases)
    assert counts == {"cmuh": 1, "tcga": 2}


# --- load_predict_cases (predict-side, gold-optional) ------------------------


def _seed_report_only(
    root: Path, dataset: str, organ_n: str, case_id: str,
) -> None:
    """Seed a reports-only case (no gold annotation)."""
    rep_dir = root / "data" / dataset / "reports" / organ_n
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / f"{case_id}.txt").write_text("dummy report", encoding="utf-8")


def test_load_predict_cases_walks_reports_without_gold(tmp_path: Path) -> None:
    # No annotations/gold/ dir at all — predict-side discovery must still work.
    _seed_report_only(tmp_path, "tcga", "1", "tcga1_1")
    _seed_report_only(tmp_path, "tcga", "2", "tcga2_1")
    cases = load_predict_cases(datasets=["tcga"], root=tmp_path)
    assert sorted(c["id"] for c in cases) == ["tcga1_1", "tcga2_1"]
    by_id = {c["id"]: c for c in cases}
    assert by_id["tcga1_1"]["organ_name"] == "breast"
    assert by_id["tcga1_1"]["cancer_category"] is None
    assert by_id["tcga1_1"]["annotation_path"] is None
    assert by_id["tcga2_1"]["organ_name"] == "colorectal"
    assert by_id["tcga2_1"]["cancer_category"] is None
    assert by_id["tcga2_1"]["annotation_path"] is None
    for c in cases:
        assert Path(c["report_path"]).exists()


def test_load_predict_cases_filters_by_organ_name(tmp_path: Path) -> None:
    _seed_report_only(tmp_path, "tcga", "1", "tcga1_1")  # breast
    _seed_report_only(tmp_path, "tcga", "2", "tcga2_1")  # colorectal
    _seed_report_only(tmp_path, "tcga", "5", "tcga5_1")  # liver
    cases = load_predict_cases(
        datasets=["tcga"], root=tmp_path, organs={"breast", "liver"},
    )
    assert sorted(c["id"] for c in cases) == ["tcga1_1", "tcga5_1"]


def test_load_predict_cases_enriches_with_gold_when_present(
    tmp_path: Path,
) -> None:
    # Both reports AND gold present — cancer_category should be populated
    # for the case with gold, but None for the one without.
    _seed_dummy_case(tmp_path, "tcga", "1", "tcga1_1", "breast")
    _seed_report_only(tmp_path, "tcga", "1", "tcga1_2")  # no gold
    cases = load_predict_cases(datasets=["tcga"], root=tmp_path)
    by_id = {c["id"]: c for c in cases}
    assert by_id["tcga1_1"]["cancer_category"] == "breast"
    assert by_id["tcga1_1"]["annotation_path"] is not None
    assert Path(by_id["tcga1_1"]["annotation_path"]).exists()
    assert by_id["tcga1_2"]["cancer_category"] is None
    assert by_id["tcga1_2"]["annotation_path"] is None


def test_load_predict_cases_returns_empty_for_missing_reports(
    tmp_path: Path,
) -> None:
    # No reports dir at all.
    cases = load_predict_cases(datasets=["tcga"], root=tmp_path)
    assert cases == []


def test_load_predict_cases_skips_unknown_organ_n(tmp_path: Path) -> None:
    # TCGA only has organs 1-5 in organ_code.yaml; 99 is unknown.
    _seed_report_only(tmp_path, "tcga", "1", "tcga1_1")
    _seed_report_only(tmp_path, "tcga", "99", "tcga99_1")
    cases = load_predict_cases(datasets=["tcga"], root=tmp_path)
    assert [c["id"] for c in cases] == ["tcga1_1"]
