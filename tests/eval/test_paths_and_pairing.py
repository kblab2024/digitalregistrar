"""Path-resolution and paired-case discovery tests against /dummy."""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.eval._common.pairing import discover_paired_cases
from scripts.eval._common.paths import from_args, parse_run_id_to_path_segment
from scripts.eval._common.stratify import (
    all_organ_indices,
    organ_index,
    organ_name,
    parse_case_id,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DUMMY = REPO_ROOT / "dummy"
HAS_DUMMY = DUMMY.is_dir()


def test_organ_round_trip_per_dataset():
    for dataset in ("cmuh", "tcga"):
        for idx in all_organ_indices(dataset):
            assert organ_index(dataset, organ_name(dataset, idx)) == idx


def test_parse_case_id():
    assert parse_case_id("cmuh1_42") == ("cmuh", 1, 42)
    assert parse_case_id("tcga3_100") == ("tcga", 3, 100)
    with pytest.raises(ValueError):
        parse_case_id("not_a_case")
    # tcga6 is rejected because TCGA only has organs 1..5.
    with pytest.raises(ValueError):
        parse_case_id("tcga6_100")


def test_run_id_validation():
    assert parse_run_id_to_path_segment("run01") == "run01"
    assert parse_run_id_to_path_segment("run02-alpha") == "run02-alpha"
    with pytest.raises(ValueError):
        parse_run_id_to_path_segment("not_a_run")
    with pytest.raises(ValueError):
        parse_run_id_to_path_segment("runABC")


@pytest.mark.skipif(not HAS_DUMMY, reason="requires /dummy fixture")
def test_paths_from_args():
    p = from_args("dummy", "cmuh")
    p.assert_exists()
    assert p.data_dir.is_dir()
    assert p.annotations_dir.is_dir()


@pytest.mark.skipif(not HAS_DUMMY, reason="requires /dummy fixture")
def test_case_ids_iter():
    p = from_args("dummy", "cmuh")
    seen = list(p.case_ids("gold"))
    if not seen:
        pytest.skip("dummy/data/cmuh/annotations/gold/ has no JSON fixtures")
    organ_idxs = {oi for oi, _ in seen}
    assert organ_idxs <= set(all_organ_indices("cmuh"))


@pytest.mark.skipif(not HAS_DUMMY, reason="requires /dummy fixture")
def test_paired_subset_relationship():
    """without_preann set should be a subset of with_preann set."""
    p = from_args("dummy", "cmuh")
    paired_nhc = discover_paired_cases(p, "nhc")
    if not paired_nhc:
        pytest.skip("no nhc paired cases in dummy")
    # All paired cases must have both with and without files.
    for pc in paired_nhc:
        assert pc.with_path.is_file()
        assert pc.without_path.is_file()
    # Sanity: paired set is non-empty and finite.
    assert len(paired_nhc) > 0


@pytest.mark.skipif(not HAS_DUMMY, reason="requires /dummy fixture")
def test_run_discovery():
    p = from_args("dummy", "cmuh")
    runs = p.discover_runs("gpt_oss_20b")
    if not runs:
        pytest.skip("no gpt_oss_20b runs in dummy")
    for rid, run_dir in runs:
        # Each run dir should be a valid runNN[-slug] form.
        parse_run_id_to_path_segment(rid)
        assert run_dir.is_dir()
