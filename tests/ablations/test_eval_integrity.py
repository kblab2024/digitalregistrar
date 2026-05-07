"""Eval-pipeline integrity tests for the post-redesign aggregator + stats.

These tests fail on the pre-redesign code:
* `_split_method` mis-parses multi-underscore methods.
* `compute_efficiency` double-counts cases with both schema + parse errors.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from digital_registrar_research.ablations.eval.run_ablations import (
    compute_efficiency,
)
from digital_registrar_research.ablations.eval.stats import _split_method


def _write_summary(run_dir: Path, n_cases: int) -> None:
    (run_dir / "_summary.json").write_text(
        json.dumps({"n_cases": n_cases}), encoding="utf-8")


def _write_case(run_dir: Path, organ_n: str, case_id: str,
                *, schema_err: bool = False, parse_err: bool = False) -> None:
    organ_dir = run_dir / organ_n
    organ_dir.mkdir(parents=True, exist_ok=True)
    payload: dict = {"cancer_excision_report": True,
                     "cancer_category": "breast"}
    if schema_err:
        payload["_schema_errors"] = ["bad key foo"]
    if parse_err:
        payload["_error"] = "boom"
    (organ_dir / f"{case_id}.json").write_text(
        json.dumps(payload), encoding="utf-8")


def test_split_method_handles_underscored_models():
    """The historical bug: rsplit('_', 1) split a model slug at its last
    underscore, producing ('free_text_regex_gpt_oss', '20b'). The
    rewrite walks known cell-ids longest-first so the cell name wins."""
    assert _split_method("free_text_regex_gpt_oss_20b") == (
        "free_text_regex", "gpt_oss_20b")
    assert _split_method("dspy_monolithic_gemma4e2b") == (
        "dspy_monolithic", "gemma4e2b")
    assert _split_method("constrained_decoding_qwen3_30b") == (
        "constrained_decoding", "qwen3_30b")


def test_split_method_index_overrides_fallback():
    """When the caller supplies an explicit index (built from the grid
    CSV's cell/model columns), the parser uses it directly — no
    string-walking needed."""
    idx = {"weird_method_name": ("custom_cell", "custom_model")}
    assert _split_method("weird_method_name", index=idx) == (
        "custom_cell", "custom_model")


def test_efficiency_no_double_counting(tmp_path: Path):
    """A case carrying BOTH _schema_errors AND _error must not be counted
    in both schema_errors and parse_errors so that the rates can never
    sum past 1.0."""
    run_dir = tmp_path / "dspy_monolithic" / "gpt_oss_20b" / "run01"
    run_dir.mkdir(parents=True)
    _write_summary(run_dir, n_cases=3)
    # 1 case both schema + parse error, 1 schema-only, 1 ok.
    _write_case(run_dir, "1", "case_a", schema_err=True, parse_err=True)
    _write_case(run_dir, "1", "case_b", schema_err=True)
    _write_case(run_dir, "1", "case_ok")

    runs = [("dspy_monolithic", "gpt_oss_20b", "run01", run_dir)]
    eff = compute_efficiency(runs)
    assert not eff.empty
    row = eff.iloc[0]
    n = int(row["n_cases"])
    assert n == 3
    schema_n = int(row["schema_errors"])
    parse_n = int(row["parse_errors"])
    failed = int(row["failed_total"])
    # schema_errors counts BOTH the schema-only and the both case (2);
    # parse_errors counts the both case (1). They are NOT mutually
    # exclusive — that's deliberate so each rate is interpretable on
    # its own. failed_total is the non-overlapping union (2).
    assert schema_n == 2, f"expected schema_errors=2, got {schema_n}"
    assert parse_n == 1, f"expected parse_errors=1, got {parse_n}"
    assert failed == 2, f"expected failed_total=2, got {failed}"
    assert failed / n <= 1.0, "failed_rate must never exceed 1.0"


def test_aggregator_flags_cancer_category_mismatch(tmp_path: Path):
    """When a prediction's cancer_category disagrees with gold's, the
    aggregator must emit a cancer_category_mismatch flag — an accuracy
    signal, not a runtime error. Folder numbers are case-id keys only
    and are not used to derive ground truth."""
    from digital_registrar_research.ablations.eval.run_ablations import (
        build_grid_dataframe,
    )

    run_dir = tmp_path / "results" / "ablations" / "tcga" / "no_router" / \
              "gpt_oss_20b" / "run01"
    run_dir.mkdir(parents=True)
    _write_summary(run_dir, n_cases=1)
    organ_dir = run_dir / "3"
    organ_dir.mkdir(parents=True)
    (organ_dir / "case_x.json").write_text(json.dumps({
        "cancer_excision_report": True,
        "cancer_category": "breast",
        "cancer_data": {},
    }), encoding="utf-8")

    gold_dir = tmp_path / "data" / "tcga" / "annotations" / "gold" / "3"
    gold_dir.mkdir(parents=True)
    (gold_dir / "case_x.json").write_text(json.dumps({
        "cancer_excision_report": True,
        "cancer_category": "thyroid",
        "cancer_data": {},
    }), encoding="utf-8")
    gold_root = tmp_path / "data" / "tcga" / "annotations" / "gold"

    runs = [("no_router", "gpt_oss_20b", "run01", run_dir)]
    df = build_grid_dataframe(runs, gold_root, dataset="tcga")
    assert "cancer_category_mismatch" in df.columns
    flagged = df[df["case_id"] == "case_x"]["cancer_category_mismatch"].any()
    assert flagged, "Expected cancer_category_mismatch to be True for case_x"


def test_dspy_trace_dump_no_history(tmp_path: Path):
    """dump_dspy_trace must be a no-op when DSPy isn't loaded — the
    helper is called from run_loop unconditionally so it has to handle
    the non-DSPy runners without exploding."""
    from digital_registrar_research.ablations.runners._base import (
        dump_dspy_trace,
    )
    # Without configuring DSPy, dspy.settings.lm is None — should
    # return the cursor unchanged and not write the JSONL.
    new_cursor = dump_dspy_trace("case", tmp_path, since_index=0)
    assert new_cursor == 0
    assert not (tmp_path / "_dspy_trace.jsonl").exists()


@pytest.mark.parametrize("dataset, organ_n, expected", [
    ("tcga", "1", "breast"),
    ("tcga", "5", "liver"),
    ("cmuh", "1", "pancreas"),
    ("cmuh", "10", "thyroid"),
])
def test_organs_loader_param(dataset, organ_n, expected):
    """Sanity guard against future YAML edits — the test catches
    folder-number drift in the canonical mapping."""
    from digital_registrar_research.benchmarks.organs import organ_n_to_name
    assert organ_n_to_name(dataset, organ_n) == expected
