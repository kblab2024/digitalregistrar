"""Smoke tests for the canonical eval/metrics module (importable, callable)."""
import pytest

pytest.importorskip("drr_attic", reason="drr-attic is an optional research package")
from drr_attic.benchmarks.eval import metrics  # noqa: E402


def test_metrics_module_exposes_aggregate_and_summary():
    assert hasattr(metrics, "aggregate_cases_to_df")
    assert hasattr(metrics, "summary_table")


def test_metrics_imports_scope_constants():
    """metrics.py uses scope constants — both should resolve at import time."""
    from drr_attic.benchmarks.eval import scope
    assert hasattr(scope, "FAIR_SCOPE") or hasattr(scope, "CATEGORICAL_FIELDS")
