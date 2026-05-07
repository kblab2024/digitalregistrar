"""Smoke tests for the canonical eval/metrics module (importable, callable)."""
from digital_registrar_research.benchmarks.eval import metrics


def test_metrics_module_exposes_aggregate_and_summary():
    assert hasattr(metrics, "aggregate_cases_to_df")
    assert hasattr(metrics, "summary_table")


def test_metrics_imports_scope_constants():
    """metrics.py uses scope constants — both should resolve at import time."""
    from digital_registrar_research.benchmarks.eval import scope
    assert hasattr(scope, "FAIR_SCOPE") or hasattr(scope, "CATEGORICAL_FIELDS")
