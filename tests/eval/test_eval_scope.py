"""Sanity tests for the canonical scoring scope (FAIR_SCOPE whitelist)."""
import pytest

pytest.importorskip("drr_attic", reason="drr-attic is an optional research package")
from drr_attic.benchmarks.eval import scope  # noqa: E402


def test_fair_scope_defined_and_nonempty():
    fields = getattr(scope, "FAIR_SCOPE", None) or getattr(scope, "CATEGORICAL_FIELDS", None)
    assert fields, "Scope module should expose either FAIR_SCOPE or CATEGORICAL_FIELDS."


def test_cancer_categories_listed():
    assert hasattr(scope, "CANCER_CATEGORIES")
    cats = scope.CANCER_CATEGORIES
    # Should at least include the ten registry-supported organs.
    for organ in ["lung", "breast", "stomach", "colorectal", "esophagus",
                  "prostate", "thyroid", "pancreas", "cervix", "liver"]:
        assert organ in cats, f"{organ!r} missing from scope.CANCER_CATEGORIES"
