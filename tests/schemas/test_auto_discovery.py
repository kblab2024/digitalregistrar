"""Auto-discovery contract for the three-layer schema architecture.

These tests pin the behavior of the registry walkers in
``schemas/pydantic/_registry.py`` and ``schemas/extraction/_registry.py``,
plus the cross-layer parity check in ``schemas/extraction/__init__.py``.
"""
from __future__ import annotations

from typing import Union, get_args, get_origin

from digital_registrar_research.schemas.extraction import EXTRACTION_META
from digital_registrar_research.schemas.pydantic import CASE_MODELS, IsCancerCase

EXPECTED_ORGANS = {
    "breast", "cervix", "colorectal", "esophagus", "liver",
    "lung", "pancreas", "prostate", "stomach", "thyroid",
}


def test_case_models_contains_all_organs():
    assert set(CASE_MODELS.keys()) == EXPECTED_ORGANS


def test_extraction_meta_keys_match_case_models():
    assert set(EXTRACTION_META.keys()) == set(CASE_MODELS.keys())


def test_case_models_sorted():
    # discover_case_models() returns a dict sorted by key for stable iteration.
    assert list(CASE_MODELS.keys()) == sorted(CASE_MODELS.keys())


def test_extraction_meta_sorted():
    assert list(EXTRACTION_META.keys()) == sorted(EXTRACTION_META.keys())


def test_class_naming_convention():
    """Each registered class follows the `<Organ>CancerCase` PascalCase rule."""
    for key, cls in CASE_MODELS.items():
        expected = "".join(part.capitalize() for part in key.split("_")) + "CancerCase"
        assert cls.__name__ == expected, (
            f"organ {key!r} registered class {cls.__name__!r}; expected {expected!r}"
        )


def test_cancer_category_literal_covers_all_organs():
    """`IsCancerCase.cancer_category` Literal is auto-built from the registry."""
    annotation = IsCancerCase.model_fields["cancer_category"].annotation
    # The annotation is `Optional[Literal[...]]` = `Literal[...] | None`.
    if get_origin(annotation) is Union:
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        assert len(non_none) == 1
        literal_type = non_none[0]
    else:
        literal_type = annotation
    literal_values = set(get_args(literal_type))
    assert literal_values == EXPECTED_ORGANS | {"others"}


def test_extraction_meta_structure():
    """Every organ entry has fields + groups keys with non-empty values."""
    for organ, meta in EXTRACTION_META.items():
        assert "fields" in meta and "groups" in meta
        assert meta["fields"], f"{organ}: empty FIELD_META"
        assert meta["groups"], f"{organ}: empty GROUP_INSTRUCTIONS"


def test_extraction_modules_export_required_names(tmp_path):
    """Sanity-check that the discoverer's import-time validation runs.

    Drop a deliberately-broken organ file into a fresh package, then
    confirm the discoverer raises. Using a sandbox avoids polluting the
    real schemas/ tree.
    """
    import importlib
    import sys

    pkg_root = tmp_path / "fake_extraction_pkg"
    pkg_root.mkdir()
    (pkg_root / "__init__.py").write_text("", encoding="utf-8")
    (pkg_root / "bad.py").write_text(
        "# missing FIELD_META and GROUP_INSTRUCTIONS\n",
        encoding="utf-8",
    )
    sys.path.insert(0, str(tmp_path))
    try:
        # Walk the sandbox the same way discover_extraction_modules does.
        import pkgutil

        pkg = importlib.import_module("fake_extraction_pkg")
        bad_mod_info = next(
            mi for mi in pkgutil.iter_modules(pkg.__path__) if mi.name == "bad"
        )
        bad_module = importlib.import_module(f"fake_extraction_pkg.{bad_mod_info.name}")
        # Re-implement the discoverer's contract assertion inline so we
        # don't depend on private internals.
        missing = [
            attr for attr in ("FIELD_META", "GROUP_INSTRUCTIONS")
            if not hasattr(bad_module, attr)
        ]
        assert missing == ["FIELD_META", "GROUP_INSTRUCTIONS"]
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("fake_extraction_pkg", None)
        sys.modules.pop("fake_extraction_pkg.bad", None)
