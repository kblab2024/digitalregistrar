"""Concordance: Pydantic case-model ↔ checked-in JSON schema ↔ DSPy signatures.

The three representations describe the same fields. This test pins that
contract so future drift surfaces loudly (in CI) rather than silently
(in production runs).
"""
from __future__ import annotations

import pytest

from digital_registrar_research.models.modellist import organmodels
from digital_registrar_research.schemas import (
    CASE_MODELS,
    list_organs,
    load_json_schema,
    load_pydantic_model,
)
from digital_registrar_research.schemas.generate import _render_model
from digital_registrar_research.schemas.pydantic._builder import _iter_signature_output_fields

ORGANS = list_organs()


def test_organ_registry_matches_modellist():
    """The 10 organs in CASE_MODELS should align with the keys of organmodels."""
    assert set(CASE_MODELS) == set(organmodels), (
        f"Pydantic registry {sorted(CASE_MODELS)} vs. organmodels {sorted(organmodels)}"
    )


@pytest.mark.parametrize("organ", ORGANS)
def test_pydantic_to_json_parity(organ):
    """`model.model_json_schema()` should match the checked-in `data/<organ>.json`."""
    on_disk = load_json_schema(organ)
    fresh = load_pydantic_model(organ).model_json_schema()
    assert on_disk == fresh, (
        f"Drift in {organ}.json. Run `python -m digital_registrar_research.schemas.generate`."
    )


@pytest.mark.parametrize("organ", ORGANS)
def test_dspy_signature_coverage(organ):
    """Union of OutputField names across signatures should equal the case-model's fields."""
    import sys
    builder_globals = sys.modules[
        "digital_registrar_research.schemas.pydantic._builder"
    ].__dict__

    sig_field_names: set[str] = set()
    for sig_name in organmodels[organ]:
        cls = builder_globals.get(sig_name)
        assert cls is not None, f"Signature {sig_name} not importable."
        sig_field_names.update(name for name, _, _ in _iter_signature_output_fields(cls))

    case_fields = set(load_pydantic_model(organ).model_fields)
    assert sig_field_names == case_fields, (
        f"{organ}: signatures {sorted(sig_field_names - case_fields)!r} not in case-model; "
        f"case-model {sorted(case_fields - sig_field_names)!r} not in signatures."
    )


@pytest.mark.parametrize("organ", ORGANS)
def test_render_is_idempotent(organ):
    """Re-rendering the same model produces byte-identical JSON."""
    model = load_pydantic_model(organ)
    a = _render_model(model)
    b = _render_model(model)
    assert a == b


def test_colorectal_uses_colon_signatures():
    """Documented naming-drift contract: ColorectalCancerCase composes Colon* nested types."""
    case_model = load_pydantic_model("colorectal")
    # The annotation names tell us about the inner types referenced.
    annotations_text = " ".join(
        repr(f.annotation) for f in case_model.model_fields.values()
    )
    assert "Colon" in annotations_text, (
        "ColorectalCancerCase should reference Colon* types from models/colon.py"
    )
