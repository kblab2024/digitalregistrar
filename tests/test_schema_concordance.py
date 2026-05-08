"""Concordance: Pydantic case-model ↔ checked-in JSON schema ↔ schema invariants.

The three representations describe the same fields. This test pins that
contract so future drift surfaces loudly (in CI) rather than silently
(in production runs).

Updated for v2: the source of truth is the hand-authored Pydantic schema
in ``schemas/pydantic/<organ>.py``. The legacy ``models/<organ>.py`` DSPy
signatures are no longer canonical (they're kept as the legacy pipeline's
extractors); concordance with them is dropped, and replaced by:
  - every field carries a ``group=`` tag;
  - every group has an entry in ``_GROUP_INSTRUCTIONS``;
  - the registry-driven router Literal matches ``CASE_MODELS``.
"""
from __future__ import annotations

import typing as _t

import pytest

from digital_registrar_research.schemas import (
    CASE_MODELS,
    list_organs,
    load_json_schema,
    load_pydantic_model,
)
from digital_registrar_research.schemas.generate import _render_model
from digital_registrar_research.schemas.pydantic import (
    field_group,
    group_instructions,
    group_order,
)

ORGANS = list_organs()


def test_organ_registry_matches_modellist():
    """``CASE_MODELS`` should align with the legacy ``organmodels`` dict.

    Soft-fails when ``organmodels`` has been retired (post-v2-cleanup PR).
    """
    try:
        from digital_registrar_research.models.modellist import organmodels
    except ImportError:  # legacy retired — that's fine.
        pytest.skip("organmodels retired; v2 registry is canonical.")
        return
    assert set(CASE_MODELS) == set(organmodels), (
        f"Pydantic registry {sorted(CASE_MODELS)} vs. organmodels {sorted(organmodels)}"
    )


@pytest.mark.parametrize("organ", ORGANS)
def test_pydantic_to_json_parity(organ):
    """``model.model_json_schema()`` should match the checked-in ``data/<organ>.json``."""
    on_disk = load_json_schema(organ)
    fresh = load_pydantic_model(organ).model_json_schema()
    assert on_disk == fresh, (
        f"Drift in {organ}.json. Run `python -m digital_registrar_research.schemas.generate`."
    )


@pytest.mark.parametrize("organ", ORGANS)
def test_render_is_idempotent(organ):
    """Re-rendering the same model produces byte-identical JSON."""
    model = load_pydantic_model(organ)
    assert _render_model(model) == _render_model(model)


# --- v2 schema invariants -------------------------------------------------

@pytest.mark.parametrize("organ", ORGANS)
def test_every_field_has_a_group(organ):
    """Every Pydantic field must be tagged with a clinical ``group=``.

    A missing tag is an authoring error: the factory needs the tag to
    decide which decomposition group the field belongs to.
    """
    model = load_pydantic_model(organ)
    untagged = [
        name for name, fi in model.model_fields.items()
        if not field_group(fi)
    ]
    assert not untagged, (
        f"{organ}: untagged fields {untagged}. Use `GroupedField(group=..., desc=...)`."
    )


@pytest.mark.parametrize("organ", ORGANS)
def test_every_group_has_an_instruction(organ):
    """Every group tag on a field must have an entry in ``_GROUP_INSTRUCTIONS``.

    Missing-instruction groups still extract (the factory falls back to
    DEFAULT_GROUP_INSTRUCTION), but the omission almost certainly
    indicates a schema authoring slip.
    """
    model = load_pydantic_model(organ)
    instr = group_instructions(model)
    used = {field_group(fi) for fi in model.model_fields.values() if field_group(fi)}
    orphans = used - set(instr)
    assert not orphans, (
        f"{organ}: groups used on fields but missing from _GROUP_INSTRUCTIONS: {sorted(orphans)}"
    )


@pytest.mark.parametrize("organ", ORGANS)
def test_group_order_is_deterministic(organ):
    """``group_order`` returns groups in a stable, deterministic order."""
    model = load_pydantic_model(organ)
    a = group_order(model)
    b = group_order(model)
    assert a == b
    assert len(a) == len(set(a)), f"{organ}: duplicate group in order: {a}"


def test_router_literal_matches_registry():
    """The factory-built router's ``cancer_category`` Literal == sorted(CASE_MODELS) + ['others']."""
    from digital_registrar_research.signatures import build_router_signature

    sig = build_router_signature()
    ann = sig.model_fields["cancer_category"].annotation
    # Unwrap Optional → Literal members.
    members: set[str] = set()
    for arg in _t.get_args(ann):
        if _t.get_origin(arg) is _t.Literal:
            members |= set(_t.get_args(arg))
    expected = set(CASE_MODELS) | {"others"}
    assert members == expected, f"router Literal {members} != expected {expected}"


def test_jsonize_literal_excludes_others():
    """Legacy asymmetry preserved: ``ReportJsonize`` Literal MUST NOT include 'others'."""
    from digital_registrar_research.signatures import build_jsonize_signature

    sig = build_jsonize_signature()
    ann = sig.model_fields["cancer_category"].annotation
    members: set[str] = set()
    for arg in _t.get_args(ann):
        if _t.get_origin(arg) is _t.Literal:
            members |= set(_t.get_args(arg))
    assert "others" not in members, (
        "ReportJsonize is intentionally narrower than is_cancer; if you fix "
        "this asymmetry, do it deliberately and update the test."
    )
    assert members == set(CASE_MODELS)


def test_colorectal_uses_colon_common_types():
    """Documented naming-drift contract: ``ColorectalCancerCase`` references ``Colon*`` nested types.

    The registry key is ``"colorectal"`` but the synthesised nested types
    use the legacy ``Colon*`` prefix (set via the ``_TYPE_PREFIX_OVERRIDES``
    table in ``schemas/pydantic/_case_builder.py``).
    """
    case_model = load_pydantic_model("colorectal")
    annotations_text = " ".join(
        repr(f.annotation) for f in case_model.model_fields.values()
    )
    assert "Colon" in annotations_text, (
        "ColorectalCancerCase should synthesise Colon* nested types via "
        "_case_builder._TYPE_PREFIX_OVERRIDES"
    )
