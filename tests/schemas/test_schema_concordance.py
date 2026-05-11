"""Concordance: Pydantic case-model ↔ checked-in JSON schema ↔ schema invariants.

These three representations describe the same fields. The tests below
pin that contract so future drift surfaces loudly (in CI) rather than
silently (in production runs).

Two-layer invariants (Phase 3+):
  - every field has a ``group`` tag in ``EXTRACTION_META[organ]["fields"]``;
  - every group used on a field has an entry in
    ``EXTRACTION_META[organ]["groups"]``;
  - the registry-driven router Literal matches ``CASE_MODELS``.

Layer-3 invariants (aliases):
  - every ``value_hints`` canonical key (loaded from
    ``schemas/aliases/<organ>.toml``) is a real enum value on the
    corresponding pydantic field.
"""
from __future__ import annotations

import json
import typing as _t
from enum import Enum

import pytest
from pydantic import BaseModel

from digital_registrar_research.schemas import (
    CASE_MODELS,
    list_organs,
    load_json_schema,
    load_pydantic_model,
)
from digital_registrar_research.schemas.extraction import EXTRACTION_META
from digital_registrar_research.schemas.extraction._resolve import (
    field_group as meta_field_group,
    group_order as meta_group_order,
)
from digital_registrar_research.schemas.generate import _render_model

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
    """The full render (Pydantic schema + metadata injection) should match the
    checked-in ``data/<organ>.json``. ``_render_model`` is the public
    renderer that does the inject + serialize round-trip."""
    on_disk = load_json_schema(organ)
    fresh = json.loads(_render_model(organ, load_pydantic_model(organ)))
    assert on_disk == fresh, (
        f"Drift in {organ}.json. Run `python -m digital_registrar_research.schemas.generate`."
    )


@pytest.mark.parametrize("organ", ORGANS)
def test_render_is_idempotent(organ):
    """Re-rendering the same model produces byte-identical JSON."""
    model = load_pydantic_model(organ)
    assert _render_model(organ, model) == _render_model(organ, model)


# --- two-layer schema invariants -----------------------------------------


@pytest.mark.parametrize("organ", ORGANS)
def test_every_field_has_a_group(organ):
    """Every Pydantic field must have a ``group`` tag in FIELD_META.

    A missing tag is an authoring error: the factory needs it to decide
    which decomposition group the field belongs to.
    """
    model = load_pydantic_model(organ)
    field_meta = EXTRACTION_META[organ]["fields"]
    untagged = [
        name for name in model.model_fields
        if not meta_field_group(field_meta, name)
    ]
    assert not untagged, (
        f"{organ}: untagged fields {untagged}. Add a `group` key to "
        f"FIELD_META in schemas/extraction/{organ}.py."
    )


@pytest.mark.parametrize("organ", ORGANS)
def test_every_group_has_an_instruction(organ):
    """Every group tag used on a field must have an entry in
    ``GROUP_INSTRUCTIONS``.

    Missing-instruction groups still extract (the factory falls back to
    DEFAULT_GROUP_INSTRUCTION), but the omission almost certainly
    indicates a schema authoring slip.
    """
    model = load_pydantic_model(organ)
    field_meta = EXTRACTION_META[organ]["fields"]
    instr = EXTRACTION_META[organ]["groups"]
    used = {
        meta_field_group(field_meta, name)
        for name in model.model_fields
        if meta_field_group(field_meta, name)
    }
    orphans = used - set(instr)
    assert not orphans, (
        f"{organ}: groups used on fields but missing from GROUP_INSTRUCTIONS: "
        f"{sorted(orphans)}"
    )


@pytest.mark.parametrize("organ", ORGANS)
def test_group_order_is_deterministic(organ):
    """``group_order`` returns groups in a stable, deterministic order.

    Order comes from ``GROUP_INSTRUCTIONS`` insertion order in the
    metadata module. The determinism contract is that re-evaluation
    yields the same tuple.
    """
    field_meta = EXTRACTION_META[organ]["fields"]
    groups = EXTRACTION_META[organ]["groups"]
    a = meta_group_order(groups, field_meta)
    b = meta_group_order(groups, field_meta)
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


# --- Layer-3 (aliases) invariants ----------------------------------------


def _unwrap_to_enum(annotation: _t.Any) -> type[Enum] | None:
    """Descend ``list[T]`` / ``X | None`` / ``Union[...]`` and return the
    inner ``Enum`` subclass, or ``None`` if the annotation is not an enum-
    valued field (e.g. ``int | None``)."""
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return annotation
    for arg in _t.get_args(annotation):
        if arg is type(None):
            continue
        inner = _unwrap_to_enum(arg)
        if inner is not None:
            return inner
    return None


def _resolve_path_annotation(model: type[BaseModel], path: str) -> _t.Any:
    """Walk a dotted ``FIELD_META`` path on ``model`` and return the leaf
    field's annotation (raw — caller unwraps to find enums)."""
    parts = path.split(".")
    cursor: _t.Any = model
    for i, part in enumerate(parts):
        if not (isinstance(cursor, type) and issubclass(cursor, BaseModel)):
            return None
        info = cursor.model_fields.get(part)
        if info is None:
            return None
        ann = info.annotation
        if i == len(parts) - 1:
            return ann
        # Descend into nested model: unwrap list[X] / X | None to find a BaseModel.
        nested = ann
        for arg in (nested,) + _t.get_args(nested):
            if arg is type(None):
                continue
            base_candidates: list[_t.Any] = [arg]
            base_candidates.extend(_t.get_args(arg))
            for cand in base_candidates:
                if isinstance(cand, type) and issubclass(cand, BaseModel):
                    nested = cand
                    break
            if isinstance(nested, type) and issubclass(nested, BaseModel):
                break
        if not (isinstance(nested, type) and issubclass(nested, BaseModel)):
            return None
        cursor = nested
    return None


@pytest.mark.parametrize("organ", ORGANS)
def test_alias_canonicals_exist_in_enum(organ):
    """Every ``value_hints`` canonical key (merged from
    ``schemas/aliases/<organ>.toml``) must be a real enum value on the
    corresponding pydantic field — otherwise the hint is silently dead
    and the LM keeps missing the surface form with no signal why."""
    model = load_pydantic_model(organ)
    meta = EXTRACTION_META[organ]["fields"]
    for path, entry in meta.items():
        hints = entry.get("value_hints")
        if not hints:
            continue
        ann = _resolve_path_annotation(model, path)
        enum_cls = _unwrap_to_enum(ann) if ann is not None else None
        assert enum_cls is not None, (
            f"aliases/{organ}.toml: [{path}] is not an enum field "
            f"(annotation: {ann!r})"
        )
        enum_values = {m.value for m in enum_cls}
        unknown = set(hints) - enum_values
        assert not unknown, (
            f"aliases/{organ}.toml: [{path}] references non-enum "
            f"canonicals {sorted(unknown)}; valid values are "
            f"{sorted(enum_values)}"
        )


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
