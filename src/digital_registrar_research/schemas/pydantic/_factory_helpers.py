"""Schema-side helpers for the signature factory.

This module is the only schema-side dependency of
``digital_registrar_research.signatures.factory``. It is dspy-free so that
schema files, the future GUI tool, and the bootstrap script can all import
these helpers without pulling in dspy's import graph.

Conventions
-----------
Every field in a case-model schema is declared with :func:`GroupedField`,
which carries a ``group`` tag in ``Field.json_schema_extra``. The factory
groups fields by tag and builds one ``dspy.Signature`` per group (the
``"per_group"`` decomposition mode), or merges them all (``"monolithic"``).

The class-level ``_GROUP_INSTRUCTIONS: ClassVar[dict[str, str]]`` carries
the per-group LM instructions. Insertion order = extraction order =
first-wins order on duplicate field names.
"""
from __future__ import annotations

import typing as _t
from typing import Any

from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo

DEFAULT_GROUP_INSTRUCTION = (
    "Extract the listed items from the pathology report. DO NOT JUST RETURN "
    "NULL. If an item is not present, return null for that item, but try your "
    "best to fill in the others."
)


def GroupedField(
    *,
    group: str,
    desc: str,
    default: Any = None,
    **kwargs: Any,
) -> Any:
    """Pydantic ``Field`` wrapper that tags the field with a clinical group.

    The ``group`` name is consumed by the signature factory to decompose a
    flat case-model into one ``dspy.Signature`` per group. ``desc``
    populates BOTH Pydantic's ``description`` (so it appears in the
    auto-generated JSON schema) and ``json_schema_extra["dspy_desc"]`` (so
    the factory can read it without parsing the JSON-schema string back).
    """
    extra: dict[str, Any] = dict(kwargs.pop("json_schema_extra", None) or {})
    extra["group"] = group
    extra["dspy_desc"] = desc
    description = kwargs.pop("description", None) or desc
    return Field(default=default, description=description, json_schema_extra=extra, **kwargs)


def field_group(field_info: FieldInfo) -> str | None:
    """Return the group tag attached by :func:`GroupedField`, or ``None``."""
    extra = field_info.json_schema_extra or {}
    if isinstance(extra, dict):
        val = extra.get("group")
        if isinstance(val, str):
            return val
    return None


def field_desc(field_info: FieldInfo) -> str:
    """Return the LM-facing description for a field.

    Reads ``json_schema_extra["dspy_desc"]`` first (set by
    :func:`GroupedField`), falls back to Pydantic's ``description``, then to
    the empty string. The factory uses this exact lookup, so updating
    ``description`` alone is not enough — go through ``GroupedField`` to
    keep both in sync.
    """
    extra = field_info.json_schema_extra or {}
    if isinstance(extra, dict):
        val = extra.get("dspy_desc")
        if isinstance(val, str) and val:
            return val
    return field_info.description or ""


def iter_fields_by_group(
    model_cls: type[BaseModel],
) -> dict[str, list[tuple[str, Any, FieldInfo]]]:
    """Group ``model_cls.model_fields`` by the field's ``group`` tag.

    Preserves declaration order within each group (Pydantic v2 keeps
    ``model_fields`` in class-body order). Fields without a group tag are
    collected under the synthetic key ``"<ungrouped>"`` so callers can
    detect orphan fields.
    """
    out: dict[str, list[tuple[str, Any, FieldInfo]]] = {}
    for name, info in model_cls.model_fields.items():
        g = field_group(info) or "<ungrouped>"
        out.setdefault(g, []).append((name, info.annotation, info))
    return out


def group_instructions(model_cls: type[BaseModel]) -> dict[str, str]:
    """Read the schema's class-level ``_GROUP_INSTRUCTIONS`` mapping.

    Returns an empty dict if the schema doesn't declare one (the factory
    will fall back to :data:`DEFAULT_GROUP_INSTRUCTION` per group).
    """
    raw = getattr(model_cls, "_GROUP_INSTRUCTIONS", None)
    if isinstance(raw, dict):
        # Filter to str -> str so a typo in the schema can't crash later.
        return {k: v for k, v in raw.items() if isinstance(k, str) and isinstance(v, str)}
    return {}


def group_order(model_cls: type[BaseModel]) -> list[str]:
    """Return the canonical extraction order for the schema's groups.

    Insertion order of ``_GROUP_INSTRUCTIONS`` is authoritative. Any group
    tag that appears on a field but is missing from ``_GROUP_INSTRUCTIONS``
    is appended at the end in first-seen order, so an authoring mistake
    surfaces but doesn't drop fields.
    """
    instr = group_instructions(model_cls)
    seen: list[str] = list(instr.keys())
    seen_set = set(seen)
    for info in model_cls.model_fields.values():
        g = field_group(info)
        if g and g not in seen_set:
            seen.append(g)
            seen_set.add(g)
    return seen


def iter_custom_types(model_cls: type[BaseModel]) -> dict[str, type]:
    """Collect every nested ``BaseModel`` referenced by the schema's annotations.

    Returns ``{ClassName: cls}``. Used by ``factory.build_signature`` to
    pass ``custom_types=`` to ``make_signature`` so DSPy can resolve
    references like ``list[BreastMargin]`` during signature construction.
    Walks recursively through ``list[...]``, ``Optional[...]``, ``Union[...]``,
    etc.
    """
    out: dict[str, type] = {}

    def visit(tp: Any) -> None:
        origin = _t.get_origin(tp)
        if origin is not None:
            for arg in _t.get_args(tp):
                visit(arg)
            return
        if isinstance(tp, type) and issubclass(tp, BaseModel) and tp is not BaseModel:
            out.setdefault(tp.__name__, tp)
            # Recurse into nested model fields too — covers schemas whose
            # nested types themselves reference further BaseModels.
            for sub in tp.model_fields.values():
                visit(sub.annotation)

    for info in model_cls.model_fields.values():
        visit(info.annotation)
    return out


__all__ = [
    "DEFAULT_GROUP_INSTRUCTION",
    "GroupedField",
    "field_desc",
    "field_group",
    "group_instructions",
    "group_order",
    "iter_custom_types",
    "iter_fields_by_group",
]
