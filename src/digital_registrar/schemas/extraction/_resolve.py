"""Metadata-dict-driven counterparts to :mod:`schemas.pydantic._factory_helpers`.

The legacy helpers read group/desc from each field's ``json_schema_extra``
(populated by ``GroupedField``). These read from a passed-in ``FIELD_META``
dict instead, leaving the Pydantic models clean.

Top-level fields are keyed by name (``"procedure"``); nested-type
sub-fields are keyed by dotted path (``"biomarkers.percentage"``). The
top-level iterators skip dotted keys.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from . import FieldMeta


def field_desc(field_meta: dict[str, FieldMeta], path: str) -> str:
    """Return the LM-facing description for a field by dotted-or-bare path.

    If the entry has ``value_hints`` (merged in from
    ``schemas/aliases/<organ>.toml`` at registry-build time), the
    rendered hint block is appended to the description so DSPy
    serialises both into the prompt JSON Schema together.
    """
    entry = field_meta.get(path)
    if entry is None:
        return ""
    desc = entry.get("desc", "")
    hints = entry.get("value_hints")
    if hints:
        return f"{desc.rstrip()}  {_render_value_hints(hints)}"
    return desc


def _render_value_hints(hints: dict[str, list[str]]) -> str:
    """Format ``{canonical: [surface_forms]}`` as a single sentence for the prompt."""
    parts = [
        f"`{canon}` (also appears as: {', '.join(forms)})"
        for canon, forms in hints.items()
    ]
    return (
        "When the report uses any of these surface forms, output the "
        "canonical value: " + "; ".join(parts) + "."
    )


def field_group(field_meta: dict[str, FieldMeta], name: str) -> str | None:
    """Return the group tag for a top-level field. ``None`` for unknown / nested keys."""
    if "." in name:
        return None
    entry = field_meta.get(name)
    if entry is None:
        return None
    return entry.get("group")


def fields_by_group(
    model_cls: type[BaseModel],
    field_meta: dict[str, FieldMeta],
) -> dict[str, list[tuple[str, Any, FieldInfo]]]:
    """Group ``model_cls.model_fields`` by ``field_meta[name]["group"]``.

    Preserves declaration order within each group. Fields without a meta
    entry — or whose entry has no ``group`` key — land under
    ``"<ungrouped>"`` so the factory can surface orphan fields rather
    than silently drop them.
    """
    out: dict[str, list[tuple[str, Any, FieldInfo]]] = {}
    for name, info in model_cls.model_fields.items():
        g = field_group(field_meta, name) or "<ungrouped>"
        out.setdefault(g, []).append((name, info.annotation, info))
    return out


def group_order(group_instructions: dict[str, str], field_meta: dict[str, FieldMeta]) -> list[str]:
    """Canonical extraction order: insertion order of ``group_instructions``,
    then any tags appearing only in ``field_meta`` appended in first-seen order.
    """
    seen: list[str] = list(group_instructions)
    seen_set = set(seen)
    for entry in field_meta.values():
        g = entry.get("group")
        if g and g not in seen_set:
            seen.append(g)
            seen_set.add(g)
    return seen


__all__ = ["field_desc", "field_group", "fields_by_group", "group_order"]
