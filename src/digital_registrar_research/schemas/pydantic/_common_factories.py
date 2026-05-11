"""Primitive factories for the recurring nested-BaseModel patterns.

Surgical margins and lymph nodes follow the same shape across all 10
organs: only the *category vocabulary* (and, for LN, whether a side
field exists) varies per organ. These factories own the boilerplate.

The lean canonical schemas import via :mod:`._case_builder`, never from
this module directly. The decorator there calls :func:`make_margin_type`
/ :func:`make_ln_type` to synthesise the per-organ nested types.

These factories emit *clean* Pydantic models — no
``Field(description=...)`` on any sub-field. Sub-field descriptions
live in the sibling extraction-metadata module under dotted keys
(``"margins.distance"``, ``"regional_lymph_node.lymph_node_side"``,
etc.) and are injected at JSON-schema-generation and DSPy-signature-
build time.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, create_model


def make_margin_type(
    name: str,
    *,
    category_type: Any,
) -> type[BaseModel]:
    """Build a Margin ``BaseModel`` with the standard four fields.

    Returns a class with ``margin_category`` (Optional StrEnum / Literal),
    ``margin_involved`` (REQUIRED bool), ``distance`` (Optional int) and
    ``description`` (Optional str). No descriptions are baked in — those
    come from the extraction-metadata layer.
    """
    fields: dict[str, Any] = {
        "margin_category": (Optional[category_type], Field(None)),
        "margin_involved": (bool, ...),
        "distance":        (Optional[int], Field(None)),
        "description":     (Optional[str], None),
    }
    return create_model(name, __base__=BaseModel, **fields)


def make_ln_type(
    name: str,
    *,
    category_type: Any | None = None,
    side_type: Any | None = None,
) -> type[BaseModel]:
    """Build a Lymph Node ``BaseModel`` with the standard fields.

    Always emits ``involved: int``, ``examined: int`` (both REQUIRED), and
    ``station_name: Optional[str]``. ``lymph_node_side`` and
    ``lymph_node_category`` are conditional on ``side_type`` /
    ``category_type`` being non-``None``. No descriptions baked in.
    """
    fields: dict[str, Any] = {}
    if side_type is not None:
        fields["lymph_node_side"] = (Optional[side_type], Field(None))
    if category_type is not None:
        fields["lymph_node_category"] = (Optional[category_type], Field(None))
    fields["involved"] = (int, ...)
    fields["examined"] = (int, ...)
    fields["station_name"] = (Optional[str], Field(None))
    return create_model(name, __base__=BaseModel, **fields)


__all__ = ["make_margin_type", "make_ln_type"]
