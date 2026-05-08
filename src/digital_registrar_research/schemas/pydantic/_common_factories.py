"""Primitive factories for the recurring nested-BaseModel patterns.

Surgical margins and lymph nodes follow the same shape across all 10
organs: only the *category vocabulary* (and, for LN, whether a side
field exists) varies per organ. These factories own the boilerplate.

The lean canonical schemas import via :mod:`._case_builder`, never from
this module directly. The decorator there calls :func:`make_margin_type`
/ :func:`make_ln_type` to synthesise the per-organ nested types.

The factories accept either a ``Literal[...]`` type (simple enum-of-strings)
or a ``str, Enum`` subclass — both serialise to the same JSON Schema. Use
whichever is more convenient at the call site.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, create_model

# --- Reusable side enum ----------------------------------------------------
# Most organs accept the same three values for "lymph_node_side". Lung is the
# only outlier (right/left only — no midline).

LymphNodeSide = Literal["right", "left", "midline"]
LymphNodeSideRL = Literal["right", "left"]


# --- Boilerplate descriptions ---------------------------------------------

_MARGIN_DISTANCE_DESC = (
    "If margin is involved, return 0. If margin is uninvolved/free, "
    "try your best to find the distance at both microscopic and macroscopic(gross) "
    "description, and specify the distance from tumor to margin in mm, rounded to "
    "integer. If the margin is uninvolved/free and, after your best effort, the "
    "distance is still not specified, return null"
)

_LN_STATION_NAME_DESC = "specify the name of the lymph node station here."


def _default_margin_category_desc(organ: str) -> str:
    return (
        f"acceptable value for surgical margins in {organ} cancer. "
        f"If not included in these standard margins, should be classified as others."
    )


def _default_ln_side_desc(organ: str) -> str:
    return (
        f"acceptable value for lymph node side in {organ} cancer. "
        f"If not included in these standard sides, should be classified as None."
    )


def _default_ln_category_desc(organ: str, *, group_word: str = "station") -> str:
    """Default LN-category description.

    ``group_word`` chooses between "station" and "group" wording for the
    handful of organs whose legacy text varied.
    """
    return (
        f"acceptable value for lymph node categories in {organ} cancer. "
        f"If not included in these standard lymph node '{group_word}' "
        f"number, should be classified as others."
    )


# --- Factories ------------------------------------------------------------

def make_margin_type(
    name: str,
    *,
    organ: str,
    category_type: Any,
    category_desc: str | None = None,
    distance_desc: str = _MARGIN_DISTANCE_DESC,
) -> type[BaseModel]:
    """Build a Margin ``BaseModel`` with the standard four fields.

    Returns a class with ``margin_category`` (Optional Literal),
    ``margin_involved`` (REQUIRED bool), ``distance`` (Optional int) and
    ``description`` (Optional str). Field descriptions are auto-templated
    from ``organ`` unless overridden.
    """
    cat_desc = category_desc or _default_margin_category_desc(organ)
    fields: dict[str, Any] = {
        "margin_category": (
            Optional[category_type],
            Field(None, description=cat_desc),
        ),
        "margin_involved": (bool, ...),
        "distance": (
            Optional[int],
            Field(None, description=distance_desc),
        ),
        "description": (Optional[str], None),
    }
    return create_model(name, __base__=BaseModel, **fields)


def make_ln_type(
    name: str,
    *,
    organ: str,
    category_type: Any | None = None,
    category_desc: str | None = None,
    category_group_word: str = "station",
    side_type: Any | None = None,
    side_desc: str | None = None,
    station_name_desc: str = _LN_STATION_NAME_DESC,
) -> type[BaseModel]:
    """Build a Lymph Node ``BaseModel`` with the standard fields.

    Always emits ``involved: int``, ``examined: int`` (both REQUIRED), and
    ``station_name: Optional[str]``. ``lymph_node_side`` and
    ``lymph_node_category`` are conditional on ``side_type`` / ``category_type``
    being non-``None``.
    """
    fields: dict[str, Any] = {}
    if side_type is not None:
        s_desc = side_desc or _default_ln_side_desc(organ)
        fields["lymph_node_side"] = (
            Optional[side_type],
            Field(None, description=s_desc),
        )
    if category_type is not None:
        c_desc = category_desc or _default_ln_category_desc(
            organ, group_word=category_group_word
        )
        fields["lymph_node_category"] = (
            Optional[category_type],
            Field(None, description=c_desc),
        )
    fields["involved"] = (int, ...)
    fields["examined"] = (int, ...)
    fields["station_name"] = (
        Optional[str],
        Field(None, description=station_name_desc),
    )
    return create_model(name, __base__=BaseModel, **fields)


__all__ = [
    "LymphNodeSide",
    "LymphNodeSideRL",
    "make_margin_type",
    "make_ln_type",
]
