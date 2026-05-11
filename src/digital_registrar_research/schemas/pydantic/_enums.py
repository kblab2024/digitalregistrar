"""Cross-organ shared StrEnums used by the clean (post-migration) case-models.

The legacy ``LymphNodeSide`` / ``LymphNodeSideRL`` ``Literal`` aliases live
in :mod:`._common_factories` and remain in place for organs still on the
``GroupedField`` design. As each organ migrates to the two-layer pattern
(see :mod:`schemas.extraction`), the field annotation flips from
``Literal[...]`` to the corresponding ``StrEnum`` here.

Every member set ends with ``NOT_STATED`` — the explicit "report didn't
say" sentinel. Field annotations remain ``MyEnum | None`` (default
``None``) to preserve JSON-Schema null-shape compatibility with the
structured-decoding path; ``NOT_STATED`` is available if the LM wants
to assert it explicitly rather than emitting null.
"""
from __future__ import annotations

from enum import StrEnum, auto


class LowerStrEnum(StrEnum):
    """StrEnum where ``auto()`` yields ``name.lower()``.

    Use this for closed vocabularies whose canonical wire-form is
    snake_case lowercase — i.e. every clinical vocabulary in this
    package after the unify-canonical refactor. Explicit string
    values are still honored (Python's enum machinery only invokes
    ``_generate_next_value_`` for ``auto()`` sentinels), but the
    intent is that every member should be ``auto()`` so member-name
    drift can never disagree with the wire value.
    """

    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return name.lower()


class LymphNodeSide(LowerStrEnum):
    """Standard ipsilateral/contralateral/midline side designation."""

    RIGHT = auto()
    LEFT = auto()
    MIDLINE = auto()
    NOT_STATED = auto()


class LymphNodeSideRL(LowerStrEnum):
    """Right/left only — used by lung (no midline lymph nodes by anatomy)."""

    RIGHT = auto()
    LEFT = auto()
    NOT_STATED = auto()


class PMCategory(LowerStrEnum):
    """Pathologic M category. ``mx`` covers the legacy "cM0/cM1 → code as mx" rule."""

    MX = auto()
    M0 = auto()
    M1 = auto()
    NOT_STATED = auto()


class TNMDescriptor(LowerStrEnum):
    """Standard TNM descriptor prefixes used across organs."""

    Y = auto()
    R = auto()
    M = auto()
    NOT_STATED = auto()


__all__ = [
    "LowerStrEnum",
    "LymphNodeSide",
    "LymphNodeSideRL",
    "PMCategory",
    "TNMDescriptor",
]
