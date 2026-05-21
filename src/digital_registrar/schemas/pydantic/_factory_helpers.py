"""Stable cross-layer constants for the schema package.

Phase 3 reduced this module to a single re-usable constant. The old
``GroupedField`` wrapper and group-iteration helpers (``field_desc``,
``field_group``, ``iter_fields_by_group``, ``group_instructions``,
``group_order``, ``iter_custom_types``) all served the legacy
"descriptions live in Pydantic" authoring style, which has been fully
replaced by the two-layer pattern (``schemas/extraction/<organ>.py``
metadata modules). They are gone.

Only :data:`DEFAULT_GROUP_INSTRUCTION` survives because every per-organ
extraction module concatenates it onto its per-group instruction
strings and the pipeline factory falls back to it for groups whose
metadata module forgot to declare an instruction.
"""
from __future__ import annotations

DEFAULT_GROUP_INSTRUCTION = (
    "Extract the listed items from the pathology report. DO NOT JUST RETURN "
    "NULL. If an item is not present, return null for that item, but try your "
    "best to fill in the others."
)


__all__ = ["DEFAULT_GROUP_INSTRUCTION"]
