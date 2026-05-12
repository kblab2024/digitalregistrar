"""Thin wrappers around existing schemas package machinery.

Isolating these in their own module keeps ``views/`` importable without
side effects, and gives tests a single seam to stub when they don't
want to touch the live ``schemas/`` directory.
"""
from __future__ import annotations

from ..schemas.create_organ import CreatedOrganReport, create_organ


def create_new_organ(
    organ_key: str,
    *,
    class_prefix: str | None = None,
) -> CreatedOrganReport:
    """Scaffold a new organ in the live schemas/ tree.

    Defers entirely to :func:`schemas.create_organ.create_organ`; the GUI
    only adds the form UI on top.
    """
    return create_organ(
        organ_key,
        class_prefix=class_prefix,
        overwrite=False,
        regen_json=True,
    )


__all__ = ["create_new_organ"]
