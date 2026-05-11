"""Surface-form alias loader (Layer 3 — data-only).

Each organ's TOML file lives next to this module and is read at import
time by ``schemas.extraction``. The values are merged into
``EXTRACTION_META[organ]['fields'][path]['value_hints']`` and surface
in the LM-facing description that DSPy serializes into the prompt.

Format
------
Top-level tables are dotted field paths (matching ``FIELD_META`` keys).
Each entry maps a canonical enum value to a list of surface-form
strings the LM might see in pathology reports::

    ["biomarkers.biomarker_category"]
    alk  = ["alk", "ALK", "ALK D5F3", "anaplastic lymphoma kinase"]
    ros1 = ["ros1", "ROS1", "ROS-1"]

Editing this TOML does NOT require regenerating any JSON or restarting
a build step — the loader runs at Python module import time, so the
next pipeline process will pick up the new aliases.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

_ALIAS_DIR = Path(__file__).parent


def load_aliases(organ: str) -> dict[str, dict[str, list[str]]]:
    """Return ``{field_path: {canonical: [surface_forms]}}`` for one organ.

    Missing file → empty dict (aliases are opt-in per organ).
    """
    path = _ALIAS_DIR / f"{organ}.toml"
    if not path.is_file():
        return {}
    with path.open("rb") as f:
        return tomllib.load(f)


__all__ = ["load_aliases"]
