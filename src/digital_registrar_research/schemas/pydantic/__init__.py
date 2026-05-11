"""Canonical Pydantic case-models — one per organ (Layer 1 of the
three-layer schema architecture).

Each ``<Organ>CancerCase`` is the SINGLE SOURCE OF TRUTH for the
organ's *shape* (types, ``LowerStrEnum`` vocabularies, defaults).
LM-facing descriptions live in
:mod:`digital_registrar_research.schemas.extraction` (Layer 2);
surface-form aliases live in
:mod:`digital_registrar_research.schemas.aliases` (Layer 3, TOML).
JSON schemas in ``schemas/data/<organ>.json`` are generated artifacts;
regenerate with ``python -m digital_registrar_research.schemas.generate``.

Registration is **automatic**: dropping a new ``<organ>.py`` into this
directory makes it appear in :data:`CASE_MODELS` on the next process
import — no manual edit to this file. See
:mod:`._registry` for the discovery rules (naming convention,
expected class name). The scaffolder
:mod:`digital_registrar_research.schemas.create_organ` writes a
default-template organ across all three layers in one shot.
"""
from ._common import IsCancerCase, ReportJsonizeOutput
from ._registry import discover_case_models

CASE_MODELS = discover_case_models()

# Re-export each discovered case-model class at the package level so existing
# ``from digital_registrar_research.schemas.pydantic import LungCancerCase``
# call sites keep working without modification.
globals().update({cls.__name__: cls for cls in CASE_MODELS.values()})

__all__ = [
    "CASE_MODELS",
    "IsCancerCase",
    "ReportJsonizeOutput",
    *sorted(cls.__name__ for cls in CASE_MODELS.values()),
]
