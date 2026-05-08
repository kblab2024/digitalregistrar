"""Signature factory: build dspy.Signature classes dynamically from Pydantic schemas.

The factory consumes Pydantic case-models from
``digital_registrar_research.schemas.pydantic`` (each field tagged with a
``group`` via :func:`GroupedField`) and emits one or more
``dspy.Signature`` subclasses per call. The factory is the only piece of
the pipeline that talks dspy directly; everything upstream of it is
schema-only.

Public API:

- :class:`ExtractionStep` — a (name, signature, output_field_names, group) record.
- :func:`build_signature` — low-level: takes pre-resolved field dicts, returns a Signature.
- :func:`build_extraction_signatures` — main entry point; decomposes a case-model into Steps.
- :func:`build_router_signature` — builds the ``is_cancer`` equivalent (registry-driven Literal).
- :func:`build_jsonize_signature` — builds the ``ReportJsonize`` equivalent (legacy asymmetry preserved).
"""
from .factory import (
    ExtractionStep,
    build_extraction_signatures,
    build_jsonize_signature,
    build_router_signature,
    build_signature,
)

__all__ = [
    "ExtractionStep",
    "build_extraction_signatures",
    "build_jsonize_signature",
    "build_router_signature",
    "build_signature",
]
