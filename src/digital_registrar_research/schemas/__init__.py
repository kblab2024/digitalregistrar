"""Consolidated schema layer.

Public API:
    load_pydantic_model(organ) -> type[pydantic.BaseModel]
        The canonical case-model for one organ.
    load_json_schema(organ) -> dict
        Pre-generated JSON schema read from the packaged `data/` folder.
    list_organs() -> list[str]
        Public organ keys (as used in `cancer_category` Literal and JSON filenames).
    build_case_model(organ) -> type[pydantic.BaseModel]
        Regenerate a case-model from the DSPy signatures (uncached alias useful in tests).

The source of truth is the Pydantic layer (`pydantic/`), derived from the
DSPy signatures in `...models/`. JSON schemas in `data/` are generated
artifacts; regenerate with `python -m digital_registrar_research.schemas.generate`.
"""
from __future__ import annotations

import json
from importlib.resources import files
from typing import TYPE_CHECKING

from .pydantic import CASE_MODELS
from .pydantic._builder import build_case_model

if TYPE_CHECKING:
    from pydantic import BaseModel


def list_organs() -> list[str]:
    """Public organ keys available in the canonical schema registry."""
    return sorted(CASE_MODELS.keys())


def load_pydantic_model(organ: str) -> type[BaseModel]:
    """Return the canonical Pydantic case-model for `organ`."""
    try:
        return CASE_MODELS[organ]
    except KeyError as e:
        raise KeyError(
            f"Unknown organ {organ!r}. Known organs: {list_organs()}"
        ) from e


def load_json_schema(organ: str) -> dict:
    """Return the pre-generated JSON schema for `organ` from packaged `data/`."""
    path = files("digital_registrar_research.schemas.data").joinpath(f"{organ}.json")
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = ["list_organs", "load_pydantic_model", "load_json_schema", "build_case_model", "CASE_MODELS"]
