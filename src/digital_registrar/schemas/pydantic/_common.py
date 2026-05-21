"""Top-of-pipeline case-models: `IsCancerCase` and `ReportJsonizeOutput`.

These mirror the shared signatures defined in `models/common.py`:
- `is_cancer` — the router that decides excision eligibility and cancer category
- `ReportJsonize` — the first-pass structuring step that produces a rough JSON

The ``cancer_category`` Literal is built dynamically from the
auto-discovered Layer-1 registry: the set of permitted values is
``sorted(CASE_MODELS.keys()) + ["others"]``. Adding a new organ file
to ``schemas/pydantic/`` automatically extends the routing vocabulary
— no manual edit of this file is needed.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ._registry import discover_case_models

_ORGAN_KEYS: tuple[str, ...] = tuple(sorted(discover_case_models().keys())) + ("others",)


class IsCancerCase(BaseModel):
    """Top-level routing decision — mirrors `is_cancer` output fields."""

    cancer_excision_report: bool = Field(
        ...,
        description=(
            "Whether this report documents a PRIMARY cancer excision eligible for "
            "registry. False for carcinoma in situ / high-grade dysplasia only, or if "
            "no viable tumor remains after excision."
        ),
    )
    cancer_category: Literal[_ORGAN_KEYS] | None = Field(
        None,
        description=(
            "Which organ the primary cancer arises from. The implemented organs "
            "are auto-discovered from `schemas/pydantic/`; anything outside the "
            "list is 'others'."
        ),
    )
    cancer_category_others_description: str | None = Field(
        None,
        description="Free-text organ name when cancer_category == 'others'.",
    )


class ReportJsonizeOutput(BaseModel):
    """Roughly-structured JSON dump produced by `ReportJsonize`."""

    output: dict = Field(
        ...,
        description=(
            "A rough JSON conversion of the raw pathology report, preserving "
            "original wording and following the organ-specific cancer checklist."
        ),
    )
