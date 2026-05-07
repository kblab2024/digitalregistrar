"""Top-of-pipeline case-models: `IsCancerCase` and `ReportJsonizeOutput`.

These mirror the shared signatures defined in `models/common.py`:
- `is_cancer` — the router that decides excision eligibility and cancer category
- `ReportJsonize` — the first-pass structuring step that produces a rough JSON
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


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
    cancer_category: Literal["stomach", "colorectal", "breast", "esophagus", "lung", "prostate", "thyroid", "pancreas", "cervix", "liver", "others"] | None = Field(
        None,
        description=(
            "Which organ the primary cancer arises from. Ten standard organs are "
            "implemented; anything outside the list is 'others'."
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
