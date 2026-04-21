# -*- coding: utf-8 -*-
"""
schemas/canonical/common.py
Canonical Pydantic schema definitions for common/shared cancer registry fields.
Authoritative data dictionary for is_cancer and ReportJsonize signatures.
"""
__version__ = "1.0.0"
__date__ = "2026-04-22"
__author__ = ["Hong-Kai (Walther) Chen", "Po-Yen Tzeng", "Kai-Po Chang"]
__copyright__ = "Copyright 2025, Med NLP Lab, China Medical University"
__license__ = "MIT"

from pydantic import BaseModel, Field
from typing import Literal

# --- TypeAliases ---

# Used in is_cancer OutputField (includes "others")
CancerCategoryOutputType = Literal[
    'stomach', 'colorectal', 'breast', 'esophagus', 'lung',
    'prostate', 'thyroid', 'pancreas', 'cervix', 'liver', 'others'
]

# Used in ReportJsonize InputField (excludes "others")
CancerCategoryInputType = Literal[
    'stomach', 'colorectal', 'breast', 'esophagus', 'lung',
    'prostate', 'pancreas', 'thyroid', 'cervix', 'liver'
]

# --- Full output schemas ---

class IsCancerSchema(BaseModel):
    cancer_excision_report: bool = Field(
        ...,
        description=(
            "identify whether or not this report belongs to PRIMARY cancer excision eligible "
            "for registry for cancer excision. If no viable tumor is present after excision, "
            "you should not register this case. If only carcinoma in situ or high-grade "
            "dysplasia, you should not register this case."
        )
    )
    cancer_category: CancerCategoryOutputType | None = Field(
        None,
        description=(
            "identify which organ the primary cancer arises from. Currently only ten are "
            "implemented, if it IS a cancer excision report, but primary site not included "
            "in these standard organs, should be classified as others."
        )
    )
    cancer_category_others_description: str | None = Field(
        None,
        description=(
            "if is cancer_excision report AND cancer_category is others, please specify the "
            "organ here. if not, return null."
        )
    )
