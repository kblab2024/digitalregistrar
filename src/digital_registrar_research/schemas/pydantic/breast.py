"""Canonical Pydantic case-model for breast cancer.

Derived from `BreastCancerNonnested`, `DCIS`, `BreastCancerGrading`,
`BreastCancerStaging`, `BreastCancerMargins`, `BreastCancerLN`,
`BreastCancerBiomarkers` via ._builder.build_case_model.
"""
from ._builder import build_case_model

BreastCancerCase = build_case_model("breast")

__all__ = ["BreastCancerCase"]
