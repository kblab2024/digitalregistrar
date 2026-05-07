"""Canonical Pydantic case-model for lung cancer.

Derived from the per-subsection DSPy signatures `LungCancerNonnested`,
`LungCancerStaging`, `LungCancerMargins`, `LungCancerLN`,
`LungCancerBiomarkers`, `LungCancerOthernested` via ._builder.build_case_model.
"""
from ._builder import build_case_model

LungCancerCase = build_case_model("lung")

__all__ = ["LungCancerCase"]
