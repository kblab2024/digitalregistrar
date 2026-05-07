"""Canonical Pydantic case-model for liver cancer."""
from ._builder import build_case_model

LiverCancerCase = build_case_model("liver")

__all__ = ["LiverCancerCase"]
