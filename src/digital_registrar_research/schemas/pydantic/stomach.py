"""Canonical Pydantic case-model for stomach cancer."""
from ._builder import build_case_model

StomachCancerCase = build_case_model("stomach")

__all__ = ["StomachCancerCase"]
