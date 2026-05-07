"""Canonical Pydantic case-model for thyroid cancer."""
from ._builder import build_case_model

ThyroidCancerCase = build_case_model("thyroid")

__all__ = ["ThyroidCancerCase"]
