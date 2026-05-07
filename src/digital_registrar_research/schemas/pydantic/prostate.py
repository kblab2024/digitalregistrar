"""Canonical Pydantic case-model for prostate cancer."""
from ._builder import build_case_model

ProstateCancerCase = build_case_model("prostate")

__all__ = ["ProstateCancerCase"]
