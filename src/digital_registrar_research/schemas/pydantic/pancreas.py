"""Canonical Pydantic case-model for pancreas cancer."""
from ._builder import build_case_model

PancreasCancerCase = build_case_model("pancreas")

__all__ = ["PancreasCancerCase"]
