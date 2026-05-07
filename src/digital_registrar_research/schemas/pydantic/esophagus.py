"""Canonical Pydantic case-model for esophagus cancer."""
from ._builder import build_case_model

EsophagusCancerCase = build_case_model("esophagus")

__all__ = ["EsophagusCancerCase"]
