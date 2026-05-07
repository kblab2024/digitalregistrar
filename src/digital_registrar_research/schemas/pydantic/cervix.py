"""Canonical Pydantic case-model for cervix cancer."""
from ._builder import build_case_model

CervixCancerCase = build_case_model("cervix")

__all__ = ["CervixCancerCase"]
