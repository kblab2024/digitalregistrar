"""Canonical Pydantic case-model for colorectal cancer.

Derived from `ColonCancer*` DSPy signatures — note the naming drift: the
public-facing organ key is `"colorectal"` while the underlying signature
classes are named `Colon*`. See docs/schemas.md.
"""
from ._builder import build_case_model

ColorectalCancerCase = build_case_model("colorectal")

__all__ = ["ColorectalCancerCase"]
