"""Canonical Pydantic case-models — one per organ.

Each `<Organ>CancerCase` is built from the corresponding DSPy signatures
listed in `models.modellist.organmodels` (see `_builder.build_case_model`).
The registry `CASE_MODELS` maps the public organ key (as used in the
`cancer_category` Literal and JSON schema filenames) to its Pydantic class.
"""
from pydantic import BaseModel

from ._common import IsCancerCase, ReportJsonizeOutput
from .breast import BreastCancerCase
from .cervix import CervixCancerCase
from .colorectal import ColorectalCancerCase
from .esophagus import EsophagusCancerCase
from .liver import LiverCancerCase
from .lung import LungCancerCase
from .pancreas import PancreasCancerCase
from .prostate import ProstateCancerCase
from .stomach import StomachCancerCase
from .thyroid import ThyroidCancerCase

CASE_MODELS: dict[str, type[BaseModel]] = {
    "breast":     BreastCancerCase,
    "cervix":     CervixCancerCase,
    "colorectal": ColorectalCancerCase,
    "esophagus":  EsophagusCancerCase,
    "liver":      LiverCancerCase,
    "lung":       LungCancerCase,
    "pancreas":   PancreasCancerCase,
    "prostate":   ProstateCancerCase,
    "stomach":    StomachCancerCase,
    "thyroid":    ThyroidCancerCase,
}

__all__ = [
    "CASE_MODELS",
    "IsCancerCase", "ReportJsonizeOutput",
    "BreastCancerCase", "CervixCancerCase", "ColorectalCancerCase",
    "EsophagusCancerCase", "LiverCancerCase", "LungCancerCase",
    "PancreasCancerCase", "ProstateCancerCase", "StomachCancerCase",
    "ThyroidCancerCase",
]
