"""Canonical case-model for liver cancer (v2, two-layer schema).

Liver outliers: lymph-node nested type has neither category nor side
(``LNSpec(category=None, side=None)``); staging stage-group field is
named ``overall_stage``; has organ-specific ``extent`` and
``vascular_invasion`` groups carried as plain ``list[StrEnum]`` fields.
"""
from enum import auto
from typing import ClassVar, Literal

from pydantic import BaseModel

from ._case_builder import (
    LNSpec,
    MarginSpec,
    StagingSpec,
    assemble_case_model,
)
from ._enums import LowerStrEnum, PMCategory, TNMDescriptor


class LiverProcedure(LowerStrEnum):
    WEDGE_RESECTION = auto()
    PARTIAL_HEPATECTOMY = auto()
    SEGMENTECTOMY = auto()
    LOBECTOMY = auto()
    TOTAL_HEPATECTOMY = auto()
    OTHERS = auto()
    NOT_STATED = auto()


class LiverTumorSite(LowerStrEnum):
    RIGHT_LOBE = auto()
    LEFT_LOBE = auto()
    CAUDATE_LOBE = auto()
    QUADRATE_LOBE = auto()
    OTHERS = auto()
    NOT_STATED = auto()


class LiverHistology(LowerStrEnum):
    HEPATOCELLULAR_CARCINOMA = auto()
    HEPATOCELLULAR_CARCINOMA_FIBROLAMELLAR = auto()
    HEPATOCELLULAR_CARCINOMA_SCIRRHOUS = auto()
    HEPATOCELLULAR_CARCINOMA_CLEAR_CELL = auto()
    OTHERS = auto()
    NOT_STATED = auto()


class LiverTumorFocality(LowerStrEnum):
    UNIFOCAL = auto()
    MULTIFOCAL = auto()
    NOT_STATED = auto()


class LiverTumorExtent(LowerStrEnum):
    HEPATIC_VEIN = auto()
    PORTAL_VEIN = auto()
    VISCERAL_PERITONEUM = auto()
    GALLBLADDER = auto()
    DIAPHRAGM = auto()
    OTHERS = auto()


class LiverVascularInvasion(LowerStrEnum):
    LARGE_HEPATIC_VEIN = auto()
    LARGE_PORTAL_VEIN = auto()
    SMALL_VESSEL = auto()


class LiverMarginSite(LowerStrEnum):
    PARENCHYMAL = auto()
    HEPATIC_VEIN = auto()
    PORTAL_VEIN = auto()
    BILE_DUCT = auto()
    OTHERS = auto()
    NOT_STATED = auto()


class LiverPT(LowerStrEnum):
    TX = auto()
    T1A = auto()
    T1B = auto()
    T2 = auto()
    T3 = auto()
    T4 = auto()
    NOT_STATED = auto()


class LiverPN(LowerStrEnum):
    NX = auto()
    N0 = auto()
    N1 = auto()
    NOT_STATED = auto()


class LiverOverallStage(LowerStrEnum):
    STAGEIA = auto()
    STAGEIB = auto()
    STAGEII = auto()
    STAGEIIIA = auto()
    STAGEIIIB = auto()
    STAGEIVA = auto()
    STAGEIVB = auto()
    NOT_STATED = auto()


@assemble_case_model()
class LiverCancerCase(BaseModel):
    """Canonical extracted case record for liver cancer."""

    procedure: LiverProcedure | None = None
    tumor_site: LiverTumorSite | None = None
    histology: LiverHistology | None = None
    grade: Literal[1, 2, 3, 4] | None = None
    tumor_size: int | None = None
    tumor_focality: LiverTumorFocality | None = None
    perineural_invasion: bool | None = None
    distant_metastasis: bool | None = None
    treatment_effect: str | None = None

    # extent — list of organ-specific extension sites (no spec marker; lean Field)
    tumor_extent: list[LiverTumorExtent] | None = None

    # vascular_invasion — list of vessels invaded
    vascular_invasion: list[LiverVascularInvasion] | None = None

    _STAGING: ClassVar = StagingSpec(
        pt=LiverPT,
        pn=LiverPN,
        pm=PMCategory,
        tnm_descriptor=TNMDescriptor,
        stage_groups={"overall_stage": LiverOverallStage},
        lean=True,
    )

    margins: LiverMarginSite | None = MarginSpec()

    # lymph_nodes (auto-expanded to list[LiverLN]; no category, no side)
    regional_lymph_node: None = LNSpec(category=None, side=None)
    extranodal_extension: bool | None = None
    maximal_ln_size: int | None = None
