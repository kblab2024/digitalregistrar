"""Canonical case-model for pancreas cancer (v2, two-layer schema).

Pancreas-specific shape: ``pm`` has only ``mx``/``m1``; the staging
stage-group field is named ``overall_stage``; LN has no side.
"""
from enum import auto
from typing import ClassVar

from pydantic import BaseModel

from ._case_builder import (
    LNSpec,
    MarginSpec,
    StagingSpec,
    assemble_case_model,
)
from ._enums import LowerStrEnum, TNMDescriptor


class PancreasProcedure(LowerStrEnum):
    PARTIAL_PANCREATECTOMY = auto()
    SSPPD = auto()
    PPPD = auto()
    WHIPPLE_PROCEDURE = auto()
    DISTAL_PANCREATECTOMY = auto()
    TOTAL_PANCREATECTOMY = auto()
    OTHERS = auto()
    NOT_STATED = auto()


class PancreasTumorSite(LowerStrEnum):
    HEAD = auto()
    NECK = auto()
    BODY = auto()
    TAIL = auto()
    UNCINATE_PROCESS = auto()
    OTHERS = auto()
    NOT_STATED = auto()


class PancreasHistology(LowerStrEnum):
    DUCTAL_ADENOCARCINOMA_NOS = auto()
    IPMN_WITH_CARCINOMA = auto()
    ITPN_WITH_CARCINOMA = auto()
    ACINAR_CELL_CARCINOMA = auto()
    SOLID_PSEUDOPAPILLARY_NEOPLASM = auto()
    UNDIFFERENTIATED_CARCINOMA = auto()
    OTHERS = auto()
    NOT_STATED = auto()


class PancreasTumorExtension(LowerStrEnum):
    WITHIN_PANCREAS = auto()
    PERIPANCREATIC_SOFT_TISSUE = auto()
    ADJACENT_ORGANS_STRUCTURES = auto()
    OTHERS = auto()
    NOT_STATED = auto()


class PancreasMarginSite(LowerStrEnum):
    DISTAL_PANCREATIC = auto()
    PROXIMAL_PANCREATIC = auto()
    PANCREATIC_NECK = auto()
    UNCINATE = auto()
    BILE_DUCT = auto()
    PROXIMAL_GASTRIC = auto()
    PROXIMAL_DUODENAL = auto()
    DISTAL_INTESTINAL = auto()
    OUTMOST = auto()
    ANTERIOR_OUTMOST = auto()
    POSTERIOR_OUTMOST = auto()
    OTHERS = auto()
    NOT_STATED = auto()


class PancreasLNCategory(LowerStrEnum):
    REGIONAL_PANCREATIC = auto()
    REGIONAL_GASTRIC = auto()
    LN1 = auto()
    LN2 = auto()
    LN3 = auto()
    LN4 = auto()
    LN5 = auto()
    LN6 = auto()
    LN7 = auto()
    LN8 = auto()
    LN9 = auto()
    LN10 = auto()
    LN11 = auto()
    LN12 = auto()
    LN13 = auto()
    LN14 = auto()
    OTHERS = auto()
    NOT_STATED = auto()


class PancreasPT(LowerStrEnum):
    TX = auto()
    TIS = auto()
    T1A = auto()
    T1B = auto()
    T1C = auto()
    T2 = auto()
    T3 = auto()
    T4 = auto()
    NOT_STATED = auto()


class PancreasPN(LowerStrEnum):
    NX = auto()
    N0 = auto()
    N1 = auto()
    N2 = auto()
    NOT_STATED = auto()


class PancreasPM(LowerStrEnum):
    MX = auto()
    M1 = auto()
    NOT_STATED = auto()


class PancreasOverallStage(LowerStrEnum):
    STAGEIA = auto()
    STAGEIB = auto()
    STAGEIIA = auto()
    STAGEIIB = auto()
    STAGEIII = auto()
    STAGEIV = auto()
    NOT_STATED = auto()


@assemble_case_model()
class PancreasCancerCase(BaseModel):
    """Canonical extracted case record for pancreas cancer."""

    procedure: PancreasProcedure | None = None
    tumor_site: PancreasTumorSite | None = None
    histology: PancreasHistology | None = None
    tumor_size: int | None = None
    tumor_extension: PancreasTumorExtension | None = None
    lymphovascular_invasion: bool | None = None
    perineural_invasion: bool | None = None
    distant_metastasis: bool | None = None
    treatment_effect: str | None = None

    _STAGING: ClassVar = StagingSpec(
        pt=PancreasPT,
        pn=PancreasPN,
        pm=PancreasPM,
        tnm_descriptor=TNMDescriptor,
        stage_groups={"overall_stage": PancreasOverallStage},
        lean=True,
    )

    margins: PancreasMarginSite | None = MarginSpec()

    regional_lymph_node: PancreasLNCategory | None = LNSpec()
    extranodal_extension: bool | None = None
    maximal_ln_size: int | None = None
