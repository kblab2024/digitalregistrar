"""Canonical case-model for cervix cancer (v2, two-layer schema).

Cervix uses FIGO stage-group wording (carried in
``schemas/extraction/cervix.py`` for ``stage_group``) and has no
biomarkers group.
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
from ._enums import LowerStrEnum, LymphNodeSide, PMCategory, TNMDescriptor


class CervixProcedure(LowerStrEnum):
    RADICAL_HYSTERECTOMY = auto()
    TOTAL_HYSTERECTOMY_BSO = auto()
    SIMPLE_HYSTERECTOMY = auto()
    EXTENTERATION = auto()
    OTHERS = auto()
    NOT_STATED = auto()


class CervixSurgicalTechnique(LowerStrEnum):
    OPEN = auto()
    LAPAROSCOPIC = auto()
    VAGINAL = auto()
    OTHERS = auto()
    NOT_STATED = auto()


class CervixPrimarySite(LowerStrEnum):
    CLOCK_12_3 = auto()
    CLOCK_3_6 = auto()
    CLOCK_6_9 = auto()
    CLOCK_9_12 = auto()
    NOT_STATED = auto()


class CervixHistology(LowerStrEnum):
    SQUAMOUS_CELL_CARCINOMA_HPV_ASSOCIATED = auto()
    SQUAMOUS_CELL_CARCINOMA_HPV_DEPENDAENT = auto()
    SQUAMOUS_CELL_CARCINOMA_NOS = auto()
    ADENOCARCINOMA_HPV_ASSOCIATED = auto()
    ADENOCARCINOMA_HPV_INDEPENDENT = auto()
    ADENOCARCINOMA_NOS = auto()
    ADENOSQUAMOUS_CARCINOMA = auto()
    NEUROENDOCRINE_CARCINOMA = auto()
    GLASSY_CELL_CARCINOMA = auto()
    SMALL_CELL_CARCINOMA = auto()
    LARGE_CELL_CARCINOMA = auto()
    OTHERS = auto()
    NOT_STATED = auto()


class CervixDOINumber(LowerStrEnum):
    LESS_THAN_3 = auto()
    THREE_TO_FIVE = auto()
    GREATER_THAN_5 = auto()
    NOT_STATED = auto()


class CervixDOIThreeTier(LowerStrEnum):
    INNER_THIRD = auto()
    MIDDLE_THIRD = auto()
    OUTER_THIRD = auto()
    NOT_STATED = auto()


class CervixMarginSite(LowerStrEnum):
    ECTOCERVICAL = auto()
    ENDOCERVICAL = auto()
    RADIAL_CIRCUMFERENTIAL = auto()
    VAGINAL_CUFF = auto()
    OTHERS = auto()
    NOT_STATED = auto()


class CervixLNCategory(LowerStrEnum):
    PELVIC = auto()
    PARA_AORTIC = auto()
    INTERNAL_ILIAC = auto()
    OBTURATOR = auto()
    EXTERNAL_ILIAC = auto()
    COMMON_ILIAC = auto()
    PARAMETRIAL = auto()
    OTHERS = auto()
    NOT_STATED = auto()


class CervixPT(LowerStrEnum):
    TX = auto()
    T1A1 = auto()
    T1A2 = auto()
    T1B1 = auto()
    T1B2 = auto()
    T1B3 = auto()
    T2A1 = auto()
    T2A2 = auto()
    T2B = auto()
    T3A = auto()
    T3B = auto()
    T4 = auto()
    NOT_STATED = auto()


class CervixPN(LowerStrEnum):
    NX = auto()
    N0 = auto()
    N1MI = auto()
    N1A = auto()
    N2MI = auto()
    N2A = auto()
    NOT_STATED = auto()


class CervixStageGroup(LowerStrEnum):
    STAGE0 = auto()
    STAGEIA1 = auto()
    STAGEIA2 = auto()
    STAGEIB1 = auto()
    STAGEIB2 = auto()
    STAGEIB3 = auto()
    STAGEIIA1 = auto()
    STAGEIIA2 = auto()
    STAGEIIB = auto()
    STAGEIIIA = auto()
    STAGEIIIB = auto()
    STAGEIIIC1 = auto()
    STAGEIIIC2 = auto()
    STAGEIVA = auto()
    STAGEIVB = auto()
    NOT_STATED = auto()


@assemble_case_model()
class CervixCancerCase(BaseModel):
    """Canonical extracted case record for cervix cancer."""

    procedure: CervixProcedure | None = None
    surgical_technique: CervixSurgicalTechnique | None = None
    cancer_primary_site: CervixPrimarySite | None = None
    histology: CervixHistology | None = None
    grade: Literal[1, 2, 3] | None = None
    tumor_size: int | None = None
    depth_of_invasion_number: CervixDOINumber | None = None
    depth_of_invasion_three_tier: CervixDOIThreeTier | None = None
    distant_metastasis: bool | None = None
    treatment_effect: str | None = None

    _STAGING: ClassVar = StagingSpec(
        pt=CervixPT,
        pn=CervixPN,
        pm=PMCategory,
        tnm_descriptor=TNMDescriptor,
        stage_groups={"stage_group": CervixStageGroup},
        lean=True,
    )

    margins: CervixMarginSite | None = MarginSpec()

    regional_lymph_node: CervixLNCategory | None = LNSpec(side=LymphNodeSide)
    extranodal_extension: bool | None = None
    maximal_ln_size: int | None = None
