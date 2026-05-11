"""Canonical case-model for prostate cancer (v2, two-layer schema).

Prostate is the only organ with a non-standard ``margins`` group:
``margin_positivity`` (bool), ``involved_margin_list`` (list of StrEnum
sites), ``margin_length``. These stay declared inline as plain fields
rather than going through :class:`MarginSpec`. ``pm_category`` also has
5 values rather than the default 3.
"""
from enum import auto
from typing import ClassVar

from pydantic import BaseModel

from ._case_builder import (
    LNSpec,
    StagingSpec,
    assemble_case_model,
)
from ._enums import LowerStrEnum, LymphNodeSide, TNMDescriptor


class ProstateProcedure(LowerStrEnum):
    RADICAL_PROSTATECTOMY = auto()
    OTHERS = auto()
    NOT_STATED = auto()


class ProstateSurgicalTechnique(LowerStrEnum):
    OPEN = auto()
    ROBOTIC = auto()
    HYBRID = auto()
    OTHERS = auto()
    NOT_STATED = auto()


class ProstateHistology(LowerStrEnum):
    ACINAR_ADENOCARCINOMA = auto()
    INTRADUCTAL_CARCINOMA = auto()
    DUCTAL_ADENOCARCINOMA = auto()
    MIXED_ACINAR_DUCTAL = auto()
    NEUROENDOCRINE_CARCINOMA_SMALL_CELL = auto()
    OTHERS = auto()
    NOT_STATED = auto()


class ProstateGleasonGroup(LowerStrEnum):
    GROUP_1_3_3 = auto()
    GROUP_2_3_4 = auto()
    GROUP_3_4_3 = auto()
    GROUP_4_4_4 = auto()
    GROUP_5_4_5 = auto()
    GROUP_5_5_4 = auto()
    GROUP_5_5_5 = auto()
    NOT_STATED = auto()


class ProstateMarginSite(LowerStrEnum):
    RIGHT_APICAL = auto()
    LEFT_APICAL = auto()
    RIGHT_BLADDER_NECK = auto()
    LEFT_BLADDER_NECK = auto()
    RIGHT_ANTERIOR = auto()
    LEFT_ANTERIOR = auto()
    RIGHT_LATERAL = auto()
    LEFT_LATERAL = auto()
    RIGHT_POSTEROLATERAL = auto()
    LEFT_POSTEROLATERAL = auto()
    RIGHT_POSTERIOR = auto()
    LEFT_POSTERIOR = auto()


class ProstateMarginLength(LowerStrEnum):
    LIMITED = auto()
    NON_LIMITED = auto()
    NOT_STATED = auto()


class ProstateLNCategory(LowerStrEnum):
    HYPOGASTRIC = auto()
    OBTURATOR = auto()
    EXTERNAL_ILIAC = auto()
    INTERNAL_ILIAC = auto()
    COMMON_ILIAC = auto()
    ILIAC_NOS = auto()
    PELVIC_NOS = auto()
    OTHERS = auto()
    NOT_STATED = auto()


class ProstatePT(LowerStrEnum):
    TX = auto()
    T2 = auto()
    T3A = auto()
    T3B = auto()
    T4 = auto()
    NOT_STATED = auto()


class ProstatePN(LowerStrEnum):
    NX = auto()
    N0 = auto()
    N1 = auto()
    NOT_STATED = auto()


class ProstatePM(LowerStrEnum):
    MX = auto()
    M0 = auto()
    M1A = auto()
    M1B = auto()
    M1C = auto()
    NOT_STATED = auto()


class ProstateStageGroup(LowerStrEnum):
    STAGE0 = auto()
    STAGEI = auto()
    STAGEIIA = auto()
    STAGEIIB = auto()
    STAGEIIC = auto()
    STAGEIIIA = auto()
    STAGEIIIB = auto()
    STAGEIIIC = auto()
    STAGEIVA = auto()
    STAGEIVB = auto()
    NOT_STATED = auto()


@assemble_case_model()
class ProstateCancerCase(BaseModel):
    """Canonical extracted case record for prostate cancer."""

    procedure: ProstateProcedure | None = None
    surgical_technique: ProstateSurgicalTechnique | None = None
    prostate_weight: int | None = None
    prostate_size: int | None = None
    histology: ProstateHistology | None = None
    grade: ProstateGleasonGroup | None = None
    gleason_4_percentage: int | None = None
    gleason_5_percentage: int | None = None
    intraductal_carcinoma_presence: bool | None = None
    cribriform_pattern_presence: bool | None = None
    tumor_percentage: int | None = None
    tumor_size: int | None = None
    extraprostatic_extension: bool | None = None
    seminal_vesicle_invasion: bool | None = None
    bladder_invasion: bool | None = None
    lymphovascular_invasion: bool | None = None
    perineural_invasion: bool | None = None
    distant_metastasis: bool | None = None
    treatment_effect: str | None = None

    _STAGING: ClassVar = StagingSpec(
        pt=ProstatePT,
        pn=ProstatePN,
        pm=ProstatePM,
        tnm_descriptor=TNMDescriptor,
        stage_groups={"stage_group": ProstateStageGroup},
        lean=True,
    )

    # margins (non-standard shape — declared inline)
    margin_positivity: bool | None = None
    involved_margin_list: list[ProstateMarginSite] | None = None
    margin_length: ProstateMarginLength | None = None

    regional_lymph_node: ProstateLNCategory | None = LNSpec(side=LymphNodeSide)
    extranodal_extension: bool | None = None
    maximal_ln_size: int | None = None
