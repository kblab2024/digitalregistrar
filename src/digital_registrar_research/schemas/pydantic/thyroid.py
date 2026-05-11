"""Canonical case-model for thyroid cancer (v2, two-layer schema).

``stage_groups`` field is named ``overall_stage``. No biomarkers.
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
from ._enums import LowerStrEnum, LymphNodeSide, PMCategory, TNMDescriptor


class ThyroidPredisposingCondition(LowerStrEnum):
    RADIATION = auto()
    FAMILY_HISTORY = auto()
    NOT_STATED = auto()


class ThyroidProcedure(LowerStrEnum):
    PARTIAL_EXCISION = auto()
    RIGHT_LOBECTOMY = auto()
    LEFT_LOBECTOMY = auto()
    TOTAL_THYROIDECTOMY = auto()
    OTHERS = auto()
    NOT_STATED = auto()


class ThyroidTumorFocality(LowerStrEnum):
    UNIFOCAL = auto()
    MULTIFOCAL = auto()
    NOT_SPECIFIED = auto()


class ThyroidTumorSite(LowerStrEnum):
    RIGHT_LOBE = auto()
    LEFT_LOBE = auto()
    ISTHMUS = auto()
    BOTH_LOBE = auto()
    OTHERS = auto()
    NOT_STATED = auto()


class ThyroidHistology(LowerStrEnum):
    PAPILLARY_THYROID_CARCINOMA = auto()
    FOLLICULAR_THYROID_CARCINOMA = auto()
    MEDULLARY_THYROID_CARCINOMA = auto()
    ANAPLASTIC_THYROID_CARCINOMA = auto()
    OTHERS = auto()
    NOT_STATED = auto()


class ThyroidMitoticActivity(LowerStrEnum):
    LESS_THAN_3 = auto()
    THREE_TO_FIVE = auto()
    MORE_THAN_5 = auto()
    NOT_STATED = auto()


class ThyroidExtrathyroidExtension(LowerStrEnum):
    MICROSCOPIC_STRAP_MUSCLE = auto()
    MACROSCOPIC_STRAP_MUSCLE_T3B = auto()
    SUBCUTANEOUS_TRACHEA_ESOPHAGUS_RLN_T4A = auto()
    PREVERTEBRAL_CAROTID_MEDIASTINAL_T4B = auto()
    NOT_STATED = auto()


class ThyroidMarginSite(LowerStrEnum):
    OUTMOST = auto()
    ANTERIOR_OUTMOST = auto()
    POSTERIOR_OUTMOST = auto()
    ISTHMUS = auto()
    OTHERS = auto()
    NOT_STATED = auto()


class ThyroidLNCategory(LowerStrEnum):
    LEVEL_VI_CENTRAL = auto()
    LEVEL_I = auto()
    LEVEL_II = auto()
    LEVEL_III = auto()
    LEVEL_IV = auto()
    LEVEL_V = auto()
    LEVEL_VII = auto()
    OTHERS = auto()
    NOT_STATED = auto()


class ThyroidPT(LowerStrEnum):
    TX = auto()
    T1A = auto()
    T1B = auto()
    T2 = auto()
    T3A = auto()
    T3B = auto()
    T4A = auto()
    T4B = auto()
    NOT_STATED = auto()


class ThyroidPN(LowerStrEnum):
    NX = auto()
    N0 = auto()
    N1A = auto()
    N1B = auto()
    NOT_STATED = auto()


class ThyroidOverallStage(LowerStrEnum):
    STAGEI = auto()
    STAGEII = auto()
    STAGEIII = auto()
    STAGEIVA = auto()
    STAGEIVB = auto()
    STAGEIVC = auto()
    NOT_STATED = auto()


@assemble_case_model()
class ThyroidCancerCase(BaseModel):
    """Canonical extracted case record for thyroid cancer."""

    predisposing_condition: ThyroidPredisposingCondition | None = None
    procedure: ThyroidProcedure | None = None
    tumor_focality: ThyroidTumorFocality | None = None
    tumor_site: ThyroidTumorSite | None = None
    histology: ThyroidHistology | None = None
    tumor_size: int | None = None
    mitotic_activity: ThyroidMitoticActivity | None = None
    extrathyroid_extension: ThyroidExtrathyroidExtension | None = None
    tumor_necrosis: bool | None = None
    lymphovascular_invasion: bool | None = None
    perineural_invasion: bool | None = None
    distant_metastasis: bool | None = None
    treatment_effect: str | None = None

    _STAGING: ClassVar = StagingSpec(
        pt=ThyroidPT,
        pn=ThyroidPN,
        pm=PMCategory,
        tnm_descriptor=TNMDescriptor,
        stage_groups={"overall_stage": ThyroidOverallStage},
        lean=True,
    )

    margins: ThyroidMarginSite | None = MarginSpec()

    regional_lymph_node: ThyroidLNCategory | None = LNSpec(side=LymphNodeSide)
    extranodal_extension: bool | None = None
    maximal_ln_size: int | None = None
