"""Canonical case-model for esophagus cancer (v2, two-layer schema).

Note: the registry key is ``"esophagus"`` but description text uses the
adjective ``"esophageal"``. No biomarkers group; LN has no side field.
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


class EsophagusProcedure(LowerStrEnum):
    ENDOSCOPIC_RESECTION = auto()
    ESOPHAGECTOMY = auto()
    ESOPHAGOGASTRECTOMY = auto()
    OTHERS = auto()
    NOT_STATED = auto()


class EsophagusSurgicalTechnique(LowerStrEnum):
    OPEN = auto()
    THORACOSCOPIC = auto()
    ROBOTIC = auto()
    HYBRID = auto()
    ENDOSCOPIC = auto()
    OTHERS = auto()
    NOT_STATED = auto()


class EsophagusPrimarySite(LowerStrEnum):
    UPPER_THIRD = auto()
    MIDDLE_THIRD = auto()
    LOWER_THIRD = auto()
    GASTROESOPHAGEAL_JUNCTION = auto()
    NOT_STATED = auto()


class EsophagusHistology(LowerStrEnum):
    SQUAMOUS_CELL_CARCINOMA = auto()
    ADENOCARCINOMA = auto()
    ADENOID_CYSTIC_CARCINOMA = auto()
    MUCOEPIDERMOID_CARCINOMA = auto()
    BASALOID_SQUAMOUS_CELL_CARCINOMA = auto()
    SMALL_CELL_CARCINOMA = auto()
    LARGE_CELL_CARCINOMA = auto()
    OTHERS = auto()
    NOT_STATED = auto()


class EsophagusTumorExtent(LowerStrEnum):
    MUCOSA = auto()
    SUBMUCOSA = auto()
    MUSCULARIS_PROPRIA = auto()
    ADVENTITIA = auto()
    ADJACENT_STRUCTURES = auto()
    NOT_STATED = auto()


class EsophagusMarginSite(LowerStrEnum):
    PROXIMAL = auto()
    DISTAL = auto()
    RADIAL = auto()
    LATERAL = auto()
    DEEP = auto()
    OTHERS = auto()
    NOT_STATED = auto()


class EsophagusLNCategory(LowerStrEnum):
    REGIONAL_ESOPHAGEAL = auto()
    REGIONAL_GASTRIC = auto()
    THORACIC_1 = auto()
    THORACIC_1R = auto()
    THORACIC_1L = auto()
    THORACIC_4 = auto()
    THORACIC_4R = auto()
    THORACIC_4L = auto()
    THORACIC_7 = auto()
    THORACIC_8U = auto()
    THORACIC_8M = auto()
    THORACIC_8L = auto()
    THORACIC_8 = auto()
    THORACIC_9 = auto()
    THORACIC_9R = auto()
    THORACIC_9L = auto()
    THORACIC_10 = auto()
    THORACIC_10R = auto()
    THORACIC_10L = auto()
    ABDOMEN_106 = auto()
    ABDOMEN_1 = auto()
    ABDOMEN_2 = auto()
    ABDOMEN_3 = auto()
    ABDOMEN_4 = auto()
    ABDOMEN_5 = auto()
    ABDOMEN_6 = auto()
    ABDOMEN_7 = auto()
    ABDOMEN_8 = auto()
    ABDOMEN_9 = auto()
    ABDOMEN_10 = auto()
    OTHERS = auto()
    NOT_STATED = auto()


class EsophagusPT(LowerStrEnum):
    TX = auto()
    T1A = auto()
    T1B = auto()
    T2 = auto()
    T3 = auto()
    T4A = auto()
    T4B = auto()
    NOT_STATED = auto()


class EsophagusPN(LowerStrEnum):
    NX = auto()
    N0 = auto()
    N1 = auto()
    N2 = auto()
    N3 = auto()
    NOT_STATED = auto()


class EsophagusStageGroup(LowerStrEnum):
    STAGE0 = auto()
    STAGEI = auto()
    STAGEIA = auto()
    STAGEIB = auto()
    STAGEIC = auto()
    STAGEIIA = auto()
    STAGEIIB = auto()
    STAGEIIIA = auto()
    STAGEIIIB = auto()
    STAGEIVA = auto()
    STAGEIVB = auto()
    NOT_STATED = auto()


@assemble_case_model()
class EsophagusCancerCase(BaseModel):
    """Canonical extracted case record for esophagus cancer."""

    procedure: EsophagusProcedure | None = None
    surgical_technique: EsophagusSurgicalTechnique | None = None
    cancer_primary_site: EsophagusPrimarySite | None = None
    histology: EsophagusHistology | None = None
    grade: Literal[1, 2, 3] | None = None
    tumor_extent: EsophagusTumorExtent | None = None
    lymphovascular_invasion: bool | None = None
    perineural_invasion: bool | None = None
    distant_metastasis: bool | None = None
    treatment_effect: str | None = None

    _STAGING: ClassVar = StagingSpec(
        pt=EsophagusPT,
        pn=EsophagusPN,
        pm=PMCategory,
        tnm_descriptor=TNMDescriptor,
        stage_groups={"stage_group": EsophagusStageGroup},
        lean=True,
    )

    margins: EsophagusMarginSite | None = MarginSpec()

    regional_lymph_node: EsophagusLNCategory | None = LNSpec()
    extranodal_extension: bool | None = None
    maximal_ln_size: int | None = None
