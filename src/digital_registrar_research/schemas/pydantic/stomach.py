"""Canonical case-model for stomach cancer (v2, two-layer schema).

Stomach uses "groups" wording for LN categories. No biomarkers.
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


class StomachProcedure(LowerStrEnum):
    ENDOSCOPIC_RESECTION = auto()
    PARTIAL_GASTRECTOMY = auto()
    TOTAL_GASTRECTOMY = auto()
    OTHERS = auto()
    NOT_STATED = auto()


class StomachSurgicalTechnique(LowerStrEnum):
    OPEN = auto()
    LAPAROSCOPIC = auto()
    ROBOTIC = auto()
    HYBRID = auto()
    OTHERS = auto()
    NOT_STATED = auto()


class StomachPrimarySite(LowerStrEnum):
    CARDIA = auto()
    FUNDUS = auto()
    BODY = auto()
    ANTRUM = auto()
    PYLORUS = auto()
    OTHERS = auto()
    NOT_STATED = auto()


class StomachHistology(LowerStrEnum):
    TUBULAR_ADENOCARCINOMA = auto()
    POORLY_COHESIVE_CARCINOMA = auto()
    MIXED_TUBULAR_POORLY_COHESIVE = auto()
    MUCINOUS_ADENOCARCINOMA = auto()
    MIXED_MUCINOUS_POORLY_COHESIVE = auto()
    HEPATOID_CARCINOMA = auto()
    OTHERS = auto()
    NOT_STATED = auto()


class StomachTumorExtent(LowerStrEnum):
    LAMINA_PROPRIA = auto()
    MUSCULARIS_MUCOSAE = auto()
    SUBMUCOSA = auto()
    MUSCULARIS_PROPRIA = auto()
    PENETRATE_SUBSEROSAL_CONNECTIVE_TISSUE_NO_SEROSA = auto()
    INVADES_SEROSA_WITHOUT_ADJACENT_STRUCTURE_INVASION = auto()
    INVADES_ADJACENT_STRUCTURES = auto()
    NOT_STATED = auto()


class StomachMarginSite(LowerStrEnum):
    PROXIMAL = auto()
    DISTAL = auto()
    RADIAL = auto()
    LATERAL = auto()
    DEEP = auto()
    OTHERS = auto()
    NOT_STATED = auto()


class StomachLNCategory(LowerStrEnum):
    REGIONAL = auto()
    REGIONAL_LESSER_CURV = auto()
    REGIONAL_GREATER_CURV = auto()
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
    OTHERS = auto()
    NOT_STATED = auto()


class StomachPT(LowerStrEnum):
    TX = auto()
    T1A = auto()
    T1B = auto()
    T2 = auto()
    T3 = auto()
    T4A = auto()
    T4B = auto()
    NOT_STATED = auto()


class StomachPN(LowerStrEnum):
    NX = auto()
    N0 = auto()
    N1 = auto()
    N2 = auto()
    N3A = auto()
    N3B = auto()
    NOT_STATED = auto()


class StomachStageGroup(LowerStrEnum):
    STAGE0 = auto()
    STAGEI = auto()
    STAGEII = auto()
    STAGEIII = auto()
    STAGEIV = auto()
    STAGEIA = auto()
    STAGEIB = auto()
    STAGEIIA = auto()
    STAGEIIB = auto()
    STAGEIIIA = auto()
    STAGEIIIB = auto()
    STAGEIIIC = auto()
    NOT_STATED = auto()


@assemble_case_model()
class StomachCancerCase(BaseModel):
    """Canonical extracted case record for stomach cancer."""

    procedure: StomachProcedure | None = None
    surgical_technique: StomachSurgicalTechnique | None = None
    cancer_primary_site: StomachPrimarySite | None = None
    histology: StomachHistology | None = None
    grade: Literal[1, 2, 3] | None = None
    tumor_extent: StomachTumorExtent | None = None
    extracellular_mucin: bool | None = None
    signet_ring: bool | None = None
    lymphovascular_invasion: bool | None = None
    perineural_invasion: bool | None = None
    distant_metastasis: bool | None = None
    treatment_effect: str | None = None

    _STAGING: ClassVar = StagingSpec(
        pt=StomachPT,
        pn=StomachPN,
        pm=PMCategory,
        tnm_descriptor=TNMDescriptor,
        stage_groups={"stage_group": StomachStageGroup},
        lean=True,
    )

    margins: StomachMarginSite | None = MarginSpec()

    regional_lymph_node: StomachLNCategory | None = LNSpec()
    extranodal_extension: bool | None = None
    maximal_ln_size: int | None = None
