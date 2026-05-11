"""Canonical case-model for colorectal cancer (v2, two-layer schema).

Note: the registry key is ``"colorectal"`` but the synthesised
nested-type names use the legacy ``Colon*`` prefix
(``ColonMargin`` / ``ColonLN`` / ``ColonBiomarker``) — handled by
``_TYPE_PREFIX_OVERRIDES`` in :mod:`._case_builder`.
"""
from enum import auto
from typing import ClassVar

from pydantic import BaseModel

from ._case_builder import (
    BiomarkerSpec,
    LNSpec,
    MarginSpec,
    StagingSpec,
    assemble_case_model,
)
from ._enums import LowerStrEnum, TNMDescriptor


class ColorectalProcedure(LowerStrEnum):
    RIGHT_HEMICOLECTOMY = auto()
    EXTENDED_RIGHT_HEMICOLECTOMY = auto()
    LEFT_HEMICOLECTOMY = auto()
    LOW_ANTERIOR_RESECTION = auto()
    ANTERIOR_RESECTION = auto()
    ABDOMINOPERINEAL_RESECTION = auto()
    TOTAL_MESORECTAL_EXCISION = auto()
    TOTAL_COLECTOMY = auto()
    SUBTOTAL_COLECTOMY = auto()
    SEGMENTAL_COLECTOMY = auto()
    TRANSANAL_LOCAL_EXCISION = auto()
    POLYPECTOMY = auto()
    OTHERS = auto()
    NOT_STATED = auto()


class ColorectalSurgicalTechnique(LowerStrEnum):
    OPEN = auto()
    LAPAROSCOPIC = auto()
    ROBOTIC = auto()
    TA_TME = auto()
    HYBRID = auto()
    OTHERS = auto()
    NOT_STATED = auto()


class ColorectalPrimarySite(LowerStrEnum):
    CECUM = auto()
    ASCENDING_COLON = auto()
    HEPATIC_FLEXURE = auto()
    TRANSVERSE_COLON = auto()
    SPLENIC_FLEXURE = auto()
    DESCENDING_COLON = auto()
    SIGMOID_COLON = auto()
    RECTOSIGMOID_JUNCTION = auto()
    RECTUM = auto()
    APPENDIX = auto()
    NOT_STATED = auto()


class ColorectalHistology(LowerStrEnum):
    ADENOCARCINOMA = auto()
    MUCINOUS_ADENOCARCINOMA = auto()
    SIGNET_RING_CELL_CARCINOMA = auto()
    MEDULLARY_CARCINOMA = auto()
    MICROPAPILLARY_ADENOCARCINOMA = auto()
    SERRATED_ADENOCARCINOMA = auto()
    ADENOSQUAMOUS_CARCINOMA = auto()
    NEUROENDOCRINE_CARCINOMA = auto()
    OTHERS = auto()
    NOT_STATED = auto()


class ColorectalTumorInvasion(LowerStrEnum):
    LAMINA_PROPRIA = auto()
    SUBMUCOSA = auto()
    MUSCULARIS_PROPRIA = auto()
    PERICOLORECTAL_TISSUE = auto()
    VISCERAL_PERITONEUM_SURFACE = auto()
    ADJACENT_ORGANS_STRUCTURES = auto()
    NOT_STATED = auto()


class ColorectalTypeOfPolyp(LowerStrEnum):
    TUBULAR_ADENOMA = auto()
    TUBULOVILLOUS_ADENOMA = auto()
    VILLOUS_ADENOMA = auto()
    SESSILE_SERRATED_ADENOMA = auto()
    TRADITIONAL_SERRATED_ADENOMA = auto()
    NOT_STATED = auto()


class ColorectalMarginSite(LowerStrEnum):
    PROXIMAL = auto()
    DISTAL = auto()
    MESENTERIC_PEDICLE = auto()
    RADIAL_OR_CIRCUMFERENCIAL = auto()
    OUTMOST_OF_ADHERED_TISSUE = auto()
    OTHERS = auto()
    NOT_STATED = auto()


class ColorectalLNCategory(LowerStrEnum):
    REGIONAL = auto()
    MESENTERIC = auto()
    OTHERS = auto()
    NOT_STATED = auto()


class ColorectalBiomarkerCategory(LowerStrEnum):
    MLH1 = auto()
    MSH2 = auto()
    MSH6 = auto()
    PMS2 = auto()
    HER2 = auto()
    OTHERS = auto()
    NOT_STATED = auto()


class ColorectalPT(LowerStrEnum):
    TX = auto()
    TIS = auto()
    T1 = auto()
    T2 = auto()
    T3 = auto()
    T4A = auto()
    T4B = auto()
    NOT_STATED = auto()


class ColorectalPN(LowerStrEnum):
    NX = auto()
    N0 = auto()
    N1A = auto()
    N1B = auto()
    N1C = auto()
    N2A = auto()
    N2B = auto()
    NOT_STATED = auto()


class ColorectalPM(LowerStrEnum):
    MX = auto()
    M0 = auto()
    M1A = auto()
    M1B = auto()
    M1C = auto()
    NOT_STATED = auto()


class ColorectalStageGroup(LowerStrEnum):
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
    STAGEIVC = auto()
    NOT_STATED = auto()


@assemble_case_model()
class ColorectalCancerCase(BaseModel):
    """Canonical extracted case record for colorectal cancer."""

    procedure: ColorectalProcedure | None = None
    surgical_technique: ColorectalSurgicalTechnique | None = None
    cancer_primary_site: ColorectalPrimarySite | None = None
    histology: ColorectalHistology | None = None
    grade: int | None = None
    tumor_invasion: ColorectalTumorInvasion | None = None
    lymphovascular_invasion: bool | None = None
    perineural_invasion: bool | None = None
    extracellular_mucin: bool | None = None
    signet_ring: bool | None = None
    tumor_budding: int | None = None
    type_of_polyp: ColorectalTypeOfPolyp | None = None
    distant_metastasis: bool | None = None
    treatment_effect: str | None = None

    _STAGING: ClassVar = StagingSpec(
        pt=ColorectalPT,
        pn=ColorectalPN,
        pm=ColorectalPM,
        tnm_descriptor=TNMDescriptor,
        stage_groups={"stage_group": ColorectalStageGroup},
        lean=True,
    )

    margins: ColorectalMarginSite | None = MarginSpec()

    regional_lymph_node: ColorectalLNCategory | None = LNSpec()
    extranodal_extension: bool | None = None
    maximal_ln_size: int | None = None

    biomarkers: list = BiomarkerSpec(categories=ColorectalBiomarkerCategory)
