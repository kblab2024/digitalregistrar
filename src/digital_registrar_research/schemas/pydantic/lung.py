"""Canonical case-model for lung cancer (v2, two-layer schema).

Lung is the only organ that uses :class:`LymphNodeSideRL` (right/left
only — no midline) and that has the additional ``othernested`` group
carrying the lung-specific :class:`LungHistologicalPattern` nested type.
The biomarker shape also differs from breast/colon: ``expression`` is
required, there's no ``score`` field, and the percentage description
says "None" instead of "null" (reflected in
:func:`make_biomarker_type` via ``percentage_null_word``).

LM-facing prompt content lives in :mod:`schemas.extraction.lung`.
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
from ._enums import LowerStrEnum, LymphNodeSideRL, TNMDescriptor

# --- Per-organ closed vocabularies ----------------------------------------


class LungProcedure(LowerStrEnum):
    WEDGE_RESECTION = auto()
    SEGMENTECTOMY = auto()
    LOBECTOMY = auto()
    COMPLETION_LOBECTOMY = auto()
    SLEEVE_LOBECTOMY = auto()
    BILOBECTOMY = auto()
    PNEUMONECTOMY = auto()
    MAJOR_AIRWAY_RESECTION = auto()
    OTHERS = auto()
    NOT_STATED = auto()


class LungSurgicalTechnique(LowerStrEnum):
    OPEN = auto()
    THORACOSCOPIC = auto()
    ROBOTIC = auto()
    HYBRID = auto()
    OTHERS = auto()
    NOT_STATED = auto()


class LungSideness(LowerStrEnum):
    RIGHT = auto()
    LEFT = auto()
    MIDLINE = auto()
    NOT_STATED = auto()


class LungPrimarySite(LowerStrEnum):
    UPPER_LOBE = auto()
    MIDDLE_LOBE = auto()
    LOWER_LOBE = auto()
    MAIN_BRONCHUS = auto()
    BRONCHUS_INTERMEDIUS = auto()
    BRONCHUS_LOBAR = auto()
    OTHERS = auto()
    NOT_STATED = auto()


class LungTumorFocality(LowerStrEnum):
    SINGLE_FOCUS = auto()
    SEPARATE_IN_SAME_LOBE_T3 = auto()
    SEPARATE_NODULE_IN_IPSILATERAL_T4 = auto()
    SEPARATE_NODULE_IN_CONTRALATERAL_M1A = auto()
    NOT_STATED = auto()


class LungHistology(LowerStrEnum):
    ADENOCARCINOMA = auto()
    SQUAMOUS_CELL_CARCINOMA = auto()
    ADENOSQUAMOUS_CARCINOMA = auto()
    LARGE_CELL_CARCINOMA = auto()
    LARGE_CELL_NEUROENDOCRINE_CARCINOMA = auto()
    SMALL_CELL_CARCINOMA = auto()
    CARCINOID_TUMOR = auto()
    SARCOMATOID_CARCINOMA = auto()
    PLEOMORPHIC_CARCINOMA = auto()
    PULMONARY_LYMPHOEPITHELIOMA_LIKE_CARCINOMA = auto()
    MUCOEPIDERMOID_CARCINOMA = auto()
    SALIVARY_GLAND_TYPE_TUMOR = auto()
    NON_SMALL_CELL_CARCINOMA_NOT_SPECIFIED = auto()
    NON_SMALL_CELL_CARCINOMA_WITH_NEUROENDOCRINE_FEATURES = auto()
    OTHER = auto()
    NOT_STATED = auto()


class LungMarginSite(LowerStrEnum):
    BRONCHIAL = auto()
    VASCULAR = auto()
    PARENCHYMAL = auto()
    CHEST_WALL = auto()
    OTHERS = auto()
    NOT_STATED = auto()


class LungLNCategory(LowerStrEnum):
    PERIBRONCHIAL = auto()
    LN1 = auto()
    LN2 = auto()
    LN4 = auto()
    LN5 = auto()
    LN6 = auto()
    LN8 = auto()
    LN9 = auto()
    LN10 = auto()
    LN11 = auto()
    LN12 = auto()
    LN13 = auto()
    LN14 = auto()
    LN3A = auto()
    LN3P = auto()
    LN7 = auto()
    OTHERS = auto()
    NOT_STATED = auto()


class LungBiomarkerCategory(LowerStrEnum):
    ALK = auto()
    ROS1 = auto()
    PDL1 = auto()
    OTHERS = auto()
    NOT_STATED = auto()


class LungHistologicalPatternName(LowerStrEnum):
    ACINAR = auto()
    LEPIDIC = auto()
    PAPILLARY = auto()
    SOLID = auto()
    MICROPAPILLARY = auto()
    OTHERS = auto()
    NOT_STATED = auto()


# --- Staging codes (organ-specific) ---------------------------------------


class LungPT(LowerStrEnum):
    TX = auto()
    TIS = auto()
    T1MI = auto()
    T1A = auto()
    T1B = auto()
    T1C = auto()
    T2A = auto()
    T2B = auto()
    T3 = auto()
    T4 = auto()
    NOT_STATED = auto()


class LungPN(LowerStrEnum):
    NX = auto()
    N0 = auto()
    N1 = auto()
    N2 = auto()
    N3 = auto()
    NOT_STATED = auto()


class LungPM(LowerStrEnum):
    MX = auto()
    M0 = auto()
    M1A = auto()
    M1B = auto()
    M1C = auto()
    NOT_STATED = auto()


class LungStageGroup(LowerStrEnum):
    STAGE0 = auto()
    STAGEIA1 = auto()
    STAGEIA2 = auto()
    STAGEIA3 = auto()
    STAGEIB = auto()
    STAGEIIA = auto()
    STAGEIIB = auto()
    STAGEIIIA = auto()
    STAGEIIIB = auto()
    STAGEIIIC = auto()
    STAGEIVA = auto()
    STAGEIVB = auto()
    STAGEIVC = auto()
    NOT_STATED = auto()


# --- Nested type with no spec marker (othernested group) ------------------


class LungHistologicalPattern(BaseModel):
    pattern_name: LungHistologicalPatternName | None = None
    pattern_percentage: int | None = None


# --- Case-model -----------------------------------------------------------


@assemble_case_model()
class LungCancerCase(BaseModel):
    """Canonical extracted case record for lung cancer."""

    # nonnested
    procedure: LungProcedure | None = None
    surgical_technique: LungSurgicalTechnique | None = None
    sideness: LungSideness | None = None
    cancer_primary_site: LungPrimarySite | None = None
    tumor_focality: LungTumorFocality | None = None
    histology: LungHistology | None = None
    grade: int | None = None
    lymphovascular_invasion: bool | None = None
    perineural_invasion: bool | None = None
    spread_through_air_spaces_stas: bool | None = None
    visceral_pleural_invasion: bool | None = None
    direct_invasion_of_adjacent_structures: bool | None = None
    distant_metastasis: bool | None = None
    treatment_effect: str | None = None

    # staging
    _STAGING: ClassVar = StagingSpec(
        pt=LungPT,
        pn=LungPN,
        pm=LungPM,
        tnm_descriptor=TNMDescriptor,
        stage_groups={"stage_group": LungStageGroup},
        lean=True,
    )

    # margins (auto-expanded to list[LungMargin])
    margins: LungMarginSite | None = MarginSpec()

    # lymph_nodes (auto-expanded to list[LungLN]; R/L only — no midline)
    regional_lymph_node: LungLNCategory | None = LNSpec(side=LymphNodeSideRL)
    extranodal_extension: bool | None = None
    maximal_ln_size: int | None = None

    # biomarkers (auto-expanded to list[LungBiomarker]; expression required, no score)
    biomarkers: list = BiomarkerSpec(
        categories=LungBiomarkerCategory,
        expression_required=True,
        has_score=False,
        percentage_null_word="None",
    )

    # othernested — declared inline (not via spec marker) so the
    # ``LungHistologicalPattern`` type is hand-authored above.
    histological_patterns: list[LungHistologicalPattern] | None = None
