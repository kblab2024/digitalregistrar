"""Canonical case-model for breast cancer (v2, two-layer schema).

LAYER 1 (this file): shape & validation only.
- Per-vocabulary StrEnum classes (each ending with ``NOT_STATED``).
- Field annotations are ``MyEnum | None`` (default ``None``); the
  ``NOT_STATED`` member exists so the LM can assert it explicitly, but
  the JSON-Schema default stays ``null`` for backward compatibility
  with downstream eval pipelines.
- No ``GroupedField``, no descriptions, no ``_GROUP_INSTRUCTIONS``
  ClassVar. Spec markers (``MarginSpec``/``LNSpec``/``BiomarkerSpec``)
  are invoked bare, putting the decorator into "lean" mode (clean
  nested types, no ``json_schema_extra`` payload on the parent field).
- The verbose nested types (``BreastMargin``/``BreastLN``/``BreastBiomarker``)
  and the staging block are still synthesised by
  :func:`assemble_case_model`; the only change is that they now carry
  no ``Field(description=...)`` either.

LAYER 2 (``schemas/extraction/breast.py``): all LM-facing prompt
content — per-field descriptions and per-group instructions. The
factory and the JSON-schema CLI both consume that layer.
"""
from enum import auto
from typing import ClassVar, Literal

from pydantic import BaseModel

from ._case_builder import (
    BiomarkerSpec,
    LNSpec,
    MarginSpec,
    StagingSpec,
    assemble_case_model,
)
from ._enums import LowerStrEnum, LymphNodeSide, PMCategory, TNMDescriptor


# --- Per-organ closed vocabularies ----------------------------------------


class BreastProcedure(LowerStrEnum):
    PARTIAL_MASTECTOMY = auto()
    SIMPLE_MASTECTOMY = auto()
    BREAST_CONSERVING_SURGERY = auto()
    MODIFIED_RADICAL_MASTECTOMY = auto()
    TOTAL_MASTECTOMY = auto()
    WIDE_EXCISION = auto()
    OTHERS = auto()
    NOT_STATED = auto()


class BreastQuadrant(LowerStrEnum):
    UPPER_OUTER_QUADRANT = auto()
    UPPER_INNER_QUADRANT = auto()
    LOWER_OUTER_QUADRANT = auto()
    LOWER_INNER_QUADRANT = auto()
    NIPPLE = auto()
    OTHERS = auto()
    NOT_STATED = auto()


class BreastLaterality(LowerStrEnum):
    RIGHT = auto()
    LEFT = auto()
    BILATERAL = auto()
    NOT_STATED = auto()


class BreastHistology(LowerStrEnum):
    INVASIVE_CARCINOMA_NO_SPECIAL_TYPE = auto()
    INVASIVE_LOBULAR_CARCINOMA = auto()
    MIXED_DUCTAL_AND_LOBULAR_CARCINOMA = auto()
    TUBULAR_ADENOCARCINOMA = auto()
    MUCINOUS_ADENOCARCINOMA = auto()
    ENCAPSULATED_PAPILLARY_CARCINOMA = auto()
    SOLID_PAPILLARY_CARCINOMA = auto()
    INFLAMMATORY_CARCINOMA = auto()
    OTHER_SPECIAL_TYPES = auto()
    NOT_STATED = auto()


class BreastMarginSite(LowerStrEnum):
    CLOCK_12_3 = auto()
    CLOCK_3_6 = auto()
    CLOCK_6_9 = auto()
    CLOCK_9_12 = auto()
    CLOCK_12 = auto()
    CLOCK_3 = auto()
    CLOCK_6 = auto()
    CLOCK_9 = auto()
    SUPERFICIAL = auto()
    BASE = auto()
    NOT_STATED = auto()


class BreastLNCategory(LowerStrEnum):
    SENTINEL = auto()
    NONSENTINEL = auto()
    OTHERS = auto()
    NOT_STATED = auto()


class BreastBiomarkerCategory(LowerStrEnum):
    ER = auto()
    PR = auto()
    HER2 = auto()
    KI67 = auto()
    OTHERS = auto()
    NOT_STATED = auto()


# --- Staging codes (organ-specific) ---------------------------------------


class BreastPT(LowerStrEnum):
    TX = auto()
    TIS = auto()
    T1MI = auto()
    T1A = auto()
    T1B = auto()
    T1C = auto()
    T2 = auto()
    T3 = auto()
    T4A = auto()
    T4B = auto()
    T4C = auto()
    NOT_STATED = auto()


class BreastPN(LowerStrEnum):
    NX = auto()
    N0 = auto()
    N1MI = auto()
    N1A = auto()
    N1B = auto()
    N1C = auto()
    N2A = auto()
    N2B = auto()
    N3A = auto()
    N3B = auto()
    N3C = auto()
    NOT_STATED = auto()


class BreastStageGroup(LowerStrEnum):
    STAGE0 = auto()
    STAGEIA = auto()
    STAGEIB = auto()
    STAGEIIA = auto()
    STAGEIIB = auto()
    STAGEIIIA = auto()
    STAGEIIIB = auto()
    STAGEIIIC = auto()
    STAGEIV = auto()
    NOT_STATED = auto()


# --- Case-model -----------------------------------------------------------


@assemble_case_model()
class BreastCancerCase(BaseModel):
    """Canonical extracted case record for breast cancer."""

    # nonnested
    procedure: BreastProcedure | None = None
    cancer_quadrant: BreastQuadrant | None = None
    cancer_clock: Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] | None = None
    cancer_laterality: BreastLaterality | None = None
    histology: BreastHistology | None = None
    tumor_size: int | None = None
    lymphovascular_invasion: bool | None = None
    perineural_invasion: bool | None = None
    distant_metastasis: bool | None = None
    treatment_effect: str | None = None

    # dcis
    dcis_present: bool | None = None
    dcis_size: int | None = None
    dcis_comedo_necrosis: bool | None = None
    dcis_grade: Literal[1, 2, 3] | None = None

    # grading
    nuclear_grade: Literal[1, 2, 3] | None = None
    tubule_formation: Literal[1, 2, 3] | None = None
    mitotic_rate: Literal[1, 2, 3] | None = None
    total_score: Literal[3, 4, 5, 6, 7, 8, 9] | None = None
    grade: Literal[1, 2, 3] | None = None

    # staging — emitted by StagingSpec, descriptions in extraction/breast.py
    _STAGING: ClassVar = StagingSpec(
        pt=BreastPT,
        pn=BreastPN,
        pm=PMCategory,
        tnm_descriptor=TNMDescriptor,
        stage_groups={
            "pathologic_stage_group": BreastStageGroup,
            "anatomic_stage_group": BreastStageGroup,
        },
        lean=True,
    )

    # margins (auto-expanded to list[BreastMargin])
    margins: BreastMarginSite | None = MarginSpec()

    # lymph_nodes (auto-expanded to list[BreastLN])
    regional_lymph_node: BreastLNCategory | None = LNSpec(side=LymphNodeSide)
    extranodal_extension: bool | None = None
    maximal_ln_size: int | None = None

    # biomarkers (auto-expanded to list[BreastBiomarker])
    biomarkers: list = BiomarkerSpec(categories=BreastBiomarkerCategory)
