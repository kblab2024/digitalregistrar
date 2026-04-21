# -*- coding: utf-8 -*-
"""
schemas/canonical/breast.py
Canonical Pydantic schema definitions for breast cancer registry extraction.
Authoritative data dictionary for BreastCancer* and DCIS DSPy signatures.
AJCC 8th edition, CAP 4.10.0.0
"""
__version__ = "1.0.0"
__date__ = "2026-04-22"
__author__ = ["Hong-Kai (Walther) Chen", "Po-Yen Tzeng", "Kai-Po Chang"]
__copyright__ = "Copyright 2025, Med NLP Lab, China Medical University"
__license__ = "MIT"
__ajcc_version__ = 8
__cap_version__ = "4.10.0.0"

from pydantic import BaseModel, Field
from typing import Literal

# --- TypeAliases ---

ProcedureType = Literal[
    'partial_mastectomy', 'simple_mastectomy', 'breast_conserving_surgery',
    'modified_radical_mastectomy', 'total_mastectomy', 'wide_excision', 'others'
]
CancerQuadrantType = Literal[
    'upper_outer_quadrant', 'upper_inner_quadrant', 'lower_outer_quadrant',
    'lower_inner_quadrant', 'nipple', 'others'
]
CancerClockType = Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
CancerLateralityType = Literal['right', 'left', 'bilateral']
HistologyType = Literal[
    "invasive_carcinoma_no_special_type", "invasive_lobular_carcinoma",
    "mixed_ductal_and_lobular_carcinoma", "tubular_adenocarcinoma",
    "mucinous_adenocarcinoma", "encapsulated_papillary_carcinoma",
    "solid_papillary_carcinoma", "inflammatory_carcinoma", "other_special_types"
]
DcisGradeType = Literal[1, 2, 3]
TnmDescriptorType = Literal['y', 'r', 'm']
PtCategoryType = Literal[
    "tx", "tis", "t1mi", "t1a", "t1b", "t1c", "t2", "t3", "t4a", "t4b", "t4c"
]
PnCategoryType = Literal[
    "nx", "n0", "n1mi", "n1a", "n1b", "n1c",
    "n2a", "n2b", "n3a", "n3b", "n3c"
]
PmCategoryType = Literal["mx", "m0", "m1"]
StageGroupType = Literal["0", "ia", "ib", "iia", "iib", "iiia", "iiib", "iiic", "iv"]
NuclearGradeType = Literal[1, 2, 3]
TotalScoreType = Literal[3, 4, 5, 6, 7, 8, 9]
MarginCategoryType = Literal[
    "12_3_clock", "3_6_clock", "6_9_clock", "9_12_clock",
    "12_clock", "3_clock", "6_clock", "9_clock", "superficial", "base"
]
LNSideType = Literal["right", "left", "midline"]
LNCategoryType = Literal["sentinel", "nonsentinel", "others"]
BiomarkerCategoryType = Literal["er", "pr", "her2", "ki67", "others"]
HER2ScoreType = Literal[0, 1, 2, 3]

# --- Pydantic nested models ---

class BreastMargin(BaseModel):
    margin_category: MarginCategoryType | None = Field(
        None,
        description="acceptable value for surgical margins in breast cancer. If not included in these standard margins, should be classified as others."
    )
    margin_involved: bool
    distance: int | None = Field(
        None,
        description="If margin is involved, return 0. If margin is uninvolved/free, try your best to find the distance at both microscopic and macroscopic(gross) description, and specify the distance from tumor to margin in mm, rounded to integer. If the margin is uninvolved/free and, after your best effort, the distance is still not specified, return null"
    )
    description: str | None

class BreastLN(BaseModel):
    lymph_node_side: LNSideType | None = Field(
        None,
        description="acceptable value for lymph node side in breast cancer. If not included in these standard sides, should be classified as None."
    )
    lymph_node_category: LNCategoryType | None = Field(
        None,
        description="acceptable value for lymph node categories (i.e. stations) in breast cancer. If not included in these standard lymph node 'station' number, should be classified as others."
    )
    involved: int
    examined: int
    station_name: str | None = Field(None, description="specify the name of the lymph node station here.")

class BreastBiomarker(BaseModel):
    biomarker_category: BiomarkerCategoryType | None = Field(
        None,
        description="acceptable value for biomarker categories in breast cancer. If not included in these standard categories, should be classified as others."
    )
    expression: bool | None = Field(
        None,
        description="specify whether or not the biomarker is expressed here. For Her-2 please refer to the score field, and don't fill in this field."
    )
    percentage: int | None = Field(
        None,
        description="the percentage of tumor cells showing positive expression of the biomarker, rounded to integer. if not specified, return null"
    )
    score: HER2ScoreType | None = Field(
        None,
        description="specify the Her-2 expression score, negative: score 0 or 1, equivocal: score 2, positive: score 3 of the biomarker here, if applicable."
    )
    biomarker_name: str | None = Field(None, description="specify the name of the biomarker here.")

# --- Full output schemas (one per DSPy Signature) ---

class BreastCancerNonnestedSchema(BaseModel):
    procedure: ProcedureType | None = Field(None, description="identify which surgery procedure was used. e.g. partial mastectomy")
    cancer_quadrant: CancerQuadrantType | None = Field(None, description="identify the primary site of cancer. e.g. upper outer quadrant. please consider what side is the breast excision specimen when you determine the quadrant")
    cancer_clock: CancerClockType | None = Field(None, description="identify the clock position of the cancer if mentioned. e.g. 3")
    cancer_laterality: CancerLateralityType | None = Field(None, description="identify the side of the cancer. e.g. right")
    histology: HistologyType | None = Field(None, description="identify the histological type of the cancer. e.g. invasive carcinoma of no special type. If the histological type is not included in the above list, please code as other_special_types and specify the histological type in the description field.")
    tumor_size: int | None = Field(None, description="identify the size of the tumor in mm, rounded, if multiple tumors are present, please provide the size of the largest tumor")
    lymphovascular_invasion: bool | None = Field(None, description="check whether or not lymphovascular invasion is present")
    perineural_invasion: bool | None = Field(None, description="check whether or not perineural invasion is present")
    distant_metastasis: bool | None = Field(None, description="check whether or not distant metastasis is present")
    treatment_effect: str | None = Field(None, description='check the treatment effect of the cancer. If you see "No known presurgical therapy", return None')

class DCISSchema(BaseModel):
    dcis_present: bool | None = Field(None, description="check whether or not ductal carcinoma in situ is present")
    dcis_size: int | None = Field(None, description="if ductal carcinoma in situ is present, identify the size of the largest DCIS in mm, rounded")
    dcis_comedo_necrosis: bool | None = Field(None, description="if ductal carcinoma in situ is present, check whether or not comedo necrosis is present")
    dcis_grade: DcisGradeType | None = Field(None, description="if ductal carcinoma in situ is present, identify the grade of DCIS, low grade (grade 1), intermediate grade (grade 2), high grade (grade 3)")

class BreastCancerStagingSchema(BaseModel):
    tnm_descriptor: TnmDescriptorType | None = Field(None, description='identify the tnm descriptor of the tumor. e.g., "y" (post-therapy), "r", etc.')
    pt_category: PtCategoryType | None = Field(None, description="identify the pt category of the tumor")
    pn_category: PnCategoryType | None = Field(None, description="identify the pn category of the tumor")
    pm_category: PmCategoryType | None = Field(None, description="identify the pm category of the tumor. if you see cM0 or cM1, etc., code as mx, since pathological M category is not available")
    pathologic_stage_group: StageGroupType | None = Field(None, description="identify the pathologic stage group of the tumor, return none if only anatomical stage group is given, dont guess")
    anatomic_stage_group: StageGroupType | None = Field(None, description="identify the anatomic stage group of the tumor")
    ajcc_version: int | None = Field(None, description="identify the ajcc version of the pathological staging")

class BreastCancerGradingSchema(BaseModel):
    nuclear_grade: NuclearGradeType | None = Field(None, description="identify the nuclear grade of the tumor")
    tubule_formation: NuclearGradeType | None = Field(None, description="identify the tubule formation score of the tumor")
    mitotic_rate: NuclearGradeType | None = Field(None, description="identify the mitotic rate score of the tumor")
    total_score: TotalScoreType | None = Field(None, description="identify the total score of the tumor")
    grade: NuclearGradeType | None = Field(None, description="identify the grade of the tumor")

class BreastCancerMarginsSchema(BaseModel):
    margins: list[BreastMargin] | None = Field(
        None,
        description="return all of the possible involved margins and its distance from cancer. If not present, just output null for every margin"
    )

class BreastCancerLNSchema(BaseModel):
    regional_lymph_node: list[BreastLN] | None = Field(
        None,
        description="return all of the involved regional lymph node. If not present, just output null for every lymph node"
    )
    extranodal_extension: bool | None = Field(None, description="check whether or not extranodal extension is present; if no lymph node metastasis, should be None")
    maximal_ln_size: int | None = Field(None, description="check the maximal size of node metastatic tumor in mm, rounded to integer; if no lymph node metastasis, should be None")

class BreastCancerBiomarkersSchema(BaseModel):
    biomarkers: list[BreastBiomarker] | None = Field(
        None,
        description="return all of the examined immunoreceptors using immunohistochemistry techniques, like er, pr, her2, ki67, etc. If not present, just output null for every biomarker"
    )
