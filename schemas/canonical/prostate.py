# -*- coding: utf-8 -*-
"""
schemas/canonical/prostate.py
Canonical Pydantic schema definitions for prostate cancer registry extraction.
Authoritative data dictionary for ProstateCancer* DSPy signatures.
AJCC 8th edition, CAP 4.3.0.0
"""
__version__ = "1.0.0"
__date__ = "2026-04-22"
__author__ = ["Hong-Kai (Walther) Chen", "Po-Yen Tzeng", "Kai-Po Chang"]
__copyright__ = "Copyright 2025, Med NLP Lab, China Medical University"
__license__ = "MIT"
__ajcc_version__ = 8
__cap_version__ = "4.3.0.0"

from pydantic import BaseModel, Field
from typing import Literal

# --- TypeAliases ---

ProcedureType = Literal['radical_prostatectomy', 'others']
SurgicalTechniqueType = Literal["open", "robotic", "hybrid", "others"]
HistologyType = Literal[
    "acinar_adenocarcinoma", "intraductal_carcinoma",
    "ductal_adenocarcinoma", "mixed_acinar_ductal",
    "neuroendocrine_carcinoma_small_cell", "others"
]
GleasonGroupType = Literal[
    "group_1_3_3", "group_2_3_4", "group_3_4_3",
    "group_4_4_4", "group_5_4_5", "group_5_5_4", "group_5_5_5"
]
TnmDescriptorType = Literal['y', 'r', 'm']
PtCategoryType = Literal["tx", "t2", "t3a", "t3b", "t4"]
PnCategoryType = Literal["nx", "n0", "n1"]
PmCategoryType = Literal["mx", "m0", "m1a", "m1b", "m1c"]
StageGroupType = Literal["0", "i", "iia", "iib", "iic", "iiia", "iiib", "iiic", "iva", "ivb"]
InvolvedMarginType = Literal[
    "right_apical", "left_apical",
    "right_bladder_neck", "left_bladder_neck",
    "right_anterior", "left_anterior",
    "right_lateral", "left_lateral",
    "right_posterolateral", "left_posterolateral",
    "right_posterior", "left_posterior"
]
MarginLengthType = Literal["limited", "non_limited"]
LNSideType = Literal["right", "left", "midline"]
LNCategoryType = Literal[
    "hypogastric", "obturator", "external_iliac", "internal_iliac",
    "common_iliac", "iliac_nos", "pelvic_nos", "others"
]

# --- Pydantic nested models ---

class ProstateLN(BaseModel):
    lymph_node_side: LNSideType | None = Field(
        None,
        description="acceptable value for lymph node side in prostate cancer. If not included in these standard sides, should be classified as None."
    )
    lymph_node_category: LNCategoryType | None = Field(
        None,
        description="acceptable value for lymph node categories (i.e. stations) in prostate cancer. If not included in these standard lymph node 'station' number, should be classified as others."
    )
    involved: int
    examined: int
    station_name: str | None = Field(None, description="specify the name of the lymph node group/station here.")

# --- Full output schemas (one per DSPy Signature) ---

class ProstateCancerNonnestedSchema(BaseModel):
    procedure: ProcedureType | None = Field(None, description="identify which surgery procedure was used. e.g. radical prostatectomy")
    surgical_technique: SurgicalTechniqueType | None = Field(None, description="identify how the surgery was taken. e.g. robotic")
    prostate_weight: int | None = Field(None, description="identify the weight of the prostate in grams. e.g. 50")
    prostate_size: int | None = Field(None, description="identify the size of the prostate in mm, largest dimension. e.g. 45 means 45mm")
    histology: HistologyType | None = Field(None, description="identify the histological type of the cancer. e.g. acinar adenocarcinoma")
    grade: GleasonGroupType | None = Field(None, description="identify the gleason group of the cancer. e.g. group_2_3_4 means gleason score 7 (3+4)")
    gleason_4_percentage: int | None = Field(None, description="identify the percentage of gleason pattern 4. e.g. 20 means 20%")
    gleason_5_percentage: int | None = Field(None, description="identify the percentage of gleason pattern 5. e.g. 0 means 0%")
    intraductal_carcinoma_presence: bool | None = Field(None, description="check whether or not intraductal carcinoma is present")
    cribriform_pattern_presence: bool | None = Field(None, description="check whether or not cribriform pattern is present. only in gleason score 7 or 8")
    tumor_percentage: int | None = Field(None, description="identify the percentage of tumor in the prostate in both lobes. e.g. 30 means 30%")
    tumor_size: int | None = Field(None, description="identify the size of the tumor in mm, largest dimension. e.g. 15 means 15mm")
    extraprostatic_extension: bool | None = Field(None, description="check whether or not extraprostatic extension is present")
    seminal_vesicle_invasion: bool | None = Field(None, description="check whether or not seminal vesicle invasion is present")
    bladder_invasion: bool | None = Field(None, description="check whether or not bladder invasion is present")
    lymphovascular_invasion: bool | None = Field(None, description="check whether or not lymphovascular invasion is present")
    perineural_invasion: bool | None = Field(None, description="check whether or not perineural invasion is present")
    distant_metastasis: bool | None = Field(None, description="check whether or not distant metastasis is present")
    treatment_effect: str | None = Field(None, description='check the treatment effect of the cancer. If you see "No known presurgical therapy", return None')

class ProstateCancerStagingSchema(BaseModel):
    tnm_descriptor: TnmDescriptorType | None = Field(None, description='identify the tnm descriptor of the tumor. e.g., "y" (post-therapy), "r", etc.')
    pt_category: PtCategoryType | None = Field(None, description="identify the pt category of the tumor")
    pn_category: PnCategoryType | None = Field(None, description="identify the pn category of the tumor")
    pm_category: PmCategoryType | None = Field(None, description="identify the pm category of the tumor. if you see cM0 or cM1, etc., code as mx, since pathological M category is not available")
    stage_group: StageGroupType | None = Field(None, description="identify the stage group of the tumor")
    ajcc_version: int | None = Field(None, description="identify the ajcc version of the pathological staging")

class ProstateCancerMarginsSchema(BaseModel):
    margin_positivity: bool | None = Field(None, description="check whether or not any margin is positive")
    involved_margin_list: list[InvolvedMarginType] | None = Field(None, description="list of involved margins")
    margin_length: MarginLengthType | None = Field(None, description="if margin is positive, check whether the length of positive margin is limited (<3mm) or non-limited (>=3mm)")

class ProstateCancerLNSchema(BaseModel):
    regional_lymph_node: list[ProstateLN] | None = Field(
        None,
        description="return all of the involved regional lymph node. If not present, just output null for every lymph node"
    )
    extranodal_extension: bool | None = Field(None, description="check whether or not extranodal extension is present; if no lymph node metastasis, should be None")
    maximal_ln_size: int | None = Field(None, description="check the maximal size of node metastatic tumor in mm, rounded to integer; if no lymph node metastasis, should be None")
