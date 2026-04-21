# -*- coding: utf-8 -*-
"""
schemas/canonical/cervix.py
Canonical Pydantic schema definitions for cervical cancer registry extraction.
Authoritative data dictionary for CervixCancer* DSPy signatures.
AJCC 8th edition / FIGO staging, CAP 5.1.0.0
"""
__version__ = "1.0.0"
__date__ = "2026-04-22"
__author__ = ["Hong-Kai (Walther) Chen", "Po-Yen Tzeng", "Kai-Po Chang"]
__copyright__ = "Copyright 2025, Med NLP Lab, China Medical University"
__license__ = "MIT"
__ajcc_version__ = 8
__cap_version__ = "5.1.0.0"

from pydantic import BaseModel, Field
from typing import Literal

# --- TypeAliases ---

ProcedureType = Literal[
    'radical_hysterectomy', 'total_hysterectomy_bso',
    'simple_hysterectomy', 'extenteration', 'others'
]
SurgicalTechniqueType = Literal["open", "laparoscopic", "vaginal", "others"]
PrimarySiteType = Literal["12_3_clock", "3_6_clock", "6_9_clock", "9_12_clock"]
HistologyType = Literal[
    "squamous_cell_carcinoma_hpv_associated",
    "squamous_cell_carcinoma_hpv_dependaent",
    "squamous_cell_carcinoma_nos",
    "adenocarcinoma_hpv_associated",
    "adenocarcinoma_hpv_independent",
    "adenocarcinoma_nos",
    "adenosquamous_carcinoma",
    "neuroendocrine_carcinoma",
    "glassy_cell_carcinoma",
    "small_cell_carcinoma",
    "large_cell_carcinoma",
    "others"
]
GradeType = Literal[1, 2, 3]
DepthOfInvasionNumberType = Literal["less_than_3", "3_to_5", "greater_than_5"]
DepthOfInvasionThreeTierType = Literal["inner_third", "middle_third", "outer_third"]
TnmDescriptorType = Literal['y', 'r', 'm']
PtCategoryType = Literal[
    "tx", "t1a1", "t1a2", "t1b1", "t1b2", "t1b3",
    "t2a1", "t2a2", "t2b", "t3a", "t3b", "t4"
]
PnCategoryType = Literal["nx", "n0", "n1mi", "n1a", "n2mi", "n2a"]
PmCategoryType = Literal["mx", "m0", "m1"]
StageGroupType = Literal[
    "0", "ia1", "ia2", "ib1", "ib2", "ib3",
    "iia1", "iia2", "iib", "iiia", "iiib",
    "iiic1", "iiic2", "iva", "ivb"
]
MarginCategoryType = Literal[
    "ectocervical", "endocervical", "radial_circumferential",
    "vaginal_cuff", "others"
]
LNSideType = Literal["right", "left", "midline"]
LNCategoryType = Literal[
    "pelvic", "para_aortic", "internal_iliac", "obturator",
    "external_iliac", "common_iliac", "parametrial", "others"
]

# --- Pydantic nested models ---

class CervixMargin(BaseModel):
    margin_category: MarginCategoryType | None = Field(
        None,
        description="acceptable value for surgical margins in cervical cancer. If not included in these standard margins, should be classified as others."
    )
    margin_involved: bool
    distance: int | None = Field(
        None,
        description="If margin is involved, return 0. If margin is uninvolved/free, try your best to find the distance at both microscopic and macroscopic(gross) description, and specify the distance from tumor to margin in mm, rounded to integer. If the margin is uninvolved/free and, after your best effort, the distance is still not specified, return null"
    )
    description: str | None

class CervixLN(BaseModel):
    lymph_node_side: LNSideType | None = Field(
        None,
        description="acceptable value for lymph node side in cervical cancer. If not included in these standard sides, should be classified as None."
    )
    lymph_node_category: LNCategoryType | None = Field(
        None,
        description="acceptable value for lymph node categories in cervical cancer. If not included in these standard lymph node 'station' number, should be classified as others."
    )
    involved: int
    examined: int
    station_name: str | None = Field(None, description="specify the name of the lymph node station here.")

# --- Full output schemas (one per DSPy Signature) ---

class CervixCancerNonnestedSchema(BaseModel):
    procedure: ProcedureType | None = Field(None, description="identify which surgery procedure was used. e.g. radical_hysterectomy")
    surgical_technique: SurgicalTechniqueType | None = Field(None, description="identify how the surgery was taken. e.g. laparoscopic")
    cancer_primary_site: PrimarySiteType | None = Field(None, description="identify the primary site of cancer. e.g. sigmoid colon")
    histology: HistologyType | None = Field(None, description="identify the histological type of the cancer. e.g. squamous cell carcinoma")
    grade: GradeType | None = Field(None, description="identify the grade of the cancer, well->1, moderate->2, poor/undiff->3")
    tumor_size: int | None = Field(None, description="identify the size of the tumor in mm, rounded to integer")
    depth_of_invasion_number: DepthOfInvasionNumberType | None = Field(
        None,
        description="identify the depth of invasion of the tumor in mm, and choose from these three categories: less_than_3, 3_to_5, greater_than_5"
    )
    depth_of_invasion_three_tier: DepthOfInvasionThreeTierType | None = Field(
        None,
        description="identify the depth of invasion of the tumor in three-tier system: inner_third, middle_third, outer_third"
    )
    distant_metastasis: bool | None = Field(None, description="check whether or not distant metastasis is present")
    treatment_effect: str | None = Field(None, description='check the treatment effect of the cancer. If you see "No known presurgical therapy", return None')

class CervixCancerStagingSchema(BaseModel):
    tnm_descriptor: TnmDescriptorType | None = Field(None, description='identify the tnm descriptor of the tumor. e.g., "y" (post-therapy), "r", etc.')
    pt_category: PtCategoryType | None = Field(None, description="identify the pt category of the tumor")
    pn_category: PnCategoryType | None = Field(None, description="identify the pn category of the tumor")
    pm_category: PmCategoryType | None = Field(None, description="identify the pm category of the tumor. if you see cM0 or cM1, etc., code as mx, since pathological M category is not available")
    stage_group: StageGroupType | None = Field(None, description="identify the FIGO stage group of the tumor")
    ajcc_version: int | None = Field(None, description="identify the ajcc version of the pathological staging")

class CervixCancerMarginsSchema(BaseModel):
    margins: list[CervixMargin] | None = Field(
        None,
        description="return all of the possible involved margins and its distance from cancer. If not present, just output null for every margin"
    )

class CervixCancerLNSchema(BaseModel):
    regional_lymph_node: list[CervixLN] | None = Field(
        None,
        description="return all of the involved regional lymph node. If not present, just output null for every lymph node"
    )
    extranodal_extension: bool | None = Field(None, description="check whether or not extranodal extension is present; if no lymph node metastasis, should be None")
    maximal_ln_size: int | None = Field(None, description="check the maximal size of node metastatic tumor in mm, rounded to integer; if no lymph node metastasis, should be None")
