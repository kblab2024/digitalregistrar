# -*- coding: utf-8 -*-
"""
schemas/canonical/esophagus.py
Canonical Pydantic schema definitions for esophageal cancer registry extraction.
Authoritative data dictionary for EsophagusCancer* DSPy signatures.
AJCC 8th edition, CAP 4.2.0.1
"""
__version__ = "1.0.0"
__date__ = "2026-04-22"
__author__ = ["Hong-Kai (Walther) Chen", "Po-Yen Tzeng", "Kai-Po Chang"]
__copyright__ = "Copyright 2025, Med NLP Lab, China Medical University"
__license__ = "MIT"
__ajcc_version__ = 8
__cap_version__ = "4.2.0.1"

from pydantic import BaseModel, Field
from typing import Literal

# --- TypeAliases ---

ProcedureType = Literal['endoscopic_resection', 'esophagectomy', 'esophagogastrectomy', 'others']
SurgicalTechniqueType = Literal["open", "thoracoscopic", "robotic", "hybrid", "endoscopic", "others"]
PrimarySiteType = Literal['upper_third', 'middle_third', 'lower_third', 'gastroesophageal_junction']
HistologyType = Literal[
    "squamous_cell_carcinoma", "adenocarcinoma", "adenoid_cystic_carcinoma",
    "mucoepidermoid_carcinoma", "basaloid_squamous_cell_carcinoma",
    "small_cell_carcinoma", "large_cell_carcinoma", "others"
]
GradeType = Literal[1, 2, 3]
TumorExtentType = Literal[
    'mucosa', 'submucosa', 'muscularis_propria', 'adventitia', 'adjacent_structures'
]
TnmDescriptorType = Literal['y', 'r', 'm']
PtCategoryType = Literal["tx", "t1a", "t1b", "t2", "t3", "t4a", "t4b"]
PnCategoryType = Literal["nx", "n0", "n1", "n2", "n3"]
PmCategoryType = Literal["mx", "m0", "m1"]
StageGroupType = Literal["0", "i", "ia", "ib", "ic", "iia", "iib", "iiia", "iiib", "iva", "ivb"]
MarginCategoryType = Literal["proximal", "distal", "radial", "lateral", "deep", "others"]
LNCategoryType = Literal[
    "regional_esophageal", "regional_gastric",
    "thoracic_1", "thoracic_1r", "thoracic_1l",
    "thoracic_4", "thoracic_4r", "thoracic_4l", "thoracic_7",
    "thoracic_8u", "thoracic_8m", "thoracic_8l", "thoracic_8",
    "thoracic_9", "thoracic_9r", "thoracic_9l",
    "thoracic_10", "thoracic_10r", "thoracic_10l",
    "abdomen_106", "abdomen_1", "abdomen_2", "abdomen_3", "abdomen_4",
    "abdomen_5", "abdomen_6", "abdomen_7", "abdomen_8", "abdomen_9",
    "abdomen_10", "others"
]

# --- Pydantic nested models ---

class EsophagusMargin(BaseModel):
    margin_category: MarginCategoryType | None = Field(
        None,
        description="acceptable value for surgical margins in esophageal cancer. If not included in these standard margins, should be classified as others."
    )
    margin_involved: bool
    distance: int | None = Field(
        None,
        description="If margin is involved, return 0. If margin is uninvolved/free, try your best to find the distance at both microscopic and macroscopic(gross) description, and specify the distance from tumor to margin in mm, rounded to integer. If the margin is uninvolved/free and, after your best effort, the distance is still not specified, return null"
    )
    description: str | None

class EsophagusLN(BaseModel):
    lymph_node_category: LNCategoryType | None = Field(
        None,
        description="acceptable value for lymph node categories (i.e. stations or groups) in esophageal cancer. Default to thoracic if not mentioned. If not included in these standard lymph node 'station' number, should be classified as others."
    )
    involved: int
    examined: int
    station_name: str | None = Field(None, description="specify the name of the lymph node station/group here.")

# --- Full output schemas (one per DSPy Signature) ---

class EsophagusCancerNonnestedSchema(BaseModel):
    procedure: ProcedureType | None = Field(None, description="identify which surgery procedure was used. e.g. polypectomy")
    surgical_technique: SurgicalTechniqueType | None = Field(None, description="identify how the surgery was taken. e.g. thoracoscopic")
    cancer_primary_site: PrimarySiteType | None = Field(None, description="identify the primary site of cancer. e.g. upper_third")
    histology: HistologyType | None = Field(None, description="identify the histological type of the cancer. e.g. squamous cell carcinoma")
    grade: GradeType | None = Field(None, description="identify the grade of the cancer, well->1, moderate->2, poor->3")
    tumor_extent: TumorExtentType | None = Field(None, description="identify how deep the tumor invades. e.g. muscularis_propria")
    lymphovascular_invasion: bool | None = Field(None, description="check whether or not lymphovascular invasion is present")
    perineural_invasion: bool | None = Field(None, description="check whether or not perineural invasion is present")
    distant_metastasis: bool | None = Field(None, description="check whether or not distant metastasis is present")
    treatment_effect: str | None = Field(None, description='check the treatment effect of the cancer. If you see "No known presurgical therapy", return None')

class EsophagusCancerStagingSchema(BaseModel):
    tnm_descriptor: TnmDescriptorType | None = Field(None, description='identify the tnm descriptor of the tumor. e.g., "y" (post-therapy), "r", etc.')
    pt_category: PtCategoryType | None = Field(None, description="identify the pt category of the tumor")
    pn_category: PnCategoryType | None = Field(None, description="identify the pn category of the tumor")
    pm_category: PmCategoryType | None = Field(None, description="identify the pm category of the tumor. if you see cM0 or cM1, etc., code as mx, since pathological M category is not available")
    stage_group: StageGroupType | None = Field(None, description="identify the stage group of the tumor")
    ajcc_version: int | None = Field(None, description="identify the ajcc version of the pathological staging")

class EsophagusCancerMarginsSchema(BaseModel):
    margins: list[EsophagusMargin] | None = Field(
        None,
        description="return all of the possible involved margins and its distance from cancer. If not present, just output null for every margin"
    )

class EsophagusCancerLNSchema(BaseModel):
    regional_lymph_node: list[EsophagusLN] | None = Field(
        None,
        description="return all of the involved regional lymph node. If not present, just output null for every lymph node"
    )
    extranodal_extension: bool | None = Field(None, description="check whether or not extranodal extension is present; if no lymph node metastasis, should be None")
    maximal_ln_size: int | None = Field(None, description="check the maximal size of node metastatic tumor in mm, rounded to integer; if no lymph node metastasis, should be None")
