# -*- coding: utf-8 -*-
"""
schemas/canonical/lung.py
Canonical Pydantic schema definitions for lung cancer registry extraction.
Authoritative data dictionary for LungCancer* DSPy signatures.
AJCC 8th edition, CAP 4.3.0.1
"""
__version__ = "1.0.0"
__date__ = "2026-04-22"
__author__ = ["Hong-Kai (Walther) Chen", "Po-Yen Tzeng", "Kai-Po Chang"]
__copyright__ = "Copyright 2025, Med NLP Lab, China Medical University"
__license__ = "MIT"
__ajcc_version__ = 8
__cap_version__ = "4.3.0.1"

from pydantic import BaseModel, Field
from typing import Literal

# --- TypeAliases ---

ProcedureType = Literal[
    'wedge_resection', 'segmentectomy', 'lobectomy', 'completion_lobectomy',
    'sleeve_lobectomy', 'bilobectomy', 'pneumonectomy', 'major_airway_resection', 'others'
]
SurgicalTechniqueType = Literal["open", "thoracoscopic", "robotic", "hybrid", "others"]
SidenessType = Literal['right', 'left', 'midline']
PrimarySiteType = Literal[
    'upper_lobe', 'middle_lobe', 'lower_lobe',
    'main_bronchus', 'bronchus_intermedius', 'bronchus_lobar', 'others'
]
TumorFocalityType = Literal[
    'single_focus', 'separate_in_same_lobe_t3',
    'separate_nodule_in_ipsilateral_t4', 'separate_nodule_in_contralateral_m1a'
]
HistologyType = Literal[
    "adenocarcinoma", "squamous_cell_carcinoma", "adenosquamous_carcinoma",
    "large_cell_carcinoma", "large_cell_neuroendocrine_carcinoma",
    "small_cell_carcinoma", "carcinoid_tumor", "sarcomatoid_carcinoma",
    "pleomorphic_carcinoma", "pulmonary_lymphoepithelioma_like_carcinoma",
    "mucoepidermoid_carcinoma", "salivary_gland_type_tumor",
    "non_small_cell_carcinoma_not_specified",
    "non_small_cell_carcinoma_with_neuroendocrine_features", "other"
]
TnmDescriptorType = Literal['y', 'r', 'm']
PtCategoryType = Literal["tx", "tis", "t1mi", "t1a", "t1b", "t1c", "t2a", "t2b", "t3", "t4"]
PnCategoryType = Literal["nx", "n0", "n1", "n2", "n3"]
PmCategoryType = Literal["mx", "m0", "m1a", "m1b", "m1c"]
StageGroupType = Literal[
    "0", "ia1", "ia2", "ia3", "ib", "iia", "iib",
    "iiia", "iiib", "iiic", "iva", "ivb", "ivc"
]
MarginCategoryType = Literal["bronchial", "vascular", "parenchymal", "chest_wall", "others"]
LNSideType = Literal["right", "left"]
LNCategoryType = Literal[
    "peribronchial", "1", "2", "4", "5", "6", "8", "9", "10", "11", "12",
    "13", "14", "3a", "3p", "7", "others"
]
PatternNameType = Literal["acinar", "lepidic", "papillary", "solid", "micropapillary", "others"]
BiomarkerCategoryType = Literal["ALK", "ROS1", "PDL1", "others"]

# --- Pydantic nested models ---

class LungMargin(BaseModel):
    margin_category: MarginCategoryType | None = Field(
        None,
        description="acceptable value for surgical margins in lung cancer. If not included in these standard margins, should be classified as others."
    )
    margin_involved: bool
    distance: int | None = Field(
        None,
        description="If margin is involved, return 0. If margin is uninvolved/free, try your best to find the distance at both microscopic and macroscopic(gross) description, and specify the distance from tumor to margin in mm, rounded to integer. If the margin is uninvolved/free and, after your best effort, the distance is still not specified, return null"
    )
    description: str | None

class LungLN(BaseModel):
    lymph_node_side: LNSideType | None = Field(
        None,
        description="acceptable value for lymph node side in lung cancer. If not included in these standard sides, should be classified as None."
    )
    lymph_node_category: LNCategoryType | None = Field(
        None,
        description="acceptable value for lymph node categories (i.e. stations or groups) in lung cancer. If not included in these standard lymph node 'stations'/'groups' should be classified as others."
    )
    involved: int
    examined: int
    station_name: str | None = Field(None, description="specify the name of the lymph node station/group here.")

class LungHistologicalPattern(BaseModel):
    pattern_name: PatternNameType | None = Field(
        None,
        description="histological pattern of invasive non-mucinous adenocarcinoma"
    )
    pattern_percentage: int | None = Field(None, description="percentage of the histological pattern")

class LungBiomarker(BaseModel):
    biomarker_category: BiomarkerCategoryType | None = Field(
        None,
        description="acceptable value for biomarker categories in lung cancer. If not included in these standard categories, should be classified as others."
    )
    expression: bool
    percentage: int | None = Field(
        None,
        description="the percentage of tumor cells showing positive expression of the biomarker, rounded to integer. if not specified, return None"
    )
    biomarker_name: str | None = Field(None, description="specify the name of the biomarker here.")

# --- Full output schemas (one per DSPy Signature) ---

class LungCancerNonnestedSchema(BaseModel):
    procedure: ProcedureType | None = Field(None, description="identify which surgery procedure was used. e.g. polypectomy")
    surgical_technique: SurgicalTechniqueType | None = Field(None, description="identify how the surgery was taken. e.g. thoracoscopic")
    sideness: SidenessType | None = Field(None, description="identify which side the tumor is located. e.g. right")
    cancer_primary_site: PrimarySiteType | None = Field(None, description="identify the primary site of cancer. e.g. upper_lobe")
    tumor_focality: TumorFocalityType | None = Field(None, description="identify whether the tumor is single focus or multiple foci; this is important for t and m category")
    histology: HistologyType | None = Field(None, description="identify the histological type of the cancer. e.g. adenocarcinoma")
    grade: int | None = Field(None, description="identify the grade of the cancer, well->1, moderate->2, poor->3, undiff->4")
    lymphovascular_invasion: bool | None = Field(None, description="check whether or not lymphovascular invasion is present")
    perineural_invasion: bool | None = Field(None, description="check whether or not perineural invasion is present")
    spread_through_air_spaces_stas: bool | None = Field(None, description="check whether or not spread through air spaces (STAS) is present")
    visceral_pleural_invasion: bool | None = Field(None, description="check whether or not visceral pleural invasion is present")
    direct_invasion_of_adjacent_structures: bool | None = Field(None, description="check whether or not direct invasion of adjacent structures is present")
    distant_metastasis: bool | None = Field(None, description="check whether or not distant metastasis is present")
    treatment_effect: str | None = Field(None, description='check the treatment effect of the cancer. If you see "No known presurgical therapy", return None')

class LungCancerStagingSchema(BaseModel):
    tnm_descriptor: TnmDescriptorType | None = Field(None, description='identify the tnm descriptor of the tumor. e.g., "y" (post-therapy), "r", etc.')
    pt_category: PtCategoryType | None = Field(None, description="identify the pt category of the tumor")
    pn_category: PnCategoryType | None = Field(None, description="identify the pn category of the tumor")
    pm_category: PmCategoryType | None = Field(None, description="identify the pm category of the tumor. if you see cM0 or cM1, etc., code as mx, since pathological M category is not available")
    stage_group: StageGroupType | None = Field(None, description="identify the stage group of the tumor")
    ajcc_version: int | None = Field(None, description="identify the ajcc version of the pathological staging")

class LungCancerMarginsSchema(BaseModel):
    margins: list[LungMargin] | None = Field(
        None,
        description="return all of the possible involved margins and its distance from cancer. If not present, just output null for every margin"
    )

class LungCancerLNSchema(BaseModel):
    regional_lymph_node: list[LungLN] | None = Field(
        None,
        description="return all of the involved regional lymph node. If not present, just output null for every lymph node"
    )
    extranodal_extension: bool | None = Field(None, description="check whether or not extranodal extension is present; if no lymph node metastasis, should be None")
    maximal_ln_size: int | None = Field(None, description="check the maximal size of node metastatic tumor in mm, rounded to integer; if no lymph node metastasis, should be None")

class LungCancerBiomarkersSchema(BaseModel):
    biomarkers: list[LungBiomarker] | None = Field(
        None,
        description="return all of the examined biomarkers using immunohistochemistry techniques, like alk, ros1, pd-l1. If not present, just output None for every biomarker"
    )

class LungCancerOthernestedSchema(BaseModel):
    histological_patterns: list[LungHistologicalPattern] | None = Field(
        None,
        description="if the histology is invasive non-mucinous adenocarcinoma, please identify all of the histological patterns and its percentage. if not, return None"
    )
