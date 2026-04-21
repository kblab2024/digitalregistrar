# -*- coding: utf-8 -*-
"""
schemas/canonical/thyroid.py
Canonical Pydantic schema definitions for thyroid cancer registry extraction.
Authoritative data dictionary for ThyroidCancer* DSPy signatures.
AJCC 8th edition, CAP 4.4.0.0
"""
__version__ = "1.0.0"
__date__ = "2026-04-22"
__author__ = ["Hong-Kai (Walther) Chen", "Po-Yen Tzeng", "Kai-Po Chang"]
__copyright__ = "Copyright 2025, Med NLP Lab, China Medical University"
__license__ = "MIT"
__ajcc_version__ = 8
__cap_version__ = "4.4.0.0"

from pydantic import BaseModel, Field
from typing import Literal

# --- TypeAliases ---

PredisposingConditionType = Literal["radiation", "family_history"]
ProcedureType = Literal[
    'partial_excision', 'right_lobectomy', 'left_lobectomy',
    'total_thyroidectomy', 'others'
]
TumorFocalityType = Literal["unifocal", "multifocal", "not_specified"]
TumorSiteType = Literal["right_lobe", "left_lobe", "isthmus", "both_lobe", "others"]
HistologyType = Literal[
    "papillary_thyroid_carcinoma", "follicular_thyroid_carcinoma",
    "medullary_thyroid_carcinoma", "anaplastic_thyroid_carcinoma", "others"
]
MitoticActivityType = Literal["less_than_3", "3_to_5", "more_than_5"]
ExtrathyroidExtensionType = Literal[
    "microscopic_strap_muscle",
    "macroscopic_strap_muscle_t3b",
    "subcutaneous_trachea_esophagus_rln_t4a",
    "prevertebral_carotid_mediastinal_t4b"
]
TnmDescriptorType = Literal['y', 'r', 'm']
PtCategoryType = Literal["tx", "t1a", "t1b", "t2", "t3a", "t3b", "t4a", "t4b"]
PnCategoryType = Literal["nx", "n0", "n1a", "n1b"]
PmCategoryType = Literal["mx", "m0", "m1"]
OverallStageType = Literal["i", "ii", "iii", "iva", "ivb", "ivc"]
MarginCategoryType = Literal["outmost", "anterior_outmost", "posterior_outmost", "isthmus", "others"]
LNSideType = Literal["right", "left", "midline"]
LNCategoryType = Literal[
    "level_vi_central", "level_i", "level_ii", "level_iii",
    "level_iv", "level_v", "level_vii", "others"
]

# --- Pydantic nested models ---

class ThyroidMargin(BaseModel):
    margin_category: MarginCategoryType | None = Field(
        None,
        description="acceptable value for surgical margins in thyroid cancer. If not included in these standard margins, should be classified as others."
    )
    margin_involved: bool
    distance: int | None = Field(
        None,
        description="If margin is involved, return 0. If margin is uninvolved/free, try your best to find the distance at both microscopic and macroscopic(gross) description, and specify the distance from tumor to margin in mm, rounded to integer. If the margin is uninvolved/free and, after your best effort, the distance is still not specified, return null"
    )
    description: str | None

class ThyroidLN(BaseModel):
    lymph_node_side: LNSideType | None = Field(
        None,
        description="acceptable value for lymph node side in thyroid cancer. If not included in these standard sides, should be classified as None."
    )
    lymph_node_category: LNCategoryType | None = Field(
        None,
        description="acceptable value for lymph node categories in thyroid cancer. If not included in these standard lymph node 'station' number, should be classified as others."
    )
    involved: int
    examined: int
    station_name: str | None = Field(None, description="specify the name of the lymph node station here.")

# --- Full output schemas (one per DSPy Signature) ---

class ThyroidCancerNonnestedSchema(BaseModel):
    predisposing_condition: PredisposingConditionType | None = Field(
        None,
        description='identify any predisposing condition mentioned in the report, such as "Hashimoto thyroiditis" or "radiation exposure". If none mentioned, return None'
    )
    procedure: ProcedureType | None = Field(None, description="identify which surgery procedure was used. e.g. partial thyroidectomy")
    tumor_focality: TumorFocalityType | None = Field(None, description="identify whether the tumor is unifocal or multifocal. If not specified, return not_specified")
    tumor_site: TumorSiteType | None = Field(None, description="identify the primary site of cancer. e.g. right lobe of thyroid")
    histology: HistologyType | None = Field(None, description="identify the histological type of the cancer. e.g. papillary thyroid carcinoma")
    tumor_size: int | None = Field(None, description="identify the size of the tumor in mm, rounded, if multiple tumors are present, please provide the size of the largest tumor")
    mitotic_activity: MitoticActivityType | None = Field(None, description="identify the mitotic activity of the tumor, in mitoses per 10 high power fields. if not specified, return null")
    extrathyroid_extension: ExtrathyroidExtensionType | None = Field(None, description="identify if clinically extrathyroid extension is present, only four categories are allowed, if none specified, return None")
    tumor_necrosis: bool | None = Field(None, description="check whether or not tumor necrosis is present")
    lymphovascular_invasion: bool | None = Field(None, description="check whether or not lymphovascular invasion is present")
    perineural_invasion: bool | None = Field(None, description="check whether or not perineural invasion is present")
    distant_metastasis: bool | None = Field(None, description="check whether or not distant metastasis is present")
    treatment_effect: str | None = Field(None, description='check the treatment effect of the cancer. If you see "No known presurgical therapy", return None')

class ThyroidCancerStagingSchema(BaseModel):
    tnm_descriptor: TnmDescriptorType | None = Field(None, description='identify the tnm descriptor of the tumor. e.g., "y" (post-therapy), "r", etc.')
    pt_category: PtCategoryType | None = Field(None, description="identify the pt category of the tumor")
    pn_category: PnCategoryType | None = Field(None, description="identify the pn category of the tumor")
    pm_category: PmCategoryType | None = Field(None, description="identify the pm category of the tumor. if you see cM0 or cM1, etc., code as mx, since pathological M category is not available")
    overall_stage: OverallStageType | None = Field(None, description="identify the overall stage of the tumor")
    ajcc_version: int | None = Field(None, description="identify the ajcc version of the pathological staging")

class ThyroidCancerMarginsSchema(BaseModel):
    margins: list[ThyroidMargin] | None = Field(
        None,
        description="return all of the possible involved margins and its distance from cancer. If not present, just output null for every margin"
    )

class ThyroidCancerLNSchema(BaseModel):
    regional_lymph_node: list[ThyroidLN] | None = Field(
        None,
        description="return all of the involved regional lymph node. If not present, just output null for every lymph node"
    )
    extranodal_extension: bool | None = Field(None, description="check whether or not extranodal extension is present; if no lymph node metastasis, should be None")
    maximal_ln_size: int | None = Field(None, description="check the maximal size of node metastatic tumor in mm, rounded to integer; if no lymph node metastasis, should be None")
