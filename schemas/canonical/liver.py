# -*- coding: utf-8 -*-
"""
schemas/canonical/liver.py
Canonical Pydantic schema definitions for hepatocellular carcinoma registry extraction.
Authoritative data dictionary for LiverCancer* DSPy signatures (HCC only).
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

ProcedureType = Literal[
    'wedge_resection', 'partial_hepatectomy', 'segmentectomy',
    'lobectomy', 'total_hepatectomy', 'others'
]
TumorSiteType = Literal['right_lobe', 'left_lobe', 'caudate_lobe', 'quadrate_lobe', 'others']
HistologyType = Literal[
    "hepatocellular_carcinoma", "hepatocellular_carcinoma_fibrolamellar",
    "hepatocellular_carcinoma_scirrhous", "hepatocellular_carcinoma_clear_cell", "others"
]
GradeType = Literal[1, 2, 3, 4]
TumorFocalityType = Literal["unifocal", "multifocal"]
TumorExtentType = Literal[
    'hepatic_vein', 'portal_vein', 'visceral_peritoneum',
    'gallbladder', 'diaphragm', 'others'
]
VascularInvasionType = Literal['large_hepatic_vein', 'large_portal_vein', 'small_vessel']
TnmDescriptorType = Literal['y', 'r', 'm']
PtCategoryType = Literal["tx", "t1a", "t1b", "t2", "t3", "t4"]
PnCategoryType = Literal["nx", "n0", "n1"]
PmCategoryType = Literal["mx", "m0", "m1"]
OverallStageType = Literal["ia", "ib", "ii", "iiia", "iiib", "iva", "ivb"]
MarginCategoryType = Literal["parenchymal", "hepatic_vein", "portal_vein", "bile_duct", "others"]

# --- Pydantic nested models ---

class LiverMargin(BaseModel):
    margin_category: MarginCategoryType | None = Field(
        None,
        description="acceptable value for surgical margins in liver cancer. If not included in these standard margins, should be classified as others."
    )
    margin_involved: bool
    distance: int | None = Field(
        None,
        description="If margin is involved, return 0. If margin is uninvolved/free, try your best to find the distance at both microscopic and macroscopic(gross) description, and specify the distance from tumor to margin in mm, rounded to integer. If the margin is uninvolved/free and, after your best effort, the distance is still not specified, return null"
    )
    description: str | None

class LiverLN(BaseModel):
    involved: int
    examined: int
    station_name: str | None = Field(None, description="specify the name of the lymph node station here.")

# --- Full output schemas (one per DSPy Signature) ---

class LiverCancerNonnestedSchema(BaseModel):
    procedure: ProcedureType | None = Field(None, description="identify which surgery procedure was used. e.g. partial hepatectomy")
    tumor_site: TumorSiteType | None = Field(None, description="identify the site of the tumor. e.g. right lobe")
    histology: HistologyType | None = Field(None, description="identify the histological type of the cancer. e.g. invasive carcinoma of no special type")
    grade: GradeType | None = Field(None, description="identify the grade of the cancer, well->1, moderate->2, poor->3, undifferentiated->4")
    tumor_size: int | None = Field(None, description="identify the size of the tumor in mm, rounded, if multiple tumors are present, please provide the size of the largest tumor")
    tumor_focality: TumorFocalityType | None = Field(None, description="identify whether the tumor is unifocal or multifocal")
    perineural_invasion: bool | None = Field(None, description="check whether or not perineural invasion is present")
    distant_metastasis: bool | None = Field(None, description="check whether or not distant metastasis is present")
    treatment_effect: str | None = Field(None, description='check the treatment effect of the cancer. If you see "No known presurgical therapy", return None')

class LiverCancerExtentSchema(BaseModel):
    tumor_extent: list[TumorExtentType] | None = Field(
        None,
        description='return all of the possible tumor extension. example: ["hepatic_vein", "gallbladder"]. If not present, just output None, do not overdiagnosis'
    )

class LiverCancerVascularInvasionSchema(BaseModel):
    vascular_invasion: list[VascularInvasionType] | None = Field(
        None,
        description='return all of the possible vascular invasion. example: ["large_hepatic_vein", "small_vessel"]. If not present, just output None, do not overdiagnosis'
    )

class LiverCancerStagingSchema(BaseModel):
    tnm_descriptor: TnmDescriptorType | None = Field(None, description='identify the tnm descriptor of the tumor. e.g., "y" (post-therapy), "r", etc.')
    pt_category: PtCategoryType | None = Field(None, description="identify the pt category of the tumor")
    pn_category: PnCategoryType | None = Field(None, description="identify the pn category of the tumor")
    pm_category: PmCategoryType | None = Field(None, description="identify the pm category of the tumor. if you see cM0 or cM1, etc., code as mx, since pathological M category is not available")
    overall_stage: OverallStageType | None = Field(None, description="identify the overall stage of the tumor")
    ajcc_version: int | None = Field(None, description="identify the ajcc version of the pathological staging")

class LiverCancerMarginsSchema(BaseModel):
    margins: list[LiverMargin] | None = Field(
        None,
        description="return all of the possible involved margins and its distance from cancer. If not present, just output null for every margin"
    )

class LiverCancerLNSchema(BaseModel):
    regional_lymph_node: list[LiverLN] | None = Field(
        None,
        description="return all of the involved regional lymph node. If not present, just output null for every lymph node"
    )
    extranodal_extension: bool | None = Field(None, description="check whether or not extranodal extension is present; if no lymph node metastasis, should be None")
    maximal_ln_size: int | None = Field(None, description="check the maximal size of node metastatic tumor in mm, rounded to integer; if no lymph node metastasis, should be None")
