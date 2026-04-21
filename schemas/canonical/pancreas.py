# -*- coding: utf-8 -*-
"""
schemas/canonical/pancreas.py
Canonical Pydantic schema definitions for pancreas cancer registry extraction.
Authoritative data dictionary for PancreasCancer* DSPy signatures.
AJCC 8th edition, CAP 4.2.0.2
"""
__version__ = "1.0.0"
__date__ = "2026-04-22"
__author__ = ["Hong-Kai (Walther) Chen", "Po-Yen Tzeng", "Kai-Po Chang"]
__copyright__ = "Copyright 2025, Med NLP Lab, China Medical University"
__license__ = "MIT"
__ajcc_version__ = 8
__cap_version__ = "4.2.0.2"

from pydantic import BaseModel, Field
from typing import Literal

# --- TypeAliases ---

ProcedureType = Literal[
    'partial_pancreatectomy', 'ssppd', 'pppd', 'whipple_procedure',
    'distal_pancreatectomy', 'total_pancreatectomy', 'others'
]
TumorSiteType = Literal["head", "neck", "body", "tail", "uncinate_process", "others"]
HistologyType = Literal[
    "ductal_adenocarcinoma_nos", "ipmn_with_carcinoma", "itpn_with_carcinoma",
    "acinar_cell_carcinoma", "solid_pseudopapillary_neoplasm",
    "undifferentiated_carcinoma", "others"
]
TumorExtensionType = Literal[
    "within_pancreas", "peripancreatic_soft_tissue",
    "adjacent_organs_structures", "others"
]
TnmDescriptorType = Literal['y', 'r', 'm']
PtCategoryType = Literal["tx", "tis", "t1a", "t1b", "t1c", "t2", "t3", "t4"]
PnCategoryType = Literal["nx", "n0", "n1", "n2"]
PmCategoryType = Literal["mx", "m1"]
OverallStageType = Literal["ia", "ib", "iia", "iib", "iii", "iv"]
MarginCategoryType = Literal[
    "distal_pancreatic", "proximal_pancreatic", "pancreatic_neck", "uncinate",
    "bile_duct", "proximal_gastric", "proximal_duodenal", "distal_intestinal",
    "outmost", "anterior_outmost", "posterior_outmost", "others"
]
LNStationType = Literal[
    "regional_pancreatic", "regional_gastric",
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14",
    "others"
]

# --- Pydantic nested models ---

class PancreasMargin(BaseModel):
    margin_category: MarginCategoryType | None = Field(
        None,
        description="acceptable value for surgical margins in pancreas cancer. If not included in these standard margins, should be classified as others."
    )
    margin_involved: bool
    distance: int | None = Field(
        None,
        description="If margin is involved, return 0. If margin is uninvolved/free, try your best to find the distance at both microscopic and macroscopic(gross) description, and specify the distance from tumor to margin in mm, rounded to integer. If the margin is uninvolved/free and, after your best effort, the distance is still not specified, return null"
    )
    description: str | None

class PancreasLN(BaseModel):
    lymph_node_category: LNStationType | None = Field(
        None,
        description="acceptable value for lymph node categories in pancreas cancer. If not included in these standard lymph node 'station' number, should be classified as others."
    )
    involved: int
    examined: int
    station_name: str | None = Field(None, description="specify the name of the lymph node station/group here.")

# --- Full output schemas (one per DSPy Signature) ---

class PancreasCancerNonnestedSchema(BaseModel):
    procedure: ProcedureType | None = Field(
        None,
        description="identify which surgery procedure was used. e.g. partial pancreatectomy, pylorus-preserving pancreaticoduodenectomy(PPPD), Subtotal stomach-preserving pancreaticoduodenectomy (SSPPD), Whipple procedure, etc."
    )
    tumor_site: TumorSiteType | None = Field(None, description="identify the primary site of cancer. e.g. head of pancreas")
    histology: HistologyType | None = Field(None, description="identify the histological type of the cancer. e.g. invasive carcinoma of no special type")
    tumor_size: int | None = Field(None, description="identify the size of the tumor in mm, rounded, if multiple tumors are present, please provide the size of the largest tumor")
    tumor_extension: TumorExtensionType | None = Field(None, description="identify the extent of tumor extension. e.g. within pancreas")
    lymphovascular_invasion: bool | None = Field(None, description="check whether or not lymphovascular invasion is present")
    perineural_invasion: bool | None = Field(None, description="check whether or not perineural invasion is present")
    distant_metastasis: bool | None = Field(None, description="check whether or not distant metastasis is present")
    treatment_effect: str | None = Field(None, description='check the treatment effect of the cancer. If you see "No known presurgical therapy", return None')

class PancreasCancerStagingSchema(BaseModel):
    tnm_descriptor: TnmDescriptorType | None = Field(None, description='identify the tnm descriptor of the tumor. e.g., "y" (post-therapy), "r", etc.')
    pt_category: PtCategoryType | None = Field(None, description="identify the pt category of the tumor")
    pn_category: PnCategoryType | None = Field(None, description="identify the pn category of the tumor")
    pm_category: PmCategoryType | None = Field(None, description="identify the pm category of the tumor. if you see cM0 or cM1, etc., code as mx, since pathological M category is not available")
    overall_stage: OverallStageType | None = Field(None, description="identify the overall stage of the tumor")
    ajcc_version: int | None = Field(None, description="identify the ajcc version of the pathological staging")

class PancreasCancerMarginsSchema(BaseModel):
    margins: list[PancreasMargin] | None = Field(
        None,
        description="return all of the possible involved margins and its distance from cancer. If not present, just output null for every margin"
    )

class PancreasCancerLNSchema(BaseModel):
    regional_lymph_node: list[PancreasLN] | None = Field(
        None,
        description="return all of the involved regional lymph node. If not present, just output null for every lymph node"
    )
    extranodal_extension: bool | None = Field(None, description="check whether or not extranodal extension is present")
    maximal_ln_size: int | None = Field(None, description="check the maximal size of node metastatic tumor in mm, rounded to integer. if not, return None")
