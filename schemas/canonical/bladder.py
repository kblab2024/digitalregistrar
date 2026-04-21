# -*- coding: utf-8 -*-
"""
schemas/canonical/bladder.py
Canonical Pydantic schema definitions for bladder cancer registry extraction.
Authoritative data dictionary for BladderCancer* DSPy signatures.
AJCC 8th edition, CAP 4.2.0.0
"""
__version__ = "1.0.0"
__date__ = "2026-04-22"
__author__ = ["Hong-Kai (Walther) Chen", "Po-Yen Tzeng", "Kai-Po Chang"]
__copyright__ = "Copyright 2025, Med NLP Lab, China Medical University"
__license__ = "MIT"
__ajcc_version__ = 8
__cap_version__ = "4.2.0.0"

from pydantic import BaseModel, Field
from typing import Literal

# --- TypeAliases ---

ProcedureType = Literal[
    'partial_cystectomy', 'radical_cystectomy',
    'radical_cystoprostatectomy', 'others'
]
TumorSiteType = Literal[
    "trigone", "dome", "anterior_wall", "posterior_wall",
    "right_lateral_wall", "left_lateral_wall",
    "bladder_neck", "ureteral_orifice", "others"
]
HistologyType = Literal[
    "noninvasive_papillary", "invasive_urothelial_carcinoma",
    "squamous_cell_carcinoma", "adenocarcinoma"
]
MarginCategoryType = Literal[
    "right_ureteral", "left_ureteral", "urethral", "outmost", "others"
]
LNSideType = Literal["right", "left", "midline", "side_not_specified"]
LNCategoryType = Literal["sentinel", "nonsentinel", "others"]

# --- Pydantic nested models ---

class BladderMargin(BaseModel):
    margin_category: MarginCategoryType | None = Field(
        None,
        description="acceptable value for surgical margins in bladder cancer. If not included in these standard margins, should be classified as others."
    )
    margin_involved: bool
    distance: int | None = Field(
        None,
        description="the distance of the margin from the tumor in mm, rounded to integer. if margin is involved, return 0. if not specified, return null"
    )
    description: str | None

class BladderLN(BaseModel):
    lymph_node_side: LNSideType | None = Field(
        None,
        description="acceptable value for lymph node side in bladder cancer. If not included in these standard sides, should be classified as side_not_specified."
    )
    lymph_node_category: LNCategoryType | None = Field(
        None,
        description="acceptable value for lymph node categories (i.e. stations) in bladder cancer. If not included in these standard lymph node 'station' number, should be classified as others."
    )
    involved: int
    examined: int
    station_name: str | None = Field(None, description="specify the name of the lymph node station here.")

# --- Full output schemas (one per DSPy Signature) ---

class BladderCancerNonnestedSchema(BaseModel):
    procedure: ProcedureType | None = Field(None, description="identify which surgery procedure was used. e.g. partial cystectomy")
    tumor_site: TumorSiteType | None = Field(None, description="identify the primary site of cancer. e.g. trigone")
    histology: HistologyType | None = Field(None, description="identify the histological type of the cancer. e.g. invasive urothelial carcinoma")
    tumor_size: int | None = Field(None, description="identify the size of the tumor in mm, rounded, if multiple tumors are present, please provide the size of the largest tumor")
    lymphovascular_invasion: bool | None = Field(None, description="check whether or not lymphovascular invasion is present")
    perineural_invasion: bool | None = Field(None, description="check whether or not perineural invasion is present")
    distant_metastasis: bool | None = Field(None, description="check whether or not distant metastasis is present")
    treatment_effect: str | None = Field(None, description='check the treatment effect of the cancer. If you see "No known presurgical therapy", return None')
