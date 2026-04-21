# -*- coding: utf-8 -*-
"""
schemas/canonical/colon.py
Canonical Pydantic schema definitions for colorectal cancer registry extraction.
Authoritative data dictionary for ColonCancer* DSPy signatures.
AJCC 8th edition, CAP 4.3.1.0
"""
__version__ = "1.0.0"
__date__ = "2026-04-22"
__author__ = ["Hong-Kai (Walther) Chen", "Po-Yen Tzeng", "Kai-Po Chang"]
__copyright__ = "Copyright 2025, Med NLP Lab, China Medical University"
__license__ = "MIT"
__ajcc_version__ = 8
__cap_version__ = "4.3.1.0"

from pydantic import BaseModel, Field
from typing import Literal

# --- TypeAliases ---

ProcedureType = Literal[
    'right_hemicolectomy', 'extended_right_hemicolectomy', 'left_hemicolectomy',
    'low_anterior_resection', 'anterior_resection', 'abdominoperineal_resection',
    'total_mesorectal_excision', 'total_colectomy', 'subtotal_colectomy',
    'segmental_colectomy', 'transanal_local_excision', 'polypectomy', 'others'
]
SurgicalTechniqueType = Literal["open", "laparoscopic", "robotic", "ta_tme", "hybrid", "others"]
PrimarySiteType = Literal[
    "cecum", "ascending_colon", "hepatic_flexure", "transverse_colon",
    "splenic_flexure", "descending_colon", "sigmoid_colon",
    "rectosigmoid_junction", "rectum", "appendix"
]
HistologyType = Literal[
    "adenocarcinoma", "mucinous_adenocarcinoma", "signet_ring_cell_carcinoma",
    "medullary_carcinoma", "micropapillary_adenocarcinoma",
    "serrated_adenocarcinoma", "adenosquamous_carcinoma",
    "neuroendocrine_carcinoma", "others"
]
TumorInvasionType = Literal[
    "lamina_propria", "submucosa", "muscularis_propria",
    "pericolorectal_tissue", "visceral_peritoneum_surface",
    "adjacent_organs_structures"
]
TypeOfPolypType = Literal[
    "tubular_adenoma", "tubulovillous_adenoma", "villous_adenoma",
    "sessile_serrated_adenoma", "traditional_serrated_adenoma"
]
TnmDescriptorType = Literal['y', 'r', 'm']
PtCategoryType = Literal["tx", "tis", "t1", "t2", "t3", "t4a", "t4b"]
PnCategoryType = Literal["nx", "n0", "n1a", "n1b", "n1c", "n2a", "n2b"]
PmCategoryType = Literal["mx", "m0", "m1a", "m1b", "m1c"]
StageGroupType = Literal[
    "0", "i", "iia", "iib", "iic",
    "iiia", "iiib", "iiic", "iva", "ivb", "ivc"
]
MarginCategoryType = Literal[
    "proximal", "distal", "mesenteric_pedicle",
    "radial_or_circumferencial", "outmost_of_adhered_tissue", "others"
]
LNCategoryType = Literal["regional", "mesenteric", "others"]
BiomarkerCategoryType = Literal["mlh1", "msh2", "msh6", "pms2", "her2", "others"]
HER2ScoreType = Literal[0, 1, 2, 3]

# --- Pydantic nested models ---

class ColonMargin(BaseModel):
    margin_category: MarginCategoryType | None = Field(
        None,
        description="acceptable value for surgical margins in colon cancer. If not included in these standard margins, should be classified as others."
    )
    margin_involved: bool
    distance: int | None = Field(
        None,
        description="If margin is involved, return 0. If margin is uninvolved/free, try your best to find the distance at both microscopic and macroscopic(gross) description, and specify the distance from tumor to margin in mm, rounded to integer. If the margin is uninvolved/free and, after your best effort, the distance is still not specified, return null"
    )
    description: str | None

class ColonLN(BaseModel):
    lymph_node_category: LNCategoryType | None = Field(
        None,
        description="acceptable value for lymph node categories in colon cancer. If not included in these standard lymph node 'station' number, should be classified as others."
    )
    involved: int
    examined: int
    station_name: str | None = Field(None, description="specify the name of the lymph node group/station here.")

class ColonBiomarker(BaseModel):
    biomarker_category: BiomarkerCategoryType | None = Field(
        None,
        description="acceptable value for biomarker categories in colon cancer. If not included in these standard categories, should be classified as others."
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

class ColonCancerNonnestedSchema(BaseModel):
    procedure: ProcedureType | None = Field(None, description="identify which surgery procedure was used. e.g. polypectomy")
    surgical_technique: SurgicalTechniqueType | None = Field(None, description="identify how the surgery was taken. e.g. laparoscopic")
    cancer_primary_site: PrimarySiteType | None = Field(None, description="identify the primary site of cancer. e.g. sigmoid colon")
    histology: HistologyType | None = Field(None, description="identify the histological type of the cancer. e.g. adenocarcinoma, if not in the list, classify as others")
    grade: int | None = Field(None, description="identify the grade of the cancer, well->1, moderate->2, poor->3, undiff-4")
    tumor_invasion: TumorInvasionType | None = Field(None, description="identify the part invasioned by tumor, e.g lamina propria")
    lymphovascular_invasion: bool | None = Field(None, description="check whether or not lymphovascular invasion is present")
    perineural_invasion: bool | None = Field(None, description="check whether or not perineural invasion is present")
    extracellular_mucin: bool | None = Field(None, description="check whether or not extracellular mucin is present")
    signet_ring: bool | None = Field(None, description="check whether or not signet ring cell is present")
    tumor_budding: int | None = Field(None, description="identify the number of tumor budding of the cancer. low->0, moderate->1, high->2")
    type_of_polyp: TypeOfPolypType | None = Field(None, description="identify the type of polyp of the tumor")
    distant_metastasis: bool | None = Field(None, description="check whether or not distant metastasis is present")
    treatment_effect: str | None = Field(None, description='check the treatment effect of the cancer. If you see "No known presurgical therapy", return None')

class ColonCancerStagingSchema(BaseModel):
    tnm_descriptor: TnmDescriptorType | None = Field(None, description='identify the tnm descriptor of the tumor. e.g., "y" means post-therapy, "m" means multiple primary tumor, "r" means recurrent tumor')
    pt_category: PtCategoryType | None = Field(None, description="identify the pt category of the tumor")
    pn_category: PnCategoryType | None = Field(None, description="identify the pn category of the tumor")
    pm_category: PmCategoryType | None = Field(None, description="identify the pm category of the tumor. if you see cM0 or cM1, etc., code as mx, since pathological M category is not available")
    stage_group: StageGroupType | None = Field(None, description="identify the stage group of the tumor")
    ajcc_version: int | None = Field(None, description="identify the ajcc version of the pathological staging")

class ColonCancerMarginsSchema(BaseModel):
    margins: list[ColonMargin] | None = Field(
        None,
        description="return all of the possible involved margins and its distance from cancer. If not present, just output null for every margin"
    )

class ColonCancerLNSchema(BaseModel):
    regional_lymph_node: list[ColonLN] | None = Field(
        None,
        description="return all of the involved regional lymph node. If not present, just output null for every lymph node"
    )
    extranodal_extension: bool | None = Field(None, description="check whether or not extranodal extension is present; if no lymph node metastasis, should be None")
    maximal_ln_size: int | None = Field(None, description="check the maximal size of node metastatic tumor in mm, rounded to integer; if no lymph node metastasis, should be None")

class ColonCancerBiomarkersSchema(BaseModel):
    biomarkers: list[ColonBiomarker] | None = Field(
        None,
        description="return all of the examined immunoreceptors using immunohistochemistry techniques, like mlh1, msh2, etc. If not present, just output null for every biomarker"
    )
