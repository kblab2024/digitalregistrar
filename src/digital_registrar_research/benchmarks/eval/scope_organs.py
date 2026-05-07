"""
Per-organ option lists for categorical fields, boolean flags, and
numeric span fields.

Derived directly from the canonical JSON schemas in
`digital_registrar_research.schemas.data.*.json` (which are in turn
generated from the DSPy signatures in `...models/`). Keep this file in
sync when adding fields or broadening enums — regenerate cues are:

    python -m digital_registrar_research.schemas.generate

The three public maps — `ORGAN_CATEGORICAL`, `ORGAN_BOOL`,
`ORGAN_SPAN` — are consumed by `scope.py` to back the
`get_allowed_values` / `get_bool_fields` / `get_span_fields`
accessors.

Integer-enum fields are stringified (e.g. grade `1,2,3` → `"1","2","3"`)
to match the normalisation rule in `eval/metrics.py:normalize`, which
lowercases and stringifies every gold/prediction value before equality
comparison.
"""
from __future__ import annotations

# --- Categorical option lists -------------------------------------------------

ORGAN_CATEGORICAL: dict[str, dict[str, list[str]]] = {
    "breast": {
        "tnm_descriptor": ["y", "r", "m"],
        "pt_category": [
            "tx", "tis", "t1mi", "t1a", "t1b", "t1c",
            "t2", "t3", "t4a", "t4b", "t4c",
        ],
        "pn_category": [
            "nx", "n0", "n1mi", "n1a", "n1b", "n1c",
            "n2a", "n2b", "n3a", "n3b", "n3c",
        ],
        "pm_category": ["mx", "m0", "m1"],
        "pathologic_stage_group": [
            "0", "ia", "ib", "iia", "iib",
            "iiia", "iiib", "iiic", "iv",
        ],
        "anatomic_stage_group": [
            "0", "ia", "ib", "iia", "iib",
            "iiia", "iiib", "iiic", "iv",
        ],
        "grade": ["1", "2", "3"],
        "nuclear_grade": ["1", "2", "3"],
        "tubule_formation": ["1", "2", "3"],
        "mitotic_rate": ["1", "2", "3"],
        "total_score": ["3", "4", "5", "6", "7", "8", "9"],
        "dcis_grade": ["1", "2", "3"],
        "cancer_clock": [str(i) for i in range(1, 13)],
        "cancer_laterality": ["right", "left", "bilateral"],
        "cancer_quadrant": [
            "upper_outer_quadrant", "upper_inner_quadrant",
            "lower_outer_quadrant", "lower_inner_quadrant",
            "nipple", "others",
        ],
        "histology": [
            "invasive_carcinoma_no_special_type",
            "invasive_lobular_carcinoma",
            "mixed_ductal_and_lobular_carcinoma",
            "tubular_adenocarcinoma", "mucinous_adenocarcinoma",
            "encapsulated_papillary_carcinoma",
            "solid_papillary_carcinoma", "inflammatory_carcinoma",
            "other_special_types",
        ],
        "procedure": [
            "partial_mastectomy", "simple_mastectomy",
            "breast_conserving_surgery", "modified_radical_mastectomy",
            "total_mastectomy", "wide_excision", "others",
        ],
    },

    "lung": {
        "tnm_descriptor": ["y", "r", "m"],
        "pt_category": [
            "tx", "tis", "t1mi", "t1a", "t1b", "t1c",
            "t2a", "t2b", "t3", "t4",
        ],
        "pn_category": ["nx", "n0", "n1", "n2", "n3"],
        "pm_category": ["mx", "m0", "m1a", "m1b", "m1c"],
        "stage_group": [
            "0", "ia1", "ia2", "ia3", "ib", "iia", "iib",
            "iiia", "iiib", "iiic", "iva", "ivb", "ivc",
        ],
        "cancer_primary_site": [
            "upper_lobe", "middle_lobe", "lower_lobe",
            "main_bronchus", "bronchus_intermedius",
            "bronchus_lobar", "others",
        ],
        "sideness": ["right", "left", "midline"],
        "histology": [
            "adenocarcinoma", "squamous_cell_carcinoma",
            "adenosquamous_carcinoma", "large_cell_carcinoma",
            "large_cell_neuroendocrine_carcinoma",
            "small_cell_carcinoma", "carcinoid_tumor",
            "sarcomatoid_carcinoma", "pleomorphic_carcinoma",
            "pulmonary_lymphoepithelioma_like_carcinoma",
            "mucoepidermoid_carcinoma", "salivary_gland_type_tumor",
            "non_small_cell_carcinoma_not_specified",
            "non_small_cell_carcinoma_with_neuroendocrine_features",
            "other",
        ],
        "tumor_focality": [
            "single_focus", "separate_in_same_lobe_t3",
            "separate_nodule_in_ipsilateral_t4",
            "separate_nodule_in_contralateral_m1a",
        ],
        "procedure": [
            "wedge_resection", "segmentectomy", "lobectomy",
            "completion_lobectomy", "sleeve_lobectomy",
            "bilobectomy", "pneumonectomy",
            "major_airway_resection", "others",
        ],
        "surgical_technique": [
            "open", "thoracoscopic", "robotic", "hybrid", "others",
        ],
    },

    "colorectal": {
        "tnm_descriptor": ["y", "r", "m"],
        "pt_category": ["tx", "tis", "t1", "t2", "t3", "t4a", "t4b"],
        "pn_category": ["nx", "n0", "n1a", "n1b", "n1c", "n2a", "n2b"],
        "pm_category": ["mx", "m0", "m1a", "m1b", "m1c"],
        "stage_group": [
            "0", "i", "iia", "iib", "iic",
            "iiia", "iiib", "iiic",
            "iva", "ivb", "ivc",
        ],
        "cancer_primary_site": [
            "cecum", "ascending_colon", "hepatic_flexure",
            "transverse_colon", "splenic_flexure",
            "descending_colon", "sigmoid_colon",
            "rectosigmoid_junction", "rectum", "appendix",
        ],
        "histology": [
            "adenocarcinoma", "mucinous_adenocarcinoma",
            "signet_ring_cell_carcinoma", "medullary_carcinoma",
            "micropapillary_adenocarcinoma", "serrated_adenocarcinoma",
            "adenosquamous_carcinoma", "neuroendocrine_carcinoma",
            "others",
        ],
        "tumor_invasion": [
            "lamina_propria", "submucosa", "muscularis_propria",
            "pericolorectal_tissue", "visceral_peritoneum_surface",
            "adjacent_organs_structures",
        ],
        "type_of_polyp": [
            "tubular_adenoma", "tubulovillous_adenoma", "villous_adenoma",
            "sessile_serrated_adenoma", "traditional_serrated_adenoma",
        ],
        "procedure": [
            "right_hemicolectomy", "extended_right_hemicolectomy",
            "left_hemicolectomy", "low_anterior_resection",
            "anterior_resection", "abdominoperineal_resection",
            "total_mesorectal_excision", "total_colectomy",
            "subtotal_colectomy", "segmental_colectomy",
            "transanal_local_excision", "polypectomy", "others",
        ],
        "surgical_technique": [
            "open", "laparoscopic", "robotic",
            "ta_tme", "hybrid", "others",
        ],
    },

    "prostate": {
        "tnm_descriptor": ["y", "r", "m"],
        "pt_category": ["tx", "t2", "t3a", "t3b", "t4"],
        "pn_category": ["nx", "n0", "n1"],
        "pm_category": ["mx", "m0", "m1a", "m1b", "m1c"],
        "stage_group": [
            "0", "i", "iia", "iib", "iic",
            "iiia", "iiib", "iiic", "iva", "ivb",
        ],
        "grade": [
            "group_1_3_3", "group_2_3_4", "group_3_4_3",
            "group_4_4_4", "group_5_4_5",
            "group_5_5_4", "group_5_5_5",
        ],
        "histology": [
            "acinar_adenocarcinoma", "intraductal_carcinoma",
            "ductal_adenocarcinoma", "mixed_acinar_ductal",
            "neuroendocrine_carcinoma_small_cell", "others",
        ],
        "margin_length": ["limited", "non_limited"],
        "procedure": ["radical_prostatectomy", "others"],
        "surgical_technique": ["open", "robotic", "hybrid", "others"],
    },

    "esophagus": {
        "tnm_descriptor": ["y", "r", "m"],
        "pt_category": ["tx", "t1a", "t1b", "t2", "t3", "t4a", "t4b"],
        "pn_category": ["nx", "n0", "n1", "n2", "n3"],
        "pm_category": ["mx", "m0", "m1"],
        "stage_group": [
            "0", "i", "ia", "ib", "ic", "iia", "iib",
            "iiia", "iiib", "iva", "ivb",
        ],
        "grade": ["1", "2", "3"],
        "cancer_primary_site": [
            "upper_third", "middle_third", "lower_third",
            "gastroesophageal_junction",
        ],
        "histology": [
            "squamous_cell_carcinoma", "adenocarcinoma",
            "adenoid_cystic_carcinoma", "mucoepidermoid_carcinoma",
            "basaloid_squamous_cell_carcinoma",
            "small_cell_carcinoma", "large_cell_carcinoma", "others",
        ],
        "tumor_extent": [
            "mucosa", "submucosa", "muscularis_propria",
            "adventitia", "adjacent_structures",
        ],
        "procedure": [
            "endoscopic_resection", "esophagectomy",
            "esophagogastrectomy", "others",
        ],
        "surgical_technique": [
            "open", "thoracoscopic", "robotic",
            "hybrid", "endoscopic", "others",
        ],
    },

    "stomach": {
        "tnm_descriptor": ["y", "r", "m"],
        "pt_category": ["tx", "t1a", "t1b", "t2", "t3", "t4a", "t4b"],
        "pn_category": ["nx", "n0", "n1", "n2", "n3a", "n3b"],
        "pm_category": ["mx", "m0", "m1"],
        "stage_group": [
            "0", "i", "ii", "iii", "iv",
            "ia", "ib", "iia", "iib",
            "iiia", "iiib", "iiic",
        ],
        "grade": ["1", "2", "3"],
        "cancer_primary_site": [
            "cardia", "fundus", "body", "antrum", "pylorus", "others",
        ],
        "histology": [
            "tubular_adenocarcinoma", "poorly_cohesive_carcinoma",
            "mixed_tubular_poorly_cohesive", "mucinous_adenocarcinoma",
            "mixed_mucinous_poorly_cohesive",
            "hepatoid_carcinoma", "others",
        ],
        "tumor_extent": [
            "lamina_propria", "muscularis_mucosae", "submucosa",
            "muscularis_propria",
            "penetrate_subserosal_connective_tissue_no_serosa",
            "invades_serosa_without_adjacent_structure_invasion",
            "invades_adjacent_structures",
        ],
        "procedure": [
            "endoscopic_resection", "partial_gastrectomy",
            "total_gastrectomy", "others",
        ],
        "surgical_technique": [
            "open", "laparoscopic", "robotic", "hybrid", "others",
        ],
    },

    "pancreas": {
        "tnm_descriptor": ["y", "r", "m"],
        "pt_category": ["tx", "tis", "t1a", "t1b", "t1c", "t2", "t3", "t4"],
        "pn_category": ["nx", "n0", "n1", "n2"],
        "pm_category": ["mx", "m1"],
        "overall_stage": ["ia", "ib", "iia", "iib", "iii", "iv"],
        "histology": [
            "ductal_adenocarcinoma_nos", "ipmn_with_carcinoma",
            "itpn_with_carcinoma", "acinar_cell_carcinoma",
            "solid_pseudopapillary_neoplasm",
            "undifferentiated_carcinoma", "others",
        ],
        "tumor_site": [
            "head", "neck", "body", "tail",
            "uncinate_process", "others",
        ],
        "tumor_extension": [
            "within_pancreas", "peripancreatic_soft_tissue",
            "adjacent_organs_structures", "others",
        ],
        "procedure": [
            "partial_pancreatectomy", "ssppd", "pppd",
            "whipple_procedure", "distal_pancreatectomy",
            "total_pancreatectomy", "others",
        ],
    },

    "thyroid": {
        "tnm_descriptor": ["y", "r", "m"],
        "pt_category": ["tx", "t1a", "t1b", "t2", "t3a", "t3b", "t4a", "t4b"],
        "pn_category": ["nx", "n0", "n1a", "n1b"],
        "pm_category": ["mx", "m0", "m1"],
        "overall_stage": ["i", "ii", "iii", "iva", "ivb", "ivc"],
        "extrathyroid_extension": [
            "microscopic_strap_muscle",
            "macroscopic_strap_muscle_t3b",
            "subcutaneous_trachea_esophagus_rln_t4a",
            "prevertebral_carotid_mediastinal_t4b",
        ],
        "histology": [
            "papillary_thyroid_carcinoma",
            "follicular_thyroid_carcinoma",
            "medullary_thyroid_carcinoma",
            "anaplastic_thyroid_carcinoma", "others",
        ],
        "mitotic_activity": ["less_than_3", "3_to_5", "more_than_5"],
        "tumor_focality": ["unifocal", "multifocal", "not_specified"],
        "tumor_site": [
            "right_lobe", "left_lobe", "isthmus",
            "both_lobe", "others",
        ],
        "predisposing_condition": ["radiation", "family_history"],
        "procedure": [
            "partial_excision", "right_lobectomy", "left_lobectomy",
            "total_thyroidectomy", "others",
        ],
    },

    "cervix": {
        "tnm_descriptor": ["y", "r", "m"],
        "pt_category": [
            "tx", "t1a1", "t1a2", "t1b1", "t1b2", "t1b3",
            "t2a1", "t2a2", "t2b", "t3a", "t3b", "t4",
        ],
        "pn_category": ["nx", "n0", "n1mi", "n1a", "n2mi", "n2a"],
        "pm_category": ["mx", "m0", "m1"],
        "stage_group": [
            "0", "ia1", "ia2", "ib1", "ib2", "ib3",
            "iia1", "iia2", "iib", "iiia", "iiib",
            "iiic1", "iiic2", "iva", "ivb",
        ],
        "grade": ["1", "2", "3"],
        "cancer_primary_site": [
            "12_3_clock", "3_6_clock", "6_9_clock", "9_12_clock",
        ],
        "histology": [
            "squamous_cell_carcinoma_hpv_associated",
            "squamous_cell_carcinoma_hpv_dependaent",
            "squamous_cell_carcinoma_nos",
            "adenocarcinoma_hpv_associated",
            "adenocarcinoma_hpv_independent",
            "adenocarcinoma_nos", "adenosquamous_carcinoma",
            "neuroendocrine_carcinoma", "glassy_cell_carcinoma",
            "small_cell_carcinoma", "large_cell_carcinoma", "others",
        ],
        "depth_of_invasion_number": [
            "less_than_3", "3_to_5", "greater_than_5",
        ],
        "depth_of_invasion_three_tier": [
            "inner_third", "middle_third", "outer_third",
        ],
        "procedure": [
            "radical_hysterectomy", "total_hysterectomy_bso",
            "simple_hysterectomy", "extenteration", "others",
        ],
        "surgical_technique": [
            "open", "laparoscopic", "vaginal", "others",
        ],
    },

    "liver": {
        "tnm_descriptor": ["y", "r", "m"],
        "pt_category": ["tx", "t1a", "t1b", "t2", "t3", "t4"],
        "pn_category": ["nx", "n0", "n1"],
        "pm_category": ["mx", "m0", "m1"],
        "overall_stage": ["ia", "ib", "ii", "iiia", "iiib", "iva", "ivb"],
        "grade": ["1", "2", "3", "4"],
        "histology": [
            "hepatocellular_carcinoma",
            "hepatocellular_carcinoma_fibrolamellar",
            "hepatocellular_carcinoma_scirrhous",
            "hepatocellular_carcinoma_clear_cell", "others",
        ],
        "tumor_focality": ["unifocal", "multifocal"],
        "tumor_site": [
            "right_lobe", "left_lobe", "caudate_lobe",
            "quadrate_lobe", "others",
        ],
        "procedure": [
            "wedge_resection", "partial_hepatectomy", "segmentectomy",
            "lobectomy", "total_hepatectomy", "others",
        ],
    },
}


# --- Boolean fields (3-way {true, false, null}) per organ --------------------

ORGAN_BOOL: dict[str, set[str]] = {
    "breast": {
        "lymphovascular_invasion", "perineural_invasion",
        "distant_metastasis", "extranodal_extension",
        "dcis_present", "dcis_comedo_necrosis",
    },
    "lung": {
        "lymphovascular_invasion", "perineural_invasion",
        "distant_metastasis", "extranodal_extension",
        "visceral_pleural_invasion", "spread_through_air_spaces_stas",
        "direct_invasion_of_adjacent_structures",
    },
    "colorectal": {
        "lymphovascular_invasion", "perineural_invasion",
        "distant_metastasis", "extranodal_extension",
        "signet_ring", "extracellular_mucin",
    },
    "prostate": {
        "lymphovascular_invasion", "perineural_invasion",
        "distant_metastasis", "extranodal_extension",
        "extraprostatic_extension", "seminal_vesicle_invasion",
        "bladder_invasion", "intraductal_carcinoma_presence",
        "cribriform_pattern_presence", "margin_positivity",
    },
    "esophagus": {
        "lymphovascular_invasion", "perineural_invasion",
        "distant_metastasis", "extranodal_extension",
    },
    "stomach": {
        "lymphovascular_invasion", "perineural_invasion",
        "distant_metastasis", "extranodal_extension",
        "signet_ring", "extracellular_mucin",
    },
    "pancreas": {
        "lymphovascular_invasion", "perineural_invasion",
        "distant_metastasis", "extranodal_extension",
    },
    "thyroid": {
        "lymphovascular_invasion", "perineural_invasion",
        "distant_metastasis", "extranodal_extension",
        "tumor_necrosis",
    },
    "cervix": {
        "distant_metastasis", "extranodal_extension",
    },
    "liver": {
        "perineural_invasion", "distant_metastasis",
        "extranodal_extension",
    },
}


# --- Integer span fields per organ (ClinicalBERT-QA head) --------------------

ORGAN_SPAN: dict[str, set[str]] = {
    "breast": {
        "tumor_size", "dcis_size", "maximal_ln_size", "ajcc_version",
    },
    "lung": {"maximal_ln_size", "grade", "ajcc_version"},
    "colorectal": {
        "maximal_ln_size", "grade", "tumor_budding", "ajcc_version",
    },
    "prostate": {
        "tumor_size", "maximal_ln_size",
        "gleason_4_percentage", "gleason_5_percentage",
        "prostate_size", "prostate_weight",
        "tumor_percentage", "ajcc_version",
    },
    "esophagus": {"maximal_ln_size", "ajcc_version"},
    "stomach": {"maximal_ln_size", "ajcc_version"},
    "pancreas": {"tumor_size", "maximal_ln_size", "ajcc_version"},
    "thyroid": {"tumor_size", "maximal_ln_size", "ajcc_version"},
    "cervix": {"tumor_size", "maximal_ln_size", "ajcc_version"},
    "liver": {"tumor_size", "maximal_ln_size", "ajcc_version"},
}


# --- Nested-list fields per organ --------------------------------------------
#
# These go to the supplementary coverage table (not scored in FAIR_SCOPE).

ORGAN_NESTED_LIST: dict[str, set[str]] = {
    "breast":     {"margins", "regional_lymph_node", "biomarkers"},
    "lung":       {"margins", "regional_lymph_node", "biomarkers",
                   "histological_patterns"},
    "colorectal": {"margins", "regional_lymph_node", "biomarkers"},
    "prostate":   {"regional_lymph_node"},
    "esophagus":  {"margins", "regional_lymph_node"},
    "stomach":    {"margins", "regional_lymph_node"},
    "pancreas":   {"margins", "regional_lymph_node"},
    "thyroid":    {"margins", "regional_lymph_node"},
    "cervix":     {"margins", "regional_lymph_node"},
    "liver":      {"margins", "regional_lymph_node"},
}


# --- List-of-literals fields per organ ---------------------------------------
#
# Distinct from ORGAN_NESTED_LIST (which is list-of-dicts). These are
# array fields whose items are plain enum strings, e.g.
# ``vascular_invasion: ["small_vessel", "large_portal_vein"]``. Scored as
# unordered sets — exact-match for headline accuracy, set-F1 for partial
# credit.

ORGAN_LIST_OF_LITERALS: dict[str, dict[str, list[str]]] = {
    "liver": {
        "tumor_extent": [
            "hepatic_vein", "portal_vein", "visceral_peritoneum",
            "gallbladder", "diaphragm", "others",
        ],
        "vascular_invasion": [
            "large_hepatic_vein", "large_portal_vein", "small_vessel",
        ],
    },
    "prostate": {
        "involved_margin_list": [
            "right_apical", "left_apical",
            "right_bladder_neck", "left_bladder_neck",
            "right_anterior", "left_anterior",
            "right_lateral", "left_lateral",
            "right_posterolateral", "left_posterolateral",
            "right_posterior", "left_posterior",
        ],
    },
}


__all__ = [
    "ORGAN_CATEGORICAL",
    "ORGAN_BOOL",
    "ORGAN_SPAN",
    "ORGAN_NESTED_LIST",
    "ORGAN_LIST_OF_LITERALS",
]
