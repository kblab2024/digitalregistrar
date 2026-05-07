"""Generate a synthetic dummy fixture covering each dataset's real organ set.

Produces a schema-valid tree under `--out` (default: ./dummy) for both
CMUH (clean structured reports) and TCGA (chaotic, dictation-style
reports), so the benchmark/eval pipeline can be exercised end-to-end
without touching real data.

Per-dataset organ scope is read from ``configs/organ_code.yaml`` — TCGA
covers 5 organs (breast, colorectal, esophagus, stomach, liver), CMUH
covers 10 (pancreas, breast, cervix, colorectal, esophagus, liver,
lung, prostate, stomach, thyroid). Numeric organ folders mirror that
yaml (so ``data/cmuh/.../1/`` is pancreas; ``data/tcga/.../1/`` is
breast).

Layout produced:

    {out}/data/{cmuh,tcga}/
        reports/{organ_n}/{case_id}.txt
        preannotation/gpt_oss_20b/{organ_n}/{case_id}.json
        annotations/{nhc,kpc}_{with,without}_preann/{organ_n}/{case_id}.json
        annotations/gold/{organ_n}/{case_id}.json
        dataset_manifest.yaml
    {out}/results/predictions/{dataset}/
        llm/{model}/run{NN}/{organ_n}/{case_id}.json
        llm/{model}/run{NN}/_summary.json  _log.jsonl
        llm/{model}/_manifest.yaml
        clinicalbert/{v1_baseline,v2_finetuned}/{organ_n}/{case_id}.json (+ _summary.json)
        rule_based/{organ_n}/{case_id}.json  _summary.json
    {out}/configs/
        datasets/{cmuh,tcga}.yaml
        models/{...}.yaml
        annotators/annotators.yaml
    {out}/models/clinicalbert/{v1_baseline,v2_finetuned}/{config.yaml,checkpoint.pt.placeholder}

Defaults match the production dummy: cmuh = 100 cases per organ,
tcga = 50 cases per organ, 80% cancer / 20% non-cancer, 3 LLM runs.

Production run (full default):
    python scripts/data/gen_dummy_skeleton.py --out dummy --clean

Cross-corpus common-5 only:
    python scripts/data/gen_dummy_skeleton.py --out dummy --clean \\
        --organs breast,colorectal,esophagus,stomach,liver

Multi-run sweep (10 runs per LLM, like the real experiments):
    python scripts/data/gen_dummy_skeleton.py --out dummy --clean --llm-runs 10

Per-dataset case counts:
    --cases-per-organ cmuh:100,tcga:50         # default
    --cases-per-organ 20                        # bare int = same for all selected
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make the in-repo `src/` importable for `from digital_registrar_research...`.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from digital_registrar_research.benchmarks import organs as _organs  # noqa: E402

# ---- Constants --------------------------------------------------------------

DATASETS = list(_organs.all_datasets())
# Union of all organ names across datasets — matches CASE_MODELS in
# src/digital_registrar_research/schemas/pydantic/__init__.py.
ALL_ORGANS = list(_organs.union_organs())
LLM_MODELS = ["gpt_oss_20b", "gemma4_30b", "qwen3_30b", "gemma4_e2b"]
BERT_MODELS = ["v1_baseline", "v2_finetuned"]
ANNOTATORS = ["nhc", "kpc"]
MODES = ["with_preann", "without_preann"]
PREANN_MODEL = "gpt_oss_20b"

DEFAULT_CASES_PER_ORGAN = "cmuh:100,tcga:50"
DEFAULT_LLM_RUNS = 3
DEFAULT_CANCER_RATE = 0.8


def organs_map_for(dataset: str,
                   restrict_to: set[str] | None = None) -> dict[str, str]:
    """Return ``{organ_n_str: organ_name}`` for the dataset, per organ_code.yaml.

    ``restrict_to`` optionally limits the result to a subset of organ
    *names* (used by ``--organs``). Indices not present in the dataset
    are silently dropped — restrictions to organs the dataset doesn't
    cover (e.g. ``cervix`` for TCGA) yield a smaller map.
    """
    full = _organs.organs_for(dataset)  # {int: str}
    items = [(str(n), name) for n, name in sorted(full.items())]
    if restrict_to is not None:
        wanted = {o.lower() for o in restrict_to}
        items = [(n, name) for n, name in items if name.lower() in wanted]
    return dict(items)


def parse_cases_per_organ(spec: str, datasets: list[str]) -> dict[str, int]:
    """Accept either a bare int (applies to all) or 'ds:n,ds:n' CSV."""
    spec = spec.strip()
    if ":" not in spec:
        n = int(spec)
        return {ds: n for ds in datasets}
    out: dict[str, int] = {}
    for part in spec.split(","):
        ds, _, n = part.partition(":")
        ds = ds.strip()
        if not ds or not n:
            raise ValueError(f"bad --cases-per-organ entry: {part!r}")
        out[ds] = int(n)
    missing = [ds for ds in datasets if ds not in out]
    if missing:
        raise ValueError(f"--cases-per-organ missing entries for: {missing}")
    return out


# ---- Schema-valid payload factories -----------------------------------------

def breast_case(tumor_size: int, grade: int, lvi: bool, margin_involved: bool, er: bool) -> dict:
    return {
        "cancer_excision_report": True,
        "cancer_category": "breast",
        "cancer_category_others_description": None,
        "cancer_data": {
            "procedure": "wide_excision",
            "cancer_quadrant": "upper_outer_quadrant",
            "cancer_clock": None,
            "cancer_laterality": "right",
            "histology": "invasive_carcinoma_no_special_type",
            "tumor_size": tumor_size,
            "lymphovascular_invasion": lvi,
            "perineural_invasion": None,
            "distant_metastasis": None,
            "treatment_effect": None,
            "dcis_present": False,
            "dcis_size": None,
            "dcis_comedo_necrosis": None,
            "dcis_grade": None,
            "nuclear_grade": grade,
            "tubule_formation": 2,
            "mitotic_rate": 2,
            "total_score": grade + 4,
            "grade": grade,
            "tnm_descriptor": None,
            "pt_category": "t1c" if tumor_size <= 20 else "t2",
            "pn_category": "n0",
            "pm_category": "mx",
            "pathologic_stage_group": None,
            "anatomic_stage_group": None,
            "ajcc_version": None,
            "margins": [
                {"margin_category": "superficial", "margin_involved": margin_involved,
                 "distance": None if margin_involved else 5, "description": None},
                {"margin_category": "base", "margin_involved": False,
                 "distance": 8, "description": "negative"},
            ],
            "regional_lymph_node": [
                {"lymph_node_side": "right", "lymph_node_category": "sentinel",
                 "involved": 0, "examined": 2, "station_name": "sentinel #1"},
            ],
            "extranodal_extension": None,
            "maximal_ln_size": None,
            "biomarkers": [
                {"biomarker_category": "er", "expression": er, "percentage": None,
                 "score": None, "biomarker_name": "estrogen receptor"},
                {"biomarker_category": "pr", "expression": er, "percentage": None,
                 "score": None, "biomarker_name": "progesterone receptor"},
                {"biomarker_category": "her2", "expression": None, "percentage": None,
                 "score": 0, "biomarker_name": "human epidermal growth factor receptor 2"},
                {"biomarker_category": "ki67", "expression": None, "percentage": None,
                 "score": None, "biomarker_name": "ki67 proliferation index"},
            ],
        },
    }


def cervix_case(grade: int, pt: str, involved_nodes: int) -> dict:
    return {
        "cancer_excision_report": True,
        "cancer_category": "cervix",
        "cancer_category_others_description": None,
        "cancer_data": {
            "procedure": "radical_hysterectomy",
            "surgical_technique": None,
            "cancer_primary_site": "12_3_clock",
            "histology": "squamous_cell_carcinoma_nos",
            "grade": grade,
            "tumor_size": 22,
            "depth_of_invasion_number": 8,
            "depth_of_invasion_three_tier": None,
            "distant_metastasis": None,
            "treatment_effect": None,
            "tnm_descriptor": None,
            "pt_category": pt,
            "pn_category": "n0" if involved_nodes == 0 else "n1a",
            "pm_category": "mx",
            "stage_group": "ib1",
            "ajcc_version": None,
            "margins": [
                {"margin_category": None, "margin_involved": False,
                 "distance": None, "description": "Negative"},
            ],
            "regional_lymph_node": [
                {"lymph_node_side": None, "lymph_node_category": None,
                 "involved": involved_nodes, "examined": 12, "station_name": None},
            ],
            "extranodal_extension": None,
            "maximal_ln_size": None,
        },
    }


def colorectal_case(grade: int, pt: str, lvi: bool, involved_nodes: int) -> dict:
    return {
        "cancer_excision_report": True,
        "cancer_category": "colorectal",
        "cancer_category_others_description": None,
        "cancer_data": {
            "procedure": "segmental_colectomy",
            "surgical_technique": None,
            "cancer_primary_site": "sigmoid_colon",
            "histology": "adenocarcinoma",
            "grade": grade,
            "tumor_invasion": "pericolorectal_tissue",
            "lymphovascular_invasion": lvi,
            "perineural_invasion": None,
            "extracellular_mucin": None,
            "signet_ring": None,
            "tumor_budding": None,
            "type_of_polyp": None,
            "distant_metastasis": None,
            "treatment_effect": None,
            "tnm_descriptor": None,
            "pt_category": pt,
            "pn_category": "n0" if involved_nodes == 0 else "n1",
            "pm_category": "mx",
            "stage_group": "iia" if involved_nodes == 0 else "iii",
            "ajcc_version": None,
            "margins": [
                {"margin_category": "proximal", "margin_involved": False,
                 "distance": None, "description": "Negative"},
                {"margin_category": "distal", "margin_involved": False,
                 "distance": None, "description": "Negative"},
                {"margin_category": "radial_or_circumferencial", "margin_involved": False,
                 "distance": None, "description": "Negative"},
            ],
            "regional_lymph_node": [
                {"lymph_node_category": "mesenteric", "involved": involved_nodes,
                 "examined": 22, "station_name": None},
            ],
            "extranodal_extension": None,
            "maximal_ln_size": None,
            "biomarkers": None,
        },
    }


def esophagus_case(grade: int, pt: str, lvi: bool, involved_nodes: int) -> dict:
    return {
        "cancer_excision_report": True,
        "cancer_category": "esophagus",
        "cancer_category_others_description": None,
        "cancer_data": {
            "procedure": "esophagectomy",
            "surgical_technique": "open",
            "cancer_primary_site": "lower_third",
            "histology": "squamous_cell_carcinoma",
            "grade": grade,
            "tumor_extent": "adventitia",
            "lymphovascular_invasion": lvi,
            "perineural_invasion": None,
            "distant_metastasis": None,
            "treatment_effect": None,
            "tnm_descriptor": None,
            "pt_category": pt,
            "pn_category": "n0" if involved_nodes == 0 else "n2",
            "pm_category": "mx",
            "stage_group": "iiib" if involved_nodes else "ii",
            "ajcc_version": 8,
            "margins": [
                {"margin_category": None, "margin_involved": False,
                 "distance": None, "description": "Negative"},
            ],
            "regional_lymph_node": [
                {"lymph_node_category": None, "involved": involved_nodes,
                 "examined": 8, "station_name": None},
            ],
            "extranodal_extension": None,
            "maximal_ln_size": None,
        },
    }


def liver_case(grade: int, pt: str, tumor_size: int) -> dict:
    return {
        "cancer_excision_report": True,
        "cancer_category": "liver",
        "cancer_category_others_description": None,
        "cancer_data": {
            "procedure": "partial_hepatectomy",
            "tumor_site": "right_lobe",
            "tumor_focality": "unifocal",
            "tumor_size": tumor_size,
            "histology": "hepatocellular_carcinoma",
            "grade": grade,
            "tumor_extent": None,
            "vascular_invasion": False,
            "perineural_invasion": None,
            "distant_metastasis": None,
            "treatment_effect": None,
            "tnm_descriptor": None,
            "pt_category": pt,
            "pn_category": "n0",
            "pm_category": "mx",
            "overall_stage": "ib",
            "ajcc_version": None,
            "margins": [
                {"margin_category": None, "margin_involved": False,
                 "distance": None, "description": "Negative"},
            ],
            "regional_lymph_node": [
                {"involved": 0, "examined": 1, "station_name": None},
            ],
            "extranodal_extension": None,
            "maximal_ln_size": None,
        },
    }


def lung_case(tumor_size: int, pt: str, lvi: bool) -> dict:
    return {
        "cancer_excision_report": True,
        "cancer_category": "lung",
        "cancer_category_others_description": None,
        "cancer_data": {
            "procedure": "lobectomy",
            "surgical_technique": None,
            "cancer_primary_site": "upper_lobe",
            "sideness": "right",
            "histology": "adenocarcinoma",
            "histological_patterns": [
                {"pattern_name": "acinar", "pattern_percentage": 70},
                {"pattern_name": "papillary", "pattern_percentage": 30},
            ],
            "grade": None,
            "tumor_size": tumor_size,
            "tumor_focality": "unifocal",
            "visceral_pleural_invasion": False,
            "spread_through_air_spaces_stas": False,
            "lymphovascular_invasion": lvi,
            "perineural_invasion": None,
            "direct_invasion_of_adjacent_structures": None,
            "distant_metastasis": None,
            "treatment_effect": None,
            "tnm_descriptor": None,
            "pt_category": pt,
            "pn_category": "n0",
            "pm_category": "mx",
            "stage_group": "ia2",
            "ajcc_version": None,
            "margins": [
                {"margin_category": None, "margin_involved": False,
                 "distance": None, "description": "Negative"},
            ],
            "regional_lymph_node": [
                {"lymph_node_side": "right", "lymph_node_category": None,
                 "involved": 0, "examined": 6, "station_name": None},
            ],
            "extranodal_extension": None,
            "maximal_ln_size": None,
            "biomarkers": [
                {"biomarker_category": "ALK", "expression": False,
                 "percentage": None, "biomarker_name": "anaplastic lymphoma kinase"},
                {"biomarker_category": "PDL1", "expression": True,
                 "percentage": 25, "biomarker_name": "programmed death-ligand 1"},
            ],
        },
    }


def pancreas_case(pt: str, lvi: bool, tumor_size: int) -> dict:
    return {
        "cancer_excision_report": True,
        "cancer_category": "pancreas",
        "cancer_category_others_description": None,
        "cancer_data": {
            "procedure": "whipple_procedure",
            "tumor_site": "head",
            "tumor_size": tumor_size,
            "histology": "ductal_adenocarcinoma_nos",
            "tumor_extension": None,
            "lymphovascular_invasion": lvi,
            "perineural_invasion": True,
            "distant_metastasis": None,
            "treatment_effect": None,
            "tnm_descriptor": None,
            "pt_category": pt,
            "pn_category": "n1",
            "pm_category": "mx",
            "overall_stage": "iib",
            "ajcc_version": None,
            "margins": [
                {"margin_category": None, "margin_involved": False,
                 "distance": None, "description": "Negative"},
            ],
            "regional_lymph_node": [
                {"lymph_node_category": None, "involved": 2,
                 "examined": 18, "station_name": None},
            ],
            "extranodal_extension": None,
            "maximal_ln_size": None,
        },
    }


def prostate_case(gleason_group: str, pt: str, lvi: bool) -> dict:
    return {
        "cancer_excision_report": True,
        "cancer_category": "prostate",
        "cancer_category_others_description": None,
        "cancer_data": {
            "procedure": "radical_prostatectomy",
            "surgical_technique": None,
            "histology": "acinar_adenocarcinoma",
            "grade": gleason_group,
            "gleason_4_percentage": 30,
            "gleason_5_percentage": 0,
            "cribriform_pattern_presence": False,
            "intraductal_carcinoma_presence": False,
            "tumor_size": 18,
            "tumor_percentage": 15,
            "prostate_size": 50,
            "prostate_weight": 45,
            "extraprostatic_extension": False,
            "seminal_vesicle_invasion": False,
            "bladder_invasion": False,
            "lymphovascular_invasion": lvi,
            "perineural_invasion": True,
            "margin_positivity": False,
            "margin_length": None,
            "involved_margin_list": None,
            "distant_metastasis": None,
            "treatment_effect": None,
            "tnm_descriptor": None,
            "pt_category": pt,
            "pn_category": "n0",
            "pm_category": "mx",
            "stage_group": "iia",
            "ajcc_version": None,
            "regional_lymph_node": [
                {"lymph_node_side": None, "lymph_node_category": None,
                 "involved": 0, "examined": 4, "station_name": None},
            ],
            "extranodal_extension": None,
            "maximal_ln_size": None,
        },
    }


def stomach_case(grade: int, pt: str, lvi: bool, involved_nodes: int) -> dict:
    return {
        "cancer_excision_report": True,
        "cancer_category": "stomach",
        "cancer_category_others_description": None,
        "cancer_data": {
            "procedure": "partial_gastrectomy",
            "surgical_technique": None,
            "cancer_primary_site": "antrum",
            "histology": "tubular_adenocarcinoma",
            "grade": grade,
            "tumor_extent": "muscularis_propria",
            "lymphovascular_invasion": lvi,
            "perineural_invasion": False,
            "extracellular_mucin": None,
            "signet_ring": False,
            "distant_metastasis": None,
            "treatment_effect": None,
            "tnm_descriptor": None,
            "pt_category": pt,
            "pn_category": "n0" if involved_nodes == 0 else "n1",
            "pm_category": "mx",
            "stage_group": "ib" if involved_nodes == 0 else "iia",
            "ajcc_version": None,
            "margins": [
                {"margin_category": None, "margin_involved": False,
                 "distance": None, "description": "Negative"},
            ],
            "regional_lymph_node": [
                {"lymph_node_category": None, "involved": involved_nodes,
                 "examined": 25, "station_name": None},
            ],
            "extranodal_extension": None,
            "maximal_ln_size": None,
        },
    }


def thyroid_case(pt: str, tumor_size: int) -> dict:
    return {
        "cancer_excision_report": True,
        "cancer_category": "thyroid",
        "cancer_category_others_description": None,
        "cancer_data": {
            "procedure": "total_thyroidectomy",
            "tumor_site": "right_lobe",
            "tumor_focality": "unifocal",
            "tumor_size": tumor_size,
            "histology": "papillary_thyroid_carcinoma",
            "extrathyroid_extension": False,
            "tumor_necrosis": False,
            "mitotic_activity": None,
            "lymphovascular_invasion": False,
            "perineural_invasion": None,
            "predisposing_condition": None,
            "distant_metastasis": None,
            "treatment_effect": None,
            "tnm_descriptor": None,
            "pt_category": pt,
            "pn_category": "n0",
            "pm_category": "mx",
            "overall_stage": "i",
            "ajcc_version": None,
            "margins": [
                {"margin_category": None, "margin_involved": False,
                 "distance": None, "description": "Negative"},
            ],
            "regional_lymph_node": [
                {"lymph_node_side": "right", "lymph_node_category": None,
                 "involved": 0, "examined": 3, "station_name": None},
            ],
            "extranodal_extension": None,
            "maximal_ln_size": None,
        },
    }


def noncancer_case() -> dict:
    """Schema-valid non-cancer case: no organ-specific data."""
    return {
        "cancer_excision_report": False,
        "cancer_category": None,
        "cancer_category_others_description": None,
        "cancer_data": {},
    }


_ORGAN_FACTORIES = {
    "breast":     lambda idx: breast_case(
        tumor_size=15 + (idx % 8) * 3,
        grade=((idx - 1) % 3) + 1,
        lvi=(idx % 4 == 0),
        margin_involved=(idx % 7 == 0),
        er=(idx % 5 != 0),
    ),
    "cervix":     lambda idx: cervix_case(
        grade=((idx - 1) % 3) + 1,
        pt=["t1b1", "t2a1", "t2b"][idx % 3],
        involved_nodes=(idx % 4 == 0) * 2,
    ),
    "colorectal": lambda idx: colorectal_case(
        grade=((idx - 1) % 3) + 1,
        pt=["t2", "t3", "t4a"][idx % 3],
        lvi=(idx % 3 == 0),
        involved_nodes=(idx % 5 == 0) * 2,
    ),
    "esophagus":  lambda idx: esophagus_case(
        grade=((idx - 1) % 3) + 1,
        pt=["t2", "t3", "t4a"][idx % 3],
        lvi=(idx % 3 == 0),
        involved_nodes=(idx % 4 == 0) * 5,
    ),
    "liver":      lambda idx: liver_case(
        grade=((idx - 1) % 4) + 1,
        pt=["t1b", "t2", "t3"][idx % 3],
        tumor_size=20 + (idx % 6) * 5,
    ),
    "lung":       lambda idx: lung_case(
        tumor_size=15 + (idx % 7) * 3,
        pt=["t1a", "t1b", "t1c", "t2a"][idx % 4],
        lvi=(idx % 5 == 0),
    ),
    "pancreas":   lambda idx: pancreas_case(
        pt=["t2", "t3"][idx % 2],
        lvi=(idx % 3 == 0),
        tumor_size=25 + (idx % 6) * 4,
    ),
    "prostate":   lambda idx: prostate_case(
        gleason_group=["group_1_3_3", "group_2_3_4", "group_3_4_3", "group_4_4_4"][idx % 4],
        pt=["t2", "t3a", "t3b"][idx % 3],
        lvi=(idx % 5 == 0),
    ),
    "stomach":    lambda idx: stomach_case(
        grade=((idx - 1) % 3) + 1,
        pt=["t2", "t3", "t4a"][idx % 3],
        lvi=(idx % 4 == 0),
        involved_nodes=(idx % 5 == 0) * 2,
    ),
    "thyroid":    lambda idx: thyroid_case(
        pt=["t1a", "t1b", "t2", "t3a"][idx % 4],
        tumor_size=8 + (idx % 6) * 4,
    ),
}


def gold_for(dataset: str, organ_n: str, idx: int,
             organs_map: dict[str, str], cases_per_organ: int,
             cancer_rate: float) -> dict:
    """Schema-valid gold for (dataset, organ, idx).

    The first round(cases_per_organ * cancer_rate) cases per organ are
    cancer (organ-specific factory); the rest are non-cancer.
    """
    organ = organs_map[organ_n]
    n_cancer = round(cases_per_organ * cancer_rate)
    if idx > n_cancer:
        return noncancer_case()
    return _ORGAN_FACTORIES[organ](idx)


def noisify(gold: dict, error_rate: float, rng: random.Random) -> dict:
    """Produce a prediction by flipping a small fraction of leaf values."""
    import copy
    out = copy.deepcopy(gold)
    data = out["cancer_data"]

    # Non-cancer cases have no organ data to perturb; leave them as-is so
    # the routing decision (`cancer_excision_report=False`) stays stable.
    if not data:
        return out

    if rng.random() < error_rate and "lymphovascular_invasion" in data:
        v = data["lymphovascular_invasion"]
        if isinstance(v, bool):
            data["lymphovascular_invasion"] = not v

    for num_field in ("tumor_size", "grade", "nuclear_grade"):
        if num_field in data and isinstance(data[num_field], int) and rng.random() < error_rate:
            data[num_field] = max(1, data[num_field] + rng.choice([-1, 1]))

    if data.get("biomarkers") and rng.random() < error_rate:
        for b in data["biomarkers"]:
            if b["biomarker_category"] == "er" and isinstance(b["expression"], bool):
                b["expression"] = not b["expression"]
                break

    return out


# ---- Report-text rendering --------------------------------------------------

def _clean_report_text(dataset: str, case_id: str, organ: str, gold: dict) -> str:
    """Structured key-value report — used for CMUH (clean dataset)."""
    cd = gold["cancer_data"]
    header = f"SYNTHETIC PATHOLOGY REPORT (dummy) — dataset={dataset}, id={case_id}, organ={organ}"

    if not gold["cancer_excision_report"]:
        return "\n".join([
            header,
            "FINAL DIAGNOSIS:",
            "  No invasive carcinoma identified.",
            "  Benign findings; specimen not eligible for cancer registry.",
        ]) + "\n"

    bits = [
        header,
        "FINAL DIAGNOSIS:",
        f"  Procedure: {cd.get('procedure')}",
        f"  Histology: {cd.get('histology')}",
    ]
    if organ == "breast":
        bits += [
            f"  Tumor size: {cd.get('tumor_size')} mm, laterality={cd.get('cancer_laterality')}",
            f"  Grade {cd.get('grade')}, LVI={cd.get('lymphovascular_invasion')}",
            f"  Margins: {'involved' if any(m['margin_involved'] for m in cd['margins']) else 'negative'}",
            f"  ER: {cd['biomarkers'][0]['expression']}, PR: {cd['biomarkers'][1]['expression']}",
            f"  pTNM: {cd.get('pt_category')} {cd.get('pn_category')} {cd.get('pm_category')}",
        ]
        return "\n".join(bits) + "\n"

    if "tumor_size" in cd and cd.get("tumor_size") is not None:
        bits.append(f"  Tumor size: {cd['tumor_size']} mm")
    if "grade" in cd and cd.get("grade") is not None:
        bits.append(f"  Grade {cd['grade']}")
    if "lymphovascular_invasion" in cd:
        bits.append(f"  LVI={cd.get('lymphovascular_invasion')}")
    stage = cd.get("stage_group") or cd.get("overall_stage")
    bits.append(
        f"  pTNM: {cd.get('pt_category')} {cd.get('pn_category')} {cd.get('pm_category')}"
        + (f", stage {stage}" if stage else "")
    )
    if cd.get("regional_lymph_node"):
        ln = cd["regional_lymph_node"][0]
        bits.append(f"  Lymph nodes: {ln['involved']}/{ln['examined']} involved")
    if cd.get("biomarkers"):
        bm = ", ".join(f"{b['biomarker_category']}={b['expression']}"
                       for b in cd["biomarkers"])
        bits.append(f"  Biomarkers: {bm}")
    return "\n".join(bits) + "\n"


# Common abbreviations real pathologists slip into dictation
_HISTOLOGY_ABBREV = {
    "squamous_cell_carcinoma": "SCC",
    "squamous_cell_carcinoma_nos": "SCC, NOS",
    "adenocarcinoma": "adenoCA",
    "invasive_carcinoma_no_special_type": "invasive ductal CA, NST",
    "tubular_adenocarcinoma": "tubular adenoCA",
    "ductal_adenocarcinoma_nos": "ductal adenoCA",
    "papillary_thyroid_carcinoma": "PTC",
    "hepatocellular_carcinoma": "HCC",
    "acinar_adenocarcinoma": "acinar adenoCA",
}


def _chaotic_report_text(dataset: str, case_id: str, organ: str, gold: dict,
                         seed: int) -> str:
    """Messy, dictation-style report — used for TCGA.

    Same factual content as the gold, but rendered with random section
    ordering, mixed-case headers, abbreviations, and prose-embedded
    fields. Determinism comes from the per-case seed.
    """
    rng = random.Random(seed)
    cd = gold["cancer_data"]

    header_styles = [
        f"FINAL DIAGNOSIS:",
        f"Final Diagnosis",
        f"DIAGNOSIS:",
        f"-- Diagnosis --",
        f"PATH DX:",
    ]
    closers = [
        "End of report.",
        "Dictated and signed electronically.",
        "Pathologist: [redacted]",
        "** END **",
        "",
    ]

    if not gold["cancer_excision_report"]:
        # Non-cancer: short, often just a one-liner
        nc_phrasings = [
            "No malignancy identified. Benign tissue only.",
            "Negative for invasive carcinoma. Reactive changes noted.",
            "Specimen does not contain viable tumor.",
            "Findings benign; not registry-eligible.",
            "no carcinoma seen in submitted material",
        ]
        return "\n".join([
            f"[synthetic-tcga] case={case_id} organ={organ}",
            rng.choice(header_styles),
            rng.choice(nc_phrasings),
            rng.choice(closers),
        ]) + "\n"

    # Build content fragments in random orders
    proc = cd.get("procedure", "?").replace("_", " ")
    hist_raw = cd.get("histology", "?")
    hist = _HISTOLOGY_ABBREV.get(hist_raw, hist_raw.replace("_", " "))
    grade = cd.get("grade")
    pt = cd.get("pt_category")
    pn = cd.get("pn_category")
    pm = cd.get("pm_category")
    stage = cd.get("stage_group") or cd.get("overall_stage")
    tumor_size = cd.get("tumor_size")
    lvi = cd.get("lymphovascular_invasion")
    pni = cd.get("perineural_invasion")
    lns = cd.get("regional_lymph_node") or []

    fragments: list[str] = []

    # Procedure / specimen: prose form
    fragments.append(rng.choice([
        f"Specimen: s/p {proc}.",
        f"Procedure performed - {proc}.",
        f"PROC: {proc}",
        f"Surgery: {proc} (received fresh).",
    ]))

    # Histology: usually labeled, but sometimes embedded in prose
    if rng.random() < 0.5:
        fragments.append(rng.choice([
            f"Microscopic: {hist}.",
            f"Histologic type: {hist}",
            f"HISTOLOGY -- {hist}",
        ]))
    else:
        fragments.append(rng.choice([
            f"Sections show {hist} arising in the {organ}.",
            f"On microscopy, {hist} is identified, infiltrating into surrounding tissue.",
            f"The lesion is consistent with {hist}.",
        ]))

    # Tumor size: usually present, sometimes embedded
    if tumor_size is not None:
        if rng.random() < 0.4:
            fragments.append(f"Tumor size: {tumor_size} mm.")
        else:
            cm = tumor_size / 10
            fragments.append(rng.choice([
                f"A {cm:.1f} cm mass was identified.",
                f"Grossly, the tumor measures {tumor_size} mm in greatest dimension.",
                f"Lesion ~{cm:.1f}cm.",
            ]))

    # Grade
    if grade is not None and rng.random() < 0.85:
        fragments.append(rng.choice([
            f"Grade: {grade}.",
            f"Histologic grade {grade}.",
            f"(grade {grade})",
        ]))

    # LVI / PNI: optional, sometimes combined
    if lvi is not None and rng.random() < 0.7:
        fragments.append(rng.choice([
            f"LVI: {'present' if lvi else 'not identified'}.",
            f"Lymphovascular invasion - {'YES' if lvi else 'no'}.",
            f"{'+' if lvi else 'No'} angiolymphatic invasion.",
        ]))
    if pni is not None and rng.random() < 0.5:
        fragments.append(f"PNI: {'+' if pni else '-'}.")

    # Lymph nodes
    if lns:
        ln = lns[0]
        inv, exam = ln.get("involved", 0), ln.get("examined", 0)
        fragments.append(rng.choice([
            f"Lymph nodes: {inv}/{exam} positive.",
            f"LNs - {inv} of {exam} involved by tumor.",
            f"Regional nodes: {inv}/{exam}.",
        ]))

    # Staging line
    stage_bits = []
    if pt: stage_bits.append(f"p{pt.upper()}")
    if pn: stage_bits.append(pn.upper())
    if pm: stage_bits.append(pm.upper())
    if stage_bits:
        s = " ".join(stage_bits)
        if stage:
            s += f" -- stage {stage}"
        fragments.append(rng.choice([
            f"Staging: {s}",
            f"AJCC -- {s}",
            f"pTNM {s}",
        ]))

    # Margins
    margins = cd.get("margins") or []
    if margins and rng.random() < 0.8:
        any_pos = any(m.get("margin_involved") for m in margins)
        fragments.append(rng.choice([
            f"Margins: {'INVOLVED' if any_pos else 'negative'}.",
            f"Resection margins {'positive' if any_pos else 'free of tumor'}.",
            f"All margins {'(+)' if any_pos else '(-)'}",
        ]))

    # Biomarkers (organ-specific, only if present)
    bms = cd.get("biomarkers") or []
    if bms and rng.random() < 0.6:
        bm_strs = []
        for b in bms[: rng.randint(1, len(bms))]:
            cat = b.get("biomarker_category", "?")
            expr = b.get("expression")
            score = b.get("score")
            pct = b.get("percentage")
            if expr is True:
                bm_strs.append(f"{cat}+")
            elif expr is False:
                bm_strs.append(f"{cat}-")
            elif score is not None:
                bm_strs.append(f"{cat} score={score}")
            elif pct is not None:
                bm_strs.append(f"{cat} {pct}%")
        if bm_strs:
            fragments.append("IHC: " + ", ".join(bm_strs))

    # Shuffle the body fragments to simulate non-canonical ordering
    rng.shuffle(fragments)

    return "\n".join([
        f"[synthetic-tcga] case={case_id} organ={organ}",
        rng.choice(header_styles),
        *fragments,
        rng.choice(closers),
    ]) + "\n"


def report_text(dataset: str, case_id: str, organ: str, gold: dict) -> str:
    """Dispatch to the dataset-appropriate report style."""
    if dataset == "tcga":
        # Per-case deterministic seed for stable chaotic text across runs
        seed = int(hashlib.sha256(case_id.encode()).hexdigest()[:8], 16)
        return _chaotic_report_text(dataset, case_id, organ, gold, seed)
    return _clean_report_text(dataset, case_id, organ, gold)


# ---- Writers ----------------------------------------------------------------

def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_yaml(path: Path, payload: dict) -> None:
    """Minimal YAML writer (avoid pyyaml dep). Only handles dicts/lists/scalars."""
    def _dump(obj, indent=0):
        pad = "  " * indent
        lines = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    lines.append(f"{pad}{k}:")
                    lines.extend(_dump(v, indent + 1))
                else:
                    lines.append(f"{pad}{k}: {_scalar(v)}")
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, (dict, list)):
                    sub = _dump(item, indent + 1)
                    if sub:
                        sub[0] = pad + "- " + sub[0].lstrip()
                        lines.extend(sub)
                else:
                    lines.append(f"{pad}- {_scalar(item)}")
        return lines

    def _scalar(v):
        if v is None:
            return "null"
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return str(v)
        s = str(v)
        if any(c in s for c in ":#\n") or s in ("true", "false", "null", ""):
            return json.dumps(s)
        return s

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(_dump(payload)) + "\n", encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# ---- Top-level builders -----------------------------------------------------

def build_dataset(root: Path, dataset: str, organs_map: dict[str, str],
                  cases_per_organ: int, cancer_rate: float,
                  rng: random.Random) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    ds_root = root / "data" / dataset
    n_cancer_per_organ = round(cases_per_organ * cancer_rate)
    all_ids: list[str] = []
    n_cancer = 0
    n_noncancer = 0

    for organ_n, organ in organs_map.items():
        for idx in range(1, cases_per_organ + 1):
            case_id = f"{dataset}{organ_n}_{idx}"
            all_ids.append(case_id)
            if idx <= n_cancer_per_organ:
                n_cancer += 1
            else:
                n_noncancer += 1
            gold = gold_for(dataset, organ_n, idx, organs_map, cases_per_organ, cancer_rate)

            # Raw report (style depends on dataset)
            write_text(ds_root / "reports" / organ_n / f"{case_id}.txt",
                       report_text(dataset, case_id, organ, gold))

            # Gold
            write_json(ds_root / "annotations" / "gold" / organ_n / f"{case_id}.json", gold)

            # Pre-annotation (gpt-oss:20b, slightly noisy)
            preann = noisify(gold, error_rate=0.15, rng=rng)
            write_json(ds_root / "preannotation" / PREANN_MODEL / organ_n / f"{case_id}.json",
                       preann)

            # Human annotations: 4 modes, each slightly different
            for annotator in ANNOTATORS:
                for mode in MODES:
                    err = 0.05 if mode == "with_preann" else 0.10
                    ann = noisify(gold, error_rate=err, rng=rng)
                    ann["_meta"] = {
                        "filename": f"{case_id}.json",
                        "annotator": annotator,
                        "mode": mode,
                        "annotated_at": datetime.now().isoformat(timespec="seconds"),
                    }
                    write_json(
                        ds_root / "annotations" / f"{annotator}_{mode}" / organ_n / f"{case_id}.json",
                        ann,
                    )

    # dataset_manifest.yaml
    write_yaml(ds_root / "dataset_manifest.yaml", {
        "dataset": dataset,
        "created_at": now,
        "n_cases": len(all_ids),
        "n_cancer_cases": n_cancer,
        "n_noncancer_cases": n_noncancer,
        "cancer_rate": cancer_rate,
        "organs": [
            {"n": n, "name": organs_map[n], "n_cases": cases_per_organ}
            for n in organs_map
        ],
        "cases_per_organ": cases_per_organ,
        "report_style": "chaotic_dictation" if dataset == "tcga" else "structured_keyvalue",
        "source": "synthetic_dummy_skeleton",
        "schema_version": "fair_plus_nested_plus_biomarkers_v1",
    })


def build_llm_predictions(root: Path, dataset: str, organs_map: dict[str, str],
                          cases_per_organ: int, cancer_rate: float,
                          llm_runs: int, rng: random.Random) -> None:
    preds_root = root / "results" / "predictions" / dataset / "llm"
    n_total = len(organs_map) * cases_per_organ
    for model in LLM_MODELS:
        run_entries = []
        for run_i in range(1, llm_runs + 1):
            run_name = f"run{run_i:02d}"
            run_dir = preds_root / model / run_name

            base_err = {
                "gpt_oss_20b": 0.05, "gemma4_30b": 0.08,
                "qwen3_30b": 0.09, "gemma4_e2b": 0.15,
            }[model]
            seed = 42 + run_i - 1

            log_rows: list[dict] = []
            total_tokens = 0

            for organ_n in organs_map:
                for idx in range(1, cases_per_organ + 1):
                    case_id = f"{dataset}{organ_n}_{idx}"
                    gold = gold_for(dataset, organ_n, idx, organs_map,
                                    cases_per_organ, cancer_rate)
                    pred = noisify(gold, error_rate=base_err,
                                   rng=random.Random(seed * 1000 + int(organ_n) * 100 + idx))
                    write_json(run_dir / organ_n / f"{case_id}.json", pred)

                    tok = rng.randint(400, 1200)
                    total_tokens += tok
                    log_rows.append({
                        "case_id": case_id,
                        "organ": organ_n,
                        "run": run_name,
                        "seed": seed,
                        "tokens": tok,
                        "parse_success": True,
                        "latency_s": round(rng.uniform(0.4, 2.5), 3),
                    })

            write_json(run_dir / "_summary.json", {
                "run": run_name,
                "model": model,
                "seed": seed,
                "dataset": dataset,
                "n_cases": n_total,
                "total_tokens": total_tokens,
                "parse_errors": 0,
                "parse_error_rate": 0.0,
                "wall_time_s": round(rng.uniform(30, 120), 1),
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })
            (run_dir / "_log.jsonl").write_text(
                "\n".join(json.dumps(r) for r in log_rows) + "\n", encoding="utf-8"
            )

            run_entries.append({
                "run": run_name,
                "seed": seed,
                "valid": True,
                "parse_error_rate": 0.0,
            })

        write_yaml(preds_root / model / "_manifest.yaml", {
            "experiment_id": f"multirun_{model}_{dataset}_v1",
            "dataset": dataset,
            "model": model,
            "k": llm_runs,
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "config_hash": hashlib.sha256(f"{model}-{dataset}-dummy".encode()).hexdigest()[:12],
            "runs": run_entries,
        })


def build_bert_predictions(root: Path, dataset: str, organs_map: dict[str, str],
                           cases_per_organ: int, cancer_rate: float,
                           rng: random.Random) -> None:
    preds_root = root / "results" / "predictions" / dataset / "clinicalbert"
    n_total = len(organs_map) * cases_per_organ
    for model_id in BERT_MODELS:
        err = 0.07 if model_id == "v1_baseline" else 0.05
        for organ_n in organs_map:
            for idx in range(1, cases_per_organ + 1):
                case_id = f"{dataset}{organ_n}_{idx}"
                gold = gold_for(dataset, organ_n, idx, organs_map,
                                cases_per_organ, cancer_rate)
                pred = noisify(gold, error_rate=err, rng=rng)
                write_json(preds_root / model_id / organ_n / f"{case_id}.json", pred)
        write_json(preds_root / model_id / "_summary.json", {
            "model": f"clinicalbert_{model_id}",
            "dataset": dataset,
            "n_cases": n_total,
            "checkpoint": f"models/clinicalbert/{model_id}/checkpoint.pt",
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })


def build_rule_based(root: Path, dataset: str, organs_map: dict[str, str],
                     cases_per_organ: int, cancer_rate: float,
                     rng: random.Random) -> None:
    preds_root = root / "results" / "predictions" / dataset / "rule_based"
    n_total = len(organs_map) * cases_per_organ
    for organ_n in organs_map:
        for idx in range(1, cases_per_organ + 1):
            case_id = f"{dataset}{organ_n}_{idx}"
            gold = gold_for(dataset, organ_n, idx, organs_map,
                            cases_per_organ, cancer_rate)
            pred = noisify(gold, error_rate=0.25, rng=rng)
            write_json(preds_root / organ_n / f"{case_id}.json", pred)
    write_json(preds_root / "_summary.json", {
        "method": "rule_based",
        "dataset": dataset,
        "n_cases": n_total,
        "implementation": "src/digital_registrar_research/benchmarks/baselines/rules.py",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })


def build_configs(root: Path, organs_per_dataset: dict[str, dict[str, str]],
                  llm_runs: int) -> None:
    cfg = root / "configs"

    for ds, organs_map in organs_per_dataset.items():
        write_yaml(cfg / "datasets" / f"{ds}.yaml", {
            "name": ds,
            "reports_dir": f"data/{ds}/reports",
            "annotations_dir": f"data/{ds}/annotations",
            "preannotation_dir": f"data/{ds}/preannotation",
            "manifest_path": f"data/{ds}/dataset_manifest.yaml",
            "organs": [{"n": n, "name": organs_map[n]} for n in organs_map],
        })

    for m in LLM_MODELS:
        write_yaml(cfg / "models" / f"{m}.yaml", {
            "name": m,
            "family": "llm",
            "backend": "ollama" if m.startswith(("gemma", "qwen")) else "vllm",
            "runs": {"k": llm_runs, "seeds": list(range(42, 42 + llm_runs))},
            "decoding": {"temperature": 0.7, "top_p": 1.0, "max_tokens": 2048},
        })
    for m in BERT_MODELS:
        write_yaml(cfg / "models" / f"clinicalbert_{m}.yaml", {
            "name": f"clinicalbert_{m}",
            "family": "clinicalbert",
            "checkpoint": f"models/clinicalbert/{m}/checkpoint.pt",
        })
    write_yaml(cfg / "models" / "rule_based.yaml", {
        "name": "rule_based",
        "family": "rule_based",
        "module": "digital_registrar_research.benchmarks.baselines.rules",
    })

    write_yaml(cfg / "annotators" / "annotators.yaml", {
        "annotators": [
            {"id": "nhc", "display": "NHC", "modes": MODES, "enabled": True},
            {"id": "kpc", "display": "KPC", "modes": MODES, "enabled": True},
        ],
        "modes": {
            "with_preann": "Annotator sees gpt-oss:20b pre-annotation first",
            "without_preann": "Annotator starts from blank template",
        },
        "preannotation_model": PREANN_MODEL,
    })


def build_model_dirs(root: Path) -> None:
    for m in BERT_MODELS:
        mdir = root / "models" / "clinicalbert" / m
        mdir.mkdir(parents=True, exist_ok=True)
        (mdir / "checkpoint.pt.placeholder").write_text(
            "Replace with real .pt checkpoint.\n", encoding="utf-8")
        write_yaml(mdir / "config.yaml", {
            "model_id": f"clinicalbert_{m}",
            "base_model": "emilyalsentzer/Bio_ClinicalBERT",
            "trained_on": "cmuh_train" if m == "v2_finetuned" else "generic_discharge",
        })


def write_skeleton_gitkeeps(root: Path) -> None:
    """Drop .gitkeep at every directory under dummy/data/ and dummy/results/.

    The dummy fixture's directory layout is the **committed backbone**:
    even though generated content is gitignored, the empty layout itself
    must survive `--clean` so a fresh checkout (or a regen on a different
    box) reproduces the same tree. We touch a `.gitkeep` at every node
    (recursively) so directory structure is always preserved by git.

    The dummy ``results/`` tree mirrors the production ``workspace/results/``
    architecture: ``predictions/``, ``eval/``, ``ablations/``, ``benchmarks/``.
    The latter three are populated only when the corresponding scripts run
    against this dummy root; the directories exist as empty placeholders so
    the layout is discoverable.
    """
    # Ensure the canonical results/ subtree exists.
    for sub in ("predictions", "eval", "ablations", "benchmarks"):
        (root / "results" / sub).mkdir(parents=True, exist_ok=True)

    for sub in ("data", "results"):
        sub_root = root / sub
        if not sub_root.exists():
            sub_root.mkdir(parents=True)
        (sub_root / ".gitkeep").touch(exist_ok=True)
        # Recurse: every directory under data/ and results/ gets a .gitkeep
        # so the committed backbone is preserved even after a wipe.
        for d in sub_root.rglob("*"):
            if d.is_dir():
                (d / ".gitkeep").touch(exist_ok=True)


def write_readme(root: Path) -> None:
    (root / "README.md").write_text("""# Dummy fixture (synthetic, regenerable)

This tree is produced by `scripts/data/gen_dummy_skeleton.py`. The toolkit is
checked in; the output is not. Regenerate any time:

```
python scripts/data/gen_dummy_skeleton.py --out dummy --clean
```

Per-dataset organ scope is read from `configs/organ_code.yaml`:

- **CMUH**: 10 organs (1=pancreas, 2=breast, 3=cervix, 4=colorectal,
  5=esophagus, 6=liver, 7=lung, 8=prostate, 9=stomach, 10=thyroid).
- **TCGA**: 5 organs (1=breast, 2=colorectal, 3=esophagus, 4=stomach, 5=liver).

Defaults: cmuh = 100 cases per organ, tcga = 50 cases per organ,
80% cancer / 20% non-cancer, 3 LLM runs.
For a 10-run sweep matching the real experiments: `--llm-runs 10`.
To restrict to a subset: `--organs breast,colorectal,esophagus,stomach,liver`
(the cross-corpus common-5 set).

CMUH reports are clean key-value text. TCGA reports are chaotic
(dictation-style, abbreviations, shuffled sections) so the two datasets
exercise different parser robustness.
""", encoding="utf-8")


# ---- Entry point ------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("dummy"),
                    help="Root for the generated fixture (default: ./dummy)")
    ap.add_argument("--clean", action="store_true",
                    help="Wipe data/<ds>/ and results/predictions/<ds>/ for each "
                         "selected dataset before generating")
    ap.add_argument("--datasets", type=str, default=",".join(DATASETS),
                    help=f"CSV of datasets to (re)generate. Default: {','.join(DATASETS)}")
    ap.add_argument("--organs", type=str, default=None,
                    help="CSV of organ NAMES to include (e.g. 'breast,colorectal'). "
                         "Default: every organ defined for each dataset in "
                         "configs/organ_code.yaml (TCGA: 5 organs, CMUH: 10).")
    ap.add_argument("--cases-per-organ", type=str, default=DEFAULT_CASES_PER_ORGAN,
                    help=f"Either a bare int (applies to all) or 'ds:N,ds:N' CSV. "
                         f"Default: {DEFAULT_CASES_PER_ORGAN}")
    ap.add_argument("--cancer-rate", type=float, default=DEFAULT_CANCER_RATE,
                    help=f"Fraction of cases that are cancer. Default: {DEFAULT_CANCER_RATE}")
    ap.add_argument("--llm-runs", type=int, default=DEFAULT_LLM_RUNS,
                    help=f"Number of LLM runs per model in results/predictions/. "
                         f"Default: {DEFAULT_LLM_RUNS} (use 10 for the full sweep)")
    ap.add_argument("--seed", type=int, default=20251117)
    args = ap.parse_args()

    selected = [d.strip() for d in args.datasets.split(",") if d.strip()]
    unknown = [d for d in selected if d not in DATASETS]
    if unknown:
        ap.error(f"unknown dataset(s): {unknown}; known: {DATASETS}")
    if not 0.0 <= args.cancer_rate <= 1.0:
        ap.error("--cancer-rate must be in [0.0, 1.0]")
    if args.llm_runs < 1:
        ap.error("--llm-runs must be >= 1")
    try:
        cases_by_ds = parse_cases_per_organ(args.cases_per_organ, selected)
    except ValueError as e:
        ap.error(str(e))

    organ_filter: set[str] | None = None
    if args.organs:
        organ_filter = {o.strip() for o in args.organs.split(",") if o.strip()}
        unknown_organs = organ_filter - set(ALL_ORGANS)
        if unknown_organs:
            ap.error(
                f"--organs unknown organ(s): {sorted(unknown_organs)}; "
                f"valid: {ALL_ORGANS}"
            )

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.clean:
        for ds in selected:
            for sub in (out / "data" / ds, out / "results" / "predictions" / ds):
                if sub.exists():
                    shutil.rmtree(sub)

    rng = random.Random(args.seed)
    organs_per_dataset: dict[str, dict[str, str]] = {
        ds: organs_map_for(ds, restrict_to=organ_filter) for ds in selected
    }
    empty = [ds for ds, m in organs_per_dataset.items() if not m]
    if empty:
        ap.error(
            f"--organs filter left dataset(s) {empty} with no organs; check "
            f"that the requested organs exist for those datasets in "
            f"configs/organ_code.yaml."
        )

    for ds in selected:
        organs_map = organs_per_dataset[ds]
        n = cases_by_ds[ds]
        build_dataset(out, ds, organs_map, n, args.cancer_rate, rng)
        build_llm_predictions(out, ds, organs_map, n, args.cancer_rate, args.llm_runs, rng)
        build_bert_predictions(out, ds, organs_map, n, args.cancer_rate, rng)
        build_rule_based(out, ds, organs_map, n, args.cancer_rate, rng)

    build_configs(out, organs_per_dataset, args.llm_runs)
    build_model_dirs(out)
    write_skeleton_gitkeeps(out)
    write_readme(out)

    counts: dict[str, int] = {}
    for p in out.rglob("*"):
        if p.is_file():
            counts[p.suffix or "(no-ext)"] = counts.get(p.suffix or "(no-ext)", 0) + 1
    print(f"Generated fixture at: {out}")
    for ds in selected:
        n_organs = len(organs_per_dataset[ds])
        print(f"  {ds}: {cases_by_ds[ds]} cases × {n_organs} organs "
              f"× {args.cancer_rate:.0%} cancer, {args.llm_runs} LLM runs")
    for ext, n in sorted(counts.items()):
        print(f"  {ext:>10}: {n}")


if __name__ == "__main__":
    main()
