"""Generate a synthetic dummy dataset for smoke-testing the compare GUI.

Produces a self-contained `dummy_data/` directory matching the folder
layout that `annotation_io.discover_folders` expects: a `fake_dataset_*`,
`fake_result_*`, and `fake_annotation_*` trio sharing one date.

Fifteen samples across three cancer types (breast, colorectal, stomach).
Per-sample disagreement pattern cycles through five flavors so every
branch of the diff UI has something to display:

  pattern 1: full agreement (A == B)
  pattern 2: scalar-field mismatch
  pattern 3: array item same key, different inner field
  pattern 4: array item present on one side only
  pattern 5: top-level cancer_excision_report disagreement

Machine output (`*_output.json`) is a deliberately imperfect copy of A so
Evaluation mode has something non-trivial to diff.

Idempotent: wipes `dummy_data/` at start.
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT / "dummy_data"
DATE = "20260423"
PREFIX = "fake"


# ── Report text templates ──────────────────────────────────────────────────────

BREAST_REPORT = """# SYNTHETIC — for GUI testing only. Not a real patient record.

Surgical Pathology Report
Specimen: Right breast, {procedure_human}
Clinical History: Palpable mass upper outer quadrant, right breast.

Gross Description:
The specimen is oriented with a suture at 12 o'clock. On sectioning, a firm
tan tumor measuring {tumor_size} mm is identified in the upper outer quadrant
at the 3 o'clock position. Margins are inked and submitted in entirety.

Microscopic Description:
Invasive carcinoma, {histology_human}, is present. The tumor measures
{tumor_size} mm. Nuclear grade {nuclear_grade}, tubule formation score
{tubule_formation}, mitotic rate {mitotic_rate}. Total Nottingham score
{total_score}, final grade {grade}.

Margins: {margin_summary}

Regional Lymph Nodes: {ln_summary}

Biomarkers (IHC):
ER: {er_status}
PR: {pr_status}
HER2: score {her2_score}

Diagnosis:
Invasive carcinoma of the right breast, see synoptic report above.
pT{pt_category} pN{pn_category} pM{pm_category}, stage group {stage_group}.
AJCC {ajcc_version}th edition.
"""


COLORECTAL_REPORT = """# SYNTHETIC — for GUI testing only. Not a real patient record.

Surgical Pathology Report
Specimen: Colon, {procedure_human}
Clinical History: Obstructing mass, right colon.

Gross Description:
The specimen consists of a segment of colon, {technique_human}ly resected.
An ulcerating tumor is identified in the {site_human}, measuring
approximately {tumor_size} mm. Margins are inked.

Microscopic Description:
{histology_human}, grade {grade}, invading through {tumor_invasion_human}.
Lymphovascular invasion: {lvi_text}. Perineural invasion: {pni_text}.
Tumor budding count: {budding}.

Margins: {margin_summary}

Regional Lymph Nodes: {ln_summary}

Immunohistochemistry (MMR panel):
MLH1: {mlh1}
MSH2: {msh2}
MSH6: {msh6}

Diagnosis:
Adenocarcinoma of the {site_human}, see synoptic report.
pT{pt_category} pN{pn_category} pM{pm_category}, stage group {stage_group}.
"""


STOMACH_REPORT = """# SYNTHETIC — for GUI testing only. Not a real patient record.

Surgical Pathology Report
Specimen: Stomach, {procedure_human}
Clinical History: Gastric mass on endoscopy.

Gross Description:
The specimen is a {procedure_human} performed {technique_human}ly. An
ulcerated tumor is present in the {site_human}, measuring approximately
{tumor_size} mm in greatest dimension.

Microscopic Description:
{histology_human}, grade {grade}, tumor extends to the
{tumor_extent_human}. Lymphovascular invasion: {lvi_text}. Perineural
invasion: {pni_text}.

Margins: {margin_summary}

Regional Lymph Nodes: {ln_summary}

Diagnosis:
Adenocarcinoma of the stomach ({site_human}).
pT{pt_category} pN{pn_category} pM{pm_category}, stage group {stage_group}.
"""


def _humanize(s: str | None) -> str:
    if s is None:
        return "not specified"
    return s.replace("_", " ")


def _yn(v: bool | None) -> str:
    if v is None:
        return "not reported"
    return "present" if v else "not identified"


def _margin_summary(margins: list[dict]) -> str:
    lines = []
    for m in margins:
        cat = _humanize(m.get("margin_category"))
        if m.get("margin_involved"):
            lines.append(f"  {cat}: involved")
        else:
            dist = m.get("distance")
            dist_txt = f"{dist} mm clear" if dist is not None else "negative"
            lines.append(f"  {cat}: {dist_txt}")
    return "\n" + "\n".join(lines) if lines else " not assessed"


def _ln_summary(lns: list[dict]) -> str:
    lines = []
    for ln in lns:
        cat = _humanize(ln.get("lymph_node_category"))
        involved = ln.get("involved", 0)
        examined = ln.get("examined", 0)
        lines.append(f"  {cat}: {involved}/{examined} involved")
    return "\n" + "\n".join(lines) if lines else " none sampled"


# ── Base annotation builders ───────────────────────────────────────────────────

def _breast_base(variant: int) -> dict:
    """Return a fully-filled breast annotation. Variant 0..4 tweaks values."""
    return {
        "cancer_excision_report": True,
        "cancer_category": "breast",
        "cancer_category_others_description": None,
        "cancer_data": {
            "procedure": "partial_mastectomy",
            "cancer_quadrant": "upper_outer_quadrant",
            "cancer_clock": 3,
            "cancer_laterality": "right",
            "histology": "invasive_carcinoma_no_special_type",
            "tumor_size": 17 + variant,
            "lymphovascular_invasion": False,
            "perineural_invasion": False,
            "distant_metastasis": False,
            "treatment_effect": None,
            "nuclear_grade": 2,
            "tubule_formation": 2,
            "mitotic_rate": 1,
            "total_score": 5,
            "grade": 2,
            "dcis_present": True,
            "dcis_size": 3,
            "dcis_comedo_necrosis": False,
            "dcis_grade": 2,
            "tnm_descriptor": None,
            "pt_category": "t1c",
            "pn_category": "n0",
            "pm_category": "m0",
            "pathologic_stage_group": "ia",
            "anatomic_stage_group": "ia",
            "ajcc_version": 8,
            "extranodal_extension": False,
            "maximal_ln_size": None,
            "margins": [
                {"margin_category": "12_3_clock", "margin_involved": False,
                 "distance": 8, "description": "negative"},
                {"margin_category": "3_6_clock", "margin_involved": False,
                 "distance": 5, "description": "negative"},
                {"margin_category": "base", "margin_involved": False,
                 "distance": 12, "description": "negative"},
            ],
            "regional_lymph_node": [
                {"lymph_node_side": "right", "lymph_node_category": "sentinel",
                 "involved": 0, "examined": 3, "station_name": "SLN"},
            ],
            "biomarkers": [
                {"biomarker_category": "er", "expression": True,
                 "percentage": 95, "score": None, "biomarker_name": "estrogen receptor"},
                {"biomarker_category": "pr", "expression": True,
                 "percentage": 80, "score": None, "biomarker_name": "progesterone receptor"},
                {"biomarker_category": "her2", "expression": None,
                 "percentage": None, "score": 1, "biomarker_name": "HER2/neu"},
                {"biomarker_category": "ki67", "expression": None,
                 "percentage": 15, "score": None, "biomarker_name": "Ki-67"},
            ],
        },
    }


def _colorectal_base(variant: int) -> dict:
    return {
        "cancer_excision_report": True,
        "cancer_category": "colorectal",
        "cancer_category_others_description": None,
        "cancer_data": {
            "procedure": "right_hemicolectomy",
            "surgical_technique": "laparoscopic",
            "cancer_primary_site": "ascending_colon",
            "histology": "adenocarcinoma",
            "grade": 2,
            "tumor_invasion": "muscularis_propria",
            "lymphovascular_invasion": True,
            "perineural_invasion": False,
            "extracellular_mucin": False,
            "signet_ring": False,
            "tumor_budding": 4 + variant,
            "type_of_polyp": None,
            "distant_metastasis": False,
            "treatment_effect": None,
            "tnm_descriptor": None,
            "pt_category": "t3",
            "pn_category": "n1a",
            "pm_category": "m0",
            "stage_group": "iiia",
            "ajcc_version": 8,
            "extranodal_extension": False,
            "maximal_ln_size": 5,
            "margins": [
                {"margin_category": "proximal", "margin_involved": False,
                 "distance": 40, "description": "negative"},
                {"margin_category": "distal", "margin_involved": False,
                 "distance": 35, "description": "negative"},
            ],
            "regional_lymph_node": [
                {"lymph_node_category": "regional", "involved": 1,
                 "examined": 18, "station_name": "pericolonic"},
            ],
            "biomarkers": [
                {"biomarker_category": "mlh1", "expression": True,
                 "percentage": None, "score": None, "biomarker_name": "MLH1"},
                {"biomarker_category": "msh2", "expression": True,
                 "percentage": None, "score": None, "biomarker_name": "MSH2"},
                {"biomarker_category": "msh6", "expression": True,
                 "percentage": None, "score": None, "biomarker_name": "MSH6"},
            ],
        },
    }


def _stomach_base(variant: int) -> dict:
    return {
        "cancer_excision_report": True,
        "cancer_category": "stomach",
        "cancer_category_others_description": None,
        "cancer_data": {
            "procedure": "partial_gastrectomy",
            "surgical_technique": "open",
            "cancer_primary_site": "body",
            "histology": "tubular_adenocarcinoma",
            "grade": 2,
            "tumor_extent": "submucosa",
            "extracellular_mucin": False,
            "signet_ring": False,
            "lymphovascular_invasion": False,
            "perineural_invasion": False,
            "distant_metastasis": False,
            "treatment_effect": None,
            "tnm_descriptor": None,
            "pt_category": "t1b",
            "pn_category": "n0",
            "pm_category": "m0",
            "stage_group": "i",
            "ajcc_version": 8,
            "extranodal_extension": False,
            "maximal_ln_size": None,
            "margins": [
                {"margin_category": "proximal", "margin_involved": False,
                 "distance": 20, "description": "negative"},
                {"margin_category": "distal", "margin_involved": False,
                 "distance": 18, "description": "negative"},
            ],
            "regional_lymph_node": [
                {"lymph_node_category": "regional", "involved": 0,
                 "examined": 12 + variant, "station_name": "perigastric"},
            ],
        },
    }


# ── Disagreement injectors ─────────────────────────────────────────────────────

def _inject_scalar(b: dict, cancer: str) -> dict:
    cd = b["cancer_data"]
    if cancer == "breast":
        cd["grade"] = 3
        cd["nuclear_grade"] = 3
    elif cancer == "colorectal":
        cd["grade"] = 3
        cd["tumor_invasion"] = "subserosa"
    else:
        cd["grade"] = 3
        cd["tumor_extent"] = "muscularis_propria"
    return b


def _inject_array_inner(b: dict, cancer: str) -> dict:
    """Same keys on both sides, different inner-field values."""
    cd = b["cancer_data"]
    if cancer == "breast" and cd.get("margins"):
        cd["margins"][0]["distance"] = 2
        cd["margins"][0]["description"] = "close"
    elif cancer == "colorectal" and cd.get("regional_lymph_node"):
        cd["regional_lymph_node"][0]["involved"] = 3
        cd["regional_lymph_node"][0]["examined"] = 20
    else:
        if cd.get("margins"):
            cd["margins"][0]["distance"] = 10
            cd["margins"][0]["description"] = "close"
    return b


def _inject_array_missing(b: dict, cancer: str) -> dict:
    """Drop one array item to create an A-only / B-only slot."""
    cd = b["cancer_data"]
    if cancer == "breast":
        cd["margins"] = cd["margins"][:-1]  # drop base margin
        if cd.get("biomarkers"):
            cd["biomarkers"] = [bm for bm in cd["biomarkers"]
                                if bm.get("biomarker_category") != "ki67"]
    elif cancer == "colorectal":
        cd["margins"] = cd["margins"][:1]
    else:
        cd["margins"] = cd["margins"][:1]
    return b


def _inject_top_disagreement(b: dict, cancer: str) -> dict:
    """Top-level classification disagreement — B says 'not a cancer report'."""
    b["cancer_excision_report"] = False
    b["cancer_category"] = None
    b["cancer_category_others_description"] = None
    b["cancer_data"] = None
    return b


# ── Machine output perturbation ────────────────────────────────────────────────

def _machine_perturb(a: dict, cancer: str, idx: int) -> dict:
    """Produce a plausible GPT-style output that's ~80% correct vs A."""
    m = copy.deepcopy(a)
    cd = m.get("cancer_data") or {}
    # Always wrong on histology treatment_effect
    cd["treatment_effect"] = "no known presurgical therapy"
    # Perturb a grade-ish field
    if cancer == "breast":
        cd["mitotic_rate"] = 2
        cd["total_score"] = 6
        if cd.get("biomarkers"):
            for bm in cd["biomarkers"]:
                if bm.get("biomarker_category") == "ki67":
                    bm["percentage"] = None  # machine missed it
    elif cancer == "colorectal":
        cd["tumor_budding"] = None
        cd["lymphovascular_invasion"] = None
    else:
        cd["grade"] = cd.get("grade") if cd.get("grade") != 1 else 2
        if cd.get("margins"):
            cd["margins"][0]["description"] = None
    # Rotate one more field based on idx so the 5 samples of each type differ
    if idx % 2 == 0 and cd.get("pn_category"):
        cd["pn_category"] = cd["pn_category"]  # leave alone for variety
    m["cancer_data"] = cd
    return m


# ── Report rendering ───────────────────────────────────────────────────────────

def _render_breast_report(a: dict) -> str:
    cd = a["cancer_data"]
    er = next((bm for bm in cd["biomarkers"] if bm["biomarker_category"] == "er"), {})
    pr = next((bm for bm in cd["biomarkers"] if bm["biomarker_category"] == "pr"), {})
    her2 = next((bm for bm in cd["biomarkers"] if bm["biomarker_category"] == "her2"), {})
    return BREAST_REPORT.format(
        procedure_human=_humanize(cd["procedure"]),
        tumor_size=cd["tumor_size"],
        histology_human=_humanize(cd["histology"]),
        nuclear_grade=cd["nuclear_grade"],
        tubule_formation=cd["tubule_formation"],
        mitotic_rate=cd["mitotic_rate"],
        total_score=cd["total_score"],
        grade=cd["grade"],
        margin_summary=_margin_summary(cd.get("margins") or []),
        ln_summary=_ln_summary(cd.get("regional_lymph_node") or []),
        er_status=f"{er.get('percentage', 'n/a')}% positive" if er.get("expression") else "negative",
        pr_status=f"{pr.get('percentage', 'n/a')}% positive" if pr.get("expression") else "negative",
        her2_score=her2.get("score", "n/a"),
        pt_category=cd["pt_category"],
        pn_category=cd["pn_category"],
        pm_category=cd["pm_category"],
        stage_group=cd["pathologic_stage_group"],
        ajcc_version=cd["ajcc_version"],
    )


def _render_colorectal_report(a: dict) -> str:
    cd = a["cancer_data"]
    bios = {bm["biomarker_category"]: bm for bm in cd.get("biomarkers") or []}
    return COLORECTAL_REPORT.format(
        procedure_human=_humanize(cd["procedure"]),
        technique_human=_humanize(cd["surgical_technique"]),
        site_human=_humanize(cd["cancer_primary_site"]),
        tumor_size=18,
        histology_human=_humanize(cd["histology"]).capitalize(),
        grade=cd["grade"],
        tumor_invasion_human=_humanize(cd["tumor_invasion"]),
        lvi_text=_yn(cd["lymphovascular_invasion"]),
        pni_text=_yn(cd["perineural_invasion"]),
        budding=cd["tumor_budding"] if cd["tumor_budding"] is not None else "not reported",
        margin_summary=_margin_summary(cd.get("margins") or []),
        ln_summary=_ln_summary(cd.get("regional_lymph_node") or []),
        mlh1="retained" if bios.get("mlh1", {}).get("expression") else "lost",
        msh2="retained" if bios.get("msh2", {}).get("expression") else "lost",
        msh6="retained" if bios.get("msh6", {}).get("expression") else "lost",
        pt_category=cd["pt_category"],
        pn_category=cd["pn_category"],
        pm_category=cd["pm_category"],
        stage_group=cd["stage_group"],
    )


def _render_stomach_report(a: dict) -> str:
    cd = a["cancer_data"]
    return STOMACH_REPORT.format(
        procedure_human=_humanize(cd["procedure"]),
        technique_human=_humanize(cd["surgical_technique"]),
        site_human=_humanize(cd["cancer_primary_site"]),
        tumor_size=14,
        histology_human=_humanize(cd["histology"]).capitalize(),
        grade=cd["grade"],
        tumor_extent_human=_humanize(cd["tumor_extent"]),
        lvi_text=_yn(cd["lymphovascular_invasion"]),
        pni_text=_yn(cd["perineural_invasion"]),
        margin_summary=_margin_summary(cd.get("margins") or []),
        ln_summary=_ln_summary(cd.get("regional_lymph_node") or []),
        pt_category=cd["pt_category"],
        pn_category=cd["pn_category"],
        pm_category=cd["pm_category"],
        stage_group=cd["stage_group"],
    )


CANCER_CONFIG = [
    ("fake1", "breast",     _breast_base,     _render_breast_report),
    ("fake2", "colorectal", _colorectal_base, _render_colorectal_report),
    ("fake3", "stomach",    _stomach_base,    _render_stomach_report),
]

PATTERN_LABELS = {
    1: "full-agreement",
    2: "scalar-mismatch",
    3: "array-inner-mismatch",
    4: "array-item-missing",
    5: "top-level-disagreement",
}


def _apply_pattern(a: dict, cancer: str, pattern: int) -> dict:
    """Return the B-side annotation given the A-side and disagreement pattern."""
    b = copy.deepcopy(a)
    if pattern == 1:
        return b  # full agreement
    if pattern == 2:
        return _inject_scalar(b, cancer)
    if pattern == 3:
        return _inject_array_inner(b, cancer)
    if pattern == 4:
        return _inject_array_missing(b, cancer)
    if pattern == 5:
        return _inject_top_disagreement(b, cancer)
    raise ValueError(pattern)


# ── Payload I/O ────────────────────────────────────────────────────────────────

def _save(path: Path, data: dict, filename: str, consolidated_from=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_meta": {
            "filename": filename,
            "annotated_at": "2026-04-23T10:00:00",
        }
    }
    if consolidated_from is not None:
        payload["_meta"]["consolidated_from"] = consolidated_from
    for k, v in data.items():
        if not str(k).startswith("_"):
            payload[k] = v
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _save_result(path: Path, data: dict) -> None:
    """Machine output — no _meta block, same as real pre-annotation files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    dataset_dir = OUT / f"{PREFIX}_dataset_{DATE}"
    result_dir = OUT / f"{PREFIX}_result_{DATE}"
    annotation_dir = OUT / f"{PREFIX}_annotation_{DATE}"

    summary_rows = []

    for stem_idx, (stem, cancer, base_fn, render_fn) in enumerate(CANCER_CONFIG, start=1):
        n = str(stem_idx)
        for i in range(1, 6):  # 5 samples per cancer
            sample_id = f"{stem}_{i}"
            pattern = i  # patterns 1..5
            a = base_fn(variant=i - 1)
            b = _apply_pattern(a, cancer, pattern)
            machine = _machine_perturb(a, cancer, i)

            # 1. report text
            report = render_fn(a)
            (dataset_dir / stem).mkdir(parents=True, exist_ok=True)
            (dataset_dir / stem / f"{sample_id}.txt").write_text(report, encoding="utf-8")

            # 2. machine output
            _save_result(result_dir / n / f"{sample_id}_output.json", machine)

            # 3. annotator A (nhc) and B (kpc)
            fname = f"{sample_id}.txt"
            _save(annotation_dir / n / f"{sample_id}_annotation_nhc.json", a, fname)
            _save(annotation_dir / n / f"{sample_id}_annotation_kpc.json", b, fname)

            summary_rows.append((sample_id, cancer, PATTERN_LABELS[pattern]))

    print(f"Wrote {len(summary_rows)} samples to {OUT}")
    print(f"  dataset    : {dataset_dir}")
    print(f"  result     : {result_dir}")
    print(f"  annotation : {annotation_dir}")
    print()
    print(f"{'sample_id':<12} {'cancer':<12} pattern")
    for row in summary_rows:
        print(f"{row[0]:<12} {row[1]:<12} {row[2]}")


if __name__ == "__main__":
    main()
