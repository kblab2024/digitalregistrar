"""Rule-based pathology extractor — comprehensive per-organ regex+lexicon baseline.

For each organ in :data:`IMPLEMENTED_ORGANS`, attempts every field in
``bert_scope_for_organ(organ)``: per-organ categorical (TNM, stage,
histology, procedure, ...), boolean (LVI, perineural, ...), and integer
span (tumor size, AJCC version, ...). Designed as a deterministic floor
for the head-to-head comparison against ClinicalBERT and LLM methods.

Coverage policy
---------------
A field is **emitted only when the regex/lexicon matches with confidence**.
Missing keys count as "not attempted" in the eval metrics — they drop
out of the accuracy denominator rather than counting as wrong. This is
the right behavior for a deterministic floor and produces honest per-
field coverage asymmetry vs ClinicalBERT (which always emits a class).

Usage
-----
This module is a pure extractor. The batch predict orchestrator lives at
``scripts/baselines/run_rule.py`` (canonical layout, mirrors
``scripts/pipeline/run_dspy_ollama_single.py``). For ad-hoc debugging on
a single report::

    python -m digital_registrar_research.benchmarks.baselines.rules \\
        --report path/to/report.txt
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from ..eval.bert_scope import bert_scope_for_organ
from ..eval.scope import IMPLEMENTED_ORGANS
from ..eval.scope_organs import ORGAN_BOOL, ORGAN_CATEGORICAL, ORGAN_SPAN

# ============================================================================
# Organ classification + excision-vs-biopsy detection
# ============================================================================

CANCER_LEXICON: dict[str, list[str]] = {
    "breast": [
        "breast", "mastectomy", "lumpectomy", "ductal carcinoma",
        "lobular carcinoma", "nipple", "estrogen receptor",
        "progesterone receptor", "her2", "her-2", "nottingham",
    ],
    "lung": [
        "lung", "lobectomy", "pulmonary", "bronchial", "pleural",
        "alveolar", "pneumonectomy", "bronchus",
    ],
    "colorectal": [
        "colon", "rectal", "rectum", "colectomy", "sigmoid",
        "colonic", "colorectal", "cecum", "appendix",
    ],
    "prostate": [
        "prostate", "prostatectomy", "seminal vesicle", "gleason",
    ],
    "esophagus": [
        "esophagus", "esophageal", "esophagectomy", "oesophagus",
        "oesophageal",
    ],
    "pancreas": [
        "pancreas", "pancreatic", "whipple", "pancreaticoduodenectomy",
        "pancreatectomy",
    ],
    "thyroid": [
        "thyroid", "thyroidectomy", "papillary thyroid",
        "follicular thyroid", "medullary thyroid",
    ],
    "cervix": [
        "cervix", "cervical carcinoma", "uterine cervix",
    ],
    "liver": [
        "liver", "hepatic", "hepatocellular", "hepatectomy",
    ],
    "stomach": [
        "stomach", "gastric", "gastrectomy",
    ],
}

EXCISION_LEXICON: list[str] = [
    "mastectomy", "lobectomy", "lumpectomy", "colectomy", "prostatectomy",
    "excision", "resection", "gastrectomy", "hepatectomy", "thyroidectomy",
    "whipple", "pancreatectomy", "hysterectomy", "esophagectomy",
    "pancreaticoduodenectomy", "pneumonectomy", "wedge resection",
    "segmentectomy", "polypectomy", "cystectomy",
]

NON_EXCISION_LEXICON: list[str] = [
    "biopsy only", "core needle biopsy", "core biopsy", "needle biopsy",
    "punch biopsy", "shave biopsy", "incisional biopsy",
    "cytology specimen", "fine needle aspirate", "fine needle aspiration",
    "addendum only", "consult only", "consultation report",
]


def classify_organ(report: str) -> str | None:
    """Lexicon-based majority vote on cancer category."""
    text = report.lower()
    scores: dict[str, int] = {}
    for organ, words in CANCER_LEXICON.items():
        scores[organ] = sum(text.count(w) for w in words)
    if not any(scores.values()):
        return None
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] >= 1 else None


def detect_excision_report(report: str) -> bool:
    """True when excision/resection signals outweigh biopsy/cytology signals."""
    text = report.lower()
    excision = sum(text.count(w) for w in EXCISION_LEXICON)
    non_excision = sum(text.count(w) for w in NON_EXCISION_LEXICON)
    if excision == 0 and non_excision == 0:
        return False
    return excision > non_excision


# ============================================================================
# Yes/no/abstain normalization for boolean fields
# ============================================================================

POSITIVE_WORDS = ("present", "positive", "identified", "involved", "yes")
NEGATIVE_WORDS = ("absent", "negative", "not identified", "no evidence",
                  "uninvolved", "free", "clear", "no ")
ABSTAIN_WORDS = ("unable to assess", "not assessed", "cannot be determined",
                 "indeterminate", "not applicable", "n/a")


def _norm_yes_no(s: str) -> bool | None:
    """Map a fragment like 'present', 'absent', 'not identified' → True/False/None."""
    g = " ".join(s.lower().split())
    for word in ABSTAIN_WORDS:
        if word in g:
            return None
    for word in NEGATIVE_WORDS:
        if word in g:
            return False
    for word in POSITIVE_WORDS:
        if word in g:
            return True
    return None


# ============================================================================
# TNM / stage extractors
# ============================================================================

def _enum_match_longest(report: str, allow_p_prefix: bool,
                         enum: list[str]) -> str | None:
    """Find the longest enum value that appears in report (longest-match wins)."""
    text = report
    for v in sorted(enum, key=lambda s: -len(s)):
        prefix = "p?" if allow_p_prefix else ""
        pat = re.compile(rf"\b{prefix}{re.escape(v)}\b", re.I)
        if pat.search(text):
            return v
    return None


def _extract_tnm(report: str, organ: str) -> dict:
    """Extract pT/pN/pM using the per-organ enum list (longest-match)."""
    out: dict = {}
    cat = ORGAN_CATEGORICAL.get(organ, {})
    for field in ("pt_category", "pn_category", "pm_category"):
        if field in cat:
            v = _enum_match_longest(report, allow_p_prefix=True, enum=cat[field])
            if v:
                out[field] = v
    return out


def _extract_tnm_descriptor(report: str, organ: str) -> str | None:
    """y/r/m prefix on the TNM string (e.g. ypT2 → 'y')."""
    if "tnm_descriptor" not in ORGAN_CATEGORICAL.get(organ, {}):
        return None
    m = re.search(r"\b([yrm])p?T[0-4xis]", report, re.I)
    return m.group(1).lower() if m else None


def _extract_stage_group(report: str, organ: str) -> dict:
    """Emit stage-group field(s) for this organ.

    Breast has two slots (pathologic_stage_group + anatomic_stage_group) and
    we duplicate the value when the report uses an unqualified phrasing.
    Other organs use one of {stage_group, overall_stage}.
    """
    out: dict = {}
    cat = ORGAN_CATEGORICAL.get(organ, {})
    target_keys = [k for k in (
        "pathologic_stage_group", "anatomic_stage_group",
        "stage_group", "overall_stage",
    ) if k in cat]
    if not target_keys:
        return out

    pathologic = re.search(
        r"\bpathologic(?:al)?\s*stage(?:\s*group)?\s*[:=]?\s*"
        r"([0-4ivx]+[abc]?\d?)\b", report, re.I)
    anatomic = re.search(
        r"\banatomic(?:al)?\s*stage(?:\s*group)?\s*[:=]?\s*"
        r"([0-4ivx]+[abc]?\d?)\b", report, re.I)
    overall = re.search(
        r"\b(?:overall\s*)?stage(?:\s*group)?\s*[:=]?\s*"
        r"([0-4ivx]+[abc]?\d?)\b", report, re.I)

    def _put(value: str) -> None:
        v = value.lower()
        if organ == "breast":
            if "pathologic_stage_group" in cat and v in cat["pathologic_stage_group"]:
                out.setdefault("pathologic_stage_group", v)
            if "anatomic_stage_group" in cat and v in cat["anatomic_stage_group"]:
                out.setdefault("anatomic_stage_group", v)
            return
        for key in target_keys:
            if v in cat[key]:
                out.setdefault(key, v)
                break

    if pathologic:
        v = pathologic.group(1).lower()
        if organ == "breast" and "pathologic_stage_group" in cat \
                and v in cat["pathologic_stage_group"]:
            out["pathologic_stage_group"] = v
        else:
            _put(v)
    if anatomic and organ == "breast":
        v = anatomic.group(1).lower()
        if "anatomic_stage_group" in cat and v in cat["anatomic_stage_group"]:
            out["anatomic_stage_group"] = v
    if not out and overall:
        _put(overall.group(1))
    return out


# ============================================================================
# Grade extractor (organ-aware: integer for most, Gleason group for prostate)
# ============================================================================

def _isup_grade(primary: int, secondary: int) -> int | None:
    """Map (Gleason primary, secondary) → ISUP grade group (1-5).

    Uses the standard 2014 ISUP/WHO mapping:
      ≤6  → 1
      3+4 → 2; 4+3 → 3
      8   → 4
      9-10 → 5
    """
    total = primary + secondary
    if total <= 6:
        return 1
    if (primary, secondary) == (3, 4):
        return 2
    if (primary, secondary) == (4, 3):
        return 3
    if total == 8:
        return 4
    if total >= 9:
        return 5
    return None


def _extract_grade(report: str, organ: str) -> dict:
    """Extract grade-family fields for this organ."""
    out: dict = {}
    cat = ORGAN_CATEGORICAL.get(organ, {})
    span = ORGAN_SPAN.get(organ, set())

    if organ == "prostate":
        m = re.search(
            r"gleason[^\n]{0,40}?(\d)\s*\+\s*(\d)(?:\s*=\s*\d{1,2})?",
            report, re.I,
        )
        if m:
            primary, secondary = int(m.group(1)), int(m.group(2))
            isup = _isup_grade(primary, secondary)
            if isup is not None:
                value = f"group_{isup}_{primary}_{secondary}"
                if value in cat.get("grade", []):
                    out["grade"] = value
        return out

    # Numeric grade. Two pass: explicit "grade <n>", then "(G<n>)" within
    # 60 chars of a grade label (handles "GRADE: POORLY DIFFERENTIATED (G3)").
    m = re.search(
        r"\b(?:nottingham\s*grade|histologic(?:al)?\s*grade|"
        r"tumou?r\s*grade|grade)\s*[:=]?\s*(?:g\s*)?([1-4])\b",
        report, re.I,
    )
    if not m:
        m = re.search(
            r"\b(?:nottingham\s*grade|histologic(?:al)?\s*grade|"
            r"tumou?r\s*grade|grade)\b[^\.\n]{0,60}?\(?\s*g\s*([1-4])\s*\)?",
            report, re.I,
        )
    if m:
        g = m.group(1)
        if "grade" in cat and g in cat["grade"]:
            out["grade"] = g
        elif "grade" in span:
            out["grade"] = int(g)

    # Breast Nottingham sub-scores.
    if organ == "breast":
        m = re.search(
            r"\b(?:nuclear\s*grade|nuclei)\s*[=:]?\s*([1-3])\b", report, re.I,
        )
        if m:
            out["nuclear_grade"] = m.group(1)
        m = re.search(r"\btubules?\s*[=:]\s*([1-3])\b", report, re.I)
        if m:
            out["tubule_formation"] = m.group(1)
        m = re.search(r"\bmitos[ie]s\s*[=:]\s*([1-3])\b", report, re.I)
        if m:
            out["mitotic_rate"] = m.group(1)
        m = re.search(
            r"\bnottingham\s*score\s*[:=]?\s*(\d)\s*/\s*9\b", report, re.I,
        )
        if m and m.group(1) in ("3", "4", "5", "6", "7", "8", "9"):
            out["total_score"] = m.group(1)
        m = re.search(r"\bdcis\s*grade\s*[:=]?\s*([1-3])\b", report, re.I)
        if m:
            out["dcis_grade"] = m.group(1)
    return out


# ============================================================================
# Boolean fields — phrase + yes/no proximity match
# ============================================================================

BOOL_PHRASES: dict[str, list[str]] = {
    "lymphovascular_invasion": [
        r"lymph[o\-]?vascular\s*invasion",
        r"venous\s*/\s*lymphatic\s*invasion",
        r"vascular\s*invasion",
    ],
    "perineural_invasion": [r"perineural\s*invasion"],
    "distant_metastasis": [
        r"distant\s*metastas[ei]s",
        r"distant\s*metastatic\s*disease",
    ],
    "extranodal_extension": [
        r"extranodal\s*extension",
        r"extracapsular\s*extension",
        r"extra[\s\-]?nodal\s*spread",
    ],
    "dcis_present": [
        r"ductal\s*carcinoma\s*in\s*situ",
        r"\bdcis\b",
    ],
    "dcis_comedo_necrosis": [
        r"comedo\s*necrosis",
        r"comedonecrosis",
    ],
    "visceral_pleural_invasion": [
        r"visceral\s*pleural\s*invasion",
        r"\bvpi\b",
    ],
    "spread_through_air_spaces_stas": [
        r"spread\s*through\s*air\s*spaces",
        r"\bstas\b",
    ],
    "direct_invasion_of_adjacent_structures": [
        r"invasion\s*of\s*adjacent\s*(?:organ|structure)",
        r"direct\s*invasion\s*of\s*adjacent",
    ],
    "signet_ring": [
        r"signet[\s\-]?ring\s*cell",
        r"signet[\s\-]?ring",
    ],
    "extracellular_mucin": [r"extracellular\s*mucin"],
    "extraprostatic_extension": [
        r"extraprostatic\s*extension",
        r"\bepe\b",
    ],
    "seminal_vesicle_invasion": [r"seminal\s*vesicle\s*invasion"],
    "bladder_invasion": [r"bladder\s*(?:wall\s*)?invasion"],
    "intraductal_carcinoma_presence": [r"intraductal\s*carcinoma"],
    "cribriform_pattern_presence": [r"cribriform\s*pattern"],
    "margin_positivity": [
        r"margins?\s+(?:are\s+)?(?:positive|involved)",
        r"(?:positive|involved)\s+margins?",
    ],
    "tumor_necrosis": [
        r"tumou?r\s*necrosis",
        r"necrosis(?:\s*present)?",
    ],
}

# Positive/negative outcome phrase patterns.
_OUTCOME_PAT = (
    r"(present|absent|positive|negative|identified|"
    r"not\s*identified|no\s*evidence|uninvolved|involved|"
    r"unable\s*to\s*assess|not\s*assessed|cannot\s*be\s*determined)"
)


def _extract_invasions(report: str, organ: str) -> dict:
    """For every boolean field in ORGAN_BOOL[organ], attempt a phrase+outcome match."""
    out: dict = {}
    bool_fields = ORGAN_BOOL.get(organ, set())
    for field in bool_fields:
        phrases = BOOL_PHRASES.get(field)
        if not phrases:
            continue
        for phrase_pat in phrases:
            # Forward: <phrase> ... <outcome>
            m = re.search(
                phrase_pat + r"[^\.\n]{0,80}?" + _OUTCOME_PAT,
                report, re.I,
            )
            if m:
                v = _norm_yes_no(m.group(1))
                if v is not None:
                    out[field] = v
                    break
            # Reverse: <outcome> ... <phrase>
            m2 = re.search(
                r"\b" + _OUTCOME_PAT + r"[^\.\n]{0,40}?" + phrase_pat,
                report, re.I,
            )
            if m2:
                v = _norm_yes_no(m2.group(1))
                if v is not None:
                    out[field] = v
                    break

    # Special: dcis_present is True if we see DCIS/intraductal mentioned at all
    # in a positive context, regardless of pairing.
    if "dcis_present" in bool_fields and "dcis_present" not in out:
        text = report.lower()
        if re.search(r"\bdcis\b|ductal\s*carcinoma\s*in\s*situ|"
                     r"intraductal\s*component", text):
            # Watch for negation.
            if re.search(r"(?:no|not|absent)[^\.\n]{0,30}?"
                         r"(?:dcis|ductal\s*carcinoma\s*in\s*situ|"
                         r"intraductal)", text):
                out["dcis_present"] = False
            else:
                out["dcis_present"] = True

    return out


# ============================================================================
# Span / numeric extractors (integer mm or count)
# ============================================================================

_SIZE_CUE = (
    r"tumou?r\s*size|greatest\s*dimension|"
    r"(?:size\s*of\s*)?invasive\s*(?:carcinoma|tumou?r)|"
    r"mass\s*measur(?:es|ing)|measur(?:es|ing)"
)


def _extract_tumor_size_mm(report: str) -> int | None:
    """Tumor size as integer mm, taking the maximum dimension on multi-dim reads.

    Recognizes ``tumor size: X cm``, ``tumor size: X x Y x Z cm``,
    ``greatest dimension X cm``, ``invasive carcinoma: X cm``,
    ``mass measuring X cm in greatest dimension``.
    """
    candidates: list[float] = []  # mm
    pat = (
        r"(?:" + _SIZE_CUE + r")[^\.\n]{0,40}?"
        r"((?:\d+(?:\.\d+)?\s*(?:[x×]\s*)?)+?)\s*(mm|cm)\b"
    )
    for m in re.finditer(pat, report, re.I):
        dims_str = m.group(1)
        unit = m.group(2).lower()
        dims = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", dims_str)]
        if not dims:
            continue
        max_dim = max(dims)
        mm = max_dim * 10.0 if unit == "cm" else max_dim
        candidates.append(mm)
    if not candidates:
        return None
    return int(round(max(candidates)))


def _to_mm(value: float, unit: str) -> float:
    return value * 10.0 if unit.lower() == "cm" else value


def _extract_spans(report: str, organ: str) -> dict:
    """All integer span fields the organ exposes (tumor_size, AJCC, LN sizes...)."""
    out: dict = {}
    span = ORGAN_SPAN.get(organ, set())

    if "tumor_size" in span:
        v = _extract_tumor_size_mm(report)
        if v is not None:
            out["tumor_size"] = v

    if "dcis_size" in span:
        m = re.search(
            r"\bdcis(?:\s*size)?[^\.\n]{0,50}?(\d+(?:\.\d+)?)\s*(mm|cm)",
            report, re.I,
        )
        if m:
            out["dcis_size"] = int(round(_to_mm(float(m.group(1)), m.group(2))))

    if "maximal_ln_size" in span:
        m = re.search(
            r"(?:largest|maximal|biggest)\s*"
            r"(?:lymph\s*node|ln|metastatic\s*deposit|metastasis)?"
            r"[^\.\n]{0,80}?(\d+(?:\.\d+)?)\s*(mm|cm)",
            report, re.I,
        )
        if m:
            out["maximal_ln_size"] = int(round(_to_mm(float(m.group(1)), m.group(2))))

    if "ajcc_version" in span:
        m = re.search(
            r"\bajcc\s*(?:version|edition|cancer\s*staging\s*manual,?)?\s*"
            r"(\d{1,2})(?:\s*th|\s*st|\s*nd|\s*rd|\s*ed)?",
            report, re.I,
        )
        if m:
            v = int(m.group(1))
            if 5 <= v <= 9:
                out["ajcc_version"] = v

    if "gleason_4_percentage" in span:
        m = re.search(
            r"gleason\s*(?:pattern\s*)?4[^\.\n]{0,40}?(\d{1,3})\s*%",
            report, re.I,
        )
        if m:
            v = int(m.group(1))
            if 0 <= v <= 100:
                out["gleason_4_percentage"] = v

    if "gleason_5_percentage" in span:
        m = re.search(
            r"gleason\s*(?:pattern\s*)?5[^\.\n]{0,40}?(\d{1,3})\s*%",
            report, re.I,
        )
        if m:
            v = int(m.group(1))
            if 0 <= v <= 100:
                out["gleason_5_percentage"] = v

    if "prostate_size" in span:
        m = re.search(
            r"prostate\s*(?:size|measur\w+|dimension)"
            r"[^\.\n]{0,40}?(\d+(?:\.\d+)?)\s*(mm|cm)",
            report, re.I,
        )
        if m:
            out["prostate_size"] = int(round(_to_mm(float(m.group(1)), m.group(2))))

    if "prostate_weight" in span:
        m = re.search(
            r"prostate\s*weigh\w+[^\.\n]{0,30}?(\d+(?:\.\d+)?)\s*g(?:ram)?",
            report, re.I,
        )
        if m:
            out["prostate_weight"] = int(round(float(m.group(1))))

    if "tumor_percentage" in span:
        m = re.search(
            r"tumou?r\s*(?:involve\w+|percentage|burden)"
            r"[^\.\n]{0,30}?(\d+(?:\.\d+)?)\s*%",
            report, re.I,
        )
        if m:
            v = int(round(float(m.group(1))))
            if 0 <= v <= 100:
                out["tumor_percentage"] = v

    if "tumor_budding" in span:
        m = re.search(
            r"tumou?r\s*budd\w+[^\.\n]{0,40}?"
            r"(?:count|score|grade|number)?[^\.\n]{0,10}?(\d+)",
            report, re.I,
        )
        if m:
            out["tumor_budding"] = int(m.group(1))

    return out


# ============================================================================
# Per-organ lexicons for histology / procedure / surgical_technique
# ============================================================================

HISTOLOGY_LEXICON: dict[str, dict[str, list[str]]] = {
    "breast": {
        "invasive_carcinoma_no_special_type": [
            "invasive ductal carcinoma", "infiltrating ductal carcinoma",
            "invasive carcinoma nst",
            "invasive carcinoma of no special type",
        ],
        "invasive_lobular_carcinoma": [
            "invasive lobular carcinoma", "infiltrating lobular carcinoma",
        ],
        "mixed_ductal_and_lobular_carcinoma": [
            "mixed ductal and lobular", "ductal and lobular carcinoma",
        ],
        "tubular_adenocarcinoma": [
            "tubular carcinoma", "tubular adenocarcinoma",
        ],
        "mucinous_adenocarcinoma": [
            "mucinous carcinoma", "mucinous adenocarcinoma",
            "colloid carcinoma",
        ],
        "encapsulated_papillary_carcinoma": ["encapsulated papillary"],
        "solid_papillary_carcinoma": ["solid papillary"],
        "inflammatory_carcinoma": ["inflammatory carcinoma"],
        "other_special_types": [],
    },
    "lung": {
        "adenocarcinoma": ["adenocarcinoma"],
        "squamous_cell_carcinoma": [
            "squamous cell carcinoma", "squamous carcinoma",
        ],
        "adenosquamous_carcinoma": ["adenosquamous"],
        "large_cell_carcinoma": ["large cell carcinoma"],
        "large_cell_neuroendocrine_carcinoma": ["large cell neuroendocrine"],
        "small_cell_carcinoma": ["small cell carcinoma"],
        "carcinoid_tumor": [
            "carcinoid tumor", "carcinoid tumour",
            "typical carcinoid", "atypical carcinoid",
        ],
        "sarcomatoid_carcinoma": ["sarcomatoid"],
        "pleomorphic_carcinoma": ["pleomorphic carcinoma"],
        "pulmonary_lymphoepithelioma_like_carcinoma": ["lymphoepithelioma"],
        "mucoepidermoid_carcinoma": ["mucoepidermoid"],
        "salivary_gland_type_tumor": ["salivary gland type"],
        "non_small_cell_carcinoma_not_specified": [
            "non-small cell", "non small cell", "nsclc",
        ],
        "non_small_cell_carcinoma_with_neuroendocrine_features": [
            "nsclc with neuroendocrine",
            "non-small cell carcinoma with neuroendocrine",
        ],
        "other": [],
    },
    "colorectal": {
        "adenocarcinoma": ["adenocarcinoma"],
        "mucinous_adenocarcinoma": [
            "mucinous adenocarcinoma", "mucinous carcinoma",
            "colloid carcinoma",
        ],
        "signet_ring_cell_carcinoma": ["signet ring cell"],
        "medullary_carcinoma": ["medullary carcinoma"],
        "micropapillary_adenocarcinoma": ["micropapillary"],
        "serrated_adenocarcinoma": ["serrated adenocarcinoma"],
        "adenosquamous_carcinoma": ["adenosquamous"],
        "neuroendocrine_carcinoma": ["neuroendocrine carcinoma"],
        "others": [],
    },
    "prostate": {
        "acinar_adenocarcinoma": [
            "acinar adenocarcinoma",
            "prostatic adenocarcinoma",
        ],
        "intraductal_carcinoma": ["intraductal carcinoma"],
        "ductal_adenocarcinoma": ["ductal adenocarcinoma"],
        "mixed_acinar_ductal": ["mixed acinar", "acinar and ductal"],
        "neuroendocrine_carcinoma_small_cell": [
            "small cell neuroendocrine",
            "neuroendocrine small cell",
        ],
        "others": [],
    },
    "esophagus": {
        "squamous_cell_carcinoma": [
            "squamous cell carcinoma", "squamous carcinoma",
        ],
        "adenocarcinoma": ["adenocarcinoma"],
        "adenoid_cystic_carcinoma": ["adenoid cystic"],
        "mucoepidermoid_carcinoma": ["mucoepidermoid"],
        "basaloid_squamous_cell_carcinoma": ["basaloid squamous"],
        "small_cell_carcinoma": ["small cell carcinoma"],
        "large_cell_carcinoma": ["large cell carcinoma"],
        "others": [],
    },
    "stomach": {
        "tubular_adenocarcinoma": ["tubular adenocarcinoma"],
        "poorly_cohesive_carcinoma": [
            "poorly cohesive", "poorly-cohesive",
        ],
        "mixed_tubular_poorly_cohesive": [
            "mixed tubular and poorly cohesive",
        ],
        "mucinous_adenocarcinoma": [
            "mucinous adenocarcinoma", "mucinous carcinoma",
        ],
        "mixed_mucinous_poorly_cohesive": [
            "mixed mucinous and poorly cohesive",
        ],
        "hepatoid_carcinoma": ["hepatoid"],
        "others": [],
    },
    "pancreas": {
        "ductal_adenocarcinoma_nos": [
            "ductal adenocarcinoma", "pancreatic ductal adenocarcinoma",
            "pdac",
        ],
        "ipmn_with_carcinoma": [
            "ipmn", "intraductal papillary mucinous neoplasm",
        ],
        "itpn_with_carcinoma": ["itpn", "intraductal tubulopapillary"],
        "acinar_cell_carcinoma": ["acinar cell carcinoma"],
        "solid_pseudopapillary_neoplasm": [
            "solid pseudopapillary", "spn",
        ],
        "undifferentiated_carcinoma": ["undifferentiated carcinoma"],
        "others": [],
    },
    "thyroid": {
        "papillary_thyroid_carcinoma": [
            "papillary thyroid carcinoma", "papillary carcinoma",
            "ptc",
        ],
        "follicular_thyroid_carcinoma": [
            "follicular thyroid carcinoma", "follicular carcinoma",
        ],
        "medullary_thyroid_carcinoma": [
            "medullary thyroid carcinoma", "medullary carcinoma",
        ],
        "anaplastic_thyroid_carcinoma": [
            "anaplastic thyroid carcinoma", "anaplastic carcinoma",
        ],
        "others": [],
    },
    "cervix": {
        "squamous_cell_carcinoma_hpv_associated": [
            "hpv-associated squamous", "hpv associated squamous",
        ],
        "squamous_cell_carcinoma_hpv_dependaent": [
            "hpv-independent squamous", "hpv independent squamous",
        ],
        "squamous_cell_carcinoma_nos": [
            "squamous cell carcinoma", "squamous carcinoma",
        ],
        "adenocarcinoma_hpv_associated": [
            "hpv-associated adenocarcinoma", "hpv associated adenocarcinoma",
        ],
        "adenocarcinoma_hpv_independent": [
            "hpv-independent adenocarcinoma",
        ],
        "adenocarcinoma_nos": ["adenocarcinoma"],
        "adenosquamous_carcinoma": ["adenosquamous"],
        "neuroendocrine_carcinoma": ["neuroendocrine carcinoma"],
        "glassy_cell_carcinoma": ["glassy cell"],
        "small_cell_carcinoma": ["small cell carcinoma"],
        "large_cell_carcinoma": ["large cell carcinoma"],
        "others": [],
    },
    "liver": {
        "hepatocellular_carcinoma": [
            "hepatocellular carcinoma", "hcc",
        ],
        "hepatocellular_carcinoma_fibrolamellar": ["fibrolamellar"],
        "hepatocellular_carcinoma_scirrhous": ["scirrhous"],
        "hepatocellular_carcinoma_clear_cell": ["clear cell hcc",
                                                  "clear cell hepatocellular"],
        "others": [],
    },
}


PROCEDURE_LEXICON: dict[str, dict[str, list[str]]] = {
    "breast": {
        "partial_mastectomy": ["partial mastectomy"],
        "simple_mastectomy": ["simple mastectomy"],
        "breast_conserving_surgery": [
            "breast-conserving", "breast conserving",
        ],
        "modified_radical_mastectomy": ["modified radical mastectomy"],
        "total_mastectomy": ["total mastectomy"],
        "wide_excision": ["wide excision", "wide local excision"],
        "others": [],
    },
    "lung": {
        "wedge_resection": ["wedge resection"],
        "segmentectomy": ["segmentectomy"],
        "lobectomy": ["lobectomy"],
        "completion_lobectomy": ["completion lobectomy"],
        "sleeve_lobectomy": ["sleeve lobectomy"],
        "bilobectomy": ["bilobectomy"],
        "pneumonectomy": ["pneumonectomy"],
        "major_airway_resection": ["airway resection"],
        "others": [],
    },
    "colorectal": {
        "right_hemicolectomy": ["right hemicolectomy"],
        "extended_right_hemicolectomy": ["extended right hemicolectomy"],
        "left_hemicolectomy": ["left hemicolectomy"],
        "low_anterior_resection": ["low anterior resection", "lar"],
        "anterior_resection": ["anterior resection"],
        "abdominoperineal_resection": ["abdominoperineal", "apr"],
        "total_mesorectal_excision": ["total mesorectal", "tme"],
        "total_colectomy": ["total colectomy"],
        "subtotal_colectomy": ["subtotal colectomy"],
        "segmental_colectomy": ["segmental colectomy"],
        "transanal_local_excision": [
            "transanal local excision", "transanal excision",
        ],
        "polypectomy": ["polypectomy"],
        "others": [],
    },
    "prostate": {
        "radical_prostatectomy": ["radical prostatectomy"],
        "others": [],
    },
    "esophagus": {
        "endoscopic_resection": [
            "endoscopic mucosal resection", "emr",
            "endoscopic submucosal dissection", "esd",
        ],
        "esophagectomy": ["esophagectomy", "oesophagectomy"],
        "esophagogastrectomy": [
            "esophagogastrectomy", "oesophagogastrectomy",
        ],
        "others": [],
    },
    "stomach": {
        "endoscopic_resection": [
            "endoscopic mucosal resection", "emr",
            "endoscopic submucosal dissection", "esd",
        ],
        "partial_gastrectomy": [
            "partial gastrectomy", "subtotal gastrectomy",
            "distal gastrectomy", "proximal gastrectomy",
        ],
        "total_gastrectomy": ["total gastrectomy"],
        "others": [],
    },
    "pancreas": {
        "partial_pancreatectomy": ["partial pancreatectomy"],
        "ssppd": ["subtotal stomach-preserving pancreaticoduodenectomy",
                   "ssppd"],
        "pppd": [
            "pylorus-preserving pancreaticoduodenectomy",
            "pylorus preserving pancreaticoduodenectomy", "pppd",
        ],
        "whipple_procedure": ["whipple"],
        "distal_pancreatectomy": ["distal pancreatectomy"],
        "total_pancreatectomy": ["total pancreatectomy"],
        "others": [],
    },
    "thyroid": {
        "partial_excision": ["partial excision"],
        "right_lobectomy": ["right lobectomy"],
        "left_lobectomy": ["left lobectomy"],
        "total_thyroidectomy": ["total thyroidectomy"],
        "others": [],
    },
    "cervix": {
        "radical_hysterectomy": ["radical hysterectomy"],
        "total_hysterectomy_bso": [
            "total hysterectomy with bilateral salpingo-oophorectomy",
            "total abdominal hysterectomy with bso",
            "tah-bso", "tah/bso",
        ],
        "simple_hysterectomy": ["simple hysterectomy"],
        "extenteration": ["pelvic exenteration", "exenteration"],
        "others": [],
    },
    "liver": {
        "wedge_resection": ["wedge resection"],
        "partial_hepatectomy": ["partial hepatectomy"],
        "segmentectomy": ["segmentectomy"],
        "lobectomy": ["lobectomy"],
        "total_hepatectomy": ["total hepatectomy"],
        "others": [],
    },
}


SURGICAL_TECHNIQUE_LEXICON: dict[str, dict[str, list[str]]] = {
    "lung": {
        "open": ["open thoracotomy", "open approach"],
        "thoracoscopic": ["thoracoscopic", "vats"],
        "robotic": ["robotic"],
        "hybrid": ["hybrid"],
        "others": [],
    },
    "colorectal": {
        "open": ["open approach", "open laparotomy"],
        "laparoscopic": ["laparoscopic"],
        "robotic": ["robotic"],
        "ta_tme": ["ta-tme", "ta tme", "transanal total mesorectal"],
        "hybrid": ["hybrid"],
        "others": [],
    },
    "prostate": {
        "open": ["open approach", "open prostatectomy"],
        "robotic": ["robotic"],
        "hybrid": ["hybrid"],
        "others": [],
    },
    "esophagus": {
        "open": ["open approach", "open thoracotomy"],
        "thoracoscopic": ["thoracoscopic"],
        "robotic": ["robotic"],
        "hybrid": ["hybrid"],
        "endoscopic": ["endoscopic"],
        "others": [],
    },
    "stomach": {
        "open": ["open approach", "open laparotomy"],
        "laparoscopic": ["laparoscopic"],
        "robotic": ["robotic"],
        "hybrid": ["hybrid"],
        "others": [],
    },
    "cervix": {
        "open": ["open approach", "open laparotomy"],
        "laparoscopic": ["laparoscopic"],
        "vaginal": ["vaginal approach"],
        "others": [],
    },
}


# ============================================================================
# Free-form categorical extractors (laterality, sites, etc.)
# ============================================================================

def _extract_categorical_lexicon(report: str, organ: str, field: str,
                                  lexicon: dict[str, list[str]]) -> str | None:
    """Score each enum value by lexicon hits and return the top scorer."""
    text = report.lower()
    enum = ORGAN_CATEGORICAL.get(organ, {}).get(field, [])
    scores: dict[str, int] = {}
    for v in enum:
        words = lexicon.get(v) or [v.replace("_", " ")]
        hits = sum(text.count(w.lower()) for w in words if w)
        if hits:
            scores[v] = hits
    if not scores:
        return None
    return max(scores, key=lambda k: scores[k])


def _extract_laterality(report: str, organ: str) -> dict:
    out: dict = {}
    cat = ORGAN_CATEGORICAL.get(organ, {})
    text = report.lower()

    if "cancer_laterality" in cat:
        opts = cat["cancer_laterality"]
        if "bilateral" in text and "bilateral" in opts:
            out["cancer_laterality"] = "bilateral"
        elif re.search(r"\bright\s+(?:breast|side|lung|kidney|lobe)", text) \
                and "right" in opts:
            out["cancer_laterality"] = "right"
        elif re.search(r"\bleft\s+(?:breast|side|lung|kidney|lobe)", text) \
                and "left" in opts:
            out["cancer_laterality"] = "left"

    if "sideness" in cat:
        opts = cat["sideness"]
        if re.search(r"\bright\s+(?:lung|lobe|main\s*bronchus)", text) \
                and "right" in opts:
            out["sideness"] = "right"
        elif re.search(r"\bleft\s+(?:lung|lobe|main\s*bronchus)", text) \
                and "left" in opts:
            out["sideness"] = "left"
        elif "midline" in text and "midline" in opts:
            out["sideness"] = "midline"

    if "cancer_quadrant" in cat:
        opts = cat["cancer_quadrant"]
        for v in opts:
            phrase = v.replace("_", " ")
            if phrase in text:
                out["cancer_quadrant"] = v
                break

    return out


def _extract_per_organ_categoricals(report: str, organ: str) -> dict:
    """Histology, procedure, surgical_technique, primary site, laterality, ..."""
    out: dict = {}
    cat = ORGAN_CATEGORICAL.get(organ, {})

    if "histology" in cat and organ in HISTOLOGY_LEXICON:
        v = _extract_categorical_lexicon(
            report, organ, "histology", HISTOLOGY_LEXICON[organ],
        )
        if v:
            out["histology"] = v

    if "procedure" in cat and organ in PROCEDURE_LEXICON:
        v = _extract_categorical_lexicon(
            report, organ, "procedure", PROCEDURE_LEXICON[organ],
        )
        if v:
            out["procedure"] = v

    if "surgical_technique" in cat and organ in SURGICAL_TECHNIQUE_LEXICON:
        v = _extract_categorical_lexicon(
            report, organ, "surgical_technique",
            SURGICAL_TECHNIQUE_LEXICON[organ],
        )
        if v:
            out["surgical_technique"] = v

    # Generic plain-name match for the remaining enum fields.
    for field in (
        "cancer_primary_site", "tumor_focality", "tumor_invasion",
        "type_of_polyp", "tumor_extent", "tumor_site", "tumor_extension",
        "extrathyroid_extension", "mitotic_activity",
        "predisposing_condition", "depth_of_invasion_number",
        "depth_of_invasion_three_tier", "margin_length",
    ):
        if field not in cat:
            continue
        # liver tumor_extent is list-of-literals (out of bert_scope) — guard.
        if organ == "liver" and field == "tumor_extent":
            continue
        text = report.lower()
        for v in sorted(cat[field], key=lambda s: -len(s)):
            phrase = v.replace("_", " ")
            if phrase in text:
                out[field] = v
                break

    out.update(_extract_laterality(report, organ))

    if "cancer_clock" in cat:
        m = re.search(
            r"(?:at|located\s+at|between)\s*(\d{1,2})\s*o['’]?\s*clock",
            report, re.I,
        )
        if m:
            v = m.group(1)
            if v in cat["cancer_clock"]:
                out["cancer_clock"] = v
    return out


# ============================================================================
# Top-level extraction dispatch
# ============================================================================

def extract_for_organ(report: str, organ: str | None) -> dict:
    """Return a flat dict in the gold-annotation shape for ``organ``.

    Fields not in ``bert_scope_for_organ(organ)`` are NOT emitted; missing
    extractions leave the key absent (not explicit null) — see
    ``metrics.field_correct`` for why.
    """
    cancer_excision_report = detect_excision_report(report)
    cancer_data: dict = {}

    if organ in IMPLEMENTED_ORGANS:
        cancer_data.update(_extract_tnm(report, organ))
        td = _extract_tnm_descriptor(report, organ)
        if td:
            cancer_data["tnm_descriptor"] = td
        cancer_data.update(_extract_stage_group(report, organ))
        cancer_data.update(_extract_grade(report, organ))
        cancer_data.update(_extract_invasions(report, organ))
        cancer_data.update(_extract_spans(report, organ))
        cancer_data.update(_extract_per_organ_categoricals(report, organ))
        # Schema-conformance guard: prune any field not in the BERT scope.
        allowed = bert_scope_for_organ(organ) - {
            "cancer_category", "cancer_excision_report",
        }
        cancer_data = {k: v for k, v in cancer_data.items() if k in allowed}

    return {
        "cancer_excision_report": cancer_excision_report,
        "cancer_category": organ,
        "cancer_category_others_description": None,
        "cancer_data": cancer_data,
    }


def extract(report: str) -> dict:
    """Backward-compatible wrapper: classify organ then dispatch.

    For breast, also emits the legacy biomarkers list-of-dicts under
    ``cancer_data.biomarkers`` so the FAIR_SCOPE comparison continues to
    work. The biomarkers field is NOT in ``bert_scope`` and is filtered
    out by ``extract_for_organ`` — we re-add it here for legacy callers.
    """
    organ = classify_organ(report)
    if organ is None:
        return {
            "cancer_excision_report": detect_excision_report(report),
            "cancer_category": None,
            "cancer_category_others_description": None,
            "cancer_data": {},
        }
    result = extract_for_organ(report, organ)
    if organ == "breast":
        bm = _extract_breast_biomarkers(report)
        if bm:
            result["cancer_data"]["biomarkers"] = bm
    return result


def _extract_breast_biomarkers(report: str) -> list[dict]:
    """Legacy fair-scope emission: ER/PR/HER2 as list-of-dicts."""
    bm: list[dict] = []
    m = re.search(
        r"\b(?:estrogen\s*receptor|\bER\b)\s*[:=]?\s*"
        r"[^\.\n]{0,40}?(positive|negative)",
        report, re.I,
    )
    if m:
        bm.append({
            "biomarker_category": "er",
            "expression": m.group(1).lower() == "positive",
            "percentage": None, "score": None,
            "biomarker_name": "estrogen receptor",
        })
    m = re.search(
        r"\b(?:progesterone\s*receptor|\bPR\b)\s*[:=]?\s*"
        r"[^\.\n]{0,40}?(positive|negative)",
        report, re.I,
    )
    if m:
        bm.append({
            "biomarker_category": "pr",
            "expression": m.group(1).lower() == "positive",
            "percentage": None, "score": None,
            "biomarker_name": "progesterone receptor",
        })
    m = re.search(
        r"\bHER[\s\-]?2\b\s*[:=]?\s*"
        r"[^\.\n]{0,80}?(positive|negative|equivocal)",
        report, re.I,
    )
    if m:
        status = m.group(1).lower()
        bm.append({
            "biomarker_category": "her2",
            "expression": None if status == "equivocal" else (status == "positive"),
            "percentage": None,
            "score": 2 if status == "equivocal" else (3 if status == "positive" else 0),
            "biomarker_name": "human epidermal growth factor receptor 2",
        })
    return bm


# ============================================================================
# Single-report debug CLI
# ============================================================================
#
# Batch prediction lives in ``scripts/baselines/run_rule.py`` (canonical
# layout). This entry point is for ad-hoc debugging only — extract the
# fields from one file and print the JSON.


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Rule-based pathology extractor — single-file debug CLI. "
                    "For batch prediction, use scripts/baselines/run_rule.py.",
    )
    ap.add_argument(
        "--report", type=Path, required=True,
        help="Path to a pathology report .txt file.",
    )
    ap.add_argument(
        "--organ", default=None,
        help="Optional: skip lexicon classification and dispatch to "
             "this organ directly. Useful for debugging per-organ helpers.",
    )
    args = ap.parse_args()

    report_text = args.report.read_text(encoding="utf-8")
    if args.organ is not None:
        result = extract_for_organ(report_text, args.organ)
    else:
        result = extract(report_text)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
