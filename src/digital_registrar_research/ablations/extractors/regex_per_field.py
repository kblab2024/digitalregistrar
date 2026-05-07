"""Regex post-extractor for the B6 ablation (free-text + regex).

The B6 cell asks the LM to produce a free-text English summary of the
report; we then claw back primary-endpoint values via regex over the
summary text. This is a deliberately weak baseline — its job is to set
the floor below which structured-output methods improve accuracy.

Patterns target the FAIR_SCOPE primary endpoints first; secondary
fields fall back to fuzzy-token matching against the per-organ enum
vocabulary.
"""
from __future__ import annotations

import re

from ...benchmarks.eval.scope import (
    BREAST_BIOMARKERS,
    FAIR_SCOPE,
    get_allowed_values,
    get_organ_scoreable_fields,
)
from .post_hoc_parser import _jaccard, _tokenise

# Roman → arabic for grades.
_ROMAN_TO_INT = {"i": 1, "ii": 2, "iii": 3, "iv": 4}

# --- Primary-endpoint patterns ------------------------------------------------

_TUMOUR_SIZE_RE = re.compile(
    r"(?i)\b(?:tumou?r\s*(?:size|measur\w+|dimension)|"
    r"(?:invasive\s+)?(?:carcinoma|lesion)\s+(?:size|measur\w+|dimension)|"
    r"size|measur\w+|greatest\s+(?:dimension|diameter))"
    r"[\s:=is]{0,8}(\d+(?:\.\d+)?)\s*(mm|cm)\b"
)

_PT_RE = re.compile(r"(?i)\b(?:p|yp)T\s?([0-4](?:[abc])?(?:is|mi)?|x|X|is)\b")
_PN_RE = re.compile(r"(?i)\b(?:p|yp)N\s?([0-3](?:[abc])?|x|X)\b")
_PM_RE = re.compile(r"(?i)\b(?:p|yp|c)M\s?([01](?:[abc])?|x|X)\b")

_GRADE_RE = re.compile(
    r"(?i)\b(?:histologic(?:al)?\s+|nottingham\s+|tumou?r\s+)?grade"
    r"\s*[:=\-]?\s*"
    r"(I{1,3}V?|IV|[1-4])\b"
)

_LVI_RE = re.compile(
    r"(?i)\blymphovascular\s+(?:space\s+)?invasion(?:\s+is)?\s*[:=\-]?\s*"
    r"(present|positive|identified|involved|seen|"
    r"absent|negative|not\s+(?:identified|seen|present))"
)
_LVI_KEYWORD_POS = re.compile(r"(?i)\blymphovascular\s+invasion\s+(?:is\s+)?(?:present|positive|identified|seen|involved)\b")
_LVI_KEYWORD_NEG = re.compile(r"(?i)\b(?:no|absent|negative)\s+lymphovascular\s+invasion\b|\blymphovascular\s+invasion\s+(?:is\s+)?(?:absent|negative|not\s+(?:identified|seen|present))\b")

_PNI_RE = re.compile(
    r"(?i)\bperineural\s+invasion(?:\s+is)?\s*[:=\-]?\s*"
    r"(present|positive|identified|involved|seen|"
    r"absent|negative|not\s+(?:identified|seen|present))"
)
_PNI_KEYWORD_POS = re.compile(r"(?i)\bperineural\s+invasion\s+(?:is\s+)?(?:present|positive|identified|seen|involved)\b")
_PNI_KEYWORD_NEG = re.compile(r"(?i)\b(?:no|absent|negative)\s+perineural\s+invasion\b|\bperineural\s+invasion\s+(?:is\s+)?(?:absent|negative|not\s+(?:identified|seen|present))\b")

_BIOMARKER_RES: dict[str, re.Pattern[str]] = {
    "er": re.compile(
        r"(?i)\b(?:estrogen\s+receptor|ER)\b\s*[:=\-(]*\s*"
        r"(positive|negative|equivocal|low|"
        r"(?:strongly?\s+)?positive|(?:weakly?\s+)?positive|"
        r"\d+\s*%)"
    ),
    "pr": re.compile(
        r"(?i)\b(?:progesterone\s+receptor|PR|PgR)\b\s*[:=\-(]*\s*"
        r"(positive|negative|equivocal|low|"
        r"(?:strongly?\s+)?positive|(?:weakly?\s+)?positive|"
        r"\d+\s*%)"
    ),
    "her2": re.compile(
        r"(?i)\b(?:HER2(?:/neu)?|c-?erbB-?2)\b\s*[:=\-(]*\s*"
        r"(positive|negative|equivocal|"
        r"score\s*[0-3]\+?|[0-3]\+)"
    ),
}

# Keyword vocabulary for cancer_category inference.
_ORGAN_KEYWORDS: dict[str, list[str]] = {
    "breast": ["breast", "mammary", "lumpectomy", "mastectomy", "nipple"],
    "lung":  ["lung", "pulmonary", "lobectomy", "bronch", "pleura"],
    "colorectal": ["colon", "rectum", "rectal", "colorectal", "sigmoid", "cecum", "cecal", "appendix"],
    "prostate": ["prostate", "prostatectomy"],
    "esophagus": ["esophagus", "esophageal", "oesophag"],
    "pancreas": ["pancreas", "pancreatic", "whipple", "pancreaticoduodenectomy"],
    "thyroid": ["thyroid", "thyroidectomy", "follicular", "papillary thyroid"],
    "cervix": ["cervix", "cervical", "uterine cervix"],
    "liver": ["liver", "hepatic", "hepatectomy", "hepatocellular"],
    "stomach": ["stomach", "gastric", "gastrectomy"],
}


# --- Helpers -----------------------------------------------------------------

def _normalise_size_to_mm(value: float, unit: str) -> float:
    return value * 10 if unit.lower() == "cm" else value


def _grade_to_int(token: str) -> int | None:
    t = token.strip().lower()
    if t in _ROMAN_TO_INT:
        return _ROMAN_TO_INT[t]
    if t.isdigit():
        return int(t)
    return None


def _bool_from_lvi_pni_token(tok: str) -> bool | None:
    t = tok.strip().lower()
    if t in {"present", "positive", "identified", "involved", "seen"}:
        return True
    if "not" in t or t in {"absent", "negative"}:
        return False
    return None


def _enum_match(text: str, allowed: list[str]) -> str | None:
    """Best fuzzy match of any phrase in ``text`` to one of ``allowed``."""
    text_l = text.lower()
    text_norm = text_l.replace("_", " ")
    best_score = 0.0
    best_opt: str | None = None
    for opt in allowed:
        opt_l = str(opt).lower().replace("_", " ")
        if not opt_l:
            continue
        if opt_l in text_norm:
            return str(opt).lower()
        score = _jaccard(_tokenise(text_l), _tokenise(opt_l))
        if score > best_score:
            best_score = score
            best_opt = str(opt).lower()
    if best_opt is not None and best_score >= 0.6:
        return best_opt
    return None


# --- Public API --------------------------------------------------------------

class RegexExtractor:
    """Stateless extractor — call :meth:`extract` per case."""

    def extract_organ(self, text: str) -> str | None:
        """Infer the cancer_category from organ-keyword frequency."""
        text_l = text.lower()
        scores: dict[str, int] = {}
        for organ, kws in _ORGAN_KEYWORDS.items():
            scores[organ] = sum(text_l.count(kw) for kw in kws)
        best_organ, best_score = max(scores.items(), key=lambda kv: kv[1])
        if best_score == 0:
            return None
        return best_organ

    def extract_primary(self, text: str, organ: str | None) -> dict:
        """Pull the primary endpoints (FAIR_SCOPE + breast biomarkers) from
        ``text`` via fixed regexes. Missing endpoints come back as None."""
        out: dict = {}

        # tumor_size — normalised to mm
        m = _TUMOUR_SIZE_RE.search(text)
        if m:
            try:
                size_mm = _normalise_size_to_mm(float(m.group(1)), m.group(2))
                out["tumor_size"] = round(size_mm, 1)
            except ValueError:
                out["tumor_size"] = None
        else:
            out["tumor_size"] = None

        # TNM categories — uppercase, strip whitespace
        for field, pattern in (
            ("pt_category", _PT_RE),
            ("pn_category", _PN_RE),
            ("pm_category", _PM_RE),
        ):
            m = pattern.search(text)
            if m:
                cat = "p" + field[1].upper() + m.group(1).strip().lower()
                out[field] = cat
            else:
                out[field] = None

        # grade — roman or arabic
        m = _GRADE_RE.search(text)
        out["grade"] = _grade_to_int(m.group(1)) if m else None

        # LVI / PNI — try the labelled regex first, then keyword scans
        for field, labelled, pos_kw, neg_kw in (
            ("lymphovascular_invasion", _LVI_RE, _LVI_KEYWORD_POS, _LVI_KEYWORD_NEG),
            ("perineural_invasion",     _PNI_RE, _PNI_KEYWORD_POS, _PNI_KEYWORD_NEG),
        ):
            m = labelled.search(text)
            value: bool | None = None
            if m:
                value = _bool_from_lvi_pni_token(m.group(1))
            elif neg_kw.search(text):
                value = False
            elif pos_kw.search(text):
                value = True
            out[field] = value

        # cancer_excision_report — assume true if we matched any cancer keyword
        # (the grader treats this as the routing decision; see
        # benchmarks/eval/scope.py:FAIR_SCOPE).
        out["cancer_excision_report"] = bool(organ) and organ != "others"
        out["cancer_category"] = organ

        # Breast biomarkers — pull as a list of dicts so it round-trips
        # through metrics.match_nested_list / score_case.
        biomarkers: list[dict] = []
        for cat in BREAST_BIOMARKERS:
            m = _BIOMARKER_RES[cat].search(text)
            if m:
                expression = m.group(1).strip().lower()
                # Map percentages and numeric scores to positive/negative.
                if expression.endswith("%"):
                    pct = float(_re_first_number(expression) or 0)
                    expression = "positive" if pct >= 1 else "negative"
                elif "score" in expression or "+" in expression:
                    score = _re_first_number(expression)
                    if score is not None:
                        expression = "positive" if score >= 3 else (
                            "equivocal" if score == 2 else "negative")
                biomarkers.append({"biomarker_category": cat,
                                   "expression": expression})
        if biomarkers:
            out["biomarkers"] = biomarkers

        return out

    def extract_secondary(self, text: str, organ: str) -> dict:
        """Fuzzy-match every other organ-scoreable field against its
        allowed-value enum. Cheap and best-effort."""
        out: dict = {}
        for field, _kind in get_organ_scoreable_fields(organ).items():
            if field in FAIR_SCOPE or field in {"tumor_size"}:
                continue
            allowed = get_allowed_values(field, organ)
            if not allowed:
                continue
            out[field] = _enum_match(text, allowed)
        return out

    def extract(self, text: str) -> dict:
        """One-shot extraction: organ inference + primary + secondary fields."""
        organ = self.extract_organ(text)
        cancer_data = self.extract_primary(text, organ)
        if organ:
            cancer_data.update(self.extract_secondary(text, organ))
        return {
            "cancer_excision_report": bool(organ) and organ != "others",
            "cancer_category": organ,
            "cancer_data": cancer_data,
        }


def _re_first_number(s: str) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


__all__ = ["RegexExtractor"]
