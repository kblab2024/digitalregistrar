"""Rule-based organ classifier — replaces folder-number lookups in
ablation runners with a deterministic keyword scorer over the report
text itself.

Use this whenever an ablation runner needs to translate a numeric organ
folder code into an organ name. The previous approach
(``IMPLEMENTED_ORGANS[idx-1]``) silently misclassified TCGA and CMUH
data because their folder numbering does not match the alphabetical
ordering of the organ list.

Fallback chain:
    1. Score the report text against per-organ keyword tables.
    2. If every score is zero, look up the dataset+organ_n via
       :mod:`digital_registrar_research.benchmarks.organs`.
    3. Otherwise return ``None``.

The keyword tables intentionally cover surgical procedures, anatomy,
and a handful of organ-specific biomarkers / staging terms. They are
not exhaustive — they only need to discriminate between the ten
organs supported by the project.
"""
from __future__ import annotations

import re
from functools import cache

from ...benchmarks.organs import organ_n_to_name

# Per-organ keyword lists (case-insensitive). Order matters only for
# tie-breaking on equal scores — first organ wins.
_KEYWORDS: dict[str, tuple[str, ...]] = {
    "breast":     ("mastectomy", "lumpectomy", "axillary", "nipple",
                   "DCIS", "ER/PR", "HER2", "breast", "ductal carcinoma"),
    "colorectal": ("colectomy", "rectum", "rectal", "colon", "polyp",
                   "MMR", "MSI", "sigmoid", "cecum", "appendix"),
    "thyroid":    ("thyroidectomy", "thyroid", "papillary thyroid",
                   "follicular", "Hurthle", "medullary thyroid"),
    "stomach":    ("gastrectomy", "stomach", "gastric", "antrum",
                   "pylorus", "cardia"),
    "liver":      ("hepatectomy", "liver", "hepato", "hepatocellular",
                   "HCC", "cholangiocarcinoma"),
    "lung":       ("lobectomy", "lung", "pulmonary", "bronch",
                   "bronchus", "alveolar", "pneumonectomy"),
    "prostate":   ("prostatectomy", "prostate", "Gleason", "seminal vesicle"),
    "cervix":     ("hysterectomy", "cervix", "cervical", "endocervical"),
    "pancreas":   ("Whipple", "pancreas", "pancreatic", "pancreatico",
                   "ampulla"),
    "esophagus":  ("esophagectomy", "esophagus", "esophageal",
                   "Barrett", "GE junction", "gastroesophageal"),
}


@cache
def _patterns() -> dict[str, list[re.Pattern]]:
    """Compile each keyword as a case-insensitive word-boundary pattern."""
    out: dict[str, list[re.Pattern]] = {}
    for organ, kws in _KEYWORDS.items():
        out[organ] = [re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE)
                      for kw in kws]
    return out


def classify_organ_from_text(report_text: str,
                             dataset: str | None = None,
                             fallback_organ_n: str | None = None,
                             ) -> str | None:
    """Return the most likely organ name for a pathology report.

    Score = number of distinct keywords that match (multiple matches of
    the same keyword count once). Highest score wins; ties resolved by
    the order in :data:`_KEYWORDS` (breast first, then colorectal, etc.).

    If no keyword matches at all, fall back to the dataset-aware folder
    lookup via :func:`benchmarks.organs.organ_n_to_name`.
    """
    text = report_text or ""
    scores: list[tuple[int, str]] = []  # (score, organ) preserving _KEYWORDS order
    for organ, patterns in _patterns().items():
        hits = sum(1 for p in patterns if p.search(text))
        scores.append((hits, organ))
    best_score, best_organ = max(scores, key=lambda t: t[0])
    if best_score > 0:
        # Stable tie-break: walk in declaration order, take first organ
        # that ties best_score.
        for hits, organ in scores:
            if hits == best_score:
                return organ
    if dataset and fallback_organ_n is not None:
        return organ_n_to_name(dataset, fallback_organ_n)
    return None


__all__ = ["classify_organ_from_text"]
