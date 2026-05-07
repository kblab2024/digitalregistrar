"""Multi-primary case detection and stratification.

A subset of cases describe more than one distinct tumor — bilateral
breast cancer, double primary lung, multifocal disease that the
schema's single ``cancer_data`` slot cannot fully represent. Eval
metrics must be reported separately on these cases so the writeup can
quantify the schema's blind spot (R2.2: "double primary malignancies
were occasionally misidentified").

Heuristics (any one is sufficient to flag ``multi_primary``):
    1. ``cancer_clock`` or ``cancer_quadrant`` indicates multiple foci
       (heuristic: presence of ``"and"``, ``";"``, multiple values, or
       a list type rather than a scalar).
    2. ``tumor_focality == "multifocal"`` or similar.
    3. ``cancer_laterality == "bilateral"``.
    4. ``regional_lymph_node`` lists nodes from sides AND specifies
       primary tumors on opposite sides.
    5. Free-text ``cancer_category_others_description`` mentions "double
       primary" or "secondary primary" (last-resort heuristic).

Returns ``True`` for multi-primary, ``False`` for single-primary, or
``None`` when the heuristics can't decide (treat as single-primary in
the default subgroup but flag for manual review).
"""
from __future__ import annotations

import re
from typing import Any

from .metrics import normalize

_MULTI_FOCALITY_VALUES = {
    "multifocal", "multiple", "multifocal_multicentric", "multicentric",
    "bilateral",
}
_BILATERAL_VALUES = {"bilateral", "both"}
_MULTI_REGEX = re.compile(
    r"\b(?:double primary|second primary|multifocal|multicentric|bilateral)\b",
    re.IGNORECASE,
)


def _value_indicates_multi(v: Any) -> bool:
    """Single-value test for multi-primary indicators."""
    if v is None:
        return False
    if isinstance(v, list):
        return len(v) > 1
    s = normalize(v)
    if not isinstance(s, str):
        return False
    if s in _MULTI_FOCALITY_VALUES or s in _BILATERAL_VALUES:
        return True
    # Multi-clock / multi-quadrant strings (e.g. "12 and 3" or "ouq;loq").
    if any(sep in s for sep in (" and ", "/", ";", ",")):
        return True
    return False


def detect_multi_primary(annotation: dict[str, Any]) -> bool | None:
    """Classify a single annotation as multi-primary, single-primary, or
    indeterminate.

    Returns ``True`` for multi-primary, ``False`` for single, ``None``
    when no information is available (cancer_excision_report is False
    or every relevant field is null).
    """
    if not annotation.get("cancer_excision_report"):
        return None
    cd = annotation.get("cancer_data") or {}

    # Direct focality indicators
    focality = cd.get("tumor_focality") or annotation.get("tumor_focality")
    if _value_indicates_multi(focality):
        return True
    laterality = cd.get("cancer_laterality") or annotation.get("cancer_laterality")
    if normalize(laterality) in _BILATERAL_VALUES:
        return True

    # Multi-quadrant / multi-clock heuristic for breast
    for k in ("cancer_clock", "cancer_quadrant", "cancer_primary_site"):
        if _value_indicates_multi(cd.get(k) or annotation.get(k)):
            return True

    # Free-text comment heuristic — last resort.
    desc = annotation.get("cancer_category_others_description") or ""
    if isinstance(desc, str) and _MULTI_REGEX.search(desc):
        return True

    # Indeterminate: cancer present but no multi-primary indicators.
    if normalize(annotation.get("cancer_category")) in ("none", None):
        return None
    return False


def subgroup_label(annotation: dict[str, Any]) -> str:
    """Map an annotation to one of ``{"multi_primary",
    "single_primary", "unknown"}`` for the subgroup column."""
    flag = detect_multi_primary(annotation)
    if flag is True:
        return "multi_primary"
    if flag is False:
        return "single_primary"
    return "unknown"


def n_tumors_estimate(annotation: dict[str, Any]) -> int:
    """Estimate the number of distinct tumors in an annotation.

    Used for the ``tumor_collapse_rate`` failure-mode metric: when gold
    has n=2 and pred has n=1, the model collapsed multi-primary into
    single-primary. Heuristics:

        - bilateral / multifocal flags → return 2.
        - multi-clock / multi-quadrant string → count the separators+1.
        - otherwise → return 1.

    This is approximate. Documented as such in
    ``docs/eval/glossary.md`` and the multi-primary-failure CSV.
    """
    cd = annotation.get("cancer_data") or {}
    laterality = normalize(cd.get("cancer_laterality") or annotation.get("cancer_laterality"))
    if laterality in _BILATERAL_VALUES:
        return 2
    focality = normalize(cd.get("tumor_focality") or annotation.get("tumor_focality"))
    if focality in _MULTI_FOCALITY_VALUES:
        return 2

    for k in ("cancer_clock", "cancer_quadrant"):
        v = cd.get(k) or annotation.get(k)
        if isinstance(v, list):
            return max(len(v), 1)
        if isinstance(v, str):
            count = 1 + sum(v.count(sep) for sep in (" and ", "/", ";", ","))
            if count > 1:
                return count
    return 1


__all__ = [
    "detect_multi_primary",
    "subgroup_label",
    "n_tumors_estimate",
]
