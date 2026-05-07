"""Biomarker scorer for the cascade.

Provides :func:`score_biomarkers` (per-case bipartite biomarker
matching) and :func:`score_biomarkers_filtered` (the same applied
through the cascade's biomarker whitelist filter so out-of-whitelist
categories are invisible on both sides).

Usage from cascade reductions:

    from scripts.eval.cascade.biomarkers import score_biomarkers_filtered
    stats = score_biomarkers_filtered(gold, pred, organ="colorectal")
"""
from __future__ import annotations

from typing import Any

from digital_registrar_research.benchmarks.eval.metrics import (
    _filter_biomarkers_by_whitelist,
    normalize,
)
from digital_registrar_research.benchmarks.eval.scope import (
    biomarkers_for_organ, get_field_value,
)


def _safe_list(v: Any) -> list[dict]:
    return v if isinstance(v, list) else []


def _bm_similarity(g: dict, p: dict) -> float:
    """Score a (gold, pred) biomarker pair.

    Primary key: biomarker_category — exact match required for any
    similarity > 0. Tie-break on expression match, then percentage
    proximity (within 5 percentage points).
    """
    score = 0.0
    gc = normalize(g.get("biomarker_category"))
    pc = normalize(p.get("biomarker_category"))
    if gc is None or pc is None or gc != pc:
        return 0.0
    score += 3.0
    g_exp = normalize(g.get("expression"))
    p_exp = normalize(p.get("expression"))
    if g_exp is not None and p_exp is not None and g_exp == p_exp:
        score += 1.0
    g_pct = g.get("percentage")
    p_pct = p.get("percentage")
    if (isinstance(g_pct, (int, float)) and isinstance(p_pct, (int, float))
            and abs(g_pct - p_pct) <= 5):
        score += 0.5
    return score


def score_biomarkers(gold: dict, pred: dict) -> dict:
    """Per-case biomarker metrics.

    Bipartite greedy match on ``biomarker_category``; matched pairs
    contribute to ``biomarker_expression_correct``,
    ``biomarker_percentage_correct_tol`` (±5 pct), and
    ``biomarker_score_correct``.

    Keys returned:
        biomarker_tp/fp/fn/matched
        biomarker_expression_correct
        biomarker_percentage_correct_tol
        biomarker_score_correct
        biomarker_n_gold / biomarker_n_pred
    """
    from digital_registrar_research.benchmarks.eval.nested_metrics import (
        _greedy_match,
    )

    g_list = _safe_list(get_field_value(gold, "biomarkers"))
    p_list = _safe_list(get_field_value(pred, "biomarkers"))

    matched, unm_g, unm_p = _greedy_match(g_list, p_list, _bm_similarity)
    tp = len(matched)
    fp = len(unm_p)
    fn = len(unm_g)

    expression_ok = percentage_ok = score_ok = 0
    for gi, pi in matched:
        g, p = g_list[gi], p_list[pi]
        if normalize(g.get("expression")) == normalize(p.get("expression")):
            expression_ok += 1
        gp, pp = g.get("percentage"), p.get("percentage")
        if gp is None and pp is None:
            percentage_ok += 1
        elif (isinstance(gp, (int, float)) and isinstance(pp, (int, float))
                and abs(gp - pp) <= 5):
            percentage_ok += 1
        if normalize(g.get("score")) == normalize(p.get("score")):
            score_ok += 1

    return {
        "biomarker_tp": tp,
        "biomarker_fp": fp,
        "biomarker_fn": fn,
        "biomarker_matched": tp,
        "biomarker_expression_correct": expression_ok,
        "biomarker_percentage_correct_tol": percentage_ok,
        "biomarker_score_correct": score_ok,
        "biomarker_n_gold": len(g_list),
        "biomarker_n_pred": len(p_list),
    }


def score_biomarkers_filtered(
    gold: dict, pred: dict, *, organ: str | None,
) -> dict:
    """Per-case biomarker metrics with whitelist filtering applied.

    Wraps :func:`score_biomarkers` so callers don't have to know about
    the whitelist. Wraps both gold and pred in a synthetic shape
    matching :func:`get_field_value`.
    """
    whitelist = biomarkers_for_organ(organ)
    if not whitelist:
        # Organ doesn't have a biomarker scope (e.g. lung, thyroid);
        # return a zero-record empty result.
        return {
            "biomarker_tp": 0,
            "biomarker_fp": 0,
            "biomarker_fn": 0,
            "biomarker_matched": 0,
            "biomarker_expression_correct": 0,
            "biomarker_percentage_correct_tol": 0,
            "biomarker_score_correct": 0,
            "biomarker_n_gold": 0,
            "biomarker_n_pred": 0,
        }

    g_bio = (gold.get("cancer_data") or {}).get("biomarkers") or gold.get("biomarkers")
    p_bio = (pred.get("cancer_data") or {}).get("biomarkers") or pred.get("biomarkers")
    g_filtered = _filter_biomarkers_by_whitelist(g_bio or [], organ)
    p_filtered = _filter_biomarkers_by_whitelist(p_bio or [], organ)

    gold_wrap = {"cancer_data": {"biomarkers": g_filtered}}
    pred_wrap = {"cancer_data": {"biomarkers": p_filtered}}
    return score_biomarkers(gold_wrap, pred_wrap)


__all__ = ["score_biomarkers", "score_biomarkers_filtered"]
