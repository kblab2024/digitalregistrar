"""
Shared scoring harness for all four methods.

Every method's predictions must live under a per-method folder as
`<method>/<case_id>.json`, with the SAME flat structure as the gold
annotations (top-level `cancer_category`, `cancer_data`, ...).

Core primitives:
    normalize(value)            — canonicalise strings/bools/ints for equality
    field_correct(g, p, field)  — per-field scoring with tolerance rules
    match_nested_list(g, p, k)  — bipartite greedy match of list-of-dicts fields
    score_case(gold, pred)      — returns {field: bool/None} across fair scope
    aggregate_cases_to_df(cases, method_to_preds, scope) — long-form correctness DataFrame

Coverage rule
-------------
A field is "attempted" if the prediction dict for that case has an
explicit entry (value OR null). Missing keys → coverage = 0 for that
field. This separates "wrong answer" from "didn't try", which is what
makes the ClinicalBERT / rules comparison honest.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path

import pandas as pd

from .scope import (
    BIOMARKER_WHITELIST,
    BREAST_BIOMARKERS,
    EVAL_EXCLUDED_FIELDS,
    EXCLUDED_INNER_KEYS_BY_FIELD,
    FAIR_SCOPE,
    LIST_OF_LITERALS_FIELDS,
    NESTED_LIST_FIELDS,
    SPAN_FIELDS,
    biomarkers_for_organ,
    get_field_value,
    get_list_of_literals_fields,
    get_organ_scoreable_fields,
)

ScopeArg = Iterable[str] | Callable[[str | None], Iterable[str]] | None

NUMERIC_TOLERANCE_MM = 2  # ±2 mm


# --- Primitives ---------------------------------------------------------------

def normalize(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v
    return str(v).strip().lower()


def is_attempted(pred_annotation: dict, field: str) -> bool:
    """Did the method produce any value (including explicit null) for the field?"""
    if field in pred_annotation:
        return True
    cd = pred_annotation.get("cancer_data") or {}
    return field in cd


def _normalize_set(v) -> frozenset:
    """Coerce a list-of-literals value to a normalised frozenset.

    ``None`` and the empty list both map to ``frozenset()`` so they
    compare equal — matches the ``outcome._normalize_set`` semantics
    (replicated here to avoid a metrics→outcome import cycle).
    """
    if v is None:
        return frozenset()
    if isinstance(v, list):
        return frozenset(normalize(x) for x in v if x is not None)
    return frozenset({normalize(v)})


def _is_list_of_literals_field(field: str, organ: str | None) -> bool:
    """Organ-aware lookup of list-of-literals field membership.

    The same name can be a list-of-literals in one organ and a regular
    categorical in another (e.g. ``tumor_extent`` is a list-of-literals
    only for liver). When ``organ`` is None, fall back to the
    cross-organ union.
    """
    if organ is None:
        return field in LIST_OF_LITERALS_FIELDS
    return field in get_list_of_literals_fields(organ)


def field_correct(
    gold: dict, pred: dict, field: str, *, organ: str | None = None,
) -> bool | None:
    """Returns True/False on attempted predictions, None on non-attempts.

    Field-kind dispatch:
      * list-of-literals (organ-aware) — unordered set equality.
      * span / numeric in :data:`scope.SPAN_FIELDS` — ±2 mm tolerance
        when both sides are numeric, else exact (``None == None``).
      * everything else — string equality after :func:`normalize`.

    ``organ`` lets the cascade scorer pick the right enum-vs-list
    interpretation. When omitted, falls back to ``gold.cancer_category``.
    """
    if not is_attempted(pred, field):
        return None
    g = get_field_value(gold, field)
    p = get_field_value(pred, field)
    organ_arg = organ if organ is not None else normalize(gold.get("cancer_category"))
    if _is_list_of_literals_field(field, organ_arg):
        return _normalize_set(g) == _normalize_set(p)
    if field in SPAN_FIELDS and isinstance(g, (int, float)) and isinstance(p, (int, float)):
        return abs(g - p) <= NUMERIC_TOLERANCE_MM
    return normalize(g) == normalize(p)


# --- Nested-list bipartite match ---------------------------------------------

NESTED_KEY = {
    "margins": "margin_category",
    "biomarkers": "biomarker_category",
    # `regional_lymph_node` is intentionally absent: cascade redesign
    # replaced bipartite-on-station_name with category-aggregation
    # scoring (see :func:`nested_metrics.score_lymph_nodes`). Callers
    # that score lymph nodes must invoke that function directly, not
    # the bipartite ``match_nested_list``.
}


def _item_eq(a, b) -> int:
    """Count how many inner fields match (for ranking bipartite candidates).

    Defensive against non-dict items: if either side isn't a dict, fall
    back to a single-value comparison (returns 1 if equal under
    ``normalize``, else 0).
    """
    if not (a and b):
        return 0
    if not (isinstance(a, dict) and isinstance(b, dict)):
        return 1 if normalize(a) == normalize(b) else 0
    return sum(1 for k in a if k in b and normalize(a[k]) == normalize(b[k]))


def match_nested_list(gold_annotation: dict, pred_annotation: dict,
                      field: str) -> dict:
    """Greedy bipartite match on the primary key, then per-inner-field F1.
    Returns {'tp': ..., 'fp': ..., 'fn': ..., 'f1': ...}.
    """
    key = NESTED_KEY.get(field, None)
    gold_list = get_field_value(gold_annotation, field) or []
    pred_list = get_field_value(pred_annotation, field) or []

    tp = fp = fn = 0
    if key is None:
        # Fallback: just match by position. Lists may differ in length — the
        # loop covers the overlap and the two sum() calls below account for the
        # leftover on each side; strict=False is required here.
        for g, p in zip(gold_list, pred_list, strict=False):
            tp += _item_eq(g, p)
            fp += len(p) - _item_eq(g, p)
            fn += len(g) - _item_eq(g, p)
        fp += sum(len(p) for p in pred_list[len(gold_list):])
        fn += sum(len(g) for g in gold_list[len(pred_list):])
    else:
        unmatched_pred = list(pred_list)
        for g in gold_list:
            gk = normalize(g.get(key))
            match = None
            for p in unmatched_pred:
                if normalize(p.get(key)) == gk:
                    match = p
                    break
            if match is None:
                fn += 1
            else:
                unmatched_pred.remove(match)
                # Score inner fields.
                inner_fields = set(g.keys()) | set(match.keys())
                for k in inner_fields:
                    if normalize(g.get(k)) == normalize(match.get(k)):
                        tp += 1
                    else:
                        fp += 1  # treat mismatches as FP to keep it strict
        fp += len(unmatched_pred)

    if tp + fp == 0 or tp + fn == 0:
        f1 = 0.0
    else:
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "f1": f1}


# --- Filtered-list wrapper (cascade redesign) -------------------------------

def _strip_excluded_inner(items: list, field: str) -> list[dict]:
    """Strip excluded inner keys from each list item for ``field``.

    Applied per-list-element so the bipartite matcher only sees keys
    that are legitimate scoring endpoints. Non-dict items pass through
    unchanged.
    """
    excluded = EXCLUDED_INNER_KEYS_BY_FIELD.get(field, frozenset())
    if not excluded:
        return list(items) if isinstance(items, list) else []
    out: list[dict] = []
    for it in items:
        if isinstance(it, dict):
            out.append({k: v for k, v in it.items() if k not in excluded})
        else:
            out.append(it)
    return out


def _filter_biomarkers_by_whitelist(items: list, organ: str | None) -> list[dict]:
    """Drop biomarker entries whose ``biomarker_category`` is not in
    the organ's whitelist.

    Applies symmetrically to gold and prediction. An entry whose
    category is not in the whitelist becomes invisible to the scorer —
    neither rewarded nor penalised. Non-dict items are dropped.
    """
    if not isinstance(items, list):
        return []
    whitelist = biomarkers_for_organ(organ)
    if not whitelist:
        return []
    return [
        it for it in items
        if isinstance(it, dict)
        and normalize(it.get("biomarker_category")) in whitelist
    ]


def match_nested_list_filtered(
    gold_annotation: dict,
    pred_annotation: dict,
    field: str,
    *,
    organ: str | None = None,
) -> dict:
    """Bipartite F1 over a nested list-of-dicts field, with cascade-wide
    exclusions applied first.

    For ``field == "biomarkers"`` the per-organ whitelist filters both
    sides. For every field, excluded inner keys
    (:data:`scope.EXCLUDED_INNER_KEYS_BY_FIELD`) are stripped from each
    item before matching.

    Returns the same ``{tp, fp, fn, f1}`` shape as
    :func:`match_nested_list`. ``regional_lymph_node`` is rejected at
    the boundary — callers must use
    :func:`nested_metrics.score_lymph_nodes` for that field.
    """
    if field == "regional_lymph_node":
        raise ValueError(
            "regional_lymph_node uses category-aggregation scoring; "
            "call nested_metrics.score_lymph_nodes instead."
        )

    gold_list = get_field_value(gold_annotation, field) or []
    pred_list = get_field_value(pred_annotation, field) or []

    if field == "biomarkers":
        gold_list = _filter_biomarkers_by_whitelist(gold_list, organ)
        pred_list = _filter_biomarkers_by_whitelist(pred_list, organ)

    gold_list = _strip_excluded_inner(gold_list, field)
    pred_list = _strip_excluded_inner(pred_list, field)

    # Build a wrapper "annotation" so the existing matcher can read via
    # ``get_field_value``. Top-level placement is fine because the matcher
    # only consults ``get_field_value(annotation, field)``.
    g_wrap = {field: gold_list}
    p_wrap = {field: pred_list}
    return match_nested_list(g_wrap, p_wrap, field)


# --- Per-case + aggregation --------------------------------------------------

def _resolve_scope(scope: ScopeArg, gold: dict) -> list[str]:
    if scope is None:
        return list(FAIR_SCOPE)
    if callable(scope):
        return list(scope(gold.get("cancer_category")))
    return list(scope)


def score_case(gold: dict, pred: dict, scope: ScopeArg = None) -> dict:
    """Score a single (gold, pred) pair under cascade gating.

    Returns a dict with the original flat shape plus three cascade
    metadata keys:

        out["stage_a"]          {"correct": bool/None, "gold": ..., "pred": ...}
        out["stage_b"]          same; absent if Stage A failed.
        out["stage_c_eligible"] bool; True iff Stage A and B both passed
                                AND neither side is "others".
        out[<field>]            field-level correctness for Stage-C scoring,
                                only populated when ``stage_c_eligible``.
        out["_nested"]          {field: {tp, fp, fn, f1}} for the nested
                                fields (margins / biomarkers); LN is
                                NOT here — callers invoke
                                :func:`nested_metrics.score_lymph_nodes`
                                directly so the cascade reductions can
                                consume the richer dict it returns.
        out["others_disposition"] one of "none", "gold_others", "pred_others",
                                    "both_others".

    Callers (cascade orchestrator, ablation aggregator) consume the
    metadata to decide which rows to emit at each stage. Legacy
    callers that pass ``scope=...`` get the legacy flat behavior with
    no cascade gating — the cascade is opt-in via ``scope=None``.

    The ``scope`` parameter retains its old semantics for backwards
    compatibility:
      * ``None`` (default) — full cascade with gating, exclusions, and
        biomarker whitelist.
      * iterable of field names — score every field on the list, no
        cascade gating, no biomarker whitelisting (used by the
        ClinicalBERT comparison).
      * callable — same, but the callable returns the field set per
        ``cancer_category``.
    """
    if scope is not None:
        # Legacy non-cascade path.
        return _score_case_legacy(gold, pred, scope)

    return _score_case_cascade(gold, pred)


def _score_case_legacy(gold: dict, pred: dict, scope: ScopeArg) -> dict:
    """Backwards-compatible flat scoring (no cascade gating)."""
    out: dict = {"_nested": {}}
    fields = _resolve_scope(scope, gold)
    for field in fields:
        out[field] = field_correct(gold, pred, field)
    return out


def _score_case_cascade(gold: dict, pred: dict) -> dict:
    """Cascade-gated scoring.

    Stage A: ``cancer_excision_report``. Failure halts the cascade.
    Stage B: ``cancer_category``. Failure halts. Both sides "others"
        is recorded as a Stage-B match but does NOT advance to Stage C.
    Stage C: ``cancer_data`` fields after exclusions and biomarker
        whitelist. Only runs when both gates passed and neither side
        is "others".
    """
    out: dict = {"_nested": {}}

    # --- Stage A: eligibility triage ---
    stage_a_correct = field_correct(gold, pred, "cancer_excision_report")
    out["stage_a"] = {
        "correct": stage_a_correct,
        "gold": gold.get("cancer_excision_report"),
        "pred": pred.get("cancer_excision_report"),
    }
    out["cancer_excision_report"] = stage_a_correct

    # Determine "others" disposition before deciding whether to run Stage C.
    g_org = normalize(gold.get("cancer_category"))
    p_org = normalize(pred.get("cancer_category"))
    if g_org == "others" and p_org == "others":
        others_disposition = "both_others"
    elif g_org == "others":
        others_disposition = "gold_others"
    elif p_org == "others":
        others_disposition = "pred_others"
    else:
        others_disposition = "none"
    out["others_disposition"] = others_disposition

    # If Stage A failed (or wasn't attempted), stop here.
    if stage_a_correct is None or not stage_a_correct:
        out["stage_c_eligible"] = False
        return out

    # If gold says ineligible and pred agreed, Stage A passed but there's
    # no cancer_data to score.
    if not bool(gold.get("cancer_excision_report")):
        out["stage_c_eligible"] = False
        return out

    # --- Stage B: organ classification ---
    stage_b_correct = field_correct(gold, pred, "cancer_category")
    out["stage_b"] = {
        "correct": stage_b_correct,
        "gold": gold.get("cancer_category"),
        "pred": pred.get("cancer_category"),
    }
    out["cancer_category"] = stage_b_correct

    if stage_b_correct is None or not stage_b_correct:
        out["stage_c_eligible"] = False
        return out

    # Both-"others" passes Stage B (the labels match) but cannot enter
    # Stage C — the schema for "others" cancers is not implemented.
    if others_disposition != "none":
        out["stage_c_eligible"] = False
        return out

    # --- Stage C: cancer_data field extraction ---
    out["stage_c_eligible"] = True
    organ = g_org

    # Use the FULL per-organ scope, not FAIR_SCOPE. FAIR_SCOPE is only
    # the head-to-head comparison set used by the legacy ClinicalBERT
    # baselines; it deliberately excludes most categorical / bool /
    # span / list-of-literals fields. Cascade chapter 3 must score
    # everything in the schema for the organ.
    fields = list(get_organ_scoreable_fields(organ).keys())
    for field in fields:
        # The two cascade gate fields are scored above; not part of Stage C.
        if field in ("cancer_excision_report", "cancer_category"):
            continue
        if field in EVAL_EXCLUDED_FIELDS:
            continue
        out[field] = field_correct(gold, pred, field, organ=organ)

    # Per-organ biomarkers (whitelist applied via biomarkers_for_organ).
    whitelist = biomarkers_for_organ(organ)
    if whitelist:
        gold_bm = {
            normalize(b.get("biomarker_category")): b
            for b in (get_field_value(gold, "biomarkers") or [])
            if isinstance(b, dict)
        }
        pred_bm = {
            normalize(b.get("biomarker_category")): b
            for b in (get_field_value(pred, "biomarkers") or [])
            if isinstance(b, dict)
        }
        for cat in whitelist:
            g = gold_bm.get(cat)
            p = pred_bm.get(cat)
            if p is None:
                out[f"biomarker_{cat}"] = None
            elif g is None:
                out[f"biomarker_{cat}"] = False
            else:
                out[f"biomarker_{cat}"] = (
                    normalize(g.get("expression")) == normalize(p.get("expression"))
                )

    # --- Nested-list F1 (margins, biomarkers; LN handled outside) ---
    for field in NESTED_LIST_FIELDS:
        if field == "regional_lymph_node":
            # Cascade reductions call nested_metrics.score_lymph_nodes
            # directly to consume the richer per-group return dict.
            continue
        if not is_attempted(pred, field):
            continue
        out["_nested"][field] = match_nested_list_filtered(
            gold, pred, field, organ=organ,
        )

    return out


def _resolve_pred_path(pred_root: Path, case: dict) -> Path | None:
    """Look for predictions in both layouts: per-dataset subdir or flat."""
    cid = case["id"]
    ds = case.get("dataset")
    if ds is not None:
        ds_path = pred_root / ds / f"{cid}.json"
        if ds_path.exists():
            return ds_path
    flat = pred_root / f"{cid}.json"
    return flat if flat.exists() else None


def aggregate_cases_to_df(
    cases: list[dict],
    method_to_preds: dict[str, Path],
    scope: ScopeArg = None,
) -> pd.DataFrame:
    """Score every (method, case, field) into long-form, with dataset / organ columns.

    Inputs:
        cases: list of case dicts already loaded via ``baselines._data.load_cases``
            (each carries ``id, dataset, cancer_category, annotation_path``).
        method_to_preds: ``{method_name: Path}`` where Path is the prediction
            root. The lookup tries ``<root>/<dataset>/<id>.json`` first, then
            falls back to ``<root>/<id>.json``.
        scope: passed straight through to ``score_case``.

    Output: DataFrame with columns
        ``method, dataset, organ, case_id, field, correct, attempted``.
    """
    rows = []
    for method, preds_root in method_to_preds.items():
        for case in cases:
            cid = case["id"]
            organ = case.get("cancer_category")
            ds = case.get("dataset")
            with open(case["annotation_path"], encoding="utf-8") as f:
                gold = json.load(f)

            pred_path = _resolve_pred_path(preds_root, case)
            fields = _resolve_scope(scope, gold)
            if pred_path is None:
                # Method didn't emit anything for this case — coverage 0.
                for field in fields:
                    rows.append({"method": method, "dataset": ds, "organ": organ,
                                 "case_id": cid, "field": field,
                                 "correct": None, "attempted": False})
                continue

            with pred_path.open(encoding="utf-8") as f:
                pred = json.load(f)
            result = score_case(gold, pred, scope=scope)

            for field in fields:
                if field not in result:
                    continue
                correct = result[field]
                rows.append({
                    "method": method, "dataset": ds, "organ": organ,
                    "case_id": cid, "field": field,
                    "correct": (bool(correct) if correct is not None else None),
                    "attempted": correct is not None,
                })
            for field, f1d in result["_nested"].items():
                rows.append({
                    "method": method, "dataset": ds, "organ": organ,
                    "case_id": cid, "field": field,
                    "correct": f1d["f1"], "attempted": True,
                })
    return pd.DataFrame(rows)


def summary_table(df: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Accuracy + coverage grouped by ``by`` (default ``["method", "field"]``).

    Pass ``by=["method", "dataset", "field"]`` for the dataset-stratified
    table when the input came from ``aggregate_cases_to_df``.
    """
    if by is None:
        by = ["method", "field"]

    def _agg(group):
        attempted = group["attempted"].sum()
        total = len(group)
        numeric = group["correct"].dropna()
        if numeric.empty:
            acc = float("nan")
        else:
            acc = float(pd.to_numeric(numeric, errors="coerce").mean())
        return pd.Series({
            "attempted": attempted,
            "total": total,
            "coverage": attempted / total if total else 0.0,
            "accuracy_attempted": acc,
        })

    return df.groupby(by).apply(_agg).reset_index()
