"""Chapter 4 (margins) and Chapter 5 (lymph nodes) reducers.

These two nested fields are first-class diagnostic targets — they
drive treatment decisions (margin status / closest distance dictates
re-excision; LN involvement count dictates adjuvant therapy) — and
warrant their own chapters with the full battery of summaries:

  * ``overall.csv`` — per (model, dataset, organ) clinical headlines
    + item-level micro precision / recall / F1 + hallucination / miss.
  * ``per_attribute.csv`` — matched-pair conditional accuracy for each
    inner attribute (margin: status / distance / category;
    LN: examined / involved / category / side).
  * ``per_category.csv`` — per-(margin_category) involved-correct or
    per-(LN side, category) presence + count metrics.
  * ``per_station.csv`` (LN only) — flatten ``ln_per_station`` dict.
  * ``confusion_matrices.csv`` — long-form gold×pred for the
    categorical attributes:
      - margins: matched bipartite pairs on margin_category and
        margin_involved.
      - LN: single-group cases on lymph_node_category and
        lymph_node_side (multi-group cases are ambiguous and skipped).
  * ``missingness.csv`` — four-level decomposition (parse_error /
    field_key_absent / empty_list / partial_list) with Wilson CIs.
  * ``multirun_consistency.csv`` — per-case F1 SD, mean per-case F1,
    missing-flip rate (when n_runs > 1).

All reducers are pure-data: they consume the in-memory cascade
sidecar (``cascade_nested.parquet``) and atomic table; the orchestrator
in ``run_cascade.py`` writes the outputs.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from digital_registrar_research.benchmarks.eval.stats import wilson_ci


# --- Common helpers --------------------------------------------------------

def _safe_proportion(k: int, n: int) -> float:
    return float(k / n) if n > 0 else float("nan")


def _wilson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    return wilson_ci(k, n, alpha)


def _prf(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * prec * rec / (prec + rec)
          if (prec and rec and (prec + rec)) else float("nan"))
    return prec, rec, f1


def _filter_nested_atomic(
    atomic: pd.DataFrame, field: str,
) -> pd.DataFrame:
    """Pick the chapter-{4,5} headline atomic rows for ``field``."""
    if atomic is None or atomic.empty:
        return pd.DataFrame()
    sub = atomic[
        (atomic["cascade_stage"] == "C")
        & (atomic.get("field_kind") == "nested_list")
        & (atomic["field"] == field)
    ]
    return sub


def _filter_nested_sidecar(
    nested: pd.DataFrame, field: str,
) -> pd.DataFrame:
    if nested is None or nested.empty:
        return pd.DataFrame()
    return nested[nested["field"] == field]


# --- Generic missingness + multirun (reused by both chapters) -------------

def _missingness_for_field(
    atomic: pd.DataFrame, *, field: str, alpha: float = 0.05,
) -> pd.DataFrame:
    """Four-level nested-field missingness with Wilson CIs.

    Output rows: one per (model, dataset, organ).
    """
    sub = _filter_nested_atomic(atomic, field)
    if sub.empty:
        return pd.DataFrame()
    levels = ("parse_error", "field_key_absent", "empty_list", "partial_list")
    rows: list[dict] = []
    for (model, dataset, organ), grp in sub.groupby(
        ["model", "dataset", "organ"], dropna=False,
    ):
        n_total = len(grp)
        row = {
            "model": model, "dataset": dataset, "organ": organ,
            "field": field, "n_total": n_total,
        }
        for lvl in levels:
            n_lvl = int((grp.get("nested_missingness_level") == lvl).sum())
            rate = _safe_proportion(n_lvl, n_total)
            lo, hi = _wilson(n_lvl, n_total, alpha)
            row[f"n_{lvl}"] = n_lvl
            row[f"{lvl}_rate"] = rate
            row[f"{lvl}_rate_lo"] = lo
            row[f"{lvl}_rate_hi"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def _multirun_for_field(
    atomic: pd.DataFrame, *, field: str,
) -> pd.DataFrame:
    """Per-(model, dataset, organ) multi-run consistency for one nested field.

    Reports: per-case F1 SD (mean, p90), mean per-case F1, missing-flip
    rate. Only emits rows when ``n_runs >= 2``.
    """
    sub = _filter_nested_atomic(atomic, field)
    if sub.empty or sub["run_id"].nunique() < 2:
        return pd.DataFrame()
    rows: list[dict] = []
    for (model, dataset, organ), grp in sub.groupby(
        ["model", "dataset", "organ"], dropna=False,
    ):
        # Pivot case_id × run_id of F1 (correct holds the F1 for nested rows).
        pivot_corr = grp.pivot_table(
            index="case_id", columns="run_id", values="correct",
            aggfunc="first",
        )
        pivot_att = grp.pivot_table(
            index="case_id", columns="run_id", values="attempted",
            aggfunc="first",
        )
        if pivot_corr.shape[1] < 2:
            continue
        m = pivot_corr.to_numpy(dtype=float)
        valid = ~np.any(np.isnan(m), axis=1)
        if valid.any():
            sd = m[valid].std(axis=1, ddof=0)
            sd_mean = float(sd.mean())
            sd_p90 = float(np.percentile(sd, 90)) if sd.size else float("nan")
            f1_mean = float(m[valid].mean())
        else:
            sd_mean = sd_p90 = f1_mean = float("nan")
        ma = pivot_att.to_numpy(dtype=float)
        missing_flip = float("nan")
        if ma.size:
            v = ~np.any(np.isnan(ma), axis=1)
            if v.any():
                flips = (np.any(ma[v] == 0, axis=1)
                         & np.any(ma[v] == 1, axis=1))
                missing_flip = float(flips.mean())
        rows.append({
            "model": model, "dataset": dataset, "organ": organ,
            "field": field,
            "n_cases": int(pivot_corr.shape[0]),
            "n_runs": int(pivot_corr.shape[1]),
            "mean_per_case_f1": f1_mean,
            "per_case_f1_sd_mean": sd_mean,
            "per_case_f1_sd_p90": sd_p90,
            "missing_flip_rate": missing_flip,
        })
    return pd.DataFrame(rows)


# --- Chapter 4: margins ----------------------------------------------------

def chapter4_margins_overall(
    nested: pd.DataFrame, atomic: pd.DataFrame, *, alpha: float = 0.05,
) -> pd.DataFrame:
    """Margins headline per (model, dataset, organ).

    Mixes atomic (coverage / attempted-rate counts) with sidecar
    (clinical headlines and item-level micro counts).
    """
    sidecar = _filter_nested_sidecar(nested, "margins")
    headline_atomic = _filter_nested_atomic(atomic, "margins")
    if sidecar.empty and headline_atomic.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    keys = ["model", "dataset", "organ"]
    # Build the union of group keys present in either source.
    all_keys = pd.concat([
        sidecar[keys].drop_duplicates() if not sidecar.empty else pd.DataFrame(columns=keys),
        headline_atomic[keys].drop_duplicates() if not headline_atomic.empty else pd.DataFrame(columns=keys),
    ]).drop_duplicates()

    for _, key_row in all_keys.iterrows():
        model, dataset, organ = key_row["model"], key_row["dataset"], key_row["organ"]
        sc = sidecar[
            (sidecar["model"] == model)
            & (sidecar["dataset"] == dataset)
            & (sidecar["organ"] == organ)
        ] if not sidecar.empty else pd.DataFrame()
        at = headline_atomic[
            (headline_atomic["model"] == model)
            & (headline_atomic["dataset"] == dataset)
            & (headline_atomic["organ"] == organ)
        ] if not headline_atomic.empty else pd.DataFrame()

        n_total = int(len(at))
        n_attempted = int(at["attempted"].sum()) if not at.empty else int(len(sc))
        coverage = _safe_proportion(n_attempted, n_total)
        cov_lo, cov_hi = _wilson(n_attempted, n_total, alpha)

        # Clinical headlines from per-case scorer payload.
        any_inv_correct = (
            float(pd.to_numeric(sc["margin_any_involved_correct"], errors="coerce").mean())
            if not sc.empty and "margin_any_involved_correct" in sc.columns else float("nan")
        )
        any_inv_n = int(sc["margin_any_involved_correct"].notna().sum()) if not sc.empty else 0
        ai_correct = int(pd.to_numeric(sc.get("margin_any_involved_correct"), errors="coerce").fillna(0).sum()) if not sc.empty else 0
        ai_lo, ai_hi = _wilson(ai_correct, any_inv_n, alpha)

        # Closest distance — only on cases where both sides have a value.
        has_both = sc[sc.get("margin_closest_distance_has_both") == 1] if not sc.empty else pd.DataFrame()
        cdist_mae = (float(has_both["margin_closest_distance_abs_err"].mean())
                     if not has_both.empty else float("nan"))
        cdist_acc_tol2 = (
            float(has_both["margin_closest_distance_correct_tol"].mean())
            if not has_both.empty else float("nan")
        )

        # Per-case attempted F1.
        f1_vals = pd.to_numeric(sc.get("f1"), errors="coerce").dropna() if not sc.empty else pd.Series([], dtype=float)
        attempted_f1 = float(f1_vals.mean()) if not f1_vals.empty else float("nan")

        # Item-level micro precision/recall/F1 + hallucination/miss.
        tp = int(pd.to_numeric(sc.get("margin_tp"), errors="coerce").fillna(0).sum()) if not sc.empty else 0
        fp = int(pd.to_numeric(sc.get("margin_fp"), errors="coerce").fillna(0).sum()) if not sc.empty else 0
        fn = int(pd.to_numeric(sc.get("margin_fn"), errors="coerce").fillna(0).sum()) if not sc.empty else 0
        prec, rec, micro_f1 = _prf(tp, fp, fn)
        # Hallucination / miss rates with Wilson CI.
        if (tp + fp) > 0:
            hall_rate = fp / (tp + fp)
            hall_lo, hall_hi = _wilson(fp, tp + fp, alpha)
        else:
            hall_rate = float("nan")
            hall_lo = hall_hi = float("nan")
        if (tp + fn) > 0:
            miss_rate = fn / (tp + fn)
            miss_lo, miss_hi = _wilson(fn, tp + fn, alpha)
        else:
            miss_rate = float("nan")
            miss_lo = miss_hi = float("nan")

        rows.append({
            "model": model, "dataset": dataset, "organ": organ,
            "n_total": n_total,
            "n_attempted": n_attempted,
            "coverage": coverage, "coverage_lo": cov_lo, "coverage_hi": cov_hi,
            "any_involved_acc": any_inv_correct,
            "any_involved_acc_lo": ai_lo, "any_involved_acc_hi": ai_hi,
            "n_any_involved_scored": any_inv_n,
            "closest_dist_mae_mm": cdist_mae,
            "closest_dist_acc_tol2": cdist_acc_tol2,
            "closest_dist_both_rate": (
                float(sc["margin_closest_distance_has_both"].mean())
                if not sc.empty else float("nan")
            ),
            "attempted_f1": attempted_f1,
            "tp": tp, "fp": fp, "fn": fn,
            "micro_precision": prec, "micro_recall": rec, "micro_f1": micro_f1,
            "hallucination_rate": hall_rate,
            "hallucination_rate_lo": hall_lo, "hallucination_rate_hi": hall_hi,
            "miss_rate": miss_rate,
            "miss_rate_lo": miss_lo, "miss_rate_hi": miss_hi,
        })
    return pd.DataFrame(rows)


# Map margin attribute → matched-correct sidecar column.
_MARGIN_ATTRIBUTE_COLS = {
    "margin_category":   "margin_category_correct",
    "margin_involved":   "margin_status_correct",
    "distance_within_2mm": "margin_distance_correct",
}


def chapter4_margins_per_attribute(
    nested: pd.DataFrame, *, alpha: float = 0.05,
) -> pd.DataFrame:
    """Matched-pair conditional accuracy per inner attribute.

    Denominator is the count of matched bipartite pairs; numerator is
    the count of those where the attribute agreed.
    """
    sub = _filter_nested_sidecar(nested, "margins")
    if sub.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for (model, dataset, organ), grp in sub.groupby(
        ["model", "dataset", "organ"], dropna=False,
    ):
        n_matched = int(pd.to_numeric(grp.get("margin_matched"), errors="coerce").fillna(0).sum())
        for attr, col in _MARGIN_ATTRIBUTE_COLS.items():
            if col not in grp.columns:
                continue
            n_correct = int(pd.to_numeric(grp[col], errors="coerce").fillna(0).sum())
            acc = _safe_proportion(n_correct, n_matched)
            lo, hi = _wilson(n_correct, n_matched, alpha)
            rows.append({
                "model": model, "dataset": dataset, "organ": organ,
                "attribute": attr,
                "n_matched_pairs": n_matched,
                "n_attribute_correct": n_correct,
                "accuracy": acc, "ci_lo": lo, "ci_hi": hi,
            })
    return pd.DataFrame(rows)


def chapter4_margins_per_category(
    nested: pd.DataFrame, *, alpha: float = 0.05,
) -> pd.DataFrame:
    """Per-margin_category involved-correct accuracy.

    The sidecar carries a ``margin_per_category`` dict per case shaped
    ``{cat: {gold_involved, pred_involved, involved_correct}}``. We
    flatten it and aggregate.
    """
    sub = _filter_nested_sidecar(nested, "margins")
    if sub.empty or "margin_per_category" not in sub.columns:
        return pd.DataFrame()
    rows: list[dict] = []
    for (model, dataset, organ), grp in sub.groupby(
        ["model", "dataset", "organ"], dropna=False,
    ):
        per_cat: dict[str, dict[str, int]] = {}
        for per in grp["margin_per_category"]:
            if not isinstance(per, dict):
                continue
            for cat, vals in per.items():
                slot = per_cat.setdefault(
                    cat, {"n_cases": 0, "n_involved_correct": 0,
                          "n_gold_involved": 0, "n_pred_involved": 0},
                )
                slot["n_cases"] += 1
                slot["n_involved_correct"] += int(vals.get("involved_correct", 0))
                slot["n_gold_involved"] += int(vals.get("gold_involved", 0))
                slot["n_pred_involved"] += int(vals.get("pred_involved", 0))
        for cat, slot in per_cat.items():
            n = slot["n_cases"]
            acc = _safe_proportion(slot["n_involved_correct"], n)
            lo, hi = _wilson(slot["n_involved_correct"], n, alpha)
            rows.append({
                "model": model, "dataset": dataset, "organ": organ,
                "margin_category": cat,
                "n_cases": n,
                "n_gold_involved": slot["n_gold_involved"],
                "n_pred_involved": slot["n_pred_involved"],
                "involved_correct_acc": acc,
                "ci_lo": lo, "ci_hi": hi,
            })
    return pd.DataFrame(rows)


def chapter4_margins_confusion(
    nested: pd.DataFrame,
) -> pd.DataFrame:
    """Long-form confusion matrices on matched bipartite pairs.

    Two attributes covered: ``margin_category`` and ``margin_involved``.
    Output columns: model, dataset, organ, attribute, gold, pred, count.
    """
    sub = _filter_nested_sidecar(nested, "margins")
    if sub.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for (model, dataset, organ), grp in sub.groupby(
        ["model", "dataset", "organ"], dropna=False,
    ):
        cat_pairs = _flatten_pair_lists(grp.get("margin_matched_pair_categories"))
        for (g, p), n in _count_pairs(cat_pairs).items():
            rows.append({
                "model": model, "dataset": dataset, "organ": organ,
                "attribute": "margin_category",
                "gold": _none_to_str(g), "pred": _none_to_str(p),
                "count": n,
            })
        status_pairs = _flatten_pair_lists(grp.get("margin_matched_pair_status"))
        for (g, p), n in _count_pairs(status_pairs).items():
            rows.append({
                "model": model, "dataset": dataset, "organ": organ,
                "attribute": "margin_involved",
                "gold": _none_to_str(g), "pred": _none_to_str(p),
                "count": n,
            })
    return pd.DataFrame(rows)


def chapter4_margins_missingness(
    atomic: pd.DataFrame, *, alpha: float = 0.05,
) -> pd.DataFrame:
    return _missingness_for_field(atomic, field="margins", alpha=alpha)


def chapter4_margins_multirun(
    atomic: pd.DataFrame,
) -> pd.DataFrame:
    return _multirun_for_field(atomic, field="margins")


# --- Chapter 5: lymph nodes ------------------------------------------------

def chapter5_lymph_nodes_overall(
    nested: pd.DataFrame, atomic: pd.DataFrame, *, alpha: float = 0.05,
) -> pd.DataFrame:
    """LN headline per (model, dataset, organ).

    Includes both the case-level totals (examined / involved MAE,
    any-positive accuracy) AND the group-level cascade-redesign
    metrics (group recall / precision, n_groups), AND item-level
    bipartite micro F1 / hallucination / miss for backward compat
    with the legacy station-level summary.
    """
    sidecar = _filter_nested_sidecar(nested, "regional_lymph_node")
    headline_atomic = _filter_nested_atomic(atomic, "regional_lymph_node")
    if sidecar.empty and headline_atomic.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    keys = ["model", "dataset", "organ"]
    all_keys = pd.concat([
        sidecar[keys].drop_duplicates() if not sidecar.empty else pd.DataFrame(columns=keys),
        headline_atomic[keys].drop_duplicates() if not headline_atomic.empty else pd.DataFrame(columns=keys),
    ]).drop_duplicates()

    for _, key_row in all_keys.iterrows():
        model, dataset, organ = key_row["model"], key_row["dataset"], key_row["organ"]
        sc = sidecar[
            (sidecar["model"] == model)
            & (sidecar["dataset"] == dataset)
            & (sidecar["organ"] == organ)
        ] if not sidecar.empty else pd.DataFrame()
        at = headline_atomic[
            (headline_atomic["model"] == model)
            & (headline_atomic["dataset"] == dataset)
            & (headline_atomic["organ"] == organ)
        ] if not headline_atomic.empty else pd.DataFrame()

        n_total = int(len(at))
        n_attempted = int(at["attempted"].sum()) if not at.empty else int(len(sc))
        coverage = _safe_proportion(n_attempted, n_total)
        cov_lo, cov_hi = _wilson(n_attempted, n_total, alpha)

        examined_mae = (float(pd.to_numeric(sc["ln_examined_total_abs_err"], errors="coerce").mean())
                        if not sc.empty and "ln_examined_total_abs_err" in sc.columns else float("nan"))
        examined_acc_tol1 = (
            float(pd.to_numeric(sc["ln_examined_total_correct_tol"], errors="coerce").mean())
            if not sc.empty and "ln_examined_total_correct_tol" in sc.columns else float("nan")
        )
        involved_mae = (float(pd.to_numeric(sc["ln_involved_total_abs_err"], errors="coerce").mean())
                        if not sc.empty and "ln_involved_total_abs_err" in sc.columns else float("nan"))
        involved_acc_tol1 = (
            float(pd.to_numeric(sc["ln_involved_total_correct_tol"], errors="coerce").mean())
            if not sc.empty and "ln_involved_total_correct_tol" in sc.columns else float("nan")
        )
        any_pos_n = int(sc["ln_any_positive_correct"].notna().sum()) if not sc.empty else 0
        any_pos_correct = int(pd.to_numeric(sc.get("ln_any_positive_correct"), errors="coerce").fillna(0).sum()) if not sc.empty else 0
        any_pos_acc = _safe_proportion(any_pos_correct, any_pos_n)
        any_pos_lo, any_pos_hi = _wilson(any_pos_correct, any_pos_n, alpha)

        # Group-level metrics (cascade redesign).
        group_recall = (float(pd.to_numeric(sc["ln_group_recall"], errors="coerce").mean())
                        if not sc.empty and "ln_group_recall" in sc.columns else float("nan"))
        group_precision = (float(pd.to_numeric(sc["ln_group_precision"], errors="coerce").mean())
                           if not sc.empty and "ln_group_precision" in sc.columns else float("nan"))
        n_groups_gold_mean = (float(pd.to_numeric(sc["ln_n_groups_gold"], errors="coerce").mean())
                              if not sc.empty and "ln_n_groups_gold" in sc.columns else float("nan"))
        n_groups_pred_mean = (float(pd.to_numeric(sc["ln_n_groups_pred"], errors="coerce").mean())
                              if not sc.empty and "ln_n_groups_pred" in sc.columns else float("nan"))

        # Item-level micro P/R/F1 + hallucination / miss.
        tp = int(pd.to_numeric(sc.get("ln_station_tp"), errors="coerce").fillna(0).sum()) if not sc.empty else 0
        fp = int(pd.to_numeric(sc.get("ln_station_fp"), errors="coerce").fillna(0).sum()) if not sc.empty else 0
        fn = int(pd.to_numeric(sc.get("ln_station_fn"), errors="coerce").fillna(0).sum()) if not sc.empty else 0
        prec, rec, micro_f1 = _prf(tp, fp, fn)
        if (tp + fp) > 0:
            hall_rate = fp / (tp + fp)
            hall_lo, hall_hi = _wilson(fp, tp + fp, alpha)
        else:
            hall_rate = hall_lo = hall_hi = float("nan")
        if (tp + fn) > 0:
            miss_rate = fn / (tp + fn)
            miss_lo, miss_hi = _wilson(fn, tp + fn, alpha)
        else:
            miss_rate = miss_lo = miss_hi = float("nan")

        attempted_f1 = (float(pd.to_numeric(sc["f1"], errors="coerce").dropna().mean())
                        if not sc.empty and "f1" in sc.columns else float("nan"))

        rows.append({
            "model": model, "dataset": dataset, "organ": organ,
            "n_total": n_total, "n_attempted": n_attempted,
            "coverage": coverage, "coverage_lo": cov_lo, "coverage_hi": cov_hi,
            "examined_mae": examined_mae, "examined_acc_tol1": examined_acc_tol1,
            "involved_mae": involved_mae, "involved_acc_tol1": involved_acc_tol1,
            "any_positive_acc": any_pos_acc,
            "any_positive_acc_lo": any_pos_lo, "any_positive_acc_hi": any_pos_hi,
            "n_any_positive_scored": any_pos_n,
            "group_recall": group_recall, "group_precision": group_precision,
            "n_groups_gold_mean": n_groups_gold_mean,
            "n_groups_pred_mean": n_groups_pred_mean,
            "attempted_f1": attempted_f1,
            "tp": tp, "fp": fp, "fn": fn,
            "micro_precision": prec, "micro_recall": rec, "micro_f1": micro_f1,
            "hallucination_rate": hall_rate,
            "hallucination_rate_lo": hall_lo, "hallucination_rate_hi": hall_hi,
            "miss_rate": miss_rate,
            "miss_rate_lo": miss_lo, "miss_rate_hi": miss_hi,
        })
    return pd.DataFrame(rows)


_LN_ATTRIBUTE_COLS = {
    "examined_within_1": "ln_station_examined_correct",
    "involved_within_1": "ln_station_involved_correct",
    "category":          "ln_station_category_correct",
    "side":              "ln_station_side_correct",
}


def chapter5_lymph_nodes_per_attribute(
    nested: pd.DataFrame, *, alpha: float = 0.05,
) -> pd.DataFrame:
    """Matched-pair conditional accuracy per inner attribute.

    Denominator: matched (side, category) groups across all cases.
    Note: under category-aggregation matching, ``category`` and
    ``side`` are deterministic 1.0 by construction (groups can only
    match if those agree). Recorded explicitly so the table is
    exhaustive rather than silently dropping the columns.
    """
    sub = _filter_nested_sidecar(nested, "regional_lymph_node")
    if sub.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for (model, dataset, organ), grp in sub.groupby(
        ["model", "dataset", "organ"], dropna=False,
    ):
        n_matched = int(pd.to_numeric(grp.get("ln_station_matched"), errors="coerce").fillna(0).sum())
        for attr, col in _LN_ATTRIBUTE_COLS.items():
            if col not in grp.columns:
                continue
            n_correct = int(pd.to_numeric(grp[col], errors="coerce").fillna(0).sum())
            acc = _safe_proportion(n_correct, n_matched)
            lo, hi = _wilson(n_correct, n_matched, alpha)
            rows.append({
                "model": model, "dataset": dataset, "organ": organ,
                "attribute": attr,
                "n_matched_groups": n_matched,
                "n_attribute_correct": n_correct,
                "accuracy": acc, "ci_lo": lo, "ci_hi": hi,
            })
    return pd.DataFrame(rows)


def chapter5_lymph_nodes_per_category(
    nested: pd.DataFrame, *, alpha: float = 0.05,
) -> pd.DataFrame:
    """Per-(side, category) presence + count metrics.

    Aggregates the per-case ``ln_per_group`` ledger across cases.
    Columns: side, category, n_gold_present_cases, n_pred_present_cases,
    n_both_cases, n_gold_only, n_pred_only, recall, precision,
    examined_acc_tol1 (matched-only), involved_acc_tol1 (matched-only).
    """
    sub = _filter_nested_sidecar(nested, "regional_lymph_node")
    if sub.empty or "ln_per_group" not in sub.columns:
        return pd.DataFrame()
    rows: list[dict] = []
    for (model, dataset, organ), grp in sub.groupby(
        ["model", "dataset", "organ"], dropna=False,
    ):
        per_group: dict[tuple, dict] = {}
        for groups in grp["ln_per_group"]:
            if not isinstance(groups, list):
                continue
            for g in groups:
                key = (g.get("side"), g.get("category"))
                slot = per_group.setdefault(
                    key,
                    {"n_gold_present": 0, "n_pred_present": 0, "n_both": 0,
                     "n_examined_correct": 0, "n_involved_correct": 0},
                )
                if g.get("gold_present"):
                    slot["n_gold_present"] += 1
                if g.get("pred_present"):
                    slot["n_pred_present"] += 1
                if g.get("gold_present") and g.get("pred_present"):
                    slot["n_both"] += 1
                    slot["n_examined_correct"] += int(g.get("examined_correct_tol", 0))
                    slot["n_involved_correct"] += int(g.get("involved_correct_tol", 0))
        for (side, cat), slot in per_group.items():
            n_g = slot["n_gold_present"]
            n_p = slot["n_pred_present"]
            n_b = slot["n_both"]
            recall = _safe_proportion(n_b, n_g)
            precision = _safe_proportion(n_b, n_p)
            ex_acc = _safe_proportion(slot["n_examined_correct"], n_b)
            inv_acc = _safe_proportion(slot["n_involved_correct"], n_b)
            rows.append({
                "model": model, "dataset": dataset, "organ": organ,
                "side": side, "category": cat,
                "n_gold_present_cases": n_g,
                "n_pred_present_cases": n_p,
                "n_both_cases": n_b,
                "n_gold_only": n_g - n_b,
                "n_pred_only": n_p - n_b,
                "recall": recall,
                "precision": precision,
                "examined_acc_tol1_matched": ex_acc,
                "involved_acc_tol1_matched": inv_acc,
            })
    return pd.DataFrame(rows)


def chapter5_lymph_nodes_per_station(
    nested: pd.DataFrame, *, alpha: float = 0.05,
) -> pd.DataFrame:
    """Flatten the per-case ``ln_per_station`` dict into per-station rows.

    Output: one row per (model, dataset, organ, station_name) with
    n_cases, examined_acc_tol1, involved_acc_tol1.
    """
    sub = _filter_nested_sidecar(nested, "regional_lymph_node")
    if sub.empty or "ln_per_station" not in sub.columns:
        return pd.DataFrame()
    rows: list[dict] = []
    for (model, dataset, organ), grp in sub.groupby(
        ["model", "dataset", "organ"], dropna=False,
    ):
        per_st: dict[str, dict] = {}
        for per in grp["ln_per_station"]:
            if not isinstance(per, dict):
                continue
            for sn, vals in per.items():
                slot = per_st.setdefault(
                    sn, {"n_cases": 0, "n_examined_ok": 0, "n_involved_ok": 0},
                )
                slot["n_cases"] += 1
                slot["n_examined_ok"] += int(vals.get("examined_correct_tol", 0))
                slot["n_involved_ok"] += int(vals.get("involved_correct_tol", 0))
        for sn, slot in per_st.items():
            n = slot["n_cases"]
            rows.append({
                "model": model, "dataset": dataset, "organ": organ,
                "station_name": sn,
                "n_cases": n,
                "examined_acc_tol1": _safe_proportion(slot["n_examined_ok"], n),
                "involved_acc_tol1": _safe_proportion(slot["n_involved_ok"], n),
            })
    return pd.DataFrame(rows)


def chapter5_lymph_nodes_confusion(
    nested: pd.DataFrame,
) -> pd.DataFrame:
    """Single-group-case confusion matrices for category and side.

    Multi-group cases are ambiguous (no canonical pairing across
    groups when counts differ on each side) and are skipped here.
    Per-category co-occurrence at the multi-group level is captured by
    ``chapter5_lymph_nodes_per_category``.
    """
    sub = _filter_nested_sidecar(nested, "regional_lymph_node")
    if sub.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for (model, dataset, organ), grp in sub.groupby(
        ["model", "dataset", "organ"], dropna=False,
    ):
        # Single-group pair columns hold tuples or None.
        side_pairs = [
            t for t in grp.get("ln_single_group_pair_side", [])
            if isinstance(t, tuple)
        ]
        cat_pairs = [
            t for t in grp.get("ln_single_group_pair_category", [])
            if isinstance(t, tuple)
        ]
        for (g, p), n in _count_pairs(side_pairs).items():
            rows.append({
                "model": model, "dataset": dataset, "organ": organ,
                "attribute": "lymph_node_side",
                "gold": _none_to_str(g), "pred": _none_to_str(p),
                "count": n,
            })
        for (g, p), n in _count_pairs(cat_pairs).items():
            rows.append({
                "model": model, "dataset": dataset, "organ": organ,
                "attribute": "lymph_node_category",
                "gold": _none_to_str(g), "pred": _none_to_str(p),
                "count": n,
            })
    return pd.DataFrame(rows)


def chapter5_lymph_nodes_missingness(
    atomic: pd.DataFrame, *, alpha: float = 0.05,
) -> pd.DataFrame:
    return _missingness_for_field(atomic, field="regional_lymph_node", alpha=alpha)


def chapter5_lymph_nodes_multirun(
    atomic: pd.DataFrame,
) -> pd.DataFrame:
    return _multirun_for_field(atomic, field="regional_lymph_node")


# --- Tuple-flatten helpers used by confusion reducers ---------------------

def _flatten_pair_lists(series) -> list[tuple]:
    """Concatenate list-of-tuples per row into one flat list."""
    out: list[tuple] = []
    if series is None:
        return out
    for entry in series:
        if isinstance(entry, list):
            out.extend(t for t in entry if isinstance(t, tuple))
    return out


def _count_pairs(pairs: Iterable[tuple]) -> dict[tuple, int]:
    out: dict[tuple, int] = {}
    for t in pairs:
        out[t] = out.get(t, 0) + 1
    return out


def _none_to_str(v) -> str:
    if v is None:
        return "<null>"
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


__all__ = [
    "chapter4_margins_overall",
    "chapter4_margins_per_attribute",
    "chapter4_margins_per_category",
    "chapter4_margins_confusion",
    "chapter4_margins_missingness",
    "chapter4_margins_multirun",
    "chapter5_lymph_nodes_overall",
    "chapter5_lymph_nodes_per_attribute",
    "chapter5_lymph_nodes_per_category",
    "chapter5_lymph_nodes_per_station",
    "chapter5_lymph_nodes_confusion",
    "chapter5_lymph_nodes_missingness",
    "chapter5_lymph_nodes_multirun",
]
