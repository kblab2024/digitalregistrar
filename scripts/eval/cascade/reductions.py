"""Reductions for the cascade chapter outputs.

Builds the chapter1 / chapter2 / chapter3 CSVs from the cascade atomic
table. Where the legacy non_nested / nested orchestrators have proven
reductions, we delegate to them rather than re-implement.

The atomic-table column contract (built by :mod:`run_cascade`):

    run_id, model, dataset, case_id, organ_idx, organ, subgroup,
    cascade_stage, gate_pass,
    others_disposition,
    field, field_kind,
    gold_present, attempted, correct, wrong, field_missing, parse_error,
    error_mode, gold_value, pred_value
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from digital_registrar_research.benchmarks.eval.stats import (
    accuracy_flip_rate,
    bca_bootstrap_ci,
    cohens_kappa,
    cronbach_alpha,
    icc_2_1,
    icc_3_k,
    krippendorff_alpha,
    matthews_corrcoef,
    missing_flip_rate,
    per_case_run_sd,
    spearman_brown,
    weighted_kappa,
    wilson_ci,
)


def _safe_proportion(k: int, n: int) -> float:
    return float(k / n) if n > 0 else float("nan")


def _wilson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    return wilson_ci(k, n, alpha)


def _stage_c_scalar(atomic: pd.DataFrame) -> pd.DataFrame:
    """Stage-C rows minus nested_list rows.

    The chapter-3 per-field accuracy tables score scalar fields under
    binary (correct / wrong) semantics. Nested fields (margins,
    regional_lymph_node, biomarkers) live in atomic with ``correct = f1``
    (a float) and are summarised separately by the nested reducers.
    """
    if atomic.empty:
        return atomic
    stage_c = atomic[atomic["cascade_stage"] == "C"]
    if "field_kind" in stage_c.columns:
        stage_c = stage_c[stage_c["field_kind"] != "nested_list"]
    return stage_c


# --- Chapter 1: eligibility triage ----------------------------------------

def chapter1_overall(
    atomic: pd.DataFrame,
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Eligibility triage headline metrics.

    Columns: model, dataset, n_total, n_correct, accuracy, ci_lo, ci_hi,
    n_positive_gold, n_negative_gold, sensitivity, specificity, mcc,
    cohens_kappa.
    """
    stage_a = atomic[atomic["cascade_stage"] == "A"]
    if stage_a.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for (model, dataset), sub in stage_a.groupby(["model", "dataset"], dropna=False):
        n_total = len(sub)
        n_correct = int(sub["correct"].fillna(False).astype(bool).sum())
        acc = _safe_proportion(n_correct, n_total)
        lo, hi = _wilson(n_correct, n_total, alpha)
        # Sens/spec on the binary eligibility decision.
        gold_pos = sub["gold_value"].astype(str).isin(["True", "true", "1"])
        pred_pos = sub["pred_value"].astype(str).isin(["True", "true", "1"])
        tp = int(((gold_pos) & (pred_pos)).sum())
        fn = int(((gold_pos) & (~pred_pos)).sum())
        fp = int(((~gold_pos) & (pred_pos)).sum())
        tn = int(((~gold_pos) & (~pred_pos)).sum())
        sens = _safe_proportion(tp, tp + fn)
        spec = _safe_proportion(tn, tn + fp)
        mcc_val = float("nan")
        try:
            mcc_val = matthews_corrcoef(
                gold_pos.tolist(), pred_pos.tolist(),
            )
        except Exception:
            pass
        kappa_val = float("nan")
        try:
            kappa_val = cohens_kappa(
                gold_pos.tolist(), pred_pos.tolist(),
            )
        except Exception:
            pass
        rows.append({
            "model": model, "dataset": dataset,
            "n_total": n_total, "n_correct": n_correct,
            "accuracy": acc, "ci_lo": lo, "ci_hi": hi,
            "n_positive_gold": int(gold_pos.sum()),
            "n_negative_gold": int((~gold_pos).sum()),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "sensitivity": sens, "specificity": spec,
            "mcc": mcc_val, "cohens_kappa": kappa_val,
        })
    return pd.DataFrame(rows)


def chapter1_confusion(atomic: pd.DataFrame) -> pd.DataFrame:
    """2x2 confusion of eligibility, per (model, dataset)."""
    stage_a = atomic[atomic["cascade_stage"] == "A"]
    if stage_a.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for (model, dataset), sub in stage_a.groupby(["model", "dataset"], dropna=False):
        gold_pos = sub["gold_value"].astype(str).isin(["True", "true", "1"])
        pred_pos = sub["pred_value"].astype(str).isin(["True", "true", "1"])
        tp = int(((gold_pos) & (pred_pos)).sum())
        fn = int(((gold_pos) & (~pred_pos)).sum())
        fp = int(((~gold_pos) & (pred_pos)).sum())
        tn = int(((~gold_pos) & (~pred_pos)).sum())
        rows.append({
            "model": model, "dataset": dataset,
            "TP_eligible_predicted_eligible": tp,
            "FP_ineligible_predicted_eligible": fp,
            "FN_eligible_predicted_ineligible": fn,
            "TN_ineligible_predicted_ineligible": tn,
            "n_total": tp + fp + fn + tn,
        })
    return pd.DataFrame(rows)


# --- Chapter 2: organ classification --------------------------------------

def chapter2_overall(
    atomic: pd.DataFrame,
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Organ-classification headline. Restricted to Stage-A passers.

    Columns: model, dataset, n_eligible_for_b, n_correct, accuracy,
    ci_lo, ci_hi, macro_f1, weighted_kappa, cohens_kappa.
    """
    stage_b = atomic[atomic["cascade_stage"] == "B"]
    if stage_b.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for (model, dataset), sub in stage_b.groupby(["model", "dataset"], dropna=False):
        n_total = len(sub)
        n_correct = int(sub["correct"].fillna(False).astype(bool).sum())
        acc = _safe_proportion(n_correct, n_total)
        lo, hi = _wilson(n_correct, n_total, alpha)
        gold = sub["gold_value"].dropna().tolist()
        pred = sub["pred_value"].dropna().tolist()
        # Pair-by-row.
        paired = list(zip(sub["gold_value"].tolist(), sub["pred_value"].tolist()))
        paired = [(g, p) for g, p in paired if g is not None and p is not None]
        kappa_val = float("nan")
        try:
            if paired:
                gv, pv = zip(*paired)
                kappa_val = cohens_kappa(list(gv), list(pv))
        except Exception:
            pass
        # Macro F1 via per-class P/R/F1 helper from stats_extra.
        macro_f1 = float("nan")
        try:
            from scripts.eval._common.stats_extra import per_class_prf1
            if paired:
                gv, pv = zip(*paired)
                pr = per_class_prf1(list(gv), list(pv))
                macro_f1 = pr.get("macro_avg", {}).get("f1", float("nan"))
        except Exception:
            pass
        rows.append({
            "model": model, "dataset": dataset,
            "n_eligible_for_b": n_total,
            "n_correct": n_correct,
            "accuracy": acc, "ci_lo": lo, "ci_hi": hi,
            "cohens_kappa": kappa_val,
            "macro_f1": macro_f1,
        })
    return pd.DataFrame(rows)


def chapter2_confusion_per_class(atomic: pd.DataFrame) -> pd.DataFrame:
    """Per-class P/R/F1 + support for the 11-class organ classifier."""
    stage_b = atomic[atomic["cascade_stage"] == "B"]
    if stage_b.empty:
        return pd.DataFrame()
    from scripts.eval._common.stats_extra import per_class_prf1
    rows: list[dict] = []
    for (model, dataset), sub in stage_b.groupby(["model", "dataset"], dropna=False):
        paired = [(g, p) for g, p in zip(sub["gold_value"], sub["pred_value"])
                  if g is not None and p is not None]
        if not paired:
            continue
        gv, pv = zip(*paired)
        pr = per_class_prf1(list(gv), list(pv))
        for label, vals in pr.items():
            if not isinstance(vals, dict):
                continue
            rows.append({
                "model": model, "dataset": dataset,
                "label": label,
                "precision": vals.get("precision"),
                "recall": vals.get("recall"),
                "f1": vals.get("f1"),
                "support": vals.get("support", float("nan")),
            })
    return pd.DataFrame(rows)


# --- Chapter 3: field extraction ------------------------------------------

def chapter3_per_field_overall(
    atomic: pd.DataFrame,
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Stage-C scalar field accuracy with Wilson CI and Cohen's kappa.

    One row per (model, dataset, field). Excludes ``nested_list`` rows
    — those carry an F1 in ``correct`` (not a bool) and are summarised
    in ``nested_per_field_per_organ.csv``.
    """
    stage_c = _stage_c_scalar(atomic)
    if stage_c.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for (model, dataset, field), sub in stage_c.groupby(
        ["model", "dataset", "field"], dropna=False,
    ):
        attempted = sub[sub["attempted"] == True]  # noqa: E712
        n_attempted = len(attempted)
        n_correct = int(attempted["correct"].fillna(False).astype(bool).sum())
        acc = _safe_proportion(n_correct, n_attempted)
        lo, hi = _wilson(n_correct, n_attempted, alpha)
        n_total = len(sub)
        coverage = _safe_proportion(n_attempted, n_total)
        # Kappa per field for paired non-null gold/pred.
        kappa_val = float("nan")
        paired = [(g, p) for g, p in zip(attempted["gold_value"],
                                          attempted["pred_value"])
                  if g is not None and p is not None]
        try:
            if paired:
                gv, pv = zip(*paired)
                kappa_val = cohens_kappa(list(gv), list(pv))
        except Exception:
            pass
        eff_acc = _safe_proportion(n_correct, n_total)
        eff_lo, eff_hi = _wilson(n_correct, n_total, alpha)
        rows.append({
            "model": model, "dataset": dataset, "field": field,
            "n_total": n_total, "n_attempted": n_attempted,
            "n_correct": n_correct,
            "coverage": coverage,
            "accuracy_attempted": acc,
            "ci_lo": lo, "ci_hi": hi,
            "effective_accuracy": eff_acc,
            "effective_acc_wilson_lo": eff_lo,
            "effective_acc_wilson_hi": eff_hi,
            "cohens_kappa": kappa_val,
        })
    return pd.DataFrame(rows)


def chapter3_per_field_by_organ(
    atomic: pd.DataFrame,
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Stage-C per-field accuracy stratified by organ.

    Excludes ``nested_list`` rows — see ``chapter3_per_field_overall``.
    """
    stage_c = _stage_c_scalar(atomic)
    if stage_c.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for (model, dataset, organ, field), sub in stage_c.groupby(
        ["model", "dataset", "organ", "field"], dropna=False,
    ):
        attempted = sub[sub["attempted"] == True]  # noqa: E712
        n_attempted = len(attempted)
        n_total = len(sub)
        n_correct = int(attempted["correct"].fillna(False).astype(bool).sum())
        acc = _safe_proportion(n_correct, n_attempted)
        lo, hi = _wilson(n_correct, n_attempted, alpha)
        eff_acc = _safe_proportion(n_correct, n_total)
        eff_lo, eff_hi = _wilson(n_correct, n_total, alpha)
        coverage = _safe_proportion(n_attempted, n_total)
        rows.append({
            "model": model, "dataset": dataset,
            "organ": organ, "field": field,
            "n_total": n_total,
            "n_attempted": n_attempted,
            "n_correct": n_correct,
            "coverage": coverage,
            "accuracy_attempted": acc,
            "attempted_acc_wilson_lo": lo,
            "attempted_acc_wilson_hi": hi,
            "effective_accuracy": eff_acc,
            "effective_acc_wilson_lo": eff_lo,
            "effective_acc_wilson_hi": eff_hi,
            # Legacy aliases preserved for older consumers.
            "accuracy": acc,
            "ci_lo": lo, "ci_hi": hi,
        })
    return pd.DataFrame(rows)


def chapter3_per_organ_overall(
    atomic: pd.DataFrame,
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Mean accuracy across all fields per organ (Stage C only).

    Excludes ``nested_list`` rows — see ``chapter3_per_field_overall``.
    """
    stage_c = _stage_c_scalar(atomic)
    if stage_c.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for (model, dataset, organ), sub in stage_c.groupby(
        ["model", "dataset", "organ"], dropna=False,
    ):
        attempted = sub[sub["attempted"] == True]  # noqa: E712
        n_attempted = len(attempted)
        n_correct = int(attempted["correct"].fillna(False).astype(bool).sum())
        acc = _safe_proportion(n_correct, n_attempted)
        lo, hi = _wilson(n_correct, n_attempted, alpha)
        rows.append({
            "model": model, "dataset": dataset, "organ": organ,
            "n_attempted": n_attempted, "n_correct": n_correct,
            "accuracy": acc, "ci_lo": lo, "ci_hi": hi,
        })
    return pd.DataFrame(rows)


# --- Chapter 3: nested-field summaries ------------------------------------

# Matched-pair attribute columns expected on the nested sidecar. Per
# field, the LN scorer ships counts of matched pairs where each inner
# attribute agrees; the margin scorer ships an analogous bundle. We
# flatten these into one row per (organ, field, attribute) for the
# per-attribute CSV.
_NESTED_ATTRIBUTE_COLS: dict[str, dict[str, str]] = {
    "regional_lymph_node": {
        "examined_correct": "ln_station_examined_correct",
        "involved_correct": "ln_station_involved_correct",
        "category_correct": "ln_station_category_correct",
        "side_correct":     "ln_station_side_correct",
    },
    "margins": {
        "status_correct":   "margin_status_correct",
        "distance_correct": "margin_distance_correct",
        "category_correct": "margin_category_correct",
    },
}

_NESTED_MATCHED_COL = {
    "regional_lymph_node": "ln_station_matched",
    "margins":             "margin_matched",
}


def chapter3_nested_per_field_per_organ(
    nested: pd.DataFrame,
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Per (model, dataset, organ, field) headline for nested fields.

    Columns: model, dataset, organ, field, n_attempted, attempted_f1,
    plus field-specific headlines pulled straight from the per-case
    scorer dicts (group_recall / group_precision / examined_mae /
    involved_mae / any_positive_acc for LN; any_involved_acc /
    closest_dist_mae for margins; tp/fp/fn micro for biomarkers).
    """
    if nested is None or nested.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    by = ["model", "dataset", "organ", "field"]
    for keys, sub in nested.groupby(by, dropna=False):
        n_attempted = int(sub["attempted"].sum()) if "attempted" in sub else len(sub)
        f1_vals = pd.to_numeric(sub.get("f1"), errors="coerce").dropna()
        attempted_f1 = float(f1_vals.mean()) if not f1_vals.empty else float("nan")
        row = dict(zip(by, keys))
        row.update({
            "n_cases": len(sub),
            "n_attempted": n_attempted,
            "attempted_f1": attempted_f1,
        })
        field = keys[3]
        if field == "regional_lymph_node":
            row.update(_ln_headline_block(sub))
        elif field == "margins":
            row.update(_margin_headline_block(sub))
        elif field == "biomarkers":
            row.update(_biomarker_headline_block(sub))
        rows.append(row)
    return pd.DataFrame(rows)


def _ln_headline_block(sub: pd.DataFrame) -> dict:
    out = {}
    for col, label in [
        ("ln_group_recall", "group_recall"),
        ("ln_group_precision", "group_precision"),
        ("ln_examined_total_abs_err", "examined_mae"),
        ("ln_examined_total_correct_tol", "examined_acc_tol1"),
        ("ln_involved_total_abs_err", "involved_mae"),
        ("ln_involved_total_correct_tol", "involved_acc_tol1"),
        ("ln_any_positive_correct", "any_positive_acc"),
    ]:
        out[label] = (
            float(pd.to_numeric(sub.get(col), errors="coerce").mean())
            if col in sub.columns else float("nan")
        )
    return out


def _margin_headline_block(sub: pd.DataFrame) -> dict:
    out = {}
    for col, label in [
        ("margin_any_involved_correct", "any_involved_acc"),
        ("margin_closest_distance_abs_err", "closest_dist_mae"),
        ("margin_closest_distance_correct_tol", "closest_dist_acc_tol2"),
        ("margin_closest_distance_has_both", "closest_dist_both_rate"),
    ]:
        out[label] = (
            float(pd.to_numeric(sub.get(col), errors="coerce").mean())
            if col in sub.columns else float("nan")
        )
    return out


def _biomarker_headline_block(sub: pd.DataFrame) -> dict:
    tp = int(pd.to_numeric(sub.get("tp"), errors="coerce").fillna(0).sum())
    fp = int(pd.to_numeric(sub.get("fp"), errors="coerce").fillna(0).sum())
    fn = int(pd.to_numeric(sub.get("fn"), errors="coerce").fillna(0).sum())
    if tp + fp == 0 or tp + fn == 0:
        prec = rec = micro_f1 = float("nan")
    else:
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        micro_f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "micro_precision": prec, "micro_recall": rec, "micro_f1": micro_f1}


def chapter3_nested_per_attribute_per_organ(
    nested: pd.DataFrame,
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Matched-pair attribute accuracy for LN and margins.

    One row per (model, dataset, organ, field, attribute). For matched
    pairs, what fraction got each inner attribute right (Wilson CI
    on the matched-pair denominator)?
    """
    if nested is None or nested.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for (model, dataset, organ, field), sub in nested.groupby(
        ["model", "dataset", "organ", "field"], dropna=False,
    ):
        if field not in _NESTED_ATTRIBUTE_COLS:
            continue
        matched_col = _NESTED_MATCHED_COL[field]
        if matched_col not in sub.columns:
            continue
        n_matched = int(pd.to_numeric(sub[matched_col], errors="coerce").fillna(0).sum())
        for attr_label, attr_col in _NESTED_ATTRIBUTE_COLS[field].items():
            if attr_col not in sub.columns:
                continue
            n_correct = int(
                pd.to_numeric(sub[attr_col], errors="coerce").fillna(0).sum()
            )
            acc = _safe_proportion(n_correct, n_matched)
            lo, hi = _wilson(n_correct, n_matched, alpha)
            rows.append({
                "model": model, "dataset": dataset, "organ": organ,
                "field": field, "attribute": attr_label,
                "n_matched_pairs": n_matched,
                "n_attribute_correct": n_correct,
                "accuracy": acc, "ci_lo": lo, "ci_hi": hi,
            })
    return pd.DataFrame(rows)


def chapter3_biomarker_per_category(
    atomic: pd.DataFrame,
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Per-category biomarker accuracy.

    The cascade scorer emits one row per ``biomarker_<cat>`` field in
    atomic (er/pr/her2/ki67 for breast, msh2/msh6/pms2/mlh1 for
    colorectal). One row per (model, dataset, organ, category) with
    attempted accuracy + Wilson CI.
    """
    if atomic is None or atomic.empty:
        return pd.DataFrame()

    bm = atomic[atomic["field"].astype(str).str.startswith("biomarker_")]
    if bm.empty:
        return pd.DataFrame()
    bm = bm.copy()
    bm["category"] = bm["field"].str[len("biomarker_"):]
    rows: list[dict] = []
    for (model, dataset, organ, category), sub in bm.groupby(
        ["model", "dataset", "organ", "category"], dropna=False,
    ):
        attempted = sub[sub["attempted"] == True]  # noqa: E712
        n_attempted = len(attempted)
        n_correct = int(attempted["correct"].fillna(False).astype(bool).sum())
        acc = _safe_proportion(n_correct, n_attempted)
        lo, hi = _wilson(n_correct, n_attempted, alpha)
        rows.append({
            "model": model, "dataset": dataset, "organ": organ,
            "category": category,
            "n_total": len(sub), "n_attempted": n_attempted,
            "n_correct": n_correct,
            "accuracy_attempted": acc, "ci_lo": lo, "ci_hi": hi,
        })
    return pd.DataFrame(rows)


# --- Multi-run reliability per chapter ------------------------------------

def _correctness_matrix(
    sub: pd.DataFrame,
    *,
    case_col: str = "case_id",
    run_col: str = "run_id",
    correct_col: str = "correct",
) -> pd.DataFrame:
    """Pivot a long-form DF to (case_id × run_id) of correctness values."""
    sub = sub.copy()
    sub[correct_col] = pd.to_numeric(sub[correct_col], errors="coerce")
    return sub.pivot_table(
        index=case_col, columns=run_col, values=correct_col, aggfunc="first",
    )


def chapter_multirun_reliability(
    atomic: pd.DataFrame,
    *,
    stage: str,
    field: str | None = None,
) -> pd.DataFrame:
    """Multi-run reliability for one cascade stage.

    Computes ICC(2,1), ICC(3,k), Cronbach alpha, accuracy_flip_rate per
    (model, dataset[, field]) over the case × run correctness matrix.

    For Stage A and B, set ``field=None`` (only one field per stage).
    For Stage C, pass a specific ``field`` or call once per field.
    """
    sub = atomic[atomic["cascade_stage"] == stage]
    if field is not None:
        sub = sub[sub["field"] == field]
    if sub.empty or sub["run_id"].nunique() < 2:
        return pd.DataFrame()
    rows: list[dict] = []
    by = ["model", "dataset"]
    if field is not None:
        by.append("field")
    for keys, grp in sub.groupby(by, dropna=False):
        mat = _correctness_matrix(grp).to_numpy(dtype=float)
        if mat.shape[0] < 2 or mat.shape[1] < 2:
            continue
        i21 = icc_2_1(mat)
        i3k = icc_3_k(mat)
        alpha_val = cronbach_alpha(mat)
        flip = accuracy_flip_rate(mat)
        sd = per_case_run_sd(mat)
        row = {
            "model": keys[0], "dataset": keys[1],
            "stage": stage,
            "icc_2_1": i21["icc"], "icc_2_1_lo": i21["ci_lo"], "icc_2_1_hi": i21["ci_hi"],
            "icc_3_k": i3k["icc"], "icc_3_k_lo": i3k["ci_lo"], "icc_3_k_hi": i3k["ci_hi"],
            "cronbach_alpha": alpha_val,
            "accuracy_flip_rate": flip,
            "per_case_sd_mean": sd["mean"],
            "per_case_sd_p90": sd["p90"],
            "n_cases": sd["n_cases"], "n_runs": int(mat.shape[1]),
        }
        if field is not None:
            row["field"] = keys[2]
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = [
    "chapter1_overall",
    "chapter1_confusion",
    "chapter2_overall",
    "chapter2_confusion_per_class",
    "chapter3_per_field_overall",
    "chapter3_per_field_by_organ",
    "chapter3_per_organ_overall",
    "chapter3_nested_per_field_per_organ",
    "chapter3_nested_per_attribute_per_organ",
    "chapter3_biomarker_per_category",
    "chapter_multirun_reliability",
]
