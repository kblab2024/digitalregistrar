"""Paired cross-model tests for the cascade.

Consumes the cascade atomic table (long-form, one row per
``(run, case, field)`` with a ``model`` column) and emits one CSV per
chapter with paired-McNemar / Cochran-Q / paired-bootstrap deltas.

The driver assumes the atomic table has been built across multiple
models (or multiple ablation cells) on the same case set. If only one
model is present, the output CSVs are empty (cleanly).

Output schema (per row):
    field, organ (or "ALL"), test_name,
    model_a, model_b,                     # filled for pairwise tests
    statistic, df, p_raw,
    p_adjusted_holm, p_adjusted_bh,
    effect_size, effect_kind,
    effect_ci_lo, effect_ci_hi,
    n, low_power, notes
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from digital_registrar_research.benchmarks.eval.stats import (
    cochran_q,
    holm,
    bh_fdr,
    mcnemar,
    paired_bootstrap_delta,
    stuart_maxwell,
)


def _collect_pairs(
    df: pd.DataFrame,
    *,
    model_col: str = "model",
    case_col: str = "case_id",
    field_col: str = "field",
    correct_col: str = "correct",
) -> dict:
    """Pivot the atomic table to a (case_id, model) -> correctness mapping
    per field."""
    out: dict[str, pd.DataFrame] = {}
    for field, sub in df.groupby(field_col):
        pivot = sub.pivot_table(
            index=case_col, columns=model_col, values=correct_col,
            aggfunc="mean",  # if multiple runs per (case, model), average.
        )
        out[field] = pivot
    return out


def pairwise_mcnemar_grid(
    atomic: pd.DataFrame,
    *,
    model_col: str = "model",
    field_col: str = "field",
    organ_col: str = "organ",
    correct_col: str = "correct",
    case_col: str = "case_id",
) -> pd.DataFrame:
    """All-pair McNemar + paired-bootstrap delta per (field, organ).

    Adds Holm- and BH-adjusted p-values within the field family per
    organ. Empty input -> empty DataFrame.
    """
    if atomic.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    organs = sorted(atomic[organ_col].dropna().unique().tolist()) + ["ALL"]
    for organ in organs:
        sub = atomic if organ == "ALL" else atomic[atomic[organ_col] == organ]
        if sub.empty:
            continue
        per_field_pivots = _collect_pairs(
            sub, model_col=model_col, case_col=case_col,
            field_col=field_col, correct_col=correct_col,
        )
        for field, pivot in per_field_pivots.items():
            models = pivot.columns.tolist()
            for i in range(len(models)):
                for j in range(i + 1, len(models)):
                    a_col = models[i]
                    b_col = models[j]
                    a_b = pivot[[a_col, b_col]].dropna()
                    if a_b.empty:
                        continue
                    a_vals = a_b[a_col].astype(float).tolist()
                    b_vals = a_b[b_col].astype(float).tolist()
                    mc = mcnemar(a_vals, b_vals)
                    boot = paired_bootstrap_delta(a_vals, b_vals)
                    rows.append({
                        "field": field, "organ": organ,
                        "test_name": "mcnemar",
                        "model_a": a_col, "model_b": b_col,
                        "statistic": mc.statistic,
                        "df": mc.df,
                        "p_raw": mc.p_raw,
                        "effect_size": boot.effect_size,
                        "effect_kind": "accuracy_delta",
                        "effect_ci_lo": boot.effect_ci_lo,
                        "effect_ci_hi": boot.effect_ci_hi,
                        "n": mc.n,
                        "notes": mc.notes,
                    })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Adjust within (organ, field) families: each pair gets adjusted
    # against the family of all (model_a, model_b) pairs for the same
    # field × organ.
    df["p_adjusted_holm"] = float("nan")
    df["p_adjusted_bh"] = float("nan")
    for (organ, field), group in df.groupby(["organ", "field"]):
        adj_holm = holm(group["p_raw"].tolist())
        adj_bh = bh_fdr(group["p_raw"].tolist())
        df.loc[group.index, "p_adjusted_holm"] = adj_holm
        df.loc[group.index, "p_adjusted_bh"] = adj_bh
    return df


def cochran_q_per_field(
    atomic: pd.DataFrame,
    *,
    model_col: str = "model",
    field_col: str = "field",
    organ_col: str = "organ",
    correct_col: str = "correct",
    case_col: str = "case_id",
) -> pd.DataFrame:
    """Cochran's Q per (field, organ) when k_models >= 3.

    Tests "all k models perform identically" before running pairwise
    McNemar — guards against inflated alpha from k(k-1)/2 pairwise tests.
    """
    if atomic.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    organs = sorted(atomic[organ_col].dropna().unique().tolist()) + ["ALL"]
    for organ in organs:
        sub = atomic if organ == "ALL" else atomic[atomic[organ_col] == organ]
        if sub.empty:
            continue
        per_field = _collect_pairs(
            sub, model_col=model_col, case_col=case_col,
            field_col=field_col, correct_col=correct_col,
        )
        for field, pivot in per_field.items():
            models = pivot.columns.tolist()
            if len(models) < 3:
                continue
            mat = pivot.dropna().to_numpy(dtype=float)
            if mat.shape[0] < 2:
                continue
            res = cochran_q(mat, method_labels=models)
            rows.append({
                "field": field, "organ": organ,
                "test_name": "cochran_q",
                "model_a": "*", "model_b": "*",
                "statistic": res.statistic, "df": res.df,
                "p_raw": res.p_raw,
                "effect_size": float("nan"), "effect_kind": "",
                "effect_ci_lo": float("nan"), "effect_ci_hi": float("nan"),
                "n": res.n, "notes": res.notes,
            })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["p_adjusted_holm"] = holm(df["p_raw"].tolist())
    df["p_adjusted_bh"] = bh_fdr(df["p_raw"].tolist())
    return df


def stuart_maxwell_per_organ(
    organ_classification: pd.DataFrame,
    *,
    model_col: str = "model",
    case_col: str = "case_id",
    gold_col: str = "gold_value",
    pred_col: str = "pred_value",
) -> pd.DataFrame:
    """Pairwise Stuart-Maxwell on Stage-B organ classification.

    For each pair of models, builds a (n_cases, k_classes) paired
    confusion of pred-A vs pred-B (using the same gold) and tests
    marginal homogeneity. df = k_classes - 1.

    Empty input -> empty DataFrame.
    """
    if organ_classification.empty:
        return pd.DataFrame()
    pivot = organ_classification.pivot_table(
        index=case_col, columns=model_col, values=pred_col, aggfunc="first",
    )
    rows: list[dict] = []
    models = pivot.columns.tolist()
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a_col, b_col = models[i], models[j]
            paired = pivot[[a_col, b_col]].dropna()
            if paired.empty:
                continue
            res = stuart_maxwell(
                paired[a_col].tolist(), paired[b_col].tolist(),
            )
            rows.append({
                "test_name": "stuart_maxwell",
                "model_a": a_col, "model_b": b_col,
                "statistic": res.statistic, "df": res.df,
                "p_raw": res.p_raw,
                "effect_size": float("nan"), "effect_kind": "",
                "effect_ci_lo": float("nan"), "effect_ci_hi": float("nan"),
                "n": res.n, "notes": res.notes,
            })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["p_adjusted_holm"] = holm(df["p_raw"].tolist())
    df["p_adjusted_bh"] = bh_fdr(df["p_raw"].tolist())
    return df


__all__ = [
    "pairwise_mcnemar_grid",
    "cochran_q_per_field",
    "stuart_maxwell_per_organ",
]
