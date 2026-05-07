"""Completeness / missingness aggregations.

Builds on ``scripts/eval/_common/outcome.classify_outcome`` (the
three-way correct/wrong/missing classifier) to produce per-method
missingness rates with CI and method-pair Δ tables. Lives in
``src/.../eval/`` rather than ``scripts/`` so the aggregation logic can
be unit-tested independently of the CLI harness.

Headline metrics:
    parse_error_rate      — whole-case load failures
    field_missing_rate    — case loaded, field absent/null
    total_missing_rate    — sum of the two
    attempted_rate        — 1 − total_missing_rate
    out_of_vocab_rate     — for categorical fields, predicted value
                            outside the field's allowed enum
    correct_refusal_rate  — pred=null AND gold=null (justified silence)
    lazy_missing_rate     — pred=null AND gold non-null (gave up)

All metrics are per (method, field, organ) and overall. Each row
carries Wilson 95% CI (``statsmodels.stats.proportion.proportion_confint
(method="wilson")``).
"""
from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

import pandas as pd

from .ci import wilson_ci
from .metrics import normalize
from .scope import get_allowed_values, get_field_value

logger = logging.getLogger(__name__)


# --- Outcome decomposition table --------------------------------------------


def aggregate_missingness(
    atomic: pd.DataFrame,
    *,
    by: Sequence[str] = ("method", "field", "organ"),
) -> pd.DataFrame:
    """Group an atomic outcome table by ``by`` and emit completeness rates.

    Required columns on ``atomic``: ``parse_error``, ``field_missing``,
    ``attempted``, ``correct``, ``wrong``, ``gold_present``, plus
    whatever's in ``by``. Output rows have:

        n_eligible, n_correct, n_wrong, n_field_missing, n_parse_error,
        attempted_rate (+ Wilson CI), parse_error_rate (+ Wilson CI),
        field_missing_rate (+ Wilson CI), total_missing_rate,
        attempted_accuracy, effective_accuracy.
    """
    if atomic.empty:
        return pd.DataFrame()

    def _agg(group: pd.DataFrame) -> pd.Series:
        eligible = int(group["gold_present"].sum())
        n_correct = int(group["correct"].sum())
        n_wrong = int(group["wrong"].sum())
        n_fm = int(group["field_missing"].sum())
        n_pe = int(group["parse_error"].sum())
        n_attempted = int(group["attempted"].sum())
        n_total = len(group)

        attempted_rate = n_attempted / n_total if n_total else float("nan")
        att_lo, att_hi = (wilson_ci(n_attempted, n_total) if n_total
                          else (float("nan"), float("nan")))
        parse_rate = n_pe / n_total if n_total else float("nan")
        pe_lo, pe_hi = (wilson_ci(n_pe, n_total) if n_total
                        else (float("nan"), float("nan")))
        fm_rate = n_fm / n_total if n_total else float("nan")
        fm_lo, fm_hi = (wilson_ci(n_fm, n_total) if n_total
                        else (float("nan"), float("nan")))

        attempted_acc = n_correct / n_attempted if n_attempted else float("nan")
        effective_acc = n_correct / n_total if n_total else float("nan")
        return pd.Series({
            "n_total": n_total,
            "n_eligible": eligible,
            "n_correct": n_correct,
            "n_wrong": n_wrong,
            "n_field_missing": n_fm,
            "n_parse_error": n_pe,
            "n_attempted": n_attempted,
            "attempted_rate": attempted_rate,
            "attempted_rate_ci_lo": att_lo,
            "attempted_rate_ci_hi": att_hi,
            "parse_error_rate": parse_rate,
            "parse_error_rate_ci_lo": pe_lo,
            "parse_error_rate_ci_hi": pe_hi,
            "field_missing_rate": fm_rate,
            "field_missing_rate_ci_lo": fm_lo,
            "field_missing_rate_ci_hi": fm_hi,
            "total_missing_rate": parse_rate + fm_rate,
            "attempted_accuracy": attempted_acc,
            "effective_accuracy": effective_acc,
        })

    grouped = (
        atomic.groupby(list(by), dropna=False)
        .apply(_agg, include_groups=False)
        .reset_index()
    )
    return grouped


# --- Method-pair Δ on missingness -------------------------------------------


def method_pair_deltas(
    atomic: pd.DataFrame,
    *,
    by: Sequence[str] = ("field", "organ"),
    device: str = "cpu",
) -> pd.DataFrame:
    """For every pair of methods × every (field, organ), compute the
    paired Δ on attempted_rate with McNemar p-value.

    Caller is responsible for restricting ``atomic`` to a single
    annotator and a single set of run_ids — the input must contain
    both methods on the SAME case set. Returns a long-form DataFrame
    suitable for the modularity-advantage table.

    The McNemar tests are batched: per-cell Python ``mcnemar_test`` calls
    (which can run thousands of times for a typical eval grid) are
    replaced with a single :func:`ci_gpu.mcnemar_batch` call after the
    (b, c) counts are accumulated. Per-cell p-values match the original
    one-at-a-time path bit-for-bit (same closed-form arithmetic / scipy
    CDFs). ``device`` is plumbed for API symmetry with the other stats
    functions.
    """
    from . import ci_gpu

    methods = sorted(atomic["method"].dropna().unique().tolist())
    if len(methods) < 2:
        return pd.DataFrame()

    rows: list[dict] = []
    bp_list: list[int] = []
    cp_list: list[int] = []
    # First pass: build all per-cell rows + flat (b, c) arrays.
    for i, m_a in enumerate(methods):
        for m_b in methods[i + 1:]:
            for keys, sub in atomic.groupby(list(by), dropna=False):
                a = (sub[sub["method"] == m_a]
                     .groupby("case_id")["attempted"].any())
                b = (sub[sub["method"] == m_b]
                     .groupby("case_id")["attempted"].any())
                shared = a.index.intersection(b.index)
                if shared.empty:
                    continue
                a_arr = a.loc[shared].astype(int).to_numpy()
                b_arr = b.loc[shared].astype(int).to_numpy()
                bp = int(((a_arr == 1) & (b_arr == 0)).sum())
                cp = int(((a_arr == 0) & (b_arr == 1)).sum())

                row = (dict(zip(by, keys, strict=True))
                       if isinstance(keys, tuple) else {by[0]: keys})
                row.update({
                    "method_a": m_a, "method_b": m_b,
                    "n_paired": int(len(shared)),
                    "attempted_rate_a": float(a_arr.mean()),
                    "attempted_rate_b": float(b_arr.mean()),
                    "delta_attempted_rate": float(a_arr.mean() - b_arr.mean()),
                    "mcnemar_b": bp, "mcnemar_c": cp,
                })
                rows.append(row)
                bp_list.append(bp)
                cp_list.append(cp)

    if not rows:
        return pd.DataFrame()

    # Second pass: one batched McNemar call across every cell.
    mc = ci_gpu.mcnemar_batch(bp_list, cp_list, device=device)
    for idx, row in enumerate(rows):
        row["mcnemar_statistic"] = float(mc["statistic"][idx])
        row["mcnemar_p_value"] = float(mc["p_value"][idx])
        row["mcnemar_method"] = str(mc["method"][idx])
    return pd.DataFrame(rows)


# --- Schema-conformance / out-of-vocab --------------------------------------


def out_of_vocab_rate(
    pred_records: Iterable[dict],
    *,
    field: str,
    organ: str | None = None,
) -> dict:
    """Fraction of attempted predictions whose value is outside the
    allowed enum for ``field``.

    Pairs with the modularity-advantage argument: schema-constrained
    pipelines (DSPy) should produce ~0% out-of-vocab values; raw-JSON
    typically does not.

    ``pred_records`` is an iterable of prediction dicts (caller filters
    to a specific organ-eligible subset). Returns counts + Wilson CI.
    """
    allowed = get_allowed_values(field, organ)
    if not allowed:
        return {"n_attempted": 0, "n_oov": 0, "oov_rate": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan")}
    allowed_norm = {normalize(v) for v in allowed}
    n_attempted = 0
    n_oov = 0
    for pred in pred_records:
        v = get_field_value(pred, field)
        if v is None:
            continue
        n_attempted += 1
        if normalize(v) not in allowed_norm:
            n_oov += 1
    if n_attempted == 0:
        return {"n_attempted": 0, "n_oov": 0, "oov_rate": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan")}
    rate = n_oov / n_attempted
    lo, hi = wilson_ci(n_oov, n_attempted)
    return {
        "n_attempted": n_attempted, "n_oov": n_oov,
        "oov_rate": rate, "ci_lo": lo, "ci_hi": hi,
    }


# --- Refusal calibration ----------------------------------------------------


def refusal_calibration(atomic: pd.DataFrame,
                        *,
                        by: Sequence[str] = ("method", "field", "organ"),
                        ) -> pd.DataFrame:
    """Distinguish ``correct_refusal`` (pred=null, gold=null) from
    ``lazy_missingness`` (pred=null, gold non-null).

    Required columns: ``attempted``, ``gold_present`` plus ``by``. Rows
    where gold is null AND pred is also null are correct refusals;
    rows where gold has a value AND pred is missing are lazy.
    """
    if atomic.empty:
        return pd.DataFrame()

    def _agg(group: pd.DataFrame) -> pd.Series:
        not_attempted = ~group["attempted"]
        n_pred_null = int(not_attempted.sum())
        gold_null = ~group["gold_present"]
        correct_refusal = int((not_attempted & gold_null).sum())
        lazy = int((not_attempted & ~gold_null).sum())
        if n_pred_null == 0:
            justified_share = float("nan")
            cr_lo = cr_hi = lazy_lo = lazy_hi = float("nan")
        else:
            justified_share = correct_refusal / n_pred_null
            cr_lo, cr_hi = wilson_ci(correct_refusal, n_pred_null)
            lazy_lo, lazy_hi = wilson_ci(lazy, n_pred_null)
        return pd.Series({
            "n_pred_null": n_pred_null,
            "n_correct_refusal": correct_refusal,
            "n_lazy_missing": lazy,
            "correct_refusal_rate": (correct_refusal / n_pred_null
                                     if n_pred_null else float("nan")),
            "correct_refusal_ci_lo": cr_lo,
            "correct_refusal_ci_hi": cr_hi,
            "lazy_missing_rate": (lazy / n_pred_null
                                  if n_pred_null else float("nan")),
            "lazy_missing_ci_lo": lazy_lo,
            "lazy_missing_ci_hi": lazy_hi,
            "justified_missingness_share": justified_share,
        })

    grouped = (
        atomic.groupby(list(by), dropna=False)
        .apply(_agg, include_groups=False)
        .reset_index()
    )
    return grouped


# --- Position-in-schema correlation -----------------------------------------


def position_in_schema_correlation(
    atomic: pd.DataFrame,
    *,
    field_order: Sequence[str],
    by: str = "method",
) -> pd.DataFrame:
    """Spearman correlation between schema position and field-missing rate.

    Tests the context-window-pressure hypothesis (R2.4: lower
    performance on quantitative fields late in the schema). Positive ρ
    means later fields are more frequently missing.

    Implementation: ``scipy.stats.spearmanr`` (Virtanen et al. 2020).
    """
    rows: list[dict] = []
    pos = {f: i for i, f in enumerate(field_order)}
    if atomic.empty or not pos:
        return pd.DataFrame()
    from scipy.stats import spearmanr
    for method, sub in atomic.groupby(by):
        per_field = (
            sub.assign(_idx=sub["field"].map(pos))
            .dropna(subset=["_idx"])
            .groupby("field")
            .agg(missing_rate=("field_missing", "mean"),
                 _idx=("_idx", "first"))
            .reset_index()
        )
        if len(per_field) < 3:
            rows.append({"method": method, "n_fields": len(per_field),
                         "spearman_rho": float("nan"), "p_value": float("nan")})
            continue
        rho, p = spearmanr(per_field["_idx"], per_field["missing_rate"])
        rows.append({
            "method": method,
            "n_fields": int(len(per_field)),
            "spearman_rho": float(rho),
            "p_value": float(p),
        })
    return pd.DataFrame(rows)


__all__ = [
    "aggregate_missingness",
    "method_pair_deltas",
    "out_of_vocab_rate",
    "refusal_calibration",
    "position_in_schema_correlation",
]
