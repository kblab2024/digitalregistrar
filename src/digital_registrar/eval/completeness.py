"""Completeness / missingness aggregations.

Backs ``registrar-eval completeness``. :func:`completeness_atomic` turns
gold/prediction pairs (:func:`folders.load_pairs`) into a per
(case, field) outcome table — correct / wrong / field missing / parse
error — and the aggregators below reduce it to per-method missingness
rates with CIs, refusal calibration and method-pair Δ tables.

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
carries a Wilson 95% CI (:func:`ci.wilson_ci`).
"""
from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

import pandas as pd

from .ci import wilson_ci
from .metrics import ScopeArg, field_correct, normalize
from .scope import (
    EVAL_EXCLUDED_FIELDS,
    IMPLEMENTED_ORGANS,
    get_allowed_values,
    get_categorical_fields,
    get_field_value,
    get_organ_scoreable_fields,
)

logger = logging.getLogger(__name__)

GATE_FIELDS = ("cancer_excision_report", "cancer_category")


# --- Atomic outcome table ---------------------------------------------------


def _has_value(v) -> bool:
    """A field counts as filled unless it is None or an empty list."""
    return v is not None and v != []


def _completeness_fields(gold: dict, scope: ScopeArg) -> list[str]:
    if scope is not None:
        return list(scope(gold.get("cancer_category")) if callable(scope) else scope)
    fields = list(GATE_FIELDS)
    organ = normalize(gold.get("cancer_category"))
    if gold.get("cancer_excision_report") and organ in IMPLEMENTED_ORGANS:
        fields += [f for f in get_organ_scoreable_fields(organ)
                   if f not in GATE_FIELDS and f not in EVAL_EXCLUDED_FIELDS]
    return fields


def completeness_atomic(
    pairs: Iterable,
    *,
    method: str = "pred",
    scope: ScopeArg = None,
) -> pd.DataFrame:
    """One row per (case, field) with the outcome flags
    :func:`aggregate_missingness` and :func:`refusal_calibration` need.

    ``pairs`` are :class:`folders.CasePair` objects. Fields come from the
    GOLD record: the two gate fields (``cancer_excision_report``,
    ``cancer_category``) for every case, plus every scalar /
    list-of-literals field of the gold organ's schema when the gold case
    is an eligible cancer report. Pass an explicit field list as
    ``scope`` to override (e.g. :data:`scope.FAIR_SCOPE`).

    Flags per row (a value counts as present unless None or ``[]``):
        parse_error    prediction file missing or not a JSON object
        field_missing  prediction loaded but the field is absent / null
        attempted      prediction has a value for the field
        correct/wrong  attempted and :func:`metrics.field_correct` is
                       True / False — scored without cascade gating
        gold_present   gold has a value for the field
    """
    rows: list[dict] = []
    for pair in pairs:
        gold, pred = pair.gold, pair.pred
        organ = normalize(gold.get("cancer_category"))
        for field in _completeness_fields(gold, scope):
            parse_error = pred is None
            attempted = not parse_error and _has_value(get_field_value(pred, field))
            correct = attempted and bool(field_correct(gold, pred, field, organ=organ))
            rows.append({
                "method": method, "case_id": pair.case_id, "organ": organ,
                "field": field,
                "gold_present": _has_value(get_field_value(gold, field)),
                "parse_error": parse_error,
                "field_missing": not parse_error and not attempted,
                "attempted": attempted,
                "correct": correct,
                "wrong": attempted and not correct,
            })
    columns = ["method", "case_id", "organ", "field", "gold_present",
               "parse_error", "field_missing", "attempted", "correct", "wrong"]
    return pd.DataFrame(rows, columns=columns)


# --- Outcome decomposition table --------------------------------------------


def _counts_as_int(df: pd.DataFrame) -> pd.DataFrame:
    """groupby-apply upcasts the ``n_*`` counts to float; cast them back."""
    for col in df.columns:
        if col.startswith("n_"):
            df[col] = df[col].astype(int)
    return df


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
    return _counts_as_int(grouped)


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


def _vocab_key(v) -> str:
    """Normalise a value for enum membership.

    Allowed-value lists are stringified (grade ``1,2,3`` → ``"1","2","3"``;
    bools → ``"true","false"``) but :func:`metrics.normalize` keeps ints
    and bools as-is, so stringify them here.
    """
    n = normalize(v)
    return n if isinstance(n, str) else str(n).lower()


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
    allowed_norm = {_vocab_key(v) for v in allowed}
    n_attempted = 0
    n_oov = 0
    for pred in pred_records:
        v = get_field_value(pred, field)
        if v is None:
            continue
        n_attempted += 1
        if _vocab_key(v) not in allowed_norm:
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


def out_of_vocab_table(pairs: Iterable, *, method: str = "pred") -> pd.DataFrame:
    """:func:`out_of_vocab_rate` for every categorical field of every organ
    the predictions chose.

    Predictions are grouped by their OWN ``cancer_category`` (the schema
    the model filled in), not the gold organ. Unreadable predictions and
    fields with no non-null prediction are skipped.
    """
    by_organ: dict[str, list[dict]] = {}
    for pair in pairs:
        if pair.pred is None:
            continue
        organ = normalize(pair.pred.get("cancer_category"))
        if organ in IMPLEMENTED_ORGANS:
            by_organ.setdefault(organ, []).append(pair.pred)
    rows: list[dict] = []
    for organ, preds in sorted(by_organ.items()):
        for field in get_categorical_fields(organ):
            res = out_of_vocab_rate(preds, field=field, organ=organ)
            if res["n_attempted"]:
                rows.append({"method": method, "organ": organ, "field": field, **res})
    columns = ["method", "organ", "field", "n_attempted", "n_oov", "oov_rate",
               "ci_lo", "ci_hi"]
    return pd.DataFrame(rows, columns=columns)


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
    return _counts_as_int(grouped)


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
    "completeness_atomic",
    "out_of_vocab_table",
    "aggregate_missingness",
    "method_pair_deltas",
    "out_of_vocab_rate",
    "refusal_calibration",
    "position_in_schema_correlation",
]
