"""Cascade-specific diagnostics.

These functions consume the cascade atomic table (one row per
``(run, case, field)`` with ``cascade_stage``, ``gate_pass`` columns)
and surface attrition, error compounding, and selection-bias
statistics that are unique to the gated evaluation.

Methods
-------
* :func:`cascade_funnel` — Cohort counts at each gate; the canonical
  flow diagram for the manuscript's §3 cascade figure.
* :func:`conditional_accuracy_grid` — ``P(stage_c_field_correct |
  stage_b_correct AND stage_a_correct)`` per field. Reveals whether
  errors compound across stages.
* :func:`attrition_propensity` — Logistic regression of "case dropped
  at gate" on case-level features (report length, organ, dataset).
  Identifies systematic biases in cohort attrition.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats as sstats


def cascade_funnel(
    atomic: pd.DataFrame,
    *,
    case_col: str = "case_id",
    stage_col: str = "cascade_stage",
    gate_pass_col: str = "gate_pass",
) -> pd.DataFrame:
    """Counts at each cascade gate.

    Returns a DataFrame with one row per stage:

        stage, n_total, n_passed, n_dropped, attrition_rate

    ``n_total`` is the entry cohort to the stage (= prior stage's
    ``n_passed``); ``n_passed`` is the exit cohort. Stage A entry is
    the full case set; the rest follow the cascade.
    """
    if atomic.empty:
        return pd.DataFrame(
            columns=["stage", "n_total", "n_passed", "n_dropped",
                     "attrition_rate"],
        )

    # Per-case stage outcomes: case is "passed" at stage X if any row
    # for that case has stage_col == X and gate_pass_col == True.
    by_case = atomic.groupby(case_col)
    n_cases_total = by_case.ngroups

    rows: list[dict] = []
    prev_passed: int | None = None
    for stage in ["A", "B", "C"]:
        stage_rows = atomic[atomic[stage_col] == stage]
        if stage_rows.empty:
            n_total = prev_passed if prev_passed is not None else n_cases_total
            rows.append({"stage": stage, "n_total": n_total,
                         "n_passed": 0, "n_dropped": n_total,
                         "attrition_rate": 1.0 if n_total else float("nan")})
            prev_passed = 0
            continue
        passed_per_case = (stage_rows.groupby(case_col)[gate_pass_col]
                                       .agg(lambda s: bool(s.any())))
        n_total = (prev_passed if prev_passed is not None
                   else len(passed_per_case))
        n_passed = int(passed_per_case.sum())
        n_dropped = n_total - n_passed
        attr = (n_dropped / n_total) if n_total > 0 else float("nan")
        rows.append({
            "stage": stage,
            "n_total": int(n_total),
            "n_passed": int(n_passed),
            "n_dropped": int(n_dropped),
            "attrition_rate": float(attr),
        })
        prev_passed = n_passed

    return pd.DataFrame(rows)


def conditional_accuracy_grid(
    atomic: pd.DataFrame,
    *,
    case_col: str = "case_id",
    stage_col: str = "cascade_stage",
    gate_pass_col: str = "gate_pass",
    field_col: str = "field",
    correct_col: str = "correct",
) -> pd.DataFrame:
    """Conditional accuracy of Stage C fields, given upstream gate passage.

    Returns one row per Stage-C field:

        field, n_eligible, accuracy_full, accuracy_given_A,
        accuracy_given_AB, gate_independence_score

    ``accuracy_full`` is the unconditional Stage-C accuracy among all
    cases the cascade scored (i.e. cases that already passed A and B
    by construction). The ``_given_A`` and ``_given_AB`` columns
    re-express the same accuracy for the cohort that entered each
    stage — useful when a downstream analysis re-bases denominators.

    ``gate_independence_score`` is ``accuracy_full - mean_accuracy``
    where ``mean_accuracy`` averages over the entry cohort of Stage A
    (treating dropped cases as wrong). Near zero means upstream
    triage failures don't predict field-level errors; large negative
    means errors compound.
    """
    if atomic.empty:
        return pd.DataFrame(
            columns=["field", "n_eligible", "accuracy_full",
                     "accuracy_given_A", "accuracy_given_AB",
                     "gate_independence_score"],
        )

    # Cases that passed A.
    a_rows = atomic[atomic[stage_col] == "A"]
    cases_passed_a = set(a_rows[a_rows[gate_pass_col] == True][case_col])  # noqa: E712
    # Cases that passed B.
    b_rows = atomic[atomic[stage_col] == "B"]
    cases_passed_b = set(b_rows[b_rows[gate_pass_col] == True][case_col])  # noqa: E712

    # Stage C rows.
    c_rows = atomic[atomic[stage_col] == "C"].copy()
    c_rows["correct_num"] = pd.to_numeric(c_rows[correct_col], errors="coerce")

    n_total_cases = atomic[case_col].nunique()
    rows: list[dict] = []
    for field, sub in c_rows.groupby(field_col):
        valid = sub.dropna(subset=["correct_num"])
        n_eligible = len(valid)
        if n_eligible == 0:
            continue
        acc_full = float(valid["correct_num"].mean())

        # given_A: among cases passed A, treat missing-from-C as wrong.
        n_a = len(cases_passed_a)
        if n_a:
            valid_a = valid[valid[case_col].isin(cases_passed_a)]
            acc_given_a = (
                float(valid_a["correct_num"].sum()) / n_a if n_a else float("nan")
            )
        else:
            acc_given_a = float("nan")

        n_ab = len(cases_passed_b)
        if n_ab:
            valid_ab = valid[valid[case_col].isin(cases_passed_b)]
            acc_given_ab = (
                float(valid_ab["correct_num"].sum()) / n_ab if n_ab else float("nan")
            )
        else:
            acc_given_ab = float("nan")

        # Independence score: accuracy_full vs accuracy on full A cohort.
        # If gate passage is independent of field correctness, these
        # match. If errors compound, the full-cohort accuracy is much
        # lower than the conditional one.
        independence = acc_full - acc_given_a if not np.isnan(acc_given_a) else float("nan")

        rows.append({
            "field": field,
            "n_eligible": n_eligible,
            "accuracy_full": acc_full,
            "accuracy_given_A": acc_given_a,
            "accuracy_given_AB": acc_given_ab,
            "gate_independence_score": independence,
        })

    return pd.DataFrame(rows)


def attrition_propensity(
    atomic: pd.DataFrame,
    case_features: pd.DataFrame,
    *,
    case_col: str = "case_id",
    stage_col: str = "cascade_stage",
    gate_pass_col: str = "gate_pass",
    feature_cols: Sequence[str] = ("organ", "dataset"),
) -> pd.DataFrame:
    """Logistic regression of "case dropped at gate" on case features.

    Helps identify whether attrition is structured (e.g. one organ is
    systematically over-rejected at Stage A) rather than random. For
    each stage, fits a logit model

        dropped ~ feature_cols

    and reports per-feature coefficients with their CIs and p-values.
    Soft-fails if statsmodels is unavailable.
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        return pd.DataFrame(
            columns=["stage", "feature", "coef", "ci_lo", "ci_hi", "p_value",
                     "notes"],
            data=[{"stage": "*", "feature": "*", "coef": float("nan"),
                   "ci_lo": float("nan"), "ci_hi": float("nan"),
                   "p_value": float("nan"),
                   "notes": "statsmodels not available"}],
        )

    out_rows: list[dict] = []
    for stage in ["A", "B"]:
        stage_rows = atomic[atomic[stage_col] == stage]
        if stage_rows.empty:
            continue
        passed_per_case = (stage_rows.groupby(case_col)[gate_pass_col]
                           .agg(lambda s: bool(s.any()))
                           .reset_index()
                           .rename(columns={gate_pass_col: "passed"}))
        passed_per_case["dropped"] = (~passed_per_case["passed"]).astype(int)
        merged = passed_per_case.merge(case_features, on=case_col, how="left")
        merged = merged.dropna(subset=list(feature_cols))
        if merged.empty or merged["dropped"].nunique() < 2:
            out_rows.append({"stage": stage, "feature": "*",
                             "coef": float("nan"), "ci_lo": float("nan"),
                             "ci_hi": float("nan"), "p_value": float("nan"),
                             "notes": "no variation"})
            continue
        formula = "dropped ~ " + " + ".join(f"C({c})" for c in feature_cols)
        try:
            fit = smf.logit(formula, data=merged).fit(disp=False)
        except Exception as e:
            out_rows.append({"stage": stage, "feature": "*",
                             "coef": float("nan"), "ci_lo": float("nan"),
                             "ci_hi": float("nan"), "p_value": float("nan"),
                             "notes": f"fit failed: {e}"})
            continue
        params = fit.params
        conf = fit.conf_int(alpha=0.05)
        for feature in params.index:
            out_rows.append({
                "stage": stage,
                "feature": str(feature),
                "coef": float(params[feature]),
                "ci_lo": float(conf.loc[feature, 0]),
                "ci_hi": float(conf.loc[feature, 1]),
                "p_value": float(fit.pvalues[feature]),
                "notes": "",
            })
    return pd.DataFrame(out_rows)


__all__ = [
    "cascade_funnel",
    "conditional_accuracy_grid",
    "attrition_propensity",
]
