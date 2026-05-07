"""Cross-cohort heterogeneity statistics for the cascade.

Reviewer 1(d) "robustness" point applied to cross-organ generalisability:
quantify how much per-organ effect estimates differ from the pooled
estimate.

Methods
-------
* :func:`q_test_heterogeneity` — Cochran's Q over k organ-specific
  effect estimates with their standard errors. Tests H0 of common
  effect.
* :func:`i_squared` — Higgins-Thompson I^2 = max(0, 100*(Q - df) / Q).
  The fraction of total variation across organs that is due to
  between-organ heterogeneity rather than within-organ sampling error.
* :func:`forest_plot_csv` — Builds a long-form DataFrame with one row
  per (organ, effect) ready for forest-plot rendering. CI columns
  follow ``ci_lo``, ``ci_hi``.
* :func:`interaction_test` — Logistic GLMM-style organ × method
  interaction test. Detects fields where one model is uniquely strong
  on a specific organ.

References
----------
- Cochran (1954) "The combination of estimates from different
  experiments," Biometrics.
- Higgins & Thompson (2002) "Quantifying heterogeneity in a meta-
  analysis," Stat. Med.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats as sstats

from .result import TestResult


def q_test_heterogeneity(
    estimates: Sequence[float],
    standard_errors: Sequence[float],
) -> TestResult:
    """Cochran's Q test for between-study (here, between-organ) heterogeneity.

    Returns a :class:`TestResult` with statistic = Q, df = k - 1,
    p_raw = chi-square p, effect_size = I^2 (as a fraction in [0, 1]).
    """
    est = np.asarray(estimates, dtype=float)
    se = np.asarray(standard_errors, dtype=float)
    mask = ~(np.isnan(est) | np.isnan(se) | (se <= 0))
    est, se = est[mask], se[mask]
    k = est.size
    if k < 2:
        return TestResult(name="q_test_heterogeneity", n=k,
                          notes="need >= 2 estimates")
    weights = 1.0 / (se * se)
    pooled = float((weights * est).sum() / weights.sum())
    Q = float((weights * (est - pooled) ** 2).sum())
    df = k - 1
    p = float(sstats.chi2.sf(Q, df=df))
    i_sq = max(0.0, (Q - df) / Q) if Q > 0 else 0.0
    return TestResult(
        name="q_test_heterogeneity",
        statistic=Q, df=df, p_raw=p,
        effect_size=i_sq, effect_kind="i_squared",
        n=k,
        extra={"pooled": pooled, "k": k},
    )


def i_squared(
    estimates: Sequence[float],
    standard_errors: Sequence[float],
) -> float:
    """I^2 alone, as a fraction in [0, 1]. See :func:`q_test_heterogeneity`."""
    res = q_test_heterogeneity(estimates, standard_errors)
    return float(res.effect_size)


def forest_plot_csv(
    rows: Sequence[dict],
    *,
    organ_key: str = "organ",
    effect_key: str = "effect_size",
    lo_key: str = "ci_lo",
    hi_key: str = "ci_hi",
    n_key: str = "n",
) -> pd.DataFrame:
    """Build a long-form forest-plot CSV from a list of per-organ rows.

    Each row should expose at least ``organ``, ``effect_size``,
    ``ci_lo``, ``ci_hi``, ``n``. Output columns:

        organ, effect_size, ci_lo, ci_hi, n, weight, pooled

    where ``weight`` is the precision (1/SE^2) and ``pooled`` is the
    inverse-variance-weighted average across all organs.
    """
    df = pd.DataFrame(list(rows))
    if df.empty:
        return df
    # Approximate SE from CI half-width assuming normal-symmetric CI.
    half = (df[hi_key] - df[lo_key]) / 2.0
    # Use 1.96 for a 95% CI; callers using a non-95% CI should
    # post-process the weight column.
    se_approx = half / 1.96
    df["weight"] = np.where(se_approx > 0, 1.0 / (se_approx * se_approx), 0.0)
    if df["weight"].sum() > 0:
        df["pooled"] = float(
            (df["weight"] * df[effect_key]).sum() / df["weight"].sum()
        )
    else:
        df["pooled"] = float("nan")
    return df[[organ_key, effect_key, lo_key, hi_key, n_key,
               "weight", "pooled"]]


def interaction_test(
    long_df: pd.DataFrame,
    *,
    outcome_col: str = "correct",
    method_col: str = "method",
    organ_col: str = "organ",
    case_col: str = "case_id",
) -> TestResult:
    """Logistic-regression organ × method interaction test.

    Tests H0 that the per-method effect is constant across organs.
    Uses ``statsmodels.formula.api.logit`` with formula::

        outcome ~ C(method) * C(organ)

    and reports the LR statistic against the no-interaction model.

    Soft-fails if statsmodels is unavailable, the design is singular,
    or any cell has zero variation.
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        return TestResult(name="interaction_test",
                          notes="statsmodels not available")

    df = long_df[[outcome_col, method_col, organ_col, case_col]].dropna()
    if df.empty or df[method_col].nunique() < 2 or df[organ_col].nunique() < 2:
        return TestResult(name="interaction_test", n=len(df),
                          notes="degenerate (need >= 2 methods × >= 2 organs)")
    df = df.copy()
    df[outcome_col] = df[outcome_col].astype(float)
    if df[outcome_col].nunique() < 2:
        return TestResult(name="interaction_test", n=len(df),
                          notes="outcome has no variation")
    try:
        fit_main = smf.logit(
            f"{outcome_col} ~ C({method_col}) + C({organ_col})", data=df,
        ).fit(disp=False)
        fit_full = smf.logit(
            f"{outcome_col} ~ C({method_col}) * C({organ_col})", data=df,
        ).fit(disp=False)
    except Exception as e:
        return TestResult(name="interaction_test", n=len(df),
                          notes=f"fit failed: {e}")

    lr = 2 * (fit_full.llf - fit_main.llf)
    df_diff = fit_full.df_model - fit_main.df_model
    if df_diff <= 0 or lr < 0:
        return TestResult(name="interaction_test", n=len(df),
                          notes="degenerate LR (df<=0 or stat<0)")
    p = float(sstats.chi2.sf(lr, df=int(df_diff)))
    return TestResult(
        name="interaction_test",
        statistic=float(lr), df=int(df_diff), p_raw=p, n=len(df),
        notes="LR test, full vs no-interaction",
    )


__all__ = [
    "q_test_heterogeneity",
    "i_squared",
    "forest_plot_csv",
    "interaction_test",
]
