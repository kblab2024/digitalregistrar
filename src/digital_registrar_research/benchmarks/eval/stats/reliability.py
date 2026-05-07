"""Multi-run reliability statistics.

Reviewer 1(d) asks for "robustness analyses across multiple runs." This
module turns the per-case correctness matrix (cases × runs) into
quantitative reliability claims.

Methods
-------
* :func:`icc_2_1` — Two-way mixed-effects ICC for absolute agreement,
  single-run reliability. The clinical reproducibility metric.
* :func:`icc_3_k` — Two-way mixed-effects ICC for the average of k runs.
  Answers "how reliable is the run-averaged headline accuracy I'm
  reporting in the manuscript?"
* :func:`cronbach_alpha` — Internal consistency across k runs.
* :func:`spearman_brown` — Predicted reliability of an n-run average.
* :func:`per_case_run_sd` — Per-case standard deviation across runs.
* :func:`accuracy_flip_rate` — Fraction of cases that change correct↔
  wrong across runs. Complements SD because SD on a 0/1 sequence is
  opaque.
* :func:`missing_flip_rate` — Fraction of cases where the field is
  attempted in some runs but missing in others.

References
----------
- Shrout & Fleiss (1979) "Intraclass correlations: uses in assessing
  rater reliability," Psychol. Bull.
- McGraw & Wong (1996) "Forming inferences about some intraclass
  correlation coefficients," Psychol. Methods.
- Cronbach (1951) "Coefficient alpha and the internal structure of
  tests," Psychometrika.
- Spearman (1910) / Brown (1910) — Spearman-Brown prophecy formula.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy import stats as sstats


# --- ICC ------------------------------------------------------------------

def _icc_components(matrix: np.ndarray) -> tuple[float, float, float, int, int]:
    """Compute the mean-square components for an ICC.

    Returns (MSR, MSE, MSC, n_subjects, k_raters).
        MSR — between-subjects mean square
        MSE — error mean square
        MSC — between-raters (between-runs) mean square
    """
    m = np.asarray(matrix, dtype=float)
    # Drop subjects with any NaN — ICC is balanced-design.
    keep = ~np.any(np.isnan(m), axis=1)
    m = m[keep]
    n, k = m.shape
    if n < 2 or k < 2:
        return (float("nan"),) * 3 + (n, k)
    grand_mean = m.mean()
    subj_mean = m.mean(axis=1)
    rat_mean = m.mean(axis=0)
    SS_T = float(((m - grand_mean) ** 2).sum())
    SS_R = float(k * ((subj_mean - grand_mean) ** 2).sum())
    SS_C = float(n * ((rat_mean - grand_mean) ** 2).sum())
    SS_E = SS_T - SS_R - SS_C
    df_R = n - 1
    df_C = k - 1
    df_E = (n - 1) * (k - 1)
    MSR = SS_R / df_R if df_R > 0 else float("nan")
    MSC = SS_C / df_C if df_C > 0 else float("nan")
    MSE = SS_E / df_E if df_E > 0 else float("nan")
    return MSR, MSE, MSC, n, k


def icc_2_1(matrix: np.ndarray | Sequence[Sequence[float]]) -> dict:
    """ICC(2,1) — two-way mixed effects, absolute agreement, single rater.

    Shrout-Fleiss 1979, McGraw-Wong 1996. Used for "if I run the model
    one more time, how much does the per-case correctness vary?"

    matrix shape: (n_subjects, k_runs).
    """
    MSR, MSE, MSC, n, k = _icc_components(np.asarray(matrix, dtype=float))
    if any(math.isnan(v) for v in (MSR, MSE, MSC)):
        return {"icc": float("nan"), "n": n, "k": k,
                "ci_lo": float("nan"), "ci_hi": float("nan"),
                "method": "ICC(2,1)"}
    denom = MSR + (k - 1) * MSE + k * (MSC - MSE) / n
    if denom == 0:
        return {"icc": float("nan"), "n": n, "k": k,
                "ci_lo": float("nan"), "ci_hi": float("nan"),
                "method": "ICC(2,1)"}
    icc = (MSR - MSE) / denom

    # 95% CI per McGraw-Wong eq. 7 for ICC(2,1).
    f = MSR / MSE if MSE > 0 else float("inf")
    df1, df2 = n - 1, (n - 1) * (k - 1)
    try:
        f_lo = f / float(sstats.f.ppf(0.975, df1, df2))
        f_hi = f * float(sstats.f.ppf(0.975, df2, df1))
        ci_lo = (f_lo - 1) / (f_lo + (k - 1)
                              + k * (MSC / MSE - 1) / n)
        ci_hi = (f_hi - 1) / (f_hi + (k - 1)
                              + k * (MSC / MSE - 1) / n)
    except (ValueError, ZeroDivisionError):
        ci_lo = ci_hi = float("nan")

    return {"icc": float(icc), "n": n, "k": k,
            "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
            "method": "ICC(2,1)"}


def icc_3_k(matrix: np.ndarray | Sequence[Sequence[float]]) -> dict:
    """ICC(3,k) — two-way mixed effects, consistency, k-run average.

    Reliability of the *averaged* k-run prediction. Answers "the
    headline accuracy I'm reporting is averaged over k runs — how
    reproducible is *that average*?" Higher than ICC(2,1) by the
    Spearman-Brown formula when reliability is positive.
    """
    MSR, MSE, _MSC, n, k = _icc_components(np.asarray(matrix, dtype=float))
    if math.isnan(MSR) or math.isnan(MSE):
        return {"icc": float("nan"), "n": n, "k": k,
                "ci_lo": float("nan"), "ci_hi": float("nan"),
                "method": "ICC(3,k)"}
    if MSR == 0:
        return {"icc": float("nan"), "n": n, "k": k,
                "ci_lo": float("nan"), "ci_hi": float("nan"),
                "method": "ICC(3,k)"}
    icc = (MSR - MSE) / MSR
    f = MSR / MSE if MSE > 0 else float("inf")
    df1, df2 = n - 1, (n - 1) * (k - 1)
    try:
        f_lo = f / float(sstats.f.ppf(0.975, df1, df2))
        f_hi = f * float(sstats.f.ppf(0.975, df2, df1))
        ci_lo = 1 - 1 / f_lo
        ci_hi = 1 - 1 / f_hi
    except (ValueError, ZeroDivisionError):
        ci_lo = ci_hi = float("nan")
    return {"icc": float(icc), "n": n, "k": k,
            "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
            "method": "ICC(3,k)"}


# --- Cronbach's alpha -----------------------------------------------------

def cronbach_alpha(matrix: np.ndarray | Sequence[Sequence[float]]) -> float:
    """Cronbach's alpha for k items (here, k runs) over n subjects.

    matrix shape: (n_subjects, k_runs). Returns the standard formula:

        alpha = k/(k-1) * (1 - sum(item_var) / total_var)

    where item_var is the per-run variance and total_var is the
    variance of per-case sums. Interpretation: > 0.7 acceptable, > 0.9
    excellent for clinical instruments.
    """
    m = np.asarray(matrix, dtype=float)
    keep = ~np.any(np.isnan(m), axis=1)
    m = m[keep]
    n, k = m.shape
    if n < 2 or k < 2:
        return float("nan")
    item_vars = m.var(axis=0, ddof=1)
    total_var = m.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return float("nan")
    return float(k / (k - 1) * (1 - item_vars.sum() / total_var))


# --- Spearman-Brown -------------------------------------------------------

def spearman_brown(rho_single: float, n_runs: int) -> float:
    """Predicted reliability of an ``n_runs``-run average.

    rho' = (n * rho) / (1 + (n - 1) * rho)

    Useful for "how many additional runs would I need to push ICC(3,k)
    above 0.95?" — solve for ``n``.
    """
    if rho_single <= -1 or rho_single >= 1 or n_runs <= 0:
        return float("nan")
    return float((n_runs * rho_single) / (1 + (n_runs - 1) * rho_single))


# --- Flip-rate metrics ----------------------------------------------------

def accuracy_flip_rate(matrix: np.ndarray | Sequence[Sequence[float]]) -> float:
    """Fraction of cases whose correctness flips across runs.

    Complementary to SD: an SD of 0.5 on a 0/1 vector is uninformative,
    but a flip rate of 0.4 says "40% of cases are unstable" — clearer
    for the rebuttal.
    """
    m = np.asarray(matrix, dtype=float)
    keep = ~np.any(np.isnan(m), axis=1)
    m = m[keep]
    n_cases = m.shape[0]
    if n_cases == 0 or m.shape[1] < 2:
        return float("nan")
    flips = (m.min(axis=1) != m.max(axis=1))
    return float(flips.mean())


def missing_flip_rate(
    attempted_matrix: np.ndarray | Sequence[Sequence[bool]],
) -> float:
    """Fraction of cases where attempted=True in some runs and False in others.

    Captures runtime-instability of the field-extraction pipeline
    independent of correctness.
    """
    m = np.asarray(attempted_matrix, dtype=float)
    keep = ~np.any(np.isnan(m), axis=1)
    m = m[keep]
    n_cases, n_runs = m.shape
    if n_cases == 0 or n_runs < 2:
        return float("nan")
    flips = ((m == 0).any(axis=1) & (m == 1).any(axis=1))
    return float(flips.mean())


def per_case_run_sd(
    matrix: np.ndarray | Sequence[Sequence[float]],
) -> dict:
    """Per-case standard deviation across runs; summary statistics.

    Returns mean, max, and 90th-percentile of the per-case SD vector.
    """
    m = np.asarray(matrix, dtype=float)
    keep = ~np.any(np.isnan(m), axis=1)
    m = m[keep]
    n_cases, n_runs = m.shape
    if n_cases == 0 or n_runs < 2:
        return {"mean": float("nan"), "max": float("nan"), "p90": float("nan"),
                "n_cases": n_cases}
    sd = m.std(axis=1, ddof=1)
    return {
        "mean": float(sd.mean()),
        "max": float(sd.max()),
        "p90": float(np.percentile(sd, 90)),
        "n_cases": n_cases,
    }


__all__ = [
    "icc_2_1",
    "icc_3_k",
    "cronbach_alpha",
    "spearman_brown",
    "accuracy_flip_rate",
    "missing_flip_rate",
    "per_case_run_sd",
]
