"""Paired hypothesis tests for cross-model comparisons.

Every test returns a :class:`TestResult` so cascade reductions can
concatenate results from heterogeneous tests into one DataFrame.

Tests covered
-------------
* :func:`mcnemar` — paired binary outcomes, two methods (wraps the
  canonical ``ci.mcnemar_test``).
* :func:`paired_bootstrap_delta` — paired-bootstrap CI on the accuracy
  or F1 delta between two methods (wraps ``ci.paired_bootstrap_diff``).
* :func:`cochran_q` — paired binary outcomes across k >= 3 methods.
  Used to reject "all three models perform identically" before doing
  pairwise McNemars.
* :func:`stuart_maxwell` — multi-class paired marginal-homogeneity
  test. The right tool when comparing two organ-classification models
  at Stage B (11 classes).
* :func:`bhapkar` — Bhapkar's modification of Stuart-Maxwell, more
  powerful when df > 1.
* :func:`friedman` — non-parametric repeated-measures ANOVA over k >= 3
  paired continuous outcomes (e.g. per-case F1 across three models).
* :func:`nemenyi_posthoc` — pairwise post-hoc after Friedman.

References
----------
- McNemar (1947), Stuart (1955), Bhapkar (1966), Cochran (1950),
  Friedman (1937), Nemenyi (1963), Demšar (2006) for the recommendation
  of Friedman + Nemenyi as the standard non-parametric machine-learning
  comparison protocol.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy import stats as sstats

from ..ci import mcnemar_test as _mcnemar_test, paired_bootstrap_diff
from .result import TestResult


# --- McNemar ---------------------------------------------------------------

def mcnemar(
    a: Sequence[int | bool],
    b: Sequence[int | bool],
    *,
    name: str = "mcnemar",
) -> TestResult:
    """McNemar's test on two paired binary outcome vectors.

    ``a[i]`` and ``b[i]`` are 0/1 (or bool) correctness for case ``i``
    under method A and B respectively. NaN-equivalent values (None,
    nan) are dropped pairwise.
    """
    pairs = [(int(x), int(y)) for x, y in zip(a, b, strict=True)
             if x is not None and y is not None
             and not (isinstance(x, float) and math.isnan(x))
             and not (isinstance(y, float) and math.isnan(y))]
    n = len(pairs)
    if n == 0:
        return TestResult(name=name, n=0, notes="empty input")
    b_count = sum(1 for x, y in pairs if x == 1 and y == 0)
    c_count = sum(1 for x, y in pairs if x == 0 and y == 1)
    res = _mcnemar_test(b_count, c_count)
    delta = float(np.mean([x - y for x, y in pairs]))
    return TestResult(
        name=name,
        statistic=float(res["statistic"]),
        df=1 if res["method"].startswith("chi2") else None,
        p_raw=float(res["p_value"]),
        effect_size=delta,
        effect_kind="accuracy_delta",
        n=n,
        notes=res["method"],
        extra={"b": b_count, "c": c_count},
    )


# --- Paired bootstrap delta ------------------------------------------------

def paired_bootstrap_delta(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    random_state: int | None = 0,
    name: str = "paired_bootstrap_delta",
) -> TestResult:
    """Paired-bootstrap CI on ``mean(a) - mean(b)`` for the same cases."""
    res = paired_bootstrap_diff(
        a, b, n_boot=n_boot, alpha=alpha, random_state=random_state,
    )
    n = sum(1 for x, y in zip(a, b, strict=True)
            if not (math.isnan(float(x)) or math.isnan(float(y))))
    return TestResult(
        name=name,
        effect_size=float(res.point),
        effect_kind="accuracy_delta",
        effect_ci_lo=float(res.lo),
        effect_ci_hi=float(res.hi),
        n=n,
        notes=res.method,
    )


# --- Cochran's Q -----------------------------------------------------------

def cochran_q(
    matrix: np.ndarray | Sequence[Sequence[int]],
    *,
    method_labels: Sequence[str] | None = None,
    name: str = "cochran_q",
) -> TestResult:
    """Cochran's Q for k >= 3 paired binary methods on the same cases.

    ``matrix`` shape: (n_cases, k_methods), values 0/1.

    Q = k(k-1) * (sum_j (T_j - mean_T)^2) / (k * sum_i row_i - sum_i row_i^2)
    where T_j is the column sum.

    Distribution: chi-square with df = k - 1 under H0 of equal column
    proportions.
    """
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2:
        return TestResult(name=name, notes="matrix must be 2-D")
    n_cases, k = arr.shape
    if k < 3:
        return TestResult(name=name, n=n_cases,
                          notes="cochran_q requires k >= 3")
    # Drop rows with any NaN (cases not scored under all methods).
    keep = ~np.any(np.isnan(arr), axis=1)
    arr = arr[keep]
    n_cases = arr.shape[0]
    if n_cases == 0:
        return TestResult(name=name, n=0,
                          notes="no fully-paired cases after NaN drop")
    col_sums = arr.sum(axis=0)
    row_sums = arr.sum(axis=1)
    sum_T = col_sums.sum()
    mean_T = col_sums.mean()
    numer = k * (k - 1) * np.sum((col_sums - mean_T) ** 2)
    denom = k * sum_T - np.sum(row_sums ** 2)
    if denom == 0:
        return TestResult(name=name, n=n_cases,
                          notes="degenerate (all 0s or all 1s across methods)")
    q = float(numer / denom)
    df = k - 1
    p = float(sstats.chi2.sf(q, df=df))
    extra = {"k_methods": k, "col_sums": col_sums.tolist()}
    if method_labels:
        extra["method_labels"] = list(method_labels)
    return TestResult(
        name=name, statistic=q, df=df, p_raw=p, n=n_cases, extra=extra,
    )


# --- Stuart-Maxwell --------------------------------------------------------

def _build_paired_table(
    a: Sequence,
    b: Sequence,
    labels: Sequence,
) -> np.ndarray:
    """k×k paired contingency table over the labels."""
    label_idx = {v: i for i, v in enumerate(labels)}
    k = len(labels)
    table = np.zeros((k, k), dtype=int)
    for x, y in zip(a, b, strict=True):
        if x in label_idx and y in label_idx:
            table[label_idx[x], label_idx[y]] += 1
    return table


def stuart_maxwell(
    a: Sequence,
    b: Sequence,
    *,
    labels: Sequence | None = None,
    name: str = "stuart_maxwell",
) -> TestResult:
    """Stuart-Maxwell test of marginal homogeneity for k×k paired tables.

    Tests whether the row marginals equal the column marginals — i.e.
    whether method A and method B disagree systematically about class
    frequencies. Use for organ classification (Stage B): McNemar is
    binary-only and not valid here.

    Computes ``D' V^{-1} D`` where D is the marginal-difference vector
    of length (k-1) and V is its asymptotic covariance. Distribution:
    chi-square with df = k - 1.
    """
    if labels is None:
        labels = sorted({*a, *b}, key=lambda v: ("" if v is None else str(v)))
    table = _build_paired_table(a, b, labels)
    k = table.shape[0]
    n = int(table.sum())
    if n == 0:
        return TestResult(name=name, n=0, notes="empty table")
    if k < 2:
        return TestResult(name=name, n=n, notes="need at least 2 classes")

    row_sums = table.sum(axis=1)
    col_sums = table.sum(axis=0)
    d_full = row_sums - col_sums  # length k

    # Use the first k-1 dimensions; the k-th is determined by the constraint
    # sum d = 0.
    d = d_full[:-1].astype(float)

    # Covariance matrix V (Maxwell 1970 form):
    #   V_ii = row_i + col_i - 2 * table_ii
    #   V_ij = -(table_ij + table_ji)  for i != j
    v = np.zeros((k - 1, k - 1), dtype=float)
    for i in range(k - 1):
        v[i, i] = row_sums[i] + col_sums[i] - 2 * table[i, i]
        for j in range(k - 1):
            if i != j:
                v[i, j] = -(table[i, j] + table[j, i])

    try:
        v_inv = np.linalg.inv(v)
    except np.linalg.LinAlgError:
        return TestResult(name=name, n=n, df=k - 1,
                          notes="covariance matrix singular")

    stat = float(d @ v_inv @ d)
    df = k - 1
    p = float(sstats.chi2.sf(stat, df=df))
    return TestResult(
        name=name, statistic=stat, df=df, p_raw=p, n=n,
        extra={"row_marginals": row_sums.tolist(),
               "col_marginals": col_sums.tolist(),
               "labels": list(labels)},
    )


def bhapkar(
    a: Sequence,
    b: Sequence,
    *,
    labels: Sequence | None = None,
    name: str = "bhapkar",
) -> TestResult:
    """Bhapkar (1966) modification of Stuart-Maxwell.

    Replaces the asymptotic covariance with a sample-based estimate
    that is more powerful for df > 1. Returns a chi-square with the
    same df as Stuart-Maxwell.
    """
    if labels is None:
        labels = sorted({*a, *b}, key=lambda v: ("" if v is None else str(v)))
    table = _build_paired_table(a, b, labels)
    k = table.shape[0]
    n = int(table.sum())
    if n == 0:
        return TestResult(name=name, n=0, notes="empty table")
    if k < 2:
        return TestResult(name=name, n=n, notes="need at least 2 classes")

    p_table = table.astype(float) / n
    row_marg = p_table.sum(axis=1)
    col_marg = p_table.sum(axis=0)
    d = (row_marg - col_marg)[:-1]

    # Bhapkar's V uses the empirical multinomial covariance with
    # diagonal terms (p_ii. + p_.ii) - (p_i. - p_.i)^2  / corrected.
    v = np.zeros((k - 1, k - 1), dtype=float)
    for i in range(k - 1):
        v[i, i] = (row_marg[i] + col_marg[i] - 2 * p_table[i, i]
                   - (row_marg[i] - col_marg[i]) ** 2)
        for j in range(k - 1):
            if i != j:
                v[i, j] = (-(p_table[i, j] + p_table[j, i])
                           - (row_marg[i] - col_marg[i])
                           * (row_marg[j] - col_marg[j]))

    try:
        v_inv = np.linalg.inv(v)
    except np.linalg.LinAlgError:
        return TestResult(name=name, n=n, df=k - 1,
                          notes="covariance matrix singular")

    stat = float(n * d @ v_inv @ d)
    df = k - 1
    p = float(sstats.chi2.sf(stat, df=df))
    return TestResult(
        name=name, statistic=stat, df=df, p_raw=p, n=n,
        extra={"labels": list(labels)},
    )


# --- Friedman + Nemenyi ----------------------------------------------------

def friedman(
    matrix: np.ndarray | Sequence[Sequence[float]],
    *,
    method_labels: Sequence[str] | None = None,
    name: str = "friedman",
) -> TestResult:
    """Friedman's non-parametric repeated-measures ANOVA.

    ``matrix`` shape: (n_cases, k_methods). Each row is one case;
    columns are method scores (e.g. per-case F1).

    Wraps :func:`scipy.stats.friedmanchisquare`. Distribution:
    chi-square with df = k - 1 under H0 of equal column distributions.
    """
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2:
        return TestResult(name=name, notes="matrix must be 2-D")
    keep = ~np.any(np.isnan(arr), axis=1)
    arr = arr[keep]
    n_cases, k = arr.shape
    if k < 3:
        return TestResult(name=name, n=n_cases,
                          notes="friedman requires k >= 3")
    if n_cases < 2:
        return TestResult(name=name, n=n_cases,
                          notes="need at least 2 cases")
    try:
        stat, p = sstats.friedmanchisquare(*[arr[:, j] for j in range(k)])
    except ValueError as e:
        return TestResult(name=name, n=n_cases, notes=str(e))
    extra = {"k_methods": k}
    if method_labels:
        extra["method_labels"] = list(method_labels)
    # Mean rank per method — useful for narrative ("model A had the highest
    # mean rank of N.NN").
    ranks = np.apply_along_axis(sstats.rankdata, 1, arr)
    extra["mean_ranks"] = ranks.mean(axis=0).tolist()
    return TestResult(
        name=name, statistic=float(stat), df=k - 1,
        p_raw=float(p), n=n_cases, extra=extra,
    )


def nemenyi_posthoc(
    matrix: np.ndarray | Sequence[Sequence[float]],
    *,
    alpha: float = 0.05,
    method_labels: Sequence[str] | None = None,
) -> list[TestResult]:
    """Pairwise post-hoc test after a significant Friedman.

    Computes critical-difference (CD) for each pair using the
    studentized-range distribution. Returns one TestResult per (i, j)
    method pair with i < j. Significance flag lives in
    ``extra["significant"]``; the p-value is approximated by inverting
    the studentized-range CDF.

    See Demšar (2006) §3.2.4 for the protocol. We use the CD-from-rank
    formulation:

        CD = q_alpha * sqrt(k(k+1) / (6 * n))

    where q_alpha is the critical value from the studentized-range
    distribution at k methods.
    """
    arr = np.asarray(matrix, dtype=float)
    keep = ~np.any(np.isnan(arr), axis=1)
    arr = arr[keep]
    n_cases, k = arr.shape
    if k < 3 or n_cases < 2:
        return []
    ranks = np.apply_along_axis(sstats.rankdata, 1, arr)
    mean_ranks = ranks.mean(axis=0)
    q = float(sstats.studentized_range.ppf(1 - alpha, k, np.inf)) / math.sqrt(2)
    cd = q * math.sqrt(k * (k + 1) / (6 * n_cases))
    out: list[TestResult] = []
    labels = list(method_labels) if method_labels else [str(i) for i in range(k)]
    for i in range(k):
        for j in range(i + 1, k):
            diff = float(abs(mean_ranks[i] - mean_ranks[j]))
            # Approximate p via the studentized-range distribution.
            stat_q = diff / math.sqrt(k * (k + 1) / (6 * n_cases))
            p = float(1.0 - sstats.studentized_range.cdf(
                stat_q * math.sqrt(2), k, np.inf,
            ))
            out.append(TestResult(
                name=f"nemenyi_{labels[i]}_vs_{labels[j]}",
                statistic=stat_q,
                df=None,
                p_raw=p,
                effect_size=mean_ranks[i] - mean_ranks[j],
                effect_kind="mean_rank_delta",
                n=n_cases,
                extra={"a": labels[i], "b": labels[j], "cd": cd,
                       "significant": diff > cd},
            ))
    return out


__all__ = [
    "mcnemar",
    "paired_bootstrap_delta",
    "cochran_q",
    "stuart_maxwell",
    "bhapkar",
    "friedman",
    "nemenyi_posthoc",
]
