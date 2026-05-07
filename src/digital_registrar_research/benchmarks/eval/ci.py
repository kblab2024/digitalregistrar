"""
Confidence-interval and hypothesis-test utilities shared by the IAA
(Part A) and multi-run accuracy (Part C) pipelines.

All functions are pure and deterministic given a seed; nothing here
depends on the pathology schema or on the existing scoring code.

Contents
--------
    wilson_ci(k, n, alpha)                — binary proportion Wilson CI
    clopper_pearson_ci(k, n, alpha)       — exact binomial CI
    t_ci(values, alpha)                   — Student-t CI for a small sample
    bootstrap_ci(values, stat, ...)       — BCa bootstrap CI with optional strata
    paired_bootstrap_diff(a, b, ...)      — paired Δ between two correctness vectors
    mcnemar_test(b, c)                    — McNemar test with continuity correction
    fisher_z_ci_for_corr(r, n, alpha)     — Fisher-z CI for a correlation
    two_source_bootstrap_ci(matrix, ...)  — nested case × run bootstrap for accuracy

These are intentionally light dependencies — only numpy and scipy.stats.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
from scipy import stats as sstats

__all__ = [
    "wilson_ci",
    "clopper_pearson_ci",
    "t_ci",
    "bootstrap_ci",
    "paired_bootstrap_diff",
    "mcnemar_test",
    "fisher_z_ci_for_corr",
    "two_source_bootstrap_ci",
    "BootstrapResult",
]


# --- Simple parametric CIs ---------------------------------------------------

def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a binary proportion. Robust at small n."""
    if n <= 0:
        return (float("nan"), float("nan"))
    z = sstats.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact binomial CI (Clopper-Pearson). Use when n is small and coverage
    guarantees matter more than width."""
    if n <= 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else sstats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else sstats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (float(lo), float(hi))


def t_ci(values: Sequence[float], alpha: float = 0.05) -> tuple[float, float, float]:
    """Student-t CI for the mean of a small sample. Returns (mean, lo, hi).

    Designed for the K-run accuracy vector in Part C.2 — "if I re-ran the
    model how much would the accuracy swing" style questions.
    """
    arr = np.asarray([v for v in values if v is not None and not math.isnan(v)],
                     dtype=float)
    if arr.size == 0:
        return (float("nan"),) * 3
    if arr.size == 1:
        return (float(arr[0]), float(arr[0]), float(arr[0]))
    m = float(arr.mean())
    se = float(arr.std(ddof=1) / math.sqrt(arr.size))
    half = float(sstats.t.ppf(1 - alpha / 2, arr.size - 1)) * se
    return (m, m - half, m + half)


# --- BCa bootstrap -----------------------------------------------------------

class BootstrapResult:
    """Light container to keep return values self-documenting."""
    __slots__ = ("point", "lo", "hi", "boot_dist", "method")

    def __init__(self, point: float, lo: float, hi: float,
                 boot_dist: np.ndarray, method: str):
        self.point = point
        self.lo = lo
        self.hi = hi
        self.boot_dist = boot_dist
        self.method = method

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.point, self.lo, self.hi)


def _quantile(a: np.ndarray, q: float) -> float:
    return float(np.quantile(a, q, method="linear")) if a.size else float("nan")


def bootstrap_ci(
    values: Sequence,
    statistic: Callable[[Sequence], float],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    method: str = "bca",
    strata: Sequence | None = None,
    random_state: int | None = 0,
) -> BootstrapResult:
    """Bootstrap CI for a statistic over a sequence of items.

    The item unit is whatever `statistic` expects — typically a list of
    per-case records (dicts, tuples, etc.). For IAA with case-stratified
    resampling (Part A, "stratified by organ"), pass the organ label per
    item via `strata`.

    method: "bca" (accelerated & bias-corrected) or "percentile".
    """
    rng = np.random.default_rng(random_state)
    items = list(values)
    n = len(items)
    if n == 0:
        return BootstrapResult(float("nan"), float("nan"), float("nan"),
                               np.array([]), method)

    theta_hat = float(statistic(items))

    # --- Stratified or simple resampling ---
    if strata is not None:
        strata_arr = np.asarray(list(strata))
        if len(strata_arr) != n:
            raise ValueError("strata length must match values length")
        unique, inverse = np.unique(strata_arr, return_inverse=True)
        indices_by_stratum = [np.where(inverse == s)[0] for s in range(len(unique))]

        def _resample():
            picks = np.concatenate([
                rng.choice(idx, size=len(idx), replace=True)
                for idx in indices_by_stratum
            ])
            return [items[i] for i in picks]
    else:
        def _resample():
            picks = rng.integers(0, n, size=n)
            return [items[i] for i in picks]

    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        try:
            boot[b] = float(statistic(_resample()))
        except Exception:
            boot[b] = np.nan
    boot = boot[~np.isnan(boot)]
    if boot.size < 2:
        return BootstrapResult(theta_hat, theta_hat, theta_hat, boot, method)

    if method == "percentile":
        lo = _quantile(boot, alpha / 2)
        hi = _quantile(boot, 1 - alpha / 2)
        return BootstrapResult(theta_hat, lo, hi, boot, "percentile")

    # --- BCa: bias correction + acceleration ---
    frac_below = float(np.mean(boot < theta_hat))
    # Clamp to avoid ±inf in the inverse-normal
    frac_below = min(max(frac_below, 1.0 / (boot.size + 1)), 1.0 - 1.0 / (boot.size + 1))
    z0 = float(sstats.norm.ppf(frac_below))

    # Jackknife acceleration
    jack = np.empty(n, dtype=float)
    for i in range(n):
        sub = items[:i] + items[i + 1:]
        try:
            jack[i] = float(statistic(sub))
        except Exception:
            jack[i] = np.nan
    jack_mean = float(np.nanmean(jack))
    diff = jack_mean - jack
    num = float(np.nansum(diff ** 3))
    den = 6.0 * (float(np.nansum(diff ** 2)) ** 1.5)
    a_hat = num / den if den > 0 else 0.0

    z_lo = sstats.norm.ppf(alpha / 2)
    z_hi = sstats.norm.ppf(1 - alpha / 2)
    denom_lo = 1 - a_hat * (z0 + z_lo)
    denom_hi = 1 - a_hat * (z0 + z_hi)
    if denom_lo <= 0 or denom_hi <= 0:
        # Acceleration blew up; degrade to percentile.
        lo = _quantile(boot, alpha / 2)
        hi = _quantile(boot, 1 - alpha / 2)
        return BootstrapResult(theta_hat, lo, hi, boot, "percentile_fallback")

    alpha_lo = float(sstats.norm.cdf(z0 + (z0 + z_lo) / denom_lo))
    alpha_hi = float(sstats.norm.cdf(z0 + (z0 + z_hi) / denom_hi))
    lo = _quantile(boot, alpha_lo)
    hi = _quantile(boot, alpha_hi)
    return BootstrapResult(theta_hat, lo, hi, boot, "bca")


# --- Paired bootstrap for Δ between correctness vectors ----------------------

def paired_bootstrap_diff(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    random_state: int | None = 0,
) -> BootstrapResult:
    """Bootstrap CI on mean(a) - mean(b) with paired case-level resampling.

    Used for: ensemble-vs-single-run, gpt-oss-vs-baseline comparisons where
    the two methods are evaluated on the SAME cases.
    """
    rng = np.random.default_rng(random_state)
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    if arr_a.shape != arr_b.shape:
        raise ValueError("a and b must have identical shape for paired bootstrap")
    mask = ~(np.isnan(arr_a) | np.isnan(arr_b))
    arr_a = arr_a[mask]
    arr_b = arr_b[mask]
    n = arr_a.size
    if n == 0:
        return BootstrapResult(float("nan"), float("nan"), float("nan"),
                               np.array([]), "bca")
    delta_hat = float(arr_a.mean() - arr_b.mean())
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = float(arr_a[idx].mean() - arr_b[idx].mean())
    # Percentile is fine here; paired Δ is near-symmetric in practice.
    lo = _quantile(boot, alpha / 2)
    hi = _quantile(boot, 1 - alpha / 2)
    return BootstrapResult(delta_hat, lo, hi, boot, "percentile")


# --- McNemar ----------------------------------------------------------------

def mcnemar_test(b: int, c: int, *, continuity: bool = True) -> dict:
    """McNemar's test on a 2×2 paired table.

    `b` = count where method A=1 and method B=0 (or annotator A yes, B no).
    `c` = count where method A=0 and method B=1.
    Uses chi-square approximation with continuity correction; falls back to
    exact binomial when b + c is small (< 25).

    Returns {'statistic', 'p_value', 'method', 'b', 'c'}.
    """
    n = b + c
    if n == 0:
        return {"statistic": 0.0, "p_value": 1.0, "method": "trivial",
                "b": b, "c": c}
    if n < 25:
        # Exact two-sided binomial test: H0 P(disagreement is b-type)=0.5
        k = min(b, c)
        p_two = 2 * sum(math.comb(n, i) * 0.5 ** n for i in range(k + 1))
        p_two = min(p_two, 1.0)
        return {"statistic": float(k), "p_value": float(p_two),
                "method": "exact_binomial", "b": b, "c": c}
    if continuity:
        stat = (abs(b - c) - 1) ** 2 / n
    else:
        stat = (b - c) ** 2 / n
    p = float(sstats.chi2.sf(stat, df=1))
    return {"statistic": float(stat), "p_value": p,
            "method": "chi2_cc" if continuity else "chi2", "b": b, "c": c}


# --- Fisher-z CI for correlation --------------------------------------------

def fisher_z_ci_for_corr(r: float, n: int, alpha: float = 0.05
                         ) -> tuple[float, float]:
    """Fisher-z transform CI for Pearson / Spearman / Kendall τ correlation.

    For Kendall τ-b, use the variance adjustment (n >= 10) and treat as
    approximate — sufficient for sanity checks alongside the primary
    bootstrap CI.
    """
    if n < 4 or abs(r) >= 1.0:
        return (float("nan"), float("nan"))
    z = 0.5 * math.log((1 + r) / (1 - r))
    se = 1 / math.sqrt(n - 3)
    crit = float(sstats.norm.ppf(1 - alpha / 2))
    lo_z = z - crit * se
    hi_z = z + crit * se
    lo = (math.exp(2 * lo_z) - 1) / (math.exp(2 * lo_z) + 1)
    hi = (math.exp(2 * hi_z) - 1) / (math.exp(2 * hi_z) + 1)
    return (float(lo), float(hi))


# --- Two-source (case × run) nested bootstrap -------------------------------

def two_source_bootstrap_ci(
    correctness_matrix: np.ndarray,
    *,
    statistic: Callable[[np.ndarray], float] | None = None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    random_state: int | None = 0,
) -> BootstrapResult:
    """Nested bootstrap over (cases × runs) for a 2-D correctness matrix.

    correctness_matrix: shape (n_cases, n_runs), values 0/1 (or NaN for
    not-applicable). NaNs propagate — the statistic must handle them
    (default uses nan-aware mean).

    Outer loop resamples cases; inner loop resamples runs. This is the
    primary headline CI when GLMM is unavailable or inappropriate (e.g.,
    for F1-valued nested-list metrics where mixed-effects on a per-case
    logit doesn't apply).
    """
    rng = np.random.default_rng(random_state)
    m = np.asarray(correctness_matrix, dtype=float)
    if m.ndim != 2:
        raise ValueError("correctness_matrix must be 2-D (cases × runs)")
    n_cases, n_runs = m.shape
    if n_cases == 0 or n_runs == 0:
        return BootstrapResult(float("nan"), float("nan"), float("nan"),
                               np.array([]), "two_source")

    if statistic is None:
        def statistic(x: np.ndarray) -> float:
            return float(np.nanmean(x))

    theta_hat = float(statistic(m))
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        case_idx = rng.integers(0, n_cases, size=n_cases)
        run_idx = rng.integers(0, n_runs, size=n_runs)
        resampled = m[np.ix_(case_idx, run_idx)]
        try:
            boot[i] = float(statistic(resampled))
        except Exception:
            boot[i] = np.nan

    boot = boot[~np.isnan(boot)]
    lo = _quantile(boot, alpha / 2)
    hi = _quantile(boot, 1 - alpha / 2)
    return BootstrapResult(theta_hat, lo, hi, boot, "two_source")
