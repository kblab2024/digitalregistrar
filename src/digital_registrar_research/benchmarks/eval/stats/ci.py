"""Confidence-interval methods for the cascade.

Re-exports the canonical primitives from
:mod:`digital_registrar_research.benchmarks.eval.ci` and adds a few
methods (Agresti-Coull, jackknife) plus a ``pick_ci`` selector that
chooses the right interval method per (metric_kind, n, p) cell.

References
----------
- Wilson (1927) "Probable Inference, the Law of Succession, and Statistical
  Inference," JASA.
- Clopper & Pearson (1934) "The use of confidence or fiducial limits illustrated
  in the case of the binomial," Biometrika.
- Agresti & Coull (1998) "Approximate is better than 'exact' for interval
  estimation of binomial proportions," The American Statistician.
- Efron (1982) "The Jackknife, the Bootstrap, and Other Resampling Plans,"
  CBMS Conf. Series.
- Efron (1987) "Better Bootstrap Confidence Intervals," JASA — for BCa.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
from scipy import stats as sstats

from ..ci import (
    BootstrapResult,
    bootstrap_ci as _bca_bootstrap_ci,
    clopper_pearson_ci,
    fisher_z_ci_for_corr,
    paired_bootstrap_diff,
    t_ci,
    two_source_bootstrap_ci,
    wilson_ci,
)


def agresti_coull_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Agresti-Coull adjusted-Wald CI for a binary proportion.

    Adds 2 successes and 2 failures (for a 95% CI) before computing a
    Wald CI on the adjusted estimate. Faster and only slightly wider
    than Wilson; unlike plain Wald, it does not collapse at p=0 or p=1.
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    z = float(sstats.norm.ppf(1 - alpha / 2))
    n_tilde = n + z * z
    p_tilde = (k + z * z / 2) / n_tilde
    half = z * math.sqrt(p_tilde * (1 - p_tilde) / n_tilde)
    return (max(0.0, p_tilde - half), min(1.0, p_tilde + half))


def jackknife_ci(
    values: Sequence,
    statistic: Callable[[Sequence], float],
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Jackknife CI for a statistic. Cheap, deterministic.

    Used as a sanity check on the bootstrap CI, especially for n < 30
    where the bootstrap distribution is sparse. Returns
    ``(theta_hat, lo, hi)``.
    """
    items = list(values)
    n = len(items)
    if n < 2:
        if n == 1:
            v = float(statistic(items))
            return (v, v, v)
        return (float("nan"),) * 3
    theta_hat = float(statistic(items))
    leave_one_out = np.empty(n, dtype=float)
    for i in range(n):
        sub = items[:i] + items[i + 1:]
        try:
            leave_one_out[i] = float(statistic(sub))
        except Exception:
            leave_one_out[i] = float("nan")
    valid = leave_one_out[~np.isnan(leave_one_out)]
    if valid.size < 2:
        return (theta_hat, theta_hat, theta_hat)
    pseudovals = n * theta_hat - (n - 1) * valid
    se = float(np.std(pseudovals, ddof=1) / math.sqrt(valid.size))
    z = float(sstats.norm.ppf(1 - alpha / 2))
    return (theta_hat, theta_hat - z * se, theta_hat + z * se)


def bca_bootstrap_ci(
    values: Sequence,
    statistic: Callable[[Sequence], float],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    strata: Sequence | None = None,
    random_state: int | None = 0,
) -> BootstrapResult:
    """BCa bootstrap CI. Thin alias over the canonical implementation
    in :mod:`benchmarks.eval.ci.bootstrap_ci`."""
    return _bca_bootstrap_ci(
        values, statistic,
        n_boot=n_boot, alpha=alpha, method="bca",
        strata=strata, random_state=random_state,
    )


CIKind = Literal["proportion", "mean", "f1", "kappa", "ccc", "delta"]


def pick_ci(
    kind: CIKind,
    *,
    n: int | None = None,
    p: float | None = None,
) -> str:
    """Recommend a CI method by metric kind and sample size.

    Returns one of ``"wilson"``, ``"clopper_pearson"``, ``"agresti_coull"``,
    ``"bca_bootstrap"``, ``"t"``. Callers wire the recommendation to the
    matching function. The selection rules are pre-registered:

    - Proportion, n >= 30, 0.05 <= p <= 0.95 → ``wilson``.
    - Proportion, n < 30 OR p < 0.05 OR p > 0.95 → ``clopper_pearson``
      (exact, conservative).
    - Mean of a small-n sample → ``t`` (Student-t).
    - F1 / kappa / Lin's CCC → ``bca_bootstrap`` (skewed sampling
      distribution, BCa stays calibrated).
    - Accuracy / F1 delta between two methods → ``bca_bootstrap`` on
      paired resamples (callers use ``paired_bootstrap_diff``).
    """
    if kind == "proportion":
        if n is None or n < 30:
            return "clopper_pearson"
        if p is None:
            return "wilson"
        if p < 0.05 or p > 0.95:
            return "clopper_pearson"
        return "wilson"
    if kind == "mean":
        return "t"
    if kind in ("f1", "kappa", "ccc", "delta"):
        return "bca_bootstrap"
    return "bca_bootstrap"


__all__ = [
    "wilson_ci",
    "clopper_pearson_ci",
    "agresti_coull_ci",
    "jackknife_ci",
    "bca_bootstrap_ci",
    "BootstrapResult",
    "t_ci",
    "fisher_z_ci_for_corr",
    "paired_bootstrap_diff",
    "two_source_bootstrap_ci",
    "pick_ci",
]
