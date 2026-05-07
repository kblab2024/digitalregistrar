"""Concordance / agreement on continuous outputs.

Used for the redesigned LN counts (``examined``, ``involved``) — now
aggregated by ``(side, category)`` group — and for ``tumor_size``.

Methods
-------
* :func:`lin_ccc` — Lin's concordance correlation coefficient. Method-
  comparison standard for paired continuous measurements; combines
  precision (Pearson r) and accuracy (location/scale shift).
* :func:`bland_altman` — Mean difference + 95% limits of agreement.
  Catches systematic bias that correlation misses.
* :func:`mape` — Mean Absolute Percentage Error. Reports bias relative
  to the magnitude of the gold count, which absolute MAE alone hides.
* :func:`spearman_rho` — Spearman rank correlation. Outlier-resistant.
* :func:`count_mae` — Plain mean absolute error on counts.

References
----------
- Lin (1989) "A concordance correlation coefficient to evaluate
  reproducibility," Biometrics.
- Bland & Altman (1986) "Statistical methods for assessing agreement
  between two methods of clinical measurement," Lancet.
- Spearman (1904) "The proof and measurement of association between
  two things," Am. J. Psychol.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy import stats as sstats


def _pair(x: Sequence[float], y: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Drop NaN pairs; return matched arrays."""
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if xa.shape != ya.shape:
        raise ValueError("inputs must have identical shape")
    mask = ~(np.isnan(xa) | np.isnan(ya))
    return xa[mask], ya[mask]


def lin_ccc(
    gold: Sequence[float],
    pred: Sequence[float],
    *,
    alpha: float = 0.05,
) -> dict:
    """Lin's concordance correlation coefficient.

    CCC = 2 * cov(g, p) / (var(g) + var(p) + (mean(g) - mean(p))^2)

    Returns ``{"ccc", "ci_lo", "ci_hi", "n", "pearson_r", "scale", "location"}``
    where ``scale`` is the precision multiplier sigma_g/sigma_p and
    ``location`` is the bias term (mean_g - mean_p) / geomean(sigma).
    Lin (1989) eq. 1; CI via Fisher-z on a transformed CCC.
    """
    g, p = _pair(gold, pred)
    n = g.size
    if n < 3:
        return {"ccc": float("nan"), "ci_lo": float("nan"),
                "ci_hi": float("nan"), "n": n,
                "pearson_r": float("nan"), "scale": float("nan"),
                "location": float("nan")}
    mean_g, mean_p = float(g.mean()), float(p.mean())
    var_g, var_p = float(g.var(ddof=1)), float(p.var(ddof=1))
    cov = float(np.cov(g, p, ddof=1)[0, 1])
    denom = var_g + var_p + (mean_g - mean_p) ** 2
    if denom == 0:
        return {"ccc": float("nan"), "ci_lo": float("nan"),
                "ci_hi": float("nan"), "n": n,
                "pearson_r": float("nan"), "scale": float("nan"),
                "location": float("nan")}
    ccc = 2 * cov / denom

    sd_g = math.sqrt(var_g) if var_g > 0 else float("nan")
    sd_p = math.sqrt(var_p) if var_p > 0 else float("nan")
    pearson_r = float(cov / (sd_g * sd_p)) if sd_g and sd_p else float("nan")
    scale = sd_p / sd_g if (sd_g and not math.isnan(sd_g)) else float("nan")
    location = ((mean_p - mean_g) / math.sqrt(sd_g * sd_p)
                if (sd_g and sd_p) else float("nan"))

    # Lin (1989) Fisher-z based CI.
    if abs(ccc) < 1:
        z = 0.5 * math.log((1 + ccc) / (1 - ccc))
        # Approximate SE per Lin eq. (4.7).
        try:
            v = ((1 - pearson_r ** 2) * ccc ** 2
                 / ((1 - ccc ** 2) * pearson_r ** 2 * (n - 2))
                 + 2 * ccc ** 3 * (1 - ccc) * (mean_g - mean_p) ** 2
                 / (pearson_r * (1 - ccc ** 2) ** 2 * sd_g ** 2 * sd_p ** 2)
                 - ccc ** 4 * (mean_g - mean_p) ** 4
                 / (2 * pearson_r ** 2 * (1 - ccc ** 2) ** 2
                    * sd_g ** 2 * sd_p ** 2))
            se_z = math.sqrt(max(v, 0) / max(n - 2, 1))
        except (ZeroDivisionError, ValueError):
            se_z = float("nan")
        if math.isfinite(se_z):
            crit = float(sstats.norm.ppf(1 - alpha / 2))
            z_lo, z_hi = z - crit * se_z, z + crit * se_z
            ci_lo = (math.exp(2 * z_lo) - 1) / (math.exp(2 * z_lo) + 1)
            ci_hi = (math.exp(2 * z_hi) - 1) / (math.exp(2 * z_hi) + 1)
        else:
            ci_lo = ci_hi = float("nan")
    else:
        ci_lo = ci_hi = float("nan")

    return {
        "ccc": float(ccc),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "n": n,
        "pearson_r": pearson_r,
        "scale": scale,
        "location": location,
    }


def bland_altman(
    gold: Sequence[float],
    pred: Sequence[float],
    *,
    alpha: float = 0.05,
) -> dict:
    """Bland-Altman summary: mean diff + 95% limits of agreement.

    Returns ``{"mean_diff", "sd_diff", "loa_lo", "loa_hi", "n"}``.
    Caller renders the full Bland-Altman scatter plot from the
    underlying paired arrays separately.
    """
    g, p = _pair(gold, pred)
    n = g.size
    if n < 2:
        return {"mean_diff": float("nan"), "sd_diff": float("nan"),
                "loa_lo": float("nan"), "loa_hi": float("nan"), "n": n}
    diff = p - g
    mean_diff = float(diff.mean())
    sd_diff = float(diff.std(ddof=1))
    z = float(sstats.norm.ppf(1 - alpha / 2))
    return {
        "mean_diff": mean_diff,
        "sd_diff": sd_diff,
        "loa_lo": mean_diff - z * sd_diff,
        "loa_hi": mean_diff + z * sd_diff,
        "n": n,
    }


def mape(gold: Sequence[float], pred: Sequence[float]) -> float:
    """Mean Absolute Percentage Error.

    Skips pairs where gold == 0 (MAPE undefined). Reports as a fraction
    (multiply by 100 in the CSV column header for percent).
    """
    g, p = _pair(gold, pred)
    nonzero = g != 0
    if not nonzero.any():
        return float("nan")
    return float(np.mean(np.abs((p[nonzero] - g[nonzero]) / g[nonzero])))


def spearman_rho(
    gold: Sequence[float],
    pred: Sequence[float],
) -> dict:
    """Spearman rank correlation with two-sided p-value.

    Returns ``{"rho", "p_value", "n"}``. Outlier-resistant; report
    alongside Pearson r when distributions are skewed.
    """
    g, p = _pair(gold, pred)
    n = g.size
    if n < 3:
        return {"rho": float("nan"), "p_value": float("nan"), "n": n}
    rho, pval = sstats.spearmanr(g, p)
    return {"rho": float(rho), "p_value": float(pval), "n": n}


def count_mae(gold: Sequence[float], pred: Sequence[float]) -> float:
    """Mean absolute error on counts."""
    g, p = _pair(gold, pred)
    if g.size == 0:
        return float("nan")
    return float(np.mean(np.abs(g - p)))


__all__ = [
    "lin_ccc",
    "bland_altman",
    "mape",
    "spearman_rho",
    "count_mae",
]
