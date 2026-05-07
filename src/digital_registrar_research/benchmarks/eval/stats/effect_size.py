"""Effect-size measures for the cascade.

Wraps the existing primitives in :mod:`scripts.eval._common.stats_extra`
where they exist (Cohen's d, Cliff's δ, MCC, balanced accuracy) and
adds Cohen's kappa, weighted kappa, Krippendorff's alpha, and Brier
score. Reductions should import these from this module so that the
public surface is uniform.

References
----------
- Cohen (1960) "A coefficient of agreement for nominal scales,"
  Educ. Psychol. Meas.
- Fleiss & Cohen (1973) "The equivalence of weighted kappa and the
  intraclass correlation coefficient as measures of reliability,"
  Educ. Psychol. Meas. — for weighted kappa.
- Krippendorff (2004) "Content Analysis," Sage. Krippendorff (2011)
  "Computing Krippendorff's Alpha-Reliability."
- Brier (1950) "Verification of forecasts expressed in terms of
  probability," Mon. Weather Rev.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import stats as sstats


# Re-exports from the legacy stats_extra module so callers have one
# import surface. The legacy module stays for backwards compat but is
# not the canonical home anymore.

def _legacy():
    """Lazy import to avoid coupling to scripts/ at module load time."""
    import sys
    import importlib.util
    from pathlib import Path
    if "scripts.eval._common.stats_extra" in sys.modules:
        return sys.modules["scripts.eval._common.stats_extra"]
    # Locate scripts/eval/_common/stats_extra.py relative to repo root
    repo_root = Path(__file__).resolve().parents[5]
    path = repo_root / "scripts" / "eval" / "_common" / "stats_extra.py"
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location(
        "scripts.eval._common.stats_extra", path,
    )
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """Cohen's d, pooled-SD form. See ``stats_extra.cohens_d``."""
    legacy = _legacy()
    if legacy is None:
        # Fallback: inline implementation.
        a_arr = np.asarray(a, dtype=float)
        b_arr = np.asarray(b, dtype=float)
        a_arr = a_arr[~np.isnan(a_arr)]
        b_arr = b_arr[~np.isnan(b_arr)]
        if a_arr.size < 2 or b_arr.size < 2:
            return float("nan")
        s_a = a_arr.std(ddof=1)
        s_b = b_arr.std(ddof=1)
        n_a, n_b = a_arr.size, b_arr.size
        pooled = np.sqrt(((n_a - 1) * s_a * s_a + (n_b - 1) * s_b * s_b)
                         / (n_a + n_b - 2))
        if pooled == 0:
            return float("nan")
        return float((a_arr.mean() - b_arr.mean()) / pooled)
    return float(legacy.cohens_d(a, b))


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Cliff's delta. See ``stats_extra.cliffs_delta``."""
    legacy = _legacy()
    if legacy is None:
        a_arr = np.asarray(a, dtype=float)
        b_arr = np.asarray(b, dtype=float)
        a_arr = a_arr[~np.isnan(a_arr)]
        b_arr = b_arr[~np.isnan(b_arr)]
        if a_arr.size == 0 or b_arr.size == 0:
            return float("nan")
        a_col = a_arr.reshape(-1, 1)
        b_row = b_arr.reshape(1, -1)
        gt = (a_col > b_row).sum()
        lt = (a_col < b_row).sum()
        return float((gt - lt) / (a_arr.size * b_arr.size))
    return float(legacy.cliffs_delta(a, b))


def matthews_corrcoef(y_true: Sequence, y_pred: Sequence) -> float:
    """Matthews correlation coefficient. See ``stats_extra``."""
    from sklearn.metrics import matthews_corrcoef as _mcc
    paired = [(t, p) for t, p in zip(y_true, y_pred, strict=True)
              if t is not None and p is not None]
    if not paired:
        return float("nan")
    y_t, y_p = zip(*paired)
    return float(_mcc(list(y_t), list(y_p)))


def balanced_accuracy(y_true: Sequence, y_pred: Sequence) -> float:
    """Balanced accuracy (mean per-class recall)."""
    from sklearn.metrics import balanced_accuracy_score
    paired = [(t, p) for t, p in zip(y_true, y_pred, strict=True)
              if t is not None and p is not None]
    if not paired:
        return float("nan")
    y_t, y_p = zip(*paired)
    return float(balanced_accuracy_score(list(y_t), list(y_p)))


# --- Cohen's kappa --------------------------------------------------------

def cohens_kappa(
    y_true: Sequence,
    y_pred: Sequence,
    *,
    labels: Sequence | None = None,
) -> float:
    """Cohen's kappa for two-rater nominal agreement.

    Wraps :func:`sklearn.metrics.cohen_kappa_score`. Used to surface
    model-vs-gold agreement on top of accuracy, since accuracy with
    imbalanced classes can mask poor kappa.
    """
    from sklearn.metrics import cohen_kappa_score
    paired = [(t, p) for t, p in zip(y_true, y_pred, strict=True)
              if t is not None and p is not None]
    if not paired:
        return float("nan")
    y_t, y_p = zip(*paired)
    if labels is not None:
        return float(cohen_kappa_score(list(y_t), list(y_p),
                                        labels=list(labels)))
    return float(cohen_kappa_score(list(y_t), list(y_p)))


def weighted_kappa(
    y_true: Sequence,
    y_pred: Sequence,
    *,
    weights: str = "quadratic",
    labels: Sequence | None = None,
) -> float:
    """Weighted Cohen's kappa for ordinal agreement.

    ``weights="linear"`` or ``"quadratic"``. Quadratic is the de-facto
    standard for staging concordance (T, N, M, grade) — it penalises
    "off by 2" four times as harshly as "off by 1."
    """
    from sklearn.metrics import cohen_kappa_score
    if weights not in ("linear", "quadratic"):
        raise ValueError(f"weights must be 'linear' or 'quadratic'; got {weights!r}")
    paired = [(t, p) for t, p in zip(y_true, y_pred, strict=True)
              if t is not None and p is not None]
    if not paired:
        return float("nan")
    y_t, y_p = zip(*paired)
    return float(cohen_kappa_score(
        list(y_t), list(y_p),
        weights=weights,
        labels=list(labels) if labels is not None else None,
    ))


# --- Krippendorff's alpha -------------------------------------------------

def krippendorff_alpha(
    raters: Sequence[Sequence],
    *,
    level: str = "nominal",
) -> float:
    """Krippendorff's alpha for k-rater reliability.

    ``raters`` is a list of length R (number of raters); each element
    is a sequence of length N (number of items) with values or None
    for missing. Coincidence-matrix construction handles missing data
    pairwise (Krippendorff 2011).

    ``level`` is one of ``"nominal"``, ``"ordinal"``, ``"interval"``,
    ``"ratio"``. Determines the difference function delta(c, k).

    Generalises Cohen's kappa to >= 2 raters and missing data; the
    direct answer to Reviewer 1(b)'s "incorporating independently
    annotated samples or reporting inter-annotator agreement metrics."

    Returns alpha in [-1, 1]. NaN on degenerate input (all values
    equal, no pairings, etc.).
    """
    R = len(raters)
    if R < 2:
        return float("nan")
    N = len(raters[0])
    if any(len(r) != N for r in raters):
        raise ValueError("all raters must have the same number of items")

    # Collect all observed values to build the value-index map.
    all_vals: list = []
    for r in raters:
        for v in r:
            if v is not None and not (isinstance(v, float)
                                      and np.isnan(v)):
                all_vals.append(v)
    if not all_vals:
        return float("nan")
    if level == "nominal":
        unique = list(dict.fromkeys(all_vals))
    else:
        unique = sorted(set(all_vals), key=lambda v: float(v))
    val_idx = {v: i for i, v in enumerate(unique)}
    K = len(unique)
    if K < 2:
        return float("nan")  # all annotators agree on a single value

    # Coincidence matrix: c[v, w] = sum over items of pair-count where
    # one rater chose v and another chose w in the same item. The
    # Krippendorff weighting is 1/(m_u - 1) per item, where m_u is the
    # number of non-missing ratings for item u.
    coin = np.zeros((K, K), dtype=float)
    for u in range(N):
        item_vals = [r[u] for r in raters
                     if r[u] is not None and not (isinstance(r[u], float)
                                                   and np.isnan(r[u]))]
        m_u = len(item_vals)
        if m_u < 2:
            continue
        for i in range(m_u):
            for j in range(m_u):
                if i == j:
                    continue
                coin[val_idx[item_vals[i]],
                     val_idx[item_vals[j]]] += 1.0 / (m_u - 1)

    n_v = coin.sum(axis=1)  # marginals
    n_total = n_v.sum()
    if n_total == 0:
        return float("nan")

    # Difference matrix.
    delta = np.zeros((K, K), dtype=float)
    if level == "nominal":
        for i in range(K):
            for j in range(K):
                delta[i, j] = 0.0 if i == j else 1.0
    elif level == "ordinal":
        # delta(c,k) = (sum of n_g for g in [c..k]) - 0.5*(n_c + n_k))^2
        for i in range(K):
            for j in range(K):
                if i == j:
                    continue
                lo, hi = (i, j) if i < j else (j, i)
                between = n_v[lo:hi + 1].sum() - 0.5 * (n_v[lo] + n_v[hi])
                delta[i, j] = between * between
    elif level in ("interval", "ratio"):
        vals = np.asarray([float(v) for v in unique], dtype=float)
        if level == "interval":
            for i in range(K):
                for j in range(K):
                    delta[i, j] = (vals[i] - vals[j]) ** 2
        else:  # ratio
            for i in range(K):
                for j in range(K):
                    if vals[i] + vals[j] == 0:
                        delta[i, j] = 0.0
                    else:
                        delta[i, j] = ((vals[i] - vals[j])
                                       / (vals[i] + vals[j])) ** 2
    else:
        raise ValueError(f"unknown level {level!r}")

    # Observed disagreement D_o = sum_{c,k} c[c,k] * delta[c,k] / n_total
    d_o = float((coin * delta).sum() / n_total)
    # Expected disagreement D_e:
    #   D_e = (1/(n*(n-1))) * sum_{c,k} n_c * n_k * delta[c,k]
    if n_total < 2:
        return float("nan")
    d_e = float((np.outer(n_v, n_v) * delta).sum() / (n_total * (n_total - 1)))
    if d_e == 0:
        return float("nan")
    return 1.0 - d_o / d_e


# --- Brier score ----------------------------------------------------------

def brier_score(y_true: Sequence[int], y_prob: Sequence[float]) -> float:
    """Brier score for binary calibration: mean((p - y)^2).

    Inert today (the cascade does not consume model probabilities) but
    scaffolded for the multimodal extension where calibrated
    confidences become a deliverable. Returns NaN on empty input.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    if y.size == 0 or y.size != p.size:
        return float("nan")
    return float(np.mean((p - y) ** 2))


__all__ = [
    "cohens_d",
    "cliffs_delta",
    "matthews_corrcoef",
    "balanced_accuracy",
    "cohens_kappa",
    "weighted_kappa",
    "krippendorff_alpha",
    "brier_score",
]
