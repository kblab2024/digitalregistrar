"""Statistical helpers not present in ``src/.../eval/ci.py``.

Default policy: prefer canonical library calls (``sklearn.metrics``,
``scipy.stats``, ``statsmodels.stats.multitest``) over custom code so
the paper's Methods section can cite well-known packages. Each helper
documents its library source for verbatim copying into
``docs/eval/methods_citations.md``.

References:
    Holm-Bonferroni: Holm (1979).
    Benjamini-Hochberg FDR: Benjamini & Hochberg (1995).
    Cohen's d: Cohen (1988).
    Cliff's δ: Cliff (1993).
    Matthews correlation coefficient: Matthews (1975).
    Balanced accuracy: see scikit-learn user guide.
"""
from __future__ import annotations

import logging
import math
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


# --- Multiple-comparisons correction ----------------------------------------

def adjust_pvalues(
    p_values: Sequence[float],
    method: str = "holm",
) -> list[float]:
    """Adjust a vector of p-values for multiple comparisons.

    Wraps :func:`statsmodels.stats.multitest.multipletests`. Method is
    one of ``"holm"`` (Holm-Bonferroni, FWER) or ``"fdr_bh"`` (Benjamini-
    Hochberg, FDR). NaN entries are preserved through the adjustment.

    Implementation: ``statsmodels.stats.multitest.multipletests``
    (Seabold & Perktold 2010). See Holm (1979) and Benjamini & Hochberg
    (1995) for the original methods.
    """
    from statsmodels.stats.multitest import multipletests

    p = np.asarray(p_values, dtype=float)
    finite = ~np.isnan(p)
    out = np.full_like(p, np.nan)
    if not finite.any():
        return out.tolist()
    _, p_adj, _, _ = multipletests(p[finite], method=method)
    out[finite] = p_adj
    return out.tolist()


# --- Effect sizes ------------------------------------------------------------

def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """Cohen's d for two independent samples (pooled SD).

    Implementation: in-house ``(mean(a) - mean(b)) / pooled_sd`` using
    pooled variance. ``pingouin.compute_effsize(eftype="cohen")`` is
    the library equivalent if available. Returns NaN on degenerate
    input. See Cohen (1988) §2.
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    a_arr = a_arr[~np.isnan(a_arr)]
    b_arr = b_arr[~np.isnan(b_arr)]
    if a_arr.size < 2 or b_arr.size < 2:
        return float("nan")
    s_a = a_arr.std(ddof=1)
    s_b = b_arr.std(ddof=1)
    n_a = a_arr.size
    n_b = b_arr.size
    pooled = math.sqrt(((n_a - 1) * s_a * s_a + (n_b - 1) * s_b * s_b)
                       / (n_a + n_b - 2))
    if pooled == 0:
        return float("nan")
    return float((a_arr.mean() - b_arr.mean()) / pooled)


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Cliff's δ — non-parametric effect size for ordinal data.

    Implementation: in-house O(n_a · n_b) loop counting dominance pairs.
    See Cliff (1993). δ ∈ [-1, 1]; |δ| < 0.147 negligible, < 0.33
    small, < 0.474 medium, ≥ 0.474 large (Romano et al. 2006 thresholds).
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    a_arr = a_arr[~np.isnan(a_arr)]
    b_arr = b_arr[~np.isnan(b_arr)]
    if a_arr.size == 0 or b_arr.size == 0:
        return float("nan")
    # Vectorised: build a 2-D comparison matrix.
    a_col = a_arr.reshape(-1, 1)
    b_row = b_arr.reshape(1, -1)
    gt = (a_col > b_row).sum()
    lt = (a_col < b_row).sum()
    return float((gt - lt) / (a_arr.size * b_arr.size))


def odds_ratio_with_ci(
    table: np.ndarray | Sequence[Sequence[int]],
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Odds ratio + Wald CI for a 2×2 contingency table.

    Implementation: in-house ``(a*d) / (b*c)`` with log-OR ± z·SE CI.
    SciPy 1.10+ exposes ``scipy.stats.contingency.odds_ratio`` which
    can replace this once we pin scipy ≥ 1.10. Returns ``(or, lo, hi)``;
    NaN on zero-count cells where the OR is undefined.

    Table layout: ``[[a, b], [c, d]]`` (case 1 row, case 2 row).
    """
    arr = np.asarray(table, dtype=float)
    if arr.shape != (2, 2):
        raise ValueError("table must be 2×2")
    a, b = arr[0]
    c, d = arr[1]
    if min(a, b, c, d) == 0:
        # Haldane–Anscombe correction for zero cells.
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    or_value = (a * d) / (b * c)
    se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    log_or = math.log(or_value)
    from scipy.stats import norm
    z = float(norm.ppf(1 - alpha / 2))
    lo = math.exp(log_or - z * se)
    hi = math.exp(log_or + z * se)
    return float(or_value), float(lo), float(hi)


# --- Classification metrics --------------------------------------------------

def _drop_none_pairs(
    y_true: Sequence, y_pred: Sequence,
) -> "tuple[list, list]":
    """Drop ``(t, p)`` pairs where either side is ``None``.

    Newer sklearn versions call ``np.unique`` inside ``_check_targets``
    when they receive ``object`` arrays, and numpy cannot sort arrays
    that mix ``str`` and ``NoneType``. Callers in this module that
    constrain their ``labels`` arg to non-``None`` values already
    intend for ``None`` rows to be excluded; pre-filtering is
    semantically equivalent and avoids the sklearn explosion on
    environments with newer numpy/sklearn.
    """
    paired = [(t, p) for t, p in zip(y_true, y_pred)
              if t is not None and p is not None]
    if not paired:
        return [], []
    a, b = zip(*paired)
    return list(a), list(b)


def confusion_matrix_long(
    y_true: Sequence,
    y_pred: Sequence,
    *,
    labels: Sequence | None = None,
) -> "list[dict]":
    """Build a long-form confusion matrix as a list of row-dicts.

    Implementation: ``sklearn.metrics.confusion_matrix(y_true, y_pred,
    labels=labels)`` (Pedregosa et al. 2011). Output rows have keys
    ``gold_value``, ``pred_value``, ``count`` for direct DataFrame
    ingestion. ``None`` pairs are filtered out — see
    :func:`_drop_none_pairs`.
    """
    from sklearn.metrics import confusion_matrix

    y_true, y_pred = _drop_none_pairs(y_true, y_pred)

    if labels is None:
        labels = sorted({*y_true, *y_pred}, key=lambda v: ("" if v is None else str(v)))
    if not y_true:
        # Empty matrix: all-zero counts at the requested labels.
        rows = []
        for gold_v in labels:
            for pred_v in labels:
                rows.append({"gold_value": gold_v,
                             "pred_value": pred_v, "count": 0})
        return rows
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    rows = []
    for i, gold_v in enumerate(labels):
        for j, pred_v in enumerate(labels):
            rows.append({
                "gold_value": gold_v,
                "pred_value": pred_v,
                "count": int(cm[i, j]),
            })
    return rows


def per_class_prf1(
    y_true: Sequence,
    y_pred: Sequence,
    *,
    labels: Sequence | None = None,
) -> dict[str, "dict | float"]:
    """Per-class precision/recall/F1 plus macro/micro averages.

    Implementation: ``sklearn.metrics.precision_recall_fscore_support``
    (Pedregosa et al. 2011). Returns a dict with one entry per label
    plus ``macro``/``micro``/``weighted`` aggregates.
    """
    from sklearn.metrics import precision_recall_fscore_support

    y_true, y_pred = _drop_none_pairs(y_true, y_pred)
    if labels is None:
        labels = sorted({*y_true, *y_pred}, key=lambda v: ("" if v is None else str(v)))
    if not y_true:
        out: dict[str, "dict | float"] = {}
        for label in labels:
            out[str(label)] = {"precision": 0.0, "recall": 0.0,
                                "f1": 0.0, "support": 0}
        for avg in ("macro", "micro", "weighted"):
            out[f"{avg}_avg"] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        return out
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(labels), zero_division=0,
    )
    out: dict[str, "dict | float"] = {}
    for i, label in enumerate(labels):
        out[str(label)] = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
    for avg in ("macro", "micro", "weighted"):
        prec_a, rec_a, f1_a, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=list(labels), average=avg, zero_division=0,
        )
        out[f"{avg}_avg"] = {
            "precision": float(prec_a), "recall": float(rec_a), "f1": float(f1_a),
        }
    return out


def matthews_corrcoef(y_true: Sequence, y_pred: Sequence) -> float:
    """Matthews correlation coefficient.

    Implementation: ``sklearn.metrics.matthews_corrcoef`` (Pedregosa et
    al. 2011). MCC handles class imbalance gracefully — the headline
    binary metric of choice here. See Matthews (1975).
    """
    from sklearn.metrics import matthews_corrcoef as _mcc
    y_true, y_pred = _drop_none_pairs(y_true, y_pred)
    if not y_true:
        return float("nan")
    return float(_mcc(y_true, y_pred))


def balanced_accuracy(y_true: Sequence, y_pred: Sequence) -> float:
    """Balanced accuracy = mean per-class recall.

    Implementation: ``sklearn.metrics.balanced_accuracy_score``
    (Pedregosa et al. 2011). Use alongside accuracy whenever class
    distribution is skewed.
    """
    from sklearn.metrics import balanced_accuracy_score
    y_true, y_pred = _drop_none_pairs(y_true, y_pred)
    if not y_true:
        return float("nan")
    return float(balanced_accuracy_score(y_true, y_pred))


def top_k_ordinal_accuracy(
    y_true: Sequence,
    y_pred: Sequence,
    *,
    ordinal_order: Sequence,
    k: int = 1,
) -> float:
    """Top-k accuracy for ordinal fields by rank distance.

    For ordinal labels, predictions within rank distance ``k`` of gold
    are counted as correct. ``ordinal_order`` is the list defining the
    rank order of categories. Returns the fraction of cases where
    ``|rank(pred) - rank(gold)| ≤ k``.

    No direct sklearn equivalent because ``top_k_accuracy_score``
    requires probability scores — this is in-house custom code.
    """
    rank_of = {v: i for i, v in enumerate(ordinal_order)}
    matches = 0
    n = 0
    for g, p in zip(y_true, y_pred, strict=True):
        if g is None or p is None or g not in rank_of or p not in rank_of:
            continue
        if abs(rank_of[g] - rank_of[p]) <= k:
            matches += 1
        n += 1
    if n == 0:
        return float("nan")
    return matches / n


def rank_distance_distribution(
    y_true: Sequence,
    y_pred: Sequence,
    *,
    ordinal_order: Sequence,
) -> dict[int, int]:
    """Histogram of ``|rank(pred) - rank(gold)|`` across cases.

    Useful for ordinal fields where "off by one" is materially less bad
    than "off by three." Returns ``{distance: count}``.
    """
    rank_of = {v: i for i, v in enumerate(ordinal_order)}
    dist: dict[int, int] = {}
    for g, p in zip(y_true, y_pred, strict=True):
        if g is None or p is None or g not in rank_of or p not in rank_of:
            continue
        d = abs(rank_of[g] - rank_of[p])
        dist[d] = dist.get(d, 0) + 1
    return dist


# --- Distribution-distance metrics (cross-dataset) --------------------------

def kl_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    """KL(P || Q) for discrete distributions.

    Implementation: ``scipy.special.rel_entr(p, q).sum()`` which handles
    p=0 correctly (Virtanen et al. 2020). Caller normalises so p, q sum
    to 1. Returns NaN if any q_i = 0 where p_i > 0 (KL undefined).
    """
    from scipy.special import rel_entr
    arr = np.asarray(rel_entr(p, q), dtype=float)
    if not np.isfinite(arr).all():
        return float("nan")
    return float(arr.sum())


def jensen_shannon(p: Sequence[float], q: Sequence[float]) -> float:
    """Jensen-Shannon distance (symmetric, bounded [0, 1] with log2).

    Implementation: ``scipy.spatial.distance.jensenshannon`` with
    ``base=2`` (Virtanen et al. 2020). See Lin (1991).
    """
    from scipy.spatial.distance import jensenshannon
    return float(jensenshannon(p, q, base=2))


def wasserstein(a: Sequence[float], b: Sequence[float]) -> float:
    """Wasserstein-1 distance between two empirical 1-D distributions.

    Implementation: ``scipy.stats.wasserstein_distance`` (Virtanen et al.
    2020). See Vaserstein (1969).
    """
    from scipy.stats import wasserstein_distance
    return float(wasserstein_distance(a, b))


__all__ = [
    "adjust_pvalues",
    "cohens_d",
    "cliffs_delta",
    "odds_ratio_with_ci",
    "confusion_matrix_long",
    "per_class_prf1",
    "matthews_corrcoef",
    "balanced_accuracy",
    "top_k_ordinal_accuracy",
    "rank_distance_distribution",
    "kl_divergence",
    "jensen_shannon",
    "wasserstein",
]
