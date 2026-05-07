"""Multiple-comparison adjustments for the cascade.

Wraps :func:`statsmodels.stats.multitest.multipletests` for the
canonical methods (Holm, BH-FDR, BY-FDR, Sidak) and adds a
permutation-based omnibus test for the "is *any* of these field-level
deltas significant" question.

References
----------
- Holm (1979) "A simple sequentially rejective multiple test procedure,"
  Scand. J. Stat.
- Sidak (1967) "Rectangular confidence regions for the means of
  multivariate normal distributions," JASA.
- Benjamini & Hochberg (1995) "Controlling the false discovery rate,"
  JRSS-B.
- Benjamini & Yekutieli (2001) "The control of the false discovery rate
  in multiple testing under dependency," Ann. Stat.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _adjust(p_values: Sequence[float], method: str) -> list[float]:
    """Common backbone for every adjustment method.

    Wraps statsmodels' multipletests; preserves NaN passthrough.
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


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down. Strong family-wise error control."""
    return _adjust(p_values, "holm")


def sidak(p_values: Sequence[float]) -> list[float]:
    """Šidák correction. Less conservative than Bonferroni when tests
    are independent. Reported as a sanity cross-check on Holm."""
    return _adjust(p_values, "sidak")


def bh_fdr(p_values: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg FDR. Use on the per-field × per-organ
    exploratory grid where strong FWER would be too conservative."""
    return _adjust(p_values, "fdr_bh")


def by_fdr(p_values: Sequence[float]) -> list[float]:
    """Benjamini-Yekutieli FDR for dependent tests. Per-organ × per-field
    cells share denominators (case overlap), so dependence is real."""
    return _adjust(p_values, "fdr_by")


def adjust_pvalues(
    p_values: Sequence[float],
    method: str = "holm",
) -> list[float]:
    """Single-entry adjuster; delegates to ``holm | sidak | bh_fdr | by_fdr``.

    Method names accepted: ``"holm"``, ``"sidak"``, ``"fdr_bh"`` (BH),
    ``"fdr_by"`` (BY). Anything else passes through to statsmodels.
    """
    aliases = {"bh": "fdr_bh", "by": "fdr_by"}
    return _adjust(p_values, aliases.get(method, method))


def permutation_omnibus(
    correctness_a: np.ndarray | Sequence[Sequence[float]],
    correctness_b: np.ndarray | Sequence[Sequence[float]],
    *,
    n_permutations: int = 2000,
    random_state: int | None = 0,
) -> dict:
    """Cross-field omnibus permutation test for "is *any* delta real?"

    ``correctness_a`` and ``correctness_b`` are shape (n_cases, n_fields)
    correctness matrices for two methods. Statistic: maximum absolute
    per-field accuracy delta. Null distribution: per-case label flips
    between the two methods. Returns the observed max-delta and a
    permutation p-value.

    Useful as a guard against cherry-picking a single field after the
    fact: if the omnibus p > 0.05, no individual field delta should be
    claimed without a paragraph of caveats.
    """
    rng = np.random.default_rng(random_state)
    a = np.asarray(correctness_a, dtype=float)
    b = np.asarray(correctness_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("correctness matrices must have identical shape")
    n_cases, n_fields = a.shape
    if n_cases == 0 or n_fields == 0:
        return {"max_abs_delta": float("nan"), "p_value": float("nan"),
                "n_cases": n_cases, "n_fields": n_fields}

    obs = np.abs(np.nanmean(a, axis=0) - np.nanmean(b, axis=0))
    obs_max = float(np.nanmax(obs))

    null_max = np.empty(n_permutations, dtype=float)
    for k in range(n_permutations):
        # Per-case sign flip: for case i, randomly swap a[i,:] and b[i,:].
        flip = rng.integers(0, 2, size=n_cases).astype(bool)
        a_perm = np.where(flip[:, None], b, a)
        b_perm = np.where(flip[:, None], a, b)
        null_max[k] = float(np.nanmax(np.abs(
            np.nanmean(a_perm, axis=0) - np.nanmean(b_perm, axis=0)
        )))

    # One-sided: null_max >= obs_max.
    p = float((null_max >= obs_max).mean())
    return {
        "max_abs_delta": obs_max,
        "p_value": p,
        "n_cases": n_cases,
        "n_fields": n_fields,
        "n_permutations": n_permutations,
    }


__all__ = [
    "holm",
    "sidak",
    "bh_fdr",
    "by_fdr",
    "adjust_pvalues",
    "permutation_omnibus",
]
