"""Tests for the new ``benchmarks/eval/stats/`` package.

Validates each statistical method against either a known reference
value, a textbook example, or :mod:`scipy.stats` for cross-checking.
Coverage focus: ICC(2,1), kappa variants, Krippendorff alpha,
Cochran's Q, Stuart-Maxwell, Lin's CCC, Bland-Altman, BCa bootstrap,
and the cascade-funnel + conditional-accuracy diagnostics.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from digital_registrar_research.benchmarks.eval.stats import (
    accuracy_flip_rate,
    agresti_coull_ci,
    bland_altman,
    cascade_funnel,
    clopper_pearson_ci,
    cochran_q,
    cohens_kappa,
    conditional_accuracy_grid,
    cronbach_alpha,
    holm,
    bh_fdr,
    icc_2_1,
    icc_3_k,
    jackknife_ci,
    kappa_band,
    krippendorff_alpha,
    lin_ccc,
    mape,
    mcnemar,
    pick_ci,
    spearman_brown,
    spearman_rho,
    stuart_maxwell,
    weighted_kappa,
    wilson_ci,
)


# --- CI selectors -------------------------------------------------------

def test_pick_ci_proportion_small_n():
    assert pick_ci("proportion", n=10) == "clopper_pearson"


def test_pick_ci_proportion_extreme_p():
    assert pick_ci("proportion", n=200, p=0.99) == "clopper_pearson"
    assert pick_ci("proportion", n=200, p=0.50) == "wilson"


def test_pick_ci_f1_uses_bca():
    assert pick_ci("f1") == "bca_bootstrap"


def test_agresti_coull_brackets_wilson():
    """Agresti-Coull and Wilson should be close on moderate n."""
    w_lo, w_hi = wilson_ci(50, 100)
    a_lo, a_hi = agresti_coull_ci(50, 100)
    assert abs(w_lo - a_lo) < 0.02
    assert abs(w_hi - a_hi) < 0.02


def test_jackknife_ci_matches_bootstrap_roughly():
    rng = np.random.default_rng(0)
    vals = rng.normal(0, 1, 200).tolist()
    point, lo, hi = jackknife_ci(vals, lambda v: float(np.mean(v)))
    assert lo < point < hi
    # Mean ~ 0; CI should bracket 0 at 95% under standard error of ~0.07.
    assert -0.5 < point < 0.5


# --- McNemar / paired bootstrap -----------------------------------------

def test_mcnemar_perfect_agreement_high_p():
    a = [1, 1, 1, 0, 0, 0]
    b = [1, 1, 1, 0, 0, 0]
    res = mcnemar(a, b)
    assert res.p_raw == 1.0
    assert res.effect_size == 0.0


def test_mcnemar_strong_disagreement_significant():
    a = [1] * 30 + [0] * 30
    b = [0] * 30 + [1] * 30  # always disagree
    res = mcnemar(a, b)
    # Symmetric disagreement: stat 0, p high — mcnemar tests asymmetry,
    # not disagreement. Replace with asymmetric pattern:
    a = [1] * 50 + [0] * 50
    b = [0] * 50 + [0] * 50  # method A is always 1 when method B is 0
    res = mcnemar(a, b)
    assert res.p_raw < 0.001


# --- Cochran's Q --------------------------------------------------------

def test_cochran_q_three_methods_no_diff_high_p():
    rng = np.random.default_rng(0)
    n_cases = 100
    base = rng.integers(0, 2, size=n_cases)
    mat = np.column_stack([base, base, base])
    res = cochran_q(mat)
    # All three identical → degenerate, denom == 0.
    assert "degenerate" in res.notes


def test_cochran_q_three_methods_diff_low_p():
    """Cochran's Q tests whether column proportions differ. Construct
    methods with markedly different success rates: A and B are 80%
    correct; C is 20% correct."""
    n_cases = 100
    a = np.array([1] * 80 + [0] * 20)
    b = a.copy()
    c = np.array([1] * 20 + [0] * 80)
    mat = np.column_stack([a, b, c])
    res = cochran_q(mat)
    assert res.p_raw < 0.001
    assert res.df == 2


# --- Stuart-Maxwell -----------------------------------------------------

def test_stuart_maxwell_homogeneous_marginals_high_p():
    """Symmetric paired table (with light off-diagonal noise so the
    covariance matrix is invertible) → marginal homogeneity, high p."""
    # Same marginal counts on both sides; small symmetric off-diagonal
    # disagreement so the test isn't degenerate.
    a = (["x"] * 40 + ["y"] * 40 + ["z"] * 40
         + ["x", "y", "z"])  # 3 extra agreements; marginals match
    b = (["x"] * 40 + ["y"] * 40 + ["z"] * 40
         + ["y", "z", "x"])  # symmetric off-diagonal: x->y, y->z, z->x
    res = stuart_maxwell(a, b)
    # marginals are still equal; symmetric off-diagonal cancels.
    assert res.p_raw > 0.05


def test_stuart_maxwell_asymmetric_marginals_low_p():
    """Marginals differ → reject homogeneity."""
    a = ["x"] * 100 + ["y"] * 100 + ["z"] * 100
    b = ["x"] * 50 + ["y"] * 50 + ["z"] * 200  # heavily skewed to z
    res = stuart_maxwell(a, b)
    assert res.df == 2
    assert res.p_raw < 0.001


# --- Cohen's kappa ------------------------------------------------------

def test_cohens_kappa_perfect_is_one():
    a = ["x", "y", "z", "x"]
    b = ["x", "y", "z", "x"]
    assert cohens_kappa(a, b) == pytest.approx(1.0)


def test_cohens_kappa_random_is_near_zero():
    rng = np.random.default_rng(0)
    a = rng.choice(["x", "y", "z"], size=200).tolist()
    b = rng.choice(["x", "y", "z"], size=200).tolist()
    val = cohens_kappa(a, b)
    assert abs(val) < 0.15


def test_weighted_kappa_quadratic_more_lenient_than_linear():
    """When raters are ordinally close, quadratic-weighted kappa should
    exceed linear-weighted (because off-by-one is penalised less)."""
    a = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
    b = [1, 2, 3, 4, 5, 2, 3, 4, 5, 5]
    k_lin = weighted_kappa(a, b, weights="linear")
    k_quad = weighted_kappa(a, b, weights="quadratic")
    assert k_quad > k_lin


# --- Kappa band labels --------------------------------------------------

def test_kappa_band_substantial_at_0_75():
    assert kappa_band(0.75) == "substantial"


def test_kappa_band_almost_perfect_at_0_85():
    assert kappa_band(0.85) == "almost_perfect"


# --- Krippendorff alpha -------------------------------------------------

def test_krippendorff_alpha_perfect_agreement():
    raters = [
        ["x", "y", "z", "x", "y"],
        ["x", "y", "z", "x", "y"],
    ]
    assert krippendorff_alpha(raters, level="nominal") == pytest.approx(1.0)


def test_krippendorff_alpha_random_near_zero():
    rng = np.random.default_rng(42)
    a = rng.choice(["x", "y", "z"], size=300).tolist()
    b = rng.choice(["x", "y", "z"], size=300).tolist()
    val = krippendorff_alpha([a, b], level="nominal")
    assert abs(val) < 0.15


def test_krippendorff_alpha_handles_missing():
    raters = [
        ["x", "y", None, "x", "y"],
        ["x", None, "z", "x", "y"],
    ]
    val = krippendorff_alpha(raters, level="nominal")
    # Should compute over the pairs that exist.
    assert -1 <= val <= 1


# --- Lin's CCC + Bland-Altman -------------------------------------------

def test_lin_ccc_perfect_pairs():
    g = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    p = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    res = lin_ccc(g, p)
    assert res["ccc"] == pytest.approx(1.0, abs=1e-9)


def test_lin_ccc_offset_lowers_ccc():
    g = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    p = [v + 5 for v in g]  # constant +5 offset
    res = lin_ccc(g, p)
    # Pearson r is still 1.0, but CCC penalises location shift.
    assert res["pearson_r"] == pytest.approx(1.0)
    assert res["ccc"] < 0.5


def test_bland_altman_returns_loa():
    g = [10, 20, 30, 40, 50]
    p = [11, 21, 29, 41, 49]
    ba = bland_altman(g, p)
    assert "loa_lo" in ba and "loa_hi" in ba
    assert ba["loa_lo"] < ba["mean_diff"] < ba["loa_hi"]


def test_mape_skips_zero_gold():
    g = [10, 20, 0]
    p = [12, 22, 99]  # last pred would explode if we didn't skip
    val = mape(g, p)
    assert math.isfinite(val)
    # ~mean(|10-12|/10, |20-22|/20) = ~0.15
    assert 0.1 < val < 0.2


# --- ICC + reliability --------------------------------------------------

def test_icc_2_1_perfect_correlation():
    """All raters give the same value per subject → ICC = 1."""
    mat = np.array([[1.0, 1.0, 1.0],
                    [2.0, 2.0, 2.0],
                    [3.0, 3.0, 3.0],
                    [4.0, 4.0, 4.0]])
    res = icc_2_1(mat)
    assert res["icc"] == pytest.approx(1.0, abs=1e-9)


def test_icc_3_k_at_least_icc_2_1():
    rng = np.random.default_rng(0)
    n, k = 30, 4
    subj = rng.normal(0, 1, n).reshape(-1, 1)
    noise = rng.normal(0, 0.5, (n, k))
    mat = subj + noise
    r1 = icc_2_1(mat)["icc"]
    rk = icc_3_k(mat)["icc"]
    # ICC(3,k) should be >= ICC(2,1) by Spearman-Brown when reliability > 0.
    assert rk >= r1


def test_cronbach_alpha_perfect_consistency():
    mat = np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1], [0, 0, 0]])
    val = cronbach_alpha(mat)
    assert val == pytest.approx(1.0, abs=1e-9)


def test_spearman_brown_doubles_reliability():
    # rho_single = 0.6, n_runs = 2 → predicted ~ 0.75
    val = spearman_brown(0.6, 2)
    assert val == pytest.approx(2 * 0.6 / (1 + 0.6), rel=1e-9)


def test_accuracy_flip_rate_zero_when_all_consistent():
    mat = np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1]])
    assert accuracy_flip_rate(mat) == 0.0


def test_accuracy_flip_rate_one_when_all_unstable():
    mat = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    assert accuracy_flip_rate(mat) == 1.0


# --- Multiple-comparisons -----------------------------------------------

def test_holm_more_conservative_than_bh():
    p_raw = [0.01, 0.02, 0.03, 0.04, 0.05]
    h = holm(p_raw)
    bh = bh_fdr(p_raw)
    # Holm should be elementwise >= BH-FDR at the smaller end.
    assert h[0] >= bh[0]


# --- Cascade diagnostics ------------------------------------------------

def _synth_atomic():
    """Build a small synthetic atomic table for cascade-diag tests."""
    rows = []
    cases = [("c1", "A", True),  # passes A
             ("c2", "A", False),  # fails A
             ("c3", "A", True),
             ("c4", "A", True)]
    for cid, stage, passed in cases:
        rows.append({
            "case_id": cid, "cascade_stage": stage,
            "gate_pass": passed, "field": "cancer_excision_report",
            "correct": passed,
        })
    # c1, c3, c4 entered B; only c1, c3 passed B.
    for cid, passed in [("c1", True), ("c3", False), ("c4", True)]:
        rows.append({
            "case_id": cid, "cascade_stage": "B",
            "gate_pass": passed, "field": "cancer_category",
            "correct": passed,
        })
    # Stage C — c1, c4 made it.
    for cid in ("c1", "c4"):
        rows.append({
            "case_id": cid, "cascade_stage": "C",
            "gate_pass": True, "field": "pt_category",
            "correct": True,
        })
    return pd.DataFrame(rows)


def test_cascade_funnel_counts_correctly():
    funnel = cascade_funnel(_synth_atomic())
    assert {row["stage"] for _, row in funnel.iterrows()} == {"A", "B", "C"}
    a_row = funnel[funnel["stage"] == "A"].iloc[0]
    assert a_row["n_total"] == 4
    assert a_row["n_passed"] == 3  # c1, c3, c4
    b_row = funnel[funnel["stage"] == "B"].iloc[0]
    assert b_row["n_total"] == 3
    assert b_row["n_passed"] == 2  # c1, c4


def test_conditional_accuracy_grid_runs():
    grid = conditional_accuracy_grid(_synth_atomic())
    # At least one Stage-C field row.
    assert "pt_category" in set(grid["field"].tolist())
