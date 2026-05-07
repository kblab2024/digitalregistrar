"""Canonical eight-table statistics suite from a long-form atomic table.

The atomic table is the single source of truth: one row per
``(case_id, organ, method, run, field)``, with
``correct, attempted, gold_present, case_status, case_flags,
field_status, field_error_detail`` columns.

Every output below is one groupby on this table plus a call into
``benchmarks.eval.ci`` and statsmodels. The suite is method-agnostic —
it accepts ablation grids, unified atomics, or any DataFrame matching
the schema. The ``modular_method`` argument names the comparator.

Outputs land under ``out_dir`` as eight CSVs plus a markdown run report
``canonical_stats_report.md``:

    headline.csv                 head-to-head method comparison
    failure_modes.csv            defect taxonomy by method
    per_field.csv                per-field accuracy decomposition
    per_organ.csv                per-organ accuracy decomposition
    seed_consistency.csv         multi-run robustness (only if N runs > 1)
    modularity_advantage.csv     modular vs best alternative method
    low_performer_diagnostics.csv  error-source decomposition for fields < 0.90
    canonical_stats_report.md    self-explaining run report

Tests apply Wilson 95% CIs on rates, paired-bootstrap CIs on
between-method deltas, McNemar paired tests with Holm correction,
Cochran's Q heterogeneity test for organ-level comparisons, and Fleiss
κ for cross-seed agreement. Every test sits inside a try/except that
emits an undefined-row with a ``note`` rather than raising on
degenerate input.
"""
from __future__ import annotations

import datetime as dt
import logging
import math
import platform
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...benchmarks.eval.ci import (
    mcnemar_test,
    paired_bootstrap_diff,
    wilson_ci,
)
from ...benchmarks.eval.scope import STATS_EXCLUDED_FIELDS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# p-value adjustment (Holm-Bonferroni)
# ---------------------------------------------------------------------------

def _adjust_pvalues(p_values: Sequence[float],
                    method: str = "holm") -> list[float]:
    """Holm-Bonferroni p-value adjustment, NaN-tolerant.

    Returns the input unchanged (with a warning) if statsmodels is not
    available, so the suite still completes.
    """
    try:
        from statsmodels.stats.multitest import multipletests
    except ImportError:
        logger.warning("statsmodels not installed — p-values uncorrected.")
        return list(p_values)
    p = np.asarray(p_values, dtype=float)
    finite = ~np.isnan(p)
    out = np.full_like(p, np.nan)
    if not finite.any():
        return out.tolist()
    _, p_adj, _, _ = multipletests(p[finite], method=method)
    out[finite] = p_adj
    return out.tolist()


# ---------------------------------------------------------------------------
# Effect-size helpers
# ---------------------------------------------------------------------------

def _cohen_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for difference between two proportions."""
    if math.isnan(p1) or math.isnan(p2):
        return float("nan")
    p1 = min(max(p1, 0.0), 1.0)
    p2 = min(max(p2, 0.0), 1.0)
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def _odds_ratio_with_ci(a: int, b: int, c: int, d: int,
                        alpha: float = 0.05) -> tuple[float, float, float]:
    """Odds ratio + Wald 95% CI for a 2×2 table.

    ``a, b, c, d`` form ``[[a, b], [c, d]]``. Returns ``(or, lo, hi)``;
    NaN on degenerate (zero-cell) input.
    """
    if min(a, b, c, d) == 0:
        # Apply Haldane-Anscombe correction for zero cells.
        a = a + 0.5
        b = b + 0.5
        c = c + 0.5
        d = d + 0.5
    if b * c == 0 or a * d == 0:
        return (float("nan"), float("nan"), float("nan"))
    or_hat = (a * d) / (b * c)
    se_log = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    try:
        from scipy.stats import norm
        z = float(norm.ppf(1 - alpha / 2))
    except Exception:
        z = 1.96
    log_or = math.log(or_hat)
    return (or_hat, math.exp(log_or - z * se_log), math.exp(log_or + z * se_log))


# ---------------------------------------------------------------------------
# Per-method scoreable subsets
# ---------------------------------------------------------------------------

def _scoreable_mask(df: pd.DataFrame) -> pd.Series:
    """Rows that are eligible for accuracy scoring.

    A row is eligible when ``case_status`` denotes a gradable case AND
    the field is not skipped. We KEEP defective cases in the eligible
    set per the project decision "defects count as wrong"; their
    correctness is False (correct=None → coerced to 0 for accuracy).
    """
    if "case_status" in df.columns:
        return df["case_status"] != "skipped_intentional"
    return pd.Series([True] * len(df), index=df.index)


def _accuracy_pair(df_method: pd.DataFrame) -> tuple[int, int, int, int]:
    """Return (n_correct, n_attempted, n_eligible, n_defective) for one method.

    n_eligible counts rows where the case is not skipped_intentional
    (defective cases included — they count as wrong).
    n_correct counts rows where ``correct`` is True (as a Python bool
    or 1.0; nested-field f1>=1 also counts as correct).
    n_defective counts rows whose case_status is not 'ok' nor
    'skipped_intentional'.
    """
    eligible = df_method[_scoreable_mask(df_method)]
    n_eligible = len(eligible)
    correct = eligible["correct"]
    # ``correct`` may be bool, float (nested f1), or NaN/None.
    n_correct = int(((correct == True) | (correct == 1.0) |  # noqa: E712
                     (pd.to_numeric(correct, errors="coerce") >= 1.0)).sum())
    n_attempted = int((eligible["attempted"] == True).sum())  # noqa: E712
    if "case_status" in eligible.columns:
        n_defective = int(eligible["case_status"]
                          .isin(["pipeline_error", "parse_error",
                                 "schema_error", "renest_error",
                                 "b2_parse_error", "section_error",
                                 "prediction_unreadable"]).sum())
    else:
        n_defective = 0
    return n_correct, n_attempted, n_eligible, n_defective


# ---------------------------------------------------------------------------
# Headline
# ---------------------------------------------------------------------------

def _headline(atomic: pd.DataFrame, modular_method: str,
              *, device: str = "cpu") -> pd.DataFrame:
    """Per-method headline: rates, deltas vs modular, McNemar, OR.

    One row per ``method``. Rates: Wilson CIs. Deltas: paired-bootstrap.
    p-values: Holm-adjusted McNemar across all methods.
    """
    if "method" not in atomic.columns or atomic.empty:
        return pd.DataFrame()
    methods = sorted(atomic["method"].dropna().unique().tolist())
    rows: list[dict] = []

    # Build a paired-correctness pivot: index=(case_id, organ, field, run)
    # if 'run' exists else (case_id, organ, field). Columns=method.
    key_cols = [c for c in ("case_id", "organ", "field", "run")
                if c in atomic.columns]
    paired = (atomic[key_cols + ["method", "correct"]]
              .copy())
    paired["acc01"] = pd.to_numeric(paired["correct"], errors="coerce")
    paired_pivot = paired.pivot_table(
        index=key_cols, columns="method", values="acc01",
        aggfunc="first")

    p_uncorrected: list[float] = []
    base_present = modular_method in paired_pivot.columns
    for m in methods:
        m_df = atomic[atomic["method"] == m]
        n_correct, n_attempted, n_eligible, n_defective = _accuracy_pair(m_df)
        n_skipped = (
            int((m_df["case_status"] == "skipped_intentional").sum())
            if "case_status" in m_df.columns else 0)
        n_total = n_eligible + n_skipped

        accuracy_eff = (n_correct / n_eligible) if n_eligible else float("nan")
        eff_lo, eff_hi = (wilson_ci(n_correct, n_eligible)
                          if n_eligible else (float("nan"), float("nan")))
        accuracy_att = (n_correct / n_attempted) if n_attempted else float("nan")
        att_lo, att_hi = (wilson_ci(n_correct, n_attempted)
                          if n_attempted else (float("nan"), float("nan")))
        defect_denom = n_eligible  # excludes skipped already
        defect_rate = (n_defective / defect_denom) if defect_denom else float("nan")
        d_lo, d_hi = (wilson_ci(n_defective, defect_denom)
                      if defect_denom else (float("nan"), float("nan")))
        cov = (n_attempted / n_eligible) if n_eligible else float("nan")
        c_lo, c_hi = (wilson_ci(n_attempted, n_eligible)
                      if n_eligible else (float("nan"), float("nan")))

        # vs modular
        delta = float("nan")
        delta_lo = float("nan")
        delta_hi = float("nan")
        mcn_p = float("nan")
        cohen_h = float("nan")
        or_hat = or_lo = or_hi = float("nan")
        note = ""
        if base_present and m != modular_method and m in paired_pivot.columns:
            try:
                a = paired_pivot[m].to_numpy(dtype=float)
                b = paired_pivot[modular_method].to_numpy(dtype=float)
                mask = ~(np.isnan(a) | np.isnan(b))
                a_v = a[mask]
                b_v = b[mask]
                if a_v.size:
                    from digital_registrar_research.benchmarks.eval import (
                        ci_gpu,
                    )
                    pb = ci_gpu.paired_bootstrap_diff(
                        a_v, b_v, device=device,
                    )
                    delta, delta_lo, delta_hi = (pb.point, pb.lo,
                                                 pb.hi)
                    # McNemar: b = method=1 & modular=0; c = inverse
                    bb = int(((a_v >= 1.0) & (b_v < 1.0)).sum())
                    cc = int(((a_v < 1.0) & (b_v >= 1.0)).sum())
                    mcn = mcnemar_test(bb, cc)
                    mcn_p = mcn["p_value"]
                    # Cohen's h on attempted-rate proportions
                    p_m = float(np.nanmean(a_v >= 1.0))
                    p_b = float(np.nanmean(b_v >= 1.0))
                    cohen_h = _cohen_h(p_m, p_b)
                    # Odds ratio: 2x2 of (method correct y/n) x (modular y/n)
                    ay = int(((a_v >= 1.0) & (b_v >= 1.0)).sum())
                    bn = int(((a_v >= 1.0) & (b_v < 1.0)).sum())
                    cn = int(((a_v < 1.0) & (b_v >= 1.0)).sum())
                    dy = int(((a_v < 1.0) & (b_v < 1.0)).sum())
                    or_hat, or_lo, or_hi = _odds_ratio_with_ci(ay, bn, cn, dy)
                else:
                    note = "undefined: no paired non-NaN cases vs modular"
            except (ValueError, np.linalg.LinAlgError) as exc:
                note = f"undefined: {type(exc).__name__}: {exc}"[:120]
        elif not base_present:
            note = f"modular_method '{modular_method}' not present in atomic"

        p_uncorrected.append(mcn_p)
        rows.append({
            "method": m,
            "n_total": n_total,
            "n_eligible": n_eligible,
            "n_attempted": n_attempted,
            "n_correct": n_correct,
            "n_defective": n_defective,
            "accuracy_eff": accuracy_eff,
            "accuracy_eff_lo": eff_lo,
            "accuracy_eff_hi": eff_hi,
            "accuracy_att": accuracy_att,
            "accuracy_att_lo": att_lo,
            "accuracy_att_hi": att_hi,
            "defect_rate": defect_rate,
            "defect_rate_lo": d_lo,
            "defect_rate_hi": d_hi,
            "coverage": cov,
            "coverage_lo": c_lo,
            "coverage_hi": c_hi,
            "delta_vs_modular": delta,
            "delta_lo": delta_lo,
            "delta_hi": delta_hi,
            "mcnemar_p": mcn_p,
            "cohen_h": cohen_h,
            "odds_ratio": or_hat,
            "odds_ratio_lo": or_lo,
            "odds_ratio_hi": or_hi,
            "note": note,
        })
    df = pd.DataFrame(rows)
    df["mcnemar_p_holm"] = _adjust_pvalues(df["mcnemar_p"].tolist())
    return df


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------

def _failure_modes(atomic: pd.DataFrame) -> pd.DataFrame:
    """Per-(method, field_status) counts and rates with Wilson CIs.

    Denominator excludes ``skipped_intentional``.
    """
    if "field_status" not in atomic.columns or atomic.empty:
        return pd.DataFrame()
    df = atomic.copy()
    if "case_status" in df.columns:
        df = df[df["case_status"] != "skipped_intentional"]
    if df.empty:
        return pd.DataFrame()
    counts = (df.groupby(["method", "field_status"]).size()
              .rename("n").reset_index())
    totals = (df.groupby("method").size().rename("n_total").reset_index())
    out = counts.merge(totals, on="method")
    out["fraction"] = out["n"] / out["n_total"]
    lo: list[float] = []
    hi: list[float] = []
    for _, r in out.iterrows():
        a, b = wilson_ci(int(r["n"]), int(r["n_total"]))
        lo.append(a)
        hi.append(b)
    out["fraction_lo"] = lo
    out["fraction_hi"] = hi
    return out[["method", "field_status", "n", "n_total",
                "fraction", "fraction_lo", "fraction_hi"]]


# ---------------------------------------------------------------------------
# Per-field
# ---------------------------------------------------------------------------

def _per_field(atomic: pd.DataFrame, modular_method: str,
               *, device: str = "cpu") -> pd.DataFrame:
    """Per-(method, field) accuracy and Δ vs modular.

    Filters out STATS_EXCLUDED_FIELDS before any test runs. Holm
    correction is applied across the whole (method × field) grid.
    """
    if atomic.empty:
        return pd.DataFrame()
    df = atomic[_scoreable_mask(atomic)]
    df = df[~df["field"].isin(STATS_EXCLUDED_FIELDS)]
    if df.empty:
        return pd.DataFrame()
    df = df.assign(acc01=pd.to_numeric(df["correct"],
                                       errors="coerce").fillna(0.0))

    methods = sorted(df["method"].dropna().unique().tolist())
    fields = sorted(df["field"].dropna().unique().tolist())

    # Pivot for paired comparisons.
    key_cols = [c for c in ("case_id", "organ", "run") if c in df.columns]
    pivot = (df.pivot_table(index=key_cols + ["field"], columns="method",
                            values="acc01", aggfunc="first"))

    rows: list[dict] = []
    p_uncorrected: list[float] = []
    base_present = modular_method in pivot.columns
    for m in methods:
        for f in fields:
            sub = df[(df["method"] == m) & (df["field"] == f)]
            n = len(sub)
            if not n:
                continue
            correct_count = int((sub["acc01"] >= 1.0).sum())
            attempted_count = int((sub["attempted"] == True).sum())  # noqa: E712
            acc = correct_count / n if n else float("nan")
            lo, hi = wilson_ci(correct_count, n) if n else (float("nan"),) * 2

            delta = delta_lo = delta_hi = float("nan")
            mcn_p = float("nan")
            note = ""
            if base_present and m != modular_method:
                try:
                    sub_pivot = pivot.xs(f, level="field")
                    if (m in sub_pivot.columns
                            and modular_method in sub_pivot.columns):
                        a = sub_pivot[m].to_numpy(dtype=float)
                        b = sub_pivot[modular_method].to_numpy(dtype=float)
                        mask = ~(np.isnan(a) | np.isnan(b))
                        a_v = a[mask]
                        b_v = b[mask]
                        if a_v.size:
                            from digital_registrar_research.benchmarks.eval \
                                import ci_gpu
                            pb = ci_gpu.paired_bootstrap_diff(
                                a_v, b_v, device=device,
                            )
                            delta, delta_lo, delta_hi = (pb.point,
                                                         pb.lo, pb.hi)
                            bb = int(((a_v >= 1.0) & (b_v < 1.0)).sum())
                            cc = int(((a_v < 1.0) & (b_v >= 1.0)).sum())
                            mcn_p = mcnemar_test(bb, cc)["p_value"]
                        else:
                            note = "undefined: no paired non-NaN cases"
                except (KeyError, ValueError,
                        np.linalg.LinAlgError) as exc:
                    note = f"undefined: {type(exc).__name__}: {exc}"[:120]
            p_uncorrected.append(mcn_p)
            rows.append({
                "method": m, "field": f,
                "n": n,
                "n_correct": correct_count,
                "n_attempted": attempted_count,
                "accuracy": acc,
                "accuracy_lo": lo,
                "accuracy_hi": hi,
                "delta_vs_modular": delta,
                "delta_lo": delta_lo,
                "delta_hi": delta_hi,
                "mcnemar_p": mcn_p,
                "note": note,
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["mcnemar_p_holm"] = _adjust_pvalues(out["mcnemar_p"].tolist())
    return out


# ---------------------------------------------------------------------------
# Per-organ
# ---------------------------------------------------------------------------

def _per_organ(atomic: pd.DataFrame, modular_method: str,
               *, device: str = "cpu") -> pd.DataFrame:
    """Per-(method, organ) accuracy plus per-method Cochran's Q.

    Cochran's Q tests heterogeneity of accuracy across organs for a
    given method (compared point-wise to the modular baseline).
    """
    if atomic.empty or "organ" not in atomic.columns:
        return pd.DataFrame()
    df = atomic[_scoreable_mask(atomic)]
    df = df[~df["field"].isin(STATS_EXCLUDED_FIELDS)]
    if df.empty:
        return pd.DataFrame()
    df = df.assign(acc01=pd.to_numeric(df["correct"],
                                       errors="coerce").fillna(0.0))
    rows: list[dict] = []
    for (m, organ), sub in df.groupby(["method", "organ"]):
        n = len(sub)
        if not n:
            continue
        correct_count = int((sub["acc01"] >= 1.0).sum())
        acc = correct_count / n if n else float("nan")
        lo, hi = wilson_ci(correct_count, n) if n else (float("nan"),) * 2
        rows.append({
            "method": m, "organ": organ,
            "n": n, "n_correct": correct_count,
            "accuracy": acc, "accuracy_lo": lo, "accuracy_hi": hi,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Cochran's Q per method across organs (vs the per-organ accuracy
    # of modular).
    if modular_method in out["method"].unique():
        modular_acc = (out[out["method"] == modular_method]
                       .set_index("organ")["accuracy"])
        cq_rows = []
        for m, sub in out.groupby("method"):
            if m == modular_method:
                continue
            try:
                deltas = sub.set_index("organ")["accuracy"] - modular_acc
                q_stat, q_p = _cochran_q(deltas.dropna().to_numpy())
                cq_rows.append({"method": m, "cochran_q": q_stat,
                                "cochran_q_p": q_p})
            except Exception as exc:
                cq_rows.append({
                    "method": m, "cochran_q": float("nan"),
                    "cochran_q_p": float("nan"),
                    "note": f"undefined: {type(exc).__name__}"[:120],
                })
        if cq_rows:
            cq = pd.DataFrame(cq_rows)
            out = out.merge(cq, on="method", how="left")
    return out


def _cochran_q(deltas: np.ndarray) -> tuple[float, float]:
    """Lightweight Q statistic for organ-level heterogeneity of deltas.

    Approximates Cochran's Q via χ² on the observed deltas treating
    each organ as one trial. NaN-tolerant.
    """
    if deltas.size < 2:
        return (float("nan"), float("nan"))
    mean_delta = float(np.mean(deltas))
    if mean_delta == 0:
        return (0.0, 1.0)
    q = float(((deltas - mean_delta) ** 2).sum() / abs(mean_delta + 1e-12))
    try:
        from scipy.stats import chi2
        p = float(chi2.sf(q, df=deltas.size - 1))
    except Exception:
        p = float("nan")
    return q, p


# ---------------------------------------------------------------------------
# Seed consistency
# ---------------------------------------------------------------------------

def _seed_consistency(atomic: pd.DataFrame) -> pd.DataFrame:
    """Per-(method, field) accuracy variance across runs, and Fleiss κ.

    Only meaningful when ``run`` has more than one unique value per
    method.
    """
    if atomic.empty or "run" not in atomic.columns:
        return pd.DataFrame()
    df = atomic[_scoreable_mask(atomic)]
    df = df[~df["field"].isin(STATS_EXCLUDED_FIELDS)]
    if df.empty:
        return pd.DataFrame()
    df = df.assign(acc01=pd.to_numeric(df["correct"],
                                       errors="coerce").fillna(0.0))
    rows: list[dict] = []
    for (m, f), sub in df.groupby(["method", "field"]):
        runs = sub["run"].nunique()
        if runs <= 1:
            continue
        per_run = (sub.groupby("run")["acc01"].mean())
        rows.append({
            "method": m, "field": f,
            "n_seeds": int(runs),
            "accuracy_mean": float(per_run.mean()),
            "accuracy_sd": float(per_run.std(ddof=0)),
            "accuracy_min": float(per_run.min()),
            "accuracy_max": float(per_run.max()),
            "fleiss_kappa": _fleiss_kappa_per_field(sub),
        })
    return pd.DataFrame(rows)


def _fleiss_kappa_per_field(sub: pd.DataFrame) -> float:
    """Fleiss κ on binary (correct/wrong) labels across runs.

    Each subject = (case_id, organ); each rater = run. Returns NaN on
    degenerate input.
    """
    try:
        if "case_id" not in sub.columns:
            return float("nan")
        labels = (sub.assign(label=(sub["acc01"] >= 1.0).astype(int))
                  .pivot_table(index=["case_id", "organ"]
                                if "organ" in sub.columns else ["case_id"],
                               columns="run", values="label",
                               aggfunc="first"))
        if labels.shape[0] < 2 or labels.shape[1] < 2:
            return float("nan")
        labels = labels.dropna()
        if labels.empty:
            return float("nan")
        n_subjects, n_raters = labels.shape
        # 2 categories: 0 / 1
        n0 = (labels == 0).sum(axis=1).to_numpy(dtype=float)
        n1 = (labels == 1).sum(axis=1).to_numpy(dtype=float)
        p_i = (n0 ** 2 + n1 ** 2 - n_raters) / (n_raters * (n_raters - 1))
        p_bar = float(p_i.mean())
        p_e = ((n0.sum() / (n_subjects * n_raters)) ** 2
               + (n1.sum() / (n_subjects * n_raters)) ** 2)
        if p_e >= 1.0:
            return float("nan")
        return (p_bar - p_e) / (1 - p_e)
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Modularity advantage
# ---------------------------------------------------------------------------

def _modularity_advantage(atomic: pd.DataFrame,
                          modular_method: str,
                          *, device: str = "cpu") -> pd.DataFrame:
    """Modular accuracy vs best-alternative accuracy, per field plus ALL.

    For each field, identifies the highest-accuracy non-modular method
    and reports its accuracy alongside the modular accuracy with a
    paired-bootstrap CI on the difference.
    """
    if atomic.empty:
        return pd.DataFrame()
    df = atomic[_scoreable_mask(atomic)]
    df = df[~df["field"].isin(STATS_EXCLUDED_FIELDS)]
    if df.empty or modular_method not in df["method"].unique():
        return pd.DataFrame()
    df = df.assign(acc01=pd.to_numeric(df["correct"],
                                       errors="coerce").fillna(0.0))
    fields = sorted(df["field"].dropna().unique().tolist())

    rows: list[dict] = []
    p_uncorrected: list[float] = []
    for f in fields + ["ALL"]:
        sub = df if f == "ALL" else df[df["field"] == f]
        if sub.empty:
            continue
        per_method = sub.groupby("method")["acc01"].agg(["sum", "size"])
        if modular_method not in per_method.index:
            continue
        modular_n = int(per_method.loc[modular_method, "size"])
        modular_correct = int(per_method.loc[modular_method, "sum"])
        modular_acc = (modular_correct / modular_n) if modular_n else float("nan")
        m_lo, m_hi = (wilson_ci(modular_correct, modular_n)
                      if modular_n else (float("nan"), float("nan")))

        alternatives = per_method.drop(modular_method, errors="ignore")
        if alternatives.empty:
            continue
        alt_acc = (alternatives["sum"] / alternatives["size"]).fillna(0.0)
        best_method = alt_acc.idxmax()
        best_n = int(alternatives.loc[best_method, "size"])
        best_correct = int(alternatives.loc[best_method, "sum"])
        best_acc = (best_correct / best_n) if best_n else float("nan")
        b_lo, b_hi = (wilson_ci(best_correct, best_n)
                      if best_n else (float("nan"), float("nan")))

        # Paired CI on (modular - best) restricted to (case_id, organ, run, field)
        delta = delta_lo = delta_hi = float("nan")
        mcn_p = float("nan")
        note = ""
        try:
            key_cols = [c for c in ("case_id", "organ", "run")
                        if c in sub.columns]
            piv = (sub[(sub["method"].isin([modular_method, best_method]))]
                   .pivot_table(index=key_cols + (["field"]
                                                   if f == "ALL" else []),
                                columns="method", values="acc01",
                                aggfunc="first"))
            if (modular_method in piv.columns
                    and best_method in piv.columns):
                a = piv[modular_method].to_numpy(dtype=float)
                b = piv[best_method].to_numpy(dtype=float)
                mask = ~(np.isnan(a) | np.isnan(b))
                a_v = a[mask]
                b_v = b[mask]
                if a_v.size:
                    from digital_registrar_research.benchmarks.eval import (
                        ci_gpu,
                    )
                    pb = ci_gpu.paired_bootstrap_diff(
                        a_v, b_v, device=device,
                    )
                    delta, delta_lo, delta_hi = (pb.point, pb.lo,
                                                 pb.hi)
                    bb = int(((a_v >= 1.0) & (b_v < 1.0)).sum())
                    cc = int(((a_v < 1.0) & (b_v >= 1.0)).sum())
                    mcn_p = mcnemar_test(bb, cc)["p_value"]
                else:
                    note = "undefined: no paired non-NaN cases"
        except (ValueError, np.linalg.LinAlgError) as exc:
            note = f"undefined: {type(exc).__name__}: {exc}"[:120]
        p_uncorrected.append(mcn_p)
        rows.append({
            "field": f,
            "modular_method": modular_method,
            "modular_accuracy": modular_acc,
            "modular_lo": m_lo,
            "modular_hi": m_hi,
            "best_alternative_method": best_method,
            "best_alternative_accuracy": best_acc,
            "best_alternative_lo": b_lo,
            "best_alternative_hi": b_hi,
            "modular_advantage": delta,
            "advantage_lo": delta_lo,
            "advantage_hi": delta_hi,
            "p_value": mcn_p,
            "note": note,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_value_holm"] = _adjust_pvalues(out["p_value"].tolist())
    return out


# ---------------------------------------------------------------------------
# Low-performer diagnostics
# ---------------------------------------------------------------------------

def _low_performer_diagnostics(atomic: pd.DataFrame,
                               modular_method: str,
                               threshold: float = 0.90) -> pd.DataFrame:
    """For each modular-method field with accuracy < threshold, decompose
    the error rate into model-silent vs attempted-but-wrong components.

    ``field_missing_rate`` answers "rate of model not producing a value".
    ``wrong_when_attempted_rate`` answers "rate of wrong value when
    model did produce one". Together they distinguish source-document
    silence from model reasoning errors.
    """
    if atomic.empty or modular_method not in atomic["method"].unique():
        return pd.DataFrame()
    df = atomic[atomic["method"] == modular_method]
    df = df[_scoreable_mask(df)]
    df = df[~df["field"].isin(STATS_EXCLUDED_FIELDS)]
    if df.empty:
        return pd.DataFrame()
    df = df.assign(acc01=pd.to_numeric(df["correct"],
                                       errors="coerce").fillna(0.0))
    rows: list[dict] = []
    for f, sub in df.groupby("field"):
        n = len(sub)
        if not n:
            continue
        correct_count = int((sub["acc01"] >= 1.0).sum())
        acc = correct_count / n
        if acc >= threshold:
            continue
        lo, hi = wilson_ci(correct_count, n)
        # Field-status decomposition.
        if "field_status" in sub.columns:
            n_correct = int((sub["field_status"] == "correct").sum())
            n_wrong = int(sub["field_status"].isin(
                ["wrong_value", "wrong_type", "misaligned_list"]).sum())
            n_field_missing = int(sub["field_status"].isin(
                ["missing_key", "null_value"]).sum())
            n_parse = int(sub["field_status"].isin(
                ["unscoreable_due_to_case_error"]).sum())
        else:
            n_correct = correct_count
            n_wrong = n - correct_count
            n_field_missing = 0
            n_parse = 0
        attempted = n_correct + n_wrong
        rows.append({
            "field": f,
            "n": n,
            "accuracy": acc,
            "accuracy_lo": lo,
            "accuracy_hi": hi,
            "n_correct": n_correct,
            "n_wrong": n_wrong,
            "n_field_missing": n_field_missing,
            "n_parse_error": n_parse,
            "field_missing_rate": (n_field_missing + n_parse) / n,
            "wrong_when_attempted_rate": (n_wrong / attempted
                                          if attempted else float("nan")),
        })
    return pd.DataFrame(rows).sort_values("accuracy", na_position="last")


# ---------------------------------------------------------------------------
# Run report
# ---------------------------------------------------------------------------

def _markdown_table(df: pd.DataFrame, max_rows: int = 5) -> str:
    """Format a DataFrame head as a markdown table."""
    if df.empty:
        return "_(empty)_"
    return df.head(max_rows).to_markdown(index=False, floatfmt=".4f")


def _git_sha(repo_root: Path) -> str:
    head = repo_root / ".git" / "HEAD"
    if not head.exists():
        return "unknown"
    try:
        ref = head.read_text(encoding="utf-8").strip()
        if ref.startswith("ref:"):
            ref_path = repo_root / ".git" / ref.split(maxsplit=1)[1]
            if ref_path.exists():
                return ref_path.read_text(encoding="utf-8").strip()[:12]
        return ref[:12]
    except Exception:
        return "unknown"


def _write_run_report(out_dir: Path,
                      atomic: pd.DataFrame,
                      modular_method: str,
                      csvs: dict[str, pd.DataFrame],
                      command_line: str | None = None,
                      ) -> Path:
    """Write a markdown report summarising what the canonical-stats
    invocation did and what to look at next.

    The report lives alongside the CSVs and is intended for hand-off:
    a future session can read it without re-deriving anything from
    code.
    """
    repo_root = Path(__file__).resolve().parents[3]
    git_sha = _git_sha(repo_root)
    timestamp = dt.datetime.utcnow().isoformat(timespec="seconds")
    cmd = command_line or " ".join(sys.argv)

    n_methods = atomic["method"].nunique() if "method" in atomic.columns else 0
    n_runs = atomic["run"].nunique() if "run" in atomic.columns else 0
    n_cases = (atomic["case_id"].nunique()
               if "case_id" in atomic.columns else 0)

    lines: list[str] = []
    lines.append("# Canonical statistics suite — run report")
    lines.append("")
    lines.append(f"**Generated (UTC):** {timestamp}")
    lines.append(f"**Git SHA:** `{git_sha}`")
    lines.append(f"**Host:** {platform.node()} ({platform.system()})")
    lines.append(f"**Command:** `{cmd}`")
    lines.append(f"**Output directory:** `{out_dir}`")
    lines.append("")
    lines.append("## Inputs")
    lines.append(f"- Atomic table rows: **{len(atomic)}**")
    lines.append(f"- Methods present: **{n_methods}**")
    lines.append(f"- Distinct runs: **{n_runs}**")
    lines.append(f"- Distinct case_ids: **{n_cases}**")
    lines.append(f"- Comparator (modular method): **`{modular_method}`**"
                 + (" — present" if modular_method in atomic.get(
                     "method", pd.Series()).unique() else " — *MISSING*"))
    lines.append("")
    lines.append("## Pipeline")
    lines.append(
        "1. Loaded atomic table from caller (long-form, "
        "one row per (case_id, organ, method, run, field)).")
    lines.append(
        "2. Filtered out `skipped_intentional` cases from accuracy "
        "denominators; kept defective cases (they count as wrong per "
        "the project decision).")
    lines.append(
        "3. Computed eight canonical statistics tables.")
    lines.append("4. Holm-Bonferroni correction across all (method) "
                 "and (method × field) families.")
    lines.append(
        "5. Wrote CSVs and this report under `out_dir`.")
    lines.append("")

    lines.append("## Filters and policies")
    lines.append(f"- `STATS_EXCLUDED_FIELDS = "
                 f"{sorted(STATS_EXCLUDED_FIELDS)}` — filtered before "
                 "any per-field test.")
    lines.append(
        "- Defects-as-wrong: `case_status` ∈ "
        "{`pipeline_error, parse_error, schema_error, renest_error, "
        "b2_parse_error, section_error, prediction_unreadable`} "
        "rows kept in eligible denominator with `correct=None` → 0 "
        "for accuracy.")
    lines.append(
        "- `skipped_intentional` (`_skip_reason ∈ {not_cancer, "
        "unknown_organ}`) — excluded from all denominators.")
    lines.append("- Holm correction scope: across rows of `headline.csv`, "
                 "across (method × field) of `per_field.csv`, across "
                 "fields of `modularity_advantage.csv`.")
    lines.append("")

    lines.append("## Tests applied")
    lines.append(
        "- Wilson 95% CI on rates → `*_lo, *_hi` columns of "
        "`headline.csv`, `failure_modes.csv`, `per_field.csv`, "
        "`per_organ.csv`, `low_performer_diagnostics.csv`.")
    lines.append(
        "- Paired-bootstrap CI on between-method deltas → `delta_lo`, "
        "`delta_hi`, `advantage_lo`, `advantage_hi`.")
    lines.append(
        "- McNemar paired test on case-level correctness → `mcnemar_p`, "
        "`mcnemar_p_holm`, `p_value`, `p_value_holm`.")
    lines.append(
        "- Cochran's Q heterogeneity across organs → "
        "`per_organ.csv:cochran_q, cochran_q_p`.")
    lines.append(
        "- Fleiss κ across runs → `seed_consistency.csv:fleiss_kappa`.")
    lines.append(
        "- Cohen's h, odds ratio with Wald 95% CI → `headline.csv`.")
    lines.append("")

    lines.append("## Output index")
    descriptions = {
        "headline.csv": "Per-method head-to-head: accuracy, defect rate, "
                        "Δ vs modular, McNemar, OR, Cohen's h.",
        "failure_modes.csv": "Per-(method, field_status) defect taxonomy "
                             "with Wilson CIs.",
        "per_field.csv": "Per-(method, field) accuracy decomposition + "
                         "Δ vs modular (after STATS_EXCLUDED_FIELDS).",
        "per_organ.csv": "Per-(method, organ) accuracy + Cochran's Q "
                         "heterogeneity test.",
        "seed_consistency.csv": "Per-(method, field) cross-seed "
                                "accuracy SD + Fleiss κ. Only when "
                                "multi-run.",
        "modularity_advantage.csv": "Modular vs best alternative method "
                                    "per field plus ALL.",
        "low_performer_diagnostics.csv": "Modular fields with accuracy "
                                          "< 0.90; decomposes errors "
                                          "into model-silent vs "
                                          "attempted-wrong.",
    }
    for csv_name, df in csvs.items():
        if df is None:
            continue
        lines.append(f"### {csv_name}")
        lines.append(descriptions.get(csv_name, ""))
        lines.append(f"Rows: **{len(df)}**")
        if not df.empty:
            lines.append("")
            lines.append("Top rows:")
            lines.append("")
            try:
                lines.append(_markdown_table(df, max_rows=5))
            except Exception as exc:
                lines.append(f"_(could not render: {exc!r})_")
            lines.append("")

    lines.append("## Anomalies detected")
    anomalies: list[str] = []
    headline = csvs.get("headline.csv")
    if headline is not None and "defect_rate" in headline.columns:
        high_defect = headline[headline["defect_rate"] > 0.5]["method"].tolist()
        if high_defect:
            anomalies.append(
                f"- Methods with `defect_rate > 0.5`: "
                f"{sorted(high_defect)}.")
    fm = csvs.get("failure_modes.csv")
    if fm is not None and not fm.empty and "field_status" in fm.columns:
        for status in ("parse_error", "pipeline_error", "schema_error"):
            top = (fm[fm["field_status"] == status]
                   .sort_values("fraction", ascending=False).head(3))
            if not top.empty and float(top["fraction"].iloc[0]) > 0.1:
                pairs = [(r["method"], round(float(r["fraction"]), 3))
                         for _, r in top.iterrows()]
                anomalies.append(
                    f"- High `{status}` rate (top): {pairs}.")
    pf = csvs.get("per_field.csv")
    if pf is not None and not pf.empty and "note" in pf.columns:
        undef = pf[pf["note"].astype(str).str.startswith("undefined")]
        if not undef.empty:
            anomalies.append(
                f"- {len(undef)} (method, field) pairs had undefined "
                "tests — see `note` column in `per_field.csv`.")
    lp = csvs.get("low_performer_diagnostics.csv")
    if lp is not None and not lp.empty:
        names = lp["field"].tolist()
        anomalies.append(
            f"- Modular fields below 0.90: {names} "
            "(see `low_performer_diagnostics.csv` for source-vs-model "
            "decomposition).")
    if not anomalies:
        anomalies.append("- None detected.")
    lines.extend(anomalies)
    lines.append("")

    lines.append("## Suggested follow-ups")
    lines.append(
        "- Inspect `low_performer_diagnostics.csv` columns "
        "`field_missing_rate` (model silent on field) vs "
        "`wrong_when_attempted_rate` (model reasoning error) to "
        "decompose error sources.")
    if n_runs > 1:
        lines.append(
            f"- Multi-run detected ({n_runs} runs). See "
            "`seed_consistency.csv` for fields with `accuracy_sd > 0.05`.")
    lines.append(
        "- For methods with high `defect_rate`, see `failure_modes.csv` "
        "to identify which `field_status` dominates.")
    lines.append(
        "- If a method is absent from a CSV, check `Anomalies` above "
        "and the run-level `_summary.json` under that method's "
        "results dir.")
    lines.append("")

    report_path = out_dir / "canonical_stats_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_canonical_stats(atomic: pd.DataFrame, modular_method: str,
                        out_dir: Path,
                        command_line: str | None = None,
                        device: str = "cpu") -> dict[str, Path]:
    """Compute the canonical eight-table statistics suite from an atomic
    long-form table and write CSVs + run report to ``out_dir``.

    Method-agnostic — accepts the ablation grid (``method`` column built
    as ``f"{cell}_{model}"``) or any unified atomic. ``modular_method``
    is the comparator. Returns a dict of csv_name → output path.

    ``device`` (``"cpu" | "cuda" | "mps" | "auto"``) routes the paired-
    bootstrap and McNemar primitives through :mod:`ci_gpu`. Default is
    ``"cpu"`` (the safety-net path through :mod:`ci`); explicit GPU
    backends accelerate the per-(method, field) bootstrap loops in
    ``_per_field`` and ``_modularity_advantage``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csvs: dict[str, pd.DataFrame] = {}
    csvs["headline.csv"] = _headline(atomic, modular_method, device=device)
    csvs["failure_modes.csv"] = _failure_modes(atomic)
    csvs["per_field.csv"] = _per_field(atomic, modular_method, device=device)
    csvs["per_organ.csv"] = _per_organ(atomic, modular_method, device=device)
    if "run" in atomic.columns and (
            atomic.groupby("method")["run"].nunique().max() > 1
            if not atomic.empty and "method" in atomic.columns else False):
        csvs["seed_consistency.csv"] = _seed_consistency(atomic)
    csvs["modularity_advantage.csv"] = _modularity_advantage(
        atomic, modular_method, device=device)
    csvs["low_performer_diagnostics.csv"] = _low_performer_diagnostics(
        atomic, modular_method)

    written: dict[str, Path] = {}
    for name, df in csvs.items():
        path = out_dir / name
        df.to_csv(path, index=False)
        written[name] = path
        print(f"Wrote {path}  ({len(df)} rows)")

    report = _write_run_report(out_dir, atomic, modular_method, csvs,
                               command_line=command_line)
    written["canonical_stats_report.md"] = report
    print(f"Wrote {report}")
    return written


__all__ = ["run_canonical_stats"]
