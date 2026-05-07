"""Reviewer-grade statistics for the ablation aggregator.

Consumes the long-form ``ablation_grid.csv`` written by
:func:`run_ablations.main` and emits a suite of statistical tables that
the rebuttal lesion table and the supplementary factorial section can
draw from directly.

Reused primitives (do NOT reimplement here):

    * :func:`benchmarks.eval.ci.wilson_ci`
    * :func:`benchmarks.eval.ci.bootstrap_ci`
    * :func:`benchmarks.eval.ci.paired_bootstrap_diff`
    * :func:`benchmarks.eval.ci.mcnemar_test`
    * :func:`benchmarks.eval.ci.two_source_bootstrap_ci`
    * :func:`benchmarks.eval.multirun.per_field_ci`  (GLMM + bootstrap fallback)
    * :func:`benchmarks.eval.multirun.fleiss_kappa`

The Holm/BH wrapper, Cohen's d, Cliff's δ, and odds-ratio helpers are
inlined here because the canonical home for them
(``scripts/eval/_common/stats_extra.py``) is not on the package import
path. The implementations call the same backing libraries
(``statsmodels.stats.multitest``, ``scipy.stats``).

Top-level orchestration: :func:`run_all`.
"""
from __future__ import annotations

import json
import logging
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ...benchmarks.eval.ci import (
    bootstrap_ci,
    mcnemar_test,
    paired_bootstrap_diff,
    wilson_ci,
)
from ...benchmarks.eval.multirun import (
    fleiss_kappa,
    per_field_ci,
)
from ...benchmarks.eval.scope import FAIR_SCOPE
from ...paths import REPO_ROOT

logger = logging.getLogger(__name__)

DEFAULT_AXES_PATH = REPO_ROOT / "configs" / "ablations" / "axes.yaml"
DEFAULT_ENDPOINTS_PATH = REPO_ROOT / "configs" / "eval_endpoints.yaml"

GRID_CSV = "ablation_grid.csv"
EFFICIENCY_CSV = "efficiency.csv"

PRIMARY_BINARY_FIELDS = {
    "cancer_excision_report",
    "lymphovascular_invasion",
    "perineural_invasion",
    "biomarker_er", "biomarker_pr", "biomarker_her2",
}


# ---------------------------------------------------------------------------
# Inlined helpers (keep this file standalone — no scripts/ imports).
# ---------------------------------------------------------------------------

def _adjust_pvalues(p_values: Sequence[float], method: str = "holm"
                    ) -> list[float]:
    """Holm-Bonferroni / BH-FDR adjustment via statsmodels.

    Returns the input unchanged (with a warning) if statsmodels is not
    installed — keeps the rest of the stats pack flowing rather than
    aborting the whole aggregator.
    """
    try:
        from statsmodels.stats.multitest import multipletests
    except ImportError:
        logger.warning("statsmodels not installed — p-values left uncorrected. "
                       "Install with `pip install statsmodels`.")
        return list(p_values)

    p = np.asarray(p_values, dtype=float)
    finite = ~np.isnan(p)
    out = np.full_like(p, np.nan)
    if not finite.any():
        return out.tolist()
    _, p_adj, _, _ = multipletests(p[finite], method=method)
    out[finite] = p_adj
    return out.tolist()


def _cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    a_arr = a_arr[~np.isnan(a_arr)]
    b_arr = b_arr[~np.isnan(b_arr)]
    if a_arr.size < 2 or b_arr.size < 2:
        return float("nan")
    s_a = a_arr.std(ddof=1)
    s_b = b_arr.std(ddof=1)
    pooled = math.sqrt(((a_arr.size - 1) * s_a * s_a
                        + (b_arr.size - 1) * s_b * s_b)
                       / (a_arr.size + b_arr.size - 2))
    if pooled == 0:
        return float("nan")
    return float((a_arr.mean() - b_arr.mean()) / pooled)


def _cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    a_arr = a_arr[~np.isnan(a_arr)]
    b_arr = b_arr[~np.isnan(b_arr)]
    if a_arr.size == 0 or b_arr.size == 0:
        return float("nan")
    a_col = a_arr.reshape(-1, 1)
    b_row = b_arr.reshape(1, -1)
    gt = int((a_col > b_row).sum())
    lt = int((a_col < b_row).sum())
    return float((gt - lt) / (a_arr.size * b_arr.size))


def _odds_ratio_with_ci(table, alpha: float = 0.05
                        ) -> tuple[float, float, float]:
    arr = np.asarray(table, dtype=float)
    if arr.shape != (2, 2):
        raise ValueError("table must be 2x2")
    a, b = arr[0]
    c, d = arr[1]
    # Haldane–Anscombe correction on zero cells
    if a == 0 or b == 0 or c == 0 or d == 0:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    odds = (a * d) / (b * c)
    se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    log_odds = math.log(odds)
    from scipy import stats as sstats
    z = float(sstats.norm.ppf(1 - alpha / 2))
    return (float(odds),
            float(math.exp(log_odds - z * se)),
            float(math.exp(log_odds + z * se)))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_method(method: str, *,
                  index: dict[str, tuple[str, str]] | None = None,
                  ) -> tuple[str, str]:
    """Split a ``<cell>_<model_slug>`` key back into ``(cell, model)``.

    Naive ``rsplit("_", 1)`` is wrong because BOTH cells (e.g.
    ``free_text_regex``, ``dspy_modular``) AND model slugs (e.g.
    ``gpt_oss_20b``) contain underscores — the rsplit silently produces
    ``("free_text_regex_gpt_oss", "20b")`` for the join.

    Pass an explicit ``index`` (built from the grid CSV's ``cell`` /
    ``model`` columns) to recover the correct mapping. As a last
    resort, walk known cell-ids longest-first and accept the first
    prefix match — at least this can never produce a partial-model-suffix
    cell name.
    """
    if index is not None and method in index:
        return index[method]
    # Defensive fallback: try every registered cell-id as a prefix
    # (longest first so multi-word cells beat their substrings).
    for cell in _KNOWN_CELL_IDS:
        if method.startswith(cell + "_"):
            return cell, method[len(cell) + 1:]
    parts = method.rsplit("_", 1)
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1]


# Sorted longest-first so e.g. ``free_text_regex`` is matched before
# ``free``. Keep in sync with ``runners/`` directory.
_KNOWN_CELL_IDS: tuple[str, ...] = tuple(sorted((
    "chain_of_thought", "compiled_dspy", "constrained_decoding",
    "dspy_modular", "dspy_monolithic", "fewshot_demos", "flat_schema",
    "free_text_regex", "minimal_prompt", "no_router", "per_section",
    "raw_json", "reuse_baseline", "str_outputs", "union_schema",
), key=len, reverse=True))


def _load_grid(results_root: Path) -> pd.DataFrame:
    """Load the long-form correctness grid for stats consumption.

    Prefers ``cascade_atomic.parquet`` (the canonical scorer output);
    derives the legacy ``method``, ``cell``, ``model``, ``run``, and
    ``seed`` columns from the cascade schema and filters to Stage-C
    scalar rows (stats functions reduce on binary correctness, not
    nested F1). Falls back to ``ablation_grid.csv`` when the parquet
    is absent.
    """
    cascade_path = results_root / "cascade_atomic.parquet"
    if cascade_path.exists():
        return _adapt_cascade_atomic_for_stats(pd.read_parquet(cascade_path))
    grid_path = results_root / GRID_CSV
    if not grid_path.exists():
        raise FileNotFoundError(
            f"neither cascade_atomic.parquet nor {grid_path} found — "
            f"run the aggregator first.")
    df = pd.read_csv(grid_path)
    # Prefer explicit cell/model columns when the aggregator emitted
    # them (current behaviour) — the underscore parsing is brittle
    # because multi-underscore cell-ids and model-slugs collide.
    if "cell" not in df.columns or "model" not in df.columns:
        cells_models = df["method"].apply(_split_method)
        df["cell"] = [c for c, _ in cells_models]
        df["model"] = [m for _, m in cells_models]
    return df


def _adapt_cascade_atomic_for_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Project ``cascade_atomic`` rows onto the stats-consumer schema.

    Stats functions reduce on binary correctness (``correct ∈ {0, 1}``)
    keyed by ``(cell, model, field, case_id)``. Cascade atomic stores
    F1 floats in nested-list rows and rows for Stage A/B that aren't
    fields-of-interest here. Filter to Stage-C scalar, then derive the
    legacy column names from cascade columns:

    - ``method`` ← ``f"{cell}_{model_slug}"`` (joint key for the
      stats consumers)
    - ``model`` ← ``model_slug`` (revert from the composite that the
      cascade walker stamped onto ``model`` for ablation runs)
    - ``run``  ← ``run_id``
    - ``seed`` ← ``run_id`` (so :func:`_detect_seed_column` finds a
      seed dimension; multi-run ablation cells use ``run_id`` as
      their seed identifier)
    """
    if df.empty:
        return df
    sub = df.copy()
    if "cascade_stage" in sub.columns:
        sub = sub[sub["cascade_stage"] == "C"]
    if "field_kind" in sub.columns:
        sub = sub[sub["field_kind"] != "nested_list"]
    if sub.empty:
        return sub
    if "model_slug" not in sub.columns:
        sub["model_slug"] = sub.get("model", "")
    sub["model"] = sub["model_slug"]
    sub["method"] = (
        sub["cell"].astype(str) + "_" + sub["model_slug"].astype(str)
    )
    sub["run"] = sub["run_id"]
    if "seed" not in sub.columns:
        sub["seed"] = sub["run_id"]
    # When cascade_atomic was written via _common.reporting.write_parquet
    # with mixed bool / float `correct` rows (scalar Stage C + nested
    # Stage C in the same column), the helper JSON-stringifies the
    # column ("true" / "false" / "0.5"). Stats consumers reduce on
    # numeric correctness, so coerce back. Stage-C nested rows have
    # already been dropped above.
    sub["correct"] = _coerce_str_correct(sub["correct"])
    sub["attempted"] = sub["attempted"].apply(_coerce_bool_str).astype(bool)
    return sub


def _coerce_str_correct(s: pd.Series) -> pd.Series:
    """Map a possibly-stringified correctness column to floats in {0, 1}."""
    def _one(v):
        if v is None:
            return float("nan")
        if isinstance(v, bool):
            return 1.0 if v else 0.0
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            lv = v.strip().lower()
            if lv == "true":
                return 1.0
            if lv == "false":
                return 0.0
            try:
                return float(lv)
            except ValueError:
                return float("nan")
        return float("nan")
    return s.apply(_one)


def _coerce_bool_str(v) -> bool:
    """Map "true" / True / 1 / etc. to bool; everything else to False."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() == "true"
    return False


def _load_axes(path: Path = DEFAULT_AXES_PATH) -> dict[str, str]:
    if not path.exists():
        return {}
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cell_to_axis: dict[str, str] = {}
    for axis, cells in (doc.get("axes") or {}).items():
        for cell in cells:
            cell_to_axis[cell] = axis
    return cell_to_axis


def _load_endpoints(path: Path = DEFAULT_ENDPOINTS_PATH
                    ) -> tuple[set[str], set[str]]:
    if not path.exists():
        # Fall back to FAIR_SCOPE as primary, nothing as secondary.
        return set(FAIR_SCOPE), set()
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    primary = set(doc.get("primary") or [])
    secondary = set(doc.get("secondary") or [])
    return primary, secondary


def _coerce_correct(series: pd.Series) -> pd.Series:
    """Cast the CSV's ``correct`` column (mixed bool / float / NaN) to float."""
    return pd.to_numeric(series, errors="coerce")


# ---------------------------------------------------------------------------
# 1. Paired deltas vs baseline
# ---------------------------------------------------------------------------

def paired_deltas_vs_baseline(
    grid_df: pd.DataFrame,
    *,
    baseline_method: str = "dspy_modular_gpt-oss",
    n_boot: int = 2000,
    alpha: float = 0.05,
    random_state: int = 0,
    device: str = "cpu",
) -> pd.DataFrame:
    """For every (cell, model, field) ≠ baseline: paired-bootstrap Δ + McNemar.

    Returns a DataFrame with one row per (target_method, field), with
    columns: ``cell, model, field, n_paired, baseline_acc, target_acc,
    delta, ci_lo, ci_hi, mcnemar_b, mcnemar_c, mcnemar_stat,
    mcnemar_p, mcnemar_method``.
    """
    if baseline_method not in grid_df["method"].unique():
        logger.warning("baseline %r not present in grid — skipping deltas",
                       baseline_method)
        return pd.DataFrame()

    base_df = grid_df[grid_df["method"] == baseline_method].copy()
    base_df["correct_f"] = _coerce_correct(base_df["correct"])
    base_lookup = base_df.set_index(["case_id", "field"])["correct_f"]
    base_attempt = base_df.set_index(["case_id", "field"])["attempted"]

    out_rows: list[dict] = []
    for method, group in grid_df.groupby("method"):
        if method == baseline_method:
            continue
        if "cell" in group.columns and "model" in group.columns:
            cell = str(group["cell"].iloc[0])
            model = str(group["model"].iloc[0])
        else:
            cell, model = _split_method(method)
        group = group.copy()
        group["correct_f"] = _coerce_correct(group["correct"])
        for field, sub in group.groupby("field"):
            sub = sub.set_index("case_id")
            try:
                base_for_field = base_lookup.xs(field, level="field")
                base_attempt_for_field = base_attempt.xs(field, level="field")
            except KeyError:
                continue
            common = sub.index.intersection(base_for_field.index)
            if not len(common):
                continue
            attempted_both = (
                sub.loc[common, "attempted"].astype(bool)
                & base_attempt_for_field.loc[common].astype(bool)
            )
            common_attempted = common[attempted_both.values]
            if not len(common_attempted):
                continue
            a = base_for_field.loc[common_attempted].to_numpy(dtype=float)
            b = sub.loc[common_attempted, "correct_f"].to_numpy(dtype=float)
            valid = ~(np.isnan(a) | np.isnan(b))
            a, b = a[valid], b[valid]
            n_paired = a.size
            if n_paired == 0:
                continue
            from digital_registrar_research.benchmarks.eval import ci_gpu
            res = ci_gpu.paired_bootstrap_diff(
                b, a,  # target − baseline
                n_boot=n_boot, alpha=alpha,
                random_state=random_state, device=device,
            )
            row = {
                "cell": cell, "model": model, "field": field,
                "n_paired": n_paired,
                "baseline_acc": float(a.mean()),
                "target_acc": float(b.mean()),
                "delta": res.point,
                "ci_lo": res.lo,
                "ci_hi": res.hi,
            }
            # McNemar for binary correctness fields
            is_binary = (
                field in PRIMARY_BINARY_FIELDS
                or set(np.unique(np.concatenate([a, b]))) <= {0.0, 1.0}
            )
            if is_binary:
                disc_b = int(np.sum((a == 1.0) & (b == 0.0)))
                disc_c = int(np.sum((a == 0.0) & (b == 1.0)))
                mc = mcnemar_test(disc_b, disc_c)
                row.update({
                    "mcnemar_b": disc_b, "mcnemar_c": disc_c,
                    "mcnemar_stat": mc["statistic"],
                    "mcnemar_p": mc["p_value"],
                    "mcnemar_method": mc["method"],
                })
            else:
                row.update({"mcnemar_b": None, "mcnemar_c": None,
                            "mcnemar_stat": None, "mcnemar_p": None,
                            "mcnemar_method": None})
            out_rows.append(row)
    return pd.DataFrame(out_rows)


# ---------------------------------------------------------------------------
# 2. Multiple-comparisons correction
# ---------------------------------------------------------------------------

def multiple_comparisons_correction(
    deltas_df: pd.DataFrame,
    *,
    axes_path: Path = DEFAULT_AXES_PATH,
    endpoints_path: Path = DEFAULT_ENDPOINTS_PATH,
) -> pd.DataFrame:
    """Group ``deltas_df`` by (axis, endpoint_tier) and adjust ``mcnemar_p``.

    primary tier → Holm-Bonferroni (FWER)
    secondary tier → Benjamini-Hochberg (FDR)
    exploratory tier → no adjustment

    Adds columns: ``axis, tier, family_size, p_holm, p_bh, reject_holm,
    reject_bh``.
    """
    if deltas_df.empty:
        return deltas_df

    cell_to_axis = _load_axes(axes_path)
    primary, secondary = _load_endpoints(endpoints_path)

    df = deltas_df.copy()
    df["axis"] = df["cell"].map(cell_to_axis).fillna("unmapped")
    df["tier"] = df["field"].map(
        lambda f: "primary" if f in primary
        else ("secondary" if f in secondary else "exploratory")
    )

    df["p_holm"] = np.nan
    df["p_bh"] = np.nan
    df["family_size"] = 0
    df["reject_holm"] = False
    df["reject_bh"] = False

    for (_axis, tier), group in df.groupby(["axis", "tier"]):
        ps = group["mcnemar_p"].astype(float).tolist()
        if not any(p == p for p in ps):  # all NaN
            continue
        idx = group.index
        df.loc[idx, "family_size"] = len(idx)
        if tier == "primary":
            adj = _adjust_pvalues(ps, method="holm")
            df.loc[idx, "p_holm"] = adj
            df.loc[idx, "reject_holm"] = [bool(p == p and p < 0.05) for p in adj]
        elif tier == "secondary":
            adj = _adjust_pvalues(ps, method="fdr_bh")
            df.loc[idx, "p_bh"] = adj
            df.loc[idx, "reject_bh"] = [bool(p == p and p < 0.05) for p in adj]
        # exploratory left as NaN
    return df


# ---------------------------------------------------------------------------
# 3. Multi-seed GLMM
# ---------------------------------------------------------------------------

def _detect_seed_column(df: pd.DataFrame) -> str | None:
    """The grid CSV doesn't carry a ``seed`` column directly. Multi-seed runs
    are encoded by repeating the same ``cell`` short name with different
    ``slug`` (model) suffixes, e.g. ``dspy_modular_gpt-oss``,
    ``dspy_modular_gpt-oss-s2``. Caller can pass a custom seed mapping via
    a ``seed`` column added upstream; if missing, return None."""
    return "seed" if "seed" in df.columns else None


def multi_seed_glmm(grid_df: pd.DataFrame,
                    *, n_boot: int = 2000) -> pd.DataFrame:
    """If ``grid_df`` carries a ``seed`` column, fit GLMM per (cell, field).

    Returns the per-field CI table from
    :func:`multirun.per_field_ci`, with cell appended. Skips when there
    are < 2 seeds for any cell.
    """
    seed_col = _detect_seed_column(grid_df)
    if seed_col is None:
        return pd.DataFrame()
    out_frames: list[pd.DataFrame] = []
    for cell, group in grid_df.groupby("cell"):
        if group[seed_col].nunique() < 2:
            continue
        sub = group.dropna(subset=["correct"]).copy()
        sub["correct"] = _coerce_correct(sub["correct"])
        sub["run_id"] = sub[seed_col].astype(str)
        sub["field_kind"] = sub["field"].apply(
            lambda f: "scalar" if f in PRIMARY_BINARY_FIELDS or f in FAIR_SCOPE
            else "scalar"  # default — nested fields rarely appear in ablation grid today
        )
        try:
            field_ci = per_field_ci(sub, n_boot=n_boot)
        except Exception as exc:  # pragma: no cover
            logger.warning("per_field_ci failed for cell=%s: %s", cell, exc)
            continue
        field_ci["cell"] = cell
        out_frames.append(field_ci)
    if not out_frames:
        return pd.DataFrame()
    return pd.concat(out_frames, ignore_index=True)


# ---------------------------------------------------------------------------
# 4. Seed consistency (Fleiss κ + flip rate)
# ---------------------------------------------------------------------------

def seed_consistency(grid_df: pd.DataFrame) -> pd.DataFrame:
    seed_col = _detect_seed_column(grid_df)
    if seed_col is None:
        return pd.DataFrame()
    rows: list[dict] = []
    for (cell, field), group in grid_df.groupby(["cell", "field"]):
        if group[seed_col].nunique() < 2:
            continue
        pivot = group.pivot_table(index="case_id", columns=seed_col,
                                  values="correct", aggfunc="first")
        matrix = pivot.to_numpy(dtype=float)
        if matrix.size == 0:
            continue
        mask = ~np.any(np.isnan(matrix), axis=1)
        clean = matrix[mask]
        if clean.size == 0:
            continue
        try:
            kappa = fleiss_kappa(clean.astype(int))
        except Exception:
            kappa = float("nan")
        any_disagree = ~(np.all(clean == clean[:, [0]], axis=1))
        flip_rate = float(any_disagree.mean())
        # Pairwise Spearman across seeds
        from scipy.stats import spearmanr
        n_seeds = clean.shape[1]
        pairs = []
        for i in range(n_seeds):
            for j in range(i + 1, n_seeds):
                if np.std(clean[:, i]) == 0 or np.std(clean[:, j]) == 0:
                    continue
                rho, _p = spearmanr(clean[:, i], clean[:, j])
                if not np.isnan(rho):
                    pairs.append(float(rho))
        min_rho = float(min(pairs)) if pairs else float("nan")
        rows.append({
            "cell": cell, "field": field,
            "n_cases": int(clean.shape[0]), "n_seeds": int(clean.shape[1]),
            "fleiss_kappa": kappa,
            "flip_rate": flip_rate,
            "min_pairwise_spearman": min_rho,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Factorial GLMM (Grid 2)
# ---------------------------------------------------------------------------

def factorial_glmm(
    grid_df: pd.DataFrame,
    *,
    axes_path: Path = DEFAULT_AXES_PATH,
    primary_fields: Sequence[str] | None = None,
    n_boot: int = 2000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit ``correct ~ A * B * C + (1|case) + (1|model)`` per primary field.

    Cells must have an axis assignment in ``axes.yaml`` of the form
    ``axes: {A: [...], B: [...], C: [...]}`` and a ``levels:`` map per
    cell (cell → {axis_A: <level>, axis_B: <level>, axis_C: <level>}).

    Returns (term_effects, marginal_means). Returns empty frames if the
    factorial structure isn't present (axes.yaml missing or only one
    level per axis).
    """
    if not axes_path.exists():
        return pd.DataFrame(), pd.DataFrame()

    axes_doc = yaml.safe_load(axes_path.read_text(encoding="utf-8")) or {}
    levels = axes_doc.get("levels") or {}
    if not levels:
        return pd.DataFrame(), pd.DataFrame()

    # Annotate grid with axis levels
    df = grid_df.copy()
    for ax_name in ("axis_A", "axis_B", "axis_C"):
        df[ax_name] = df["cell"].map(
            lambda c, ax_name=ax_name: (levels.get(c) or {}).get(ax_name)
        )
    df = df.dropna(subset=["axis_A", "axis_B", "axis_C"])
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    primary_fields = (
        list(primary_fields) if primary_fields
        else sorted(df["field"].unique())
    )

    term_rows: list[dict] = []
    marginal_rows: list[dict] = []

    try:
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    except ImportError:
        logger.warning("statsmodels not available — factorial_glmm skipped")
        return pd.DataFrame(), pd.DataFrame()

    for field in primary_fields:
        sub = df[df["field"] == field].dropna(subset=["correct"]).copy()
        sub["correct_int"] = pd.to_numeric(sub["correct"], errors="coerce")
        sub = sub.dropna(subset=["correct_int"])
        sub["correct_int"] = sub["correct_int"].astype(int)
        if sub["correct_int"].nunique() < 2:
            continue
        if sub.empty:
            continue
        try:
            vc = {"case_id": "0 + C(case_id)", "model": "0 + C(model)"}
            full = BinomialBayesMixedGLM.from_formula(
                "correct_int ~ C(axis_A) * C(axis_B) * C(axis_C)",
                vc_formulas=vc, data=sub).fit_vb()
            for name, coef, se in zip(
                full.model.exog_names, full.fe_mean, full.fe_sd, strict=True
            ):
                term_rows.append({
                    "field": field, "term": name,
                    "estimate_logit": float(coef),
                    "se_logit": float(se),
                    "z": float(coef / se) if se else float("nan"),
                })
            # Marginal means per axis level (empirical via predicted prob).
            for axis_name in ("axis_A", "axis_B", "axis_C"):
                for level, lvl_sub in sub.groupby(axis_name):
                    point = float(lvl_sub["correct_int"].mean())
                    n = len(lvl_sub)
                    k = int(lvl_sub["correct_int"].sum())
                    lo, hi = wilson_ci(k, n)
                    marginal_rows.append({
                        "field": field, "axis": axis_name,
                        "level": level, "n": n,
                        "marginal_acc": point,
                        "ci_lo": lo, "ci_hi": hi,
                    })
        except Exception as exc:
            logger.warning("factorial_glmm failed for field=%s: %s", field, exc)
            term_rows.append({
                "field": field, "term": "_error",
                "estimate_logit": float("nan"),
                "se_logit": float("nan"),
                "z": float("nan"),
                "error": str(exc),
            })
    return pd.DataFrame(term_rows), pd.DataFrame(marginal_rows)


# ---------------------------------------------------------------------------
# 6. Efficiency stats
# ---------------------------------------------------------------------------

def efficiency_stats(results_root: Path) -> pd.DataFrame:
    """Per-cell schema/parse error rate (Wilson CI) + median latency CI."""
    eff_path = results_root / EFFICIENCY_CSV
    if not eff_path.exists():
        return pd.DataFrame()
    eff = pd.read_csv(eff_path)

    out_rows: list[dict] = []
    for _, row in eff.iterrows():
        cell = row["cell"]
        model = row["model"]
        n_cases = int(row.get("n_cases") or 0)
        if n_cases == 0:
            continue
        schema_n = int(row.get("schema_errors") or 0)
        parse_n = int(row.get("parse_errors") or 0)
        s_lo, s_hi = wilson_ci(schema_n, n_cases)
        p_lo, p_hi = wilson_ci(parse_n, n_cases)

        # Latency CI on the median — aggregate per-case latencies from
        # every run's ``_log.jsonl`` under the canonical layout
        # ``results_root/{cell}/{model}/{run_id}/_log.jsonl``. The old
        # path ``{cell}_{model}/_ledger.json`` did not exist, so the CI
        # was always None.
        median_ci_lo = median_ci_hi = None
        median_latency = row.get("median_latency_s")
        cell_model_dir = results_root / cell / model
        lats: list[float] = []
        if cell_model_dir.is_dir():
            for run_dir in sorted(cell_model_dir.iterdir()):
                if not run_dir.is_dir() or run_dir.name.startswith("_"):
                    continue
                log_path = run_dir / "_log.jsonl"
                if not log_path.exists():
                    continue
                try:
                    with log_path.open(encoding="utf-8") as f:
                        for line in f:
                            try:
                                rec = json.loads(line)
                            except Exception:
                                continue
                            lat = rec.get("latency_s")
                            if isinstance(lat, (int, float)):
                                lats.append(float(lat))
                except Exception:
                    continue
        if len(lats) >= 5:
            try:
                res = bootstrap_ci(
                    lats, lambda xs: float(np.median(xs)),
                    n_boot=1000, method="percentile", random_state=0,
                )
                median_ci_lo = res.lo
                median_ci_hi = res.hi
            except Exception:
                pass

        out_rows.append({
            "cell": cell, "model": model, "n_cases": n_cases,
            "schema_error_rate": schema_n / n_cases if n_cases else float("nan"),
            "schema_ci_lo": s_lo, "schema_ci_hi": s_hi,
            "parse_error_rate": parse_n / n_cases if n_cases else float("nan"),
            "parse_ci_lo": p_lo, "parse_ci_hi": p_hi,
            "median_latency_s": median_latency,
            "median_ci_lo": median_ci_lo,
            "median_ci_hi": median_ci_hi,
            "mean_latency_s": row.get("mean_latency_s"),
        })
    return pd.DataFrame(out_rows)


# ---------------------------------------------------------------------------
# 7. Effect sizes per field
# ---------------------------------------------------------------------------

def effect_sizes_per_field(
    grid_df: pd.DataFrame,
    *,
    baseline_method: str = "dspy_modular_gpt-oss",
) -> pd.DataFrame:
    if baseline_method not in grid_df["method"].unique():
        return pd.DataFrame()

    base_df = grid_df[grid_df["method"] == baseline_method].copy()
    base_df["correct_f"] = _coerce_correct(base_df["correct"])
    base_lookup = base_df.set_index(["case_id", "field"])["correct_f"]

    out_rows: list[dict] = []
    for method, group in grid_df.groupby("method"):
        if method == baseline_method:
            continue
        if "cell" in group.columns and "model" in group.columns:
            cell = str(group["cell"].iloc[0])
            model = str(group["model"].iloc[0])
        else:
            cell, model = _split_method(method)
        group = group.copy()
        group["correct_f"] = _coerce_correct(group["correct"])
        for field, sub in group.groupby("field"):
            sub = sub.set_index("case_id")
            try:
                base_for_field = base_lookup.xs(field, level="field")
            except KeyError:
                continue
            common = sub.index.intersection(base_for_field.index)
            if not len(common):
                continue
            a = base_for_field.loc[common].to_numpy(dtype=float)
            b = sub.loc[common, "correct_f"].to_numpy(dtype=float)
            valid = ~(np.isnan(a) | np.isnan(b))
            a, b = a[valid], b[valid]
            if a.size == 0:
                continue
            d = _cohens_d(b, a)
            cliff = _cliffs_delta(b, a)

            row = {"cell": cell, "model": model, "field": field,
                   "n": int(a.size),
                   "cohens_d": d, "cliffs_delta": cliff}
            # Odds ratio for binary correctness
            if (
                field in PRIMARY_BINARY_FIELDS
                or set(np.unique(np.concatenate([a, b]))) <= {0.0, 1.0}
            ):
                a_correct = int((a == 1.0).sum())
                a_wrong = int(a.size - a_correct)
                b_correct = int((b == 1.0).sum())
                b_wrong = int(b.size - b_correct)
                table = [[b_correct, b_wrong], [a_correct, a_wrong]]
                try:
                    odds, lo, hi = _odds_ratio_with_ci(table)
                    row.update({"odds_ratio": odds,
                                "or_ci_lo": lo, "or_ci_hi": hi})
                except ValueError:
                    pass
            out_rows.append(row)
    return pd.DataFrame(out_rows)


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def run_all(results_root: Path,
            *, baseline_method: str = "dspy_modular_gpt-oss",
            n_boot: int = 2000,
            device: str = "cpu") -> dict[str, Path]:
    """Read ``ablation_grid.csv`` from ``results_root`` and write all stats CSVs.

    Returns a mapping of stage → output path.

    ``device`` (``cpu`` | ``cuda`` | ``mps`` | ``auto``) routes the
    paired-bootstrap call sites through :mod:`ci_gpu`. Default is
    ``cpu`` (the safety-net path).
    """
    results_root = Path(results_root)
    grid_df = _load_grid(results_root)
    out: dict[str, Path] = {}

    deltas = paired_deltas_vs_baseline(grid_df, baseline_method=baseline_method,
                                       n_boot=n_boot, device=device)
    if not deltas.empty:
        path = results_root / "ablation_paired_deltas.csv"
        deltas.to_csv(path, index=False)
        out["paired_deltas"] = path

        corrected = multiple_comparisons_correction(deltas)
        path = results_root / "ablation_paired_deltas_corrected.csv"
        corrected.to_csv(path, index=False)
        out["paired_deltas_corrected"] = path

    glmm = multi_seed_glmm(grid_df, n_boot=n_boot)
    if not glmm.empty:
        path = results_root / "ablation_glmm.csv"
        glmm.to_csv(path, index=False)
        out["glmm"] = path

    seed = seed_consistency(grid_df)
    if not seed.empty:
        path = results_root / "ablation_seed_consistency.csv"
        seed.to_csv(path, index=False)
        out["seed_consistency"] = path

    terms, marginals = factorial_glmm(grid_df)
    if not terms.empty:
        path = results_root / "ablation_factorial.csv"
        terms.to_csv(path, index=False)
        out["factorial"] = path
    if not marginals.empty:
        path = results_root / "ablation_marginal_means.csv"
        marginals.to_csv(path, index=False)
        out["marginal_means"] = path

    eff = efficiency_stats(results_root)
    if not eff.empty:
        path = results_root / "ablation_efficiency_stats.csv"
        eff.to_csv(path, index=False)
        out["efficiency_stats"] = path

    effect = effect_sizes_per_field(grid_df, baseline_method=baseline_method)
    if not effect.empty:
        path = results_root / "ablation_effect_sizes.csv"
        effect.to_csv(path, index=False)
        out["effect_sizes"] = path

    # NOTE: ``cancer_category_mismatch_stats`` was removed in the
    # cascade-conformance migration. Cascade gating means Stage-C only
    # emits when Stage B was correct, so the per-case mismatch rate is
    # always 0.0 for Stage-C rows — the metric collapsed to Stage B
    # accuracy, which is now reported by the ``compare`` subcommand's
    # ``chapter2_organ_classification/comparison.csv``.
    return out


__all__ = [
    "paired_deltas_vs_baseline",
    "multiple_comparisons_correction",
    "multi_seed_glmm",
    "seed_consistency",
    "factorial_glmm",
    "efficiency_stats",
    "effect_sizes_per_field",
    "run_all",
]
