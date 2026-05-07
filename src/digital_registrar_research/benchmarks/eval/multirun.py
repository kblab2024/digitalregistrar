"""
Multi-run accuracy analysis for gpt-oss — Part C of the CMUH statistical
plan.

Inputs
------
Expects the Part B artifact layout:

    workspace/results/benchmarks/gpt_oss/
    ├── _manifest.yaml
    ├── run1/<case_id>.json
    ├── run2/<case_id>.json
    ├── ...
    └── runK/<case_id>.json

Gold files are matched by the shared `_gold.json` suffix under the
annotations root (or any legacy gold folder; the glob accepts both).

Public entry points
-------------------
    discover_runs(root)                 — list [(run_id, Path)] sorted
    build_correctness_table(runs, ...)  — long-form DataFrame, atomic table
    per_field_ci(df, method)            — point + case-CI + run-CI + total-CI
    majority_vote_ensemble(runs_preds, out_dir) — write ensemble JSONs
    run_consistency(df)                 — Fleiss κ, flip rate, per-run corr
    ensemble_vs_single(df_single, df_ens) — paired-bootstrap Δ with CI

Design note
-----------
The atomic table shape (one row per run × case × field) matches the
existing `aggregate_cases_to_df` long-form so downstream tooling can consume
both interchangeably — just treat `run_id` as a sibling of `method`.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .ci import (
    bootstrap_ci,
    paired_bootstrap_diff,
    t_ci,
    two_source_bootstrap_ci,
    wilson_ci,
)
from .metrics import (
    normalize,
    score_case,
)
from .scope import (
    BREAST_BIOMARKERS,
    FAIR_SCOPE,
    NESTED_LIST_FIELDS,
)

__all__ = [
    "discover_runs",
    "build_correctness_table",
    "per_field_ci",
    "majority_vote_ensemble",
    "fleiss_kappa",
    "run_consistency",
    "ensemble_vs_single",
    "per_organ_ci",
    "per_fieldtype_ci",
]


# --- Discovery ---------------------------------------------------------------

def discover_runs(root: Path) -> list[tuple[str, Path]]:
    """Return [(run_id, run_dir)] for every `run*` subfolder of `root`."""
    root = Path(root)
    runs: list[tuple[str, Path]] = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and p.name.startswith("run"):
            runs.append((p.name, p))
    return runs


# --- Correctness table -------------------------------------------------------

def build_correctness_table(
    runs: Sequence[tuple[str, Path]],
    gold_root: Path,
    *,
    case_ids: Sequence[str] | None = None,
    gold_suffix: str = "_gold",
) -> pd.DataFrame:
    """Score every (run, case, field) tuple and return a long-form DataFrame.

    Columns: run_id, case_id, organ, field, field_kind, correct, attempted.
        - field_kind = "scalar" or "nested"
        - correct: True/False for scalars, float F1 for nested, NaN when
          not attempted.

    Cases are discovered by walking ``gold_root`` for ``*_gold.json``
    files (or bare-stem JSON files for the legacy single-annotator
    layout).
    """
    gold_map = _index_gold(gold_root, gold_suffix)
    selected_ids = set(case_ids) if case_ids else set(gold_map.keys())

    rows: list[dict] = []
    for run_id, run_dir in runs:
        for cid in selected_ids:
            gold_path = gold_map.get(cid)
            if gold_path is None:
                continue
            pred_path = run_dir / f"{cid}.json"
            with gold_path.open(encoding="utf-8") as f:
                gold = json.load(f)
            organ = normalize(gold.get("cancer_category"))

            if not pred_path.exists() or _is_parse_error(pred_path):
                # Coverage failure → emit NaN rows for the FAIR_SCOPE
                # fields so downstream aggregation sees the gap.
                for field in FAIR_SCOPE:
                    rows.append({
                        "run_id": run_id, "case_id": cid, "organ": organ,
                        "field": field, "field_kind": "scalar",
                        "correct": None, "attempted": False,
                    })
                continue

            with pred_path.open(encoding="utf-8") as f:
                pred = json.load(f)
            result = score_case(gold, pred)
            for field in FAIR_SCOPE + [f"biomarker_{b}" for b in BREAST_BIOMARKERS]:
                if field not in result:
                    continue
                val = result[field]
                rows.append({
                    "run_id": run_id, "case_id": cid, "organ": organ,
                    "field": field, "field_kind": "scalar",
                    "correct": (bool(val) if val is not None else None),
                    "attempted": val is not None,
                })
            for field, f1dict in result.get("_nested", {}).items():
                rows.append({
                    "run_id": run_id, "case_id": cid, "organ": organ,
                    "field": field, "field_kind": "nested",
                    "correct": float(f1dict["f1"]),
                    "attempted": True,
                })
    return pd.DataFrame(rows)


def _index_gold(gold_root: Path, gold_suffix: str) -> dict[str, Path]:
    """Build {case_id: gold_path} by walking ``<gold_root>/**/<id>{suffix}.json``."""
    mapping: dict[str, Path] = {}
    for p in Path(gold_root).rglob("*.json"):
        stem = p.stem
        if stem.endswith(gold_suffix):
            cid = stem[: -len(gold_suffix)]
            mapping[cid] = p
        elif not any(s in stem for s in ("_nhc", "_kpc", "_gold")):
            # Legacy case: single-annotator file — use the raw stem.
            mapping[stem] = p
    return mapping


def _is_parse_error(path: Path) -> bool:
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return True
    return bool(isinstance(data, dict) and data.get("_parse_error"))


# --- GLMM two-source CI (primary headline) -----------------------------------

def _glmm_marginal_accuracy(sub: pd.DataFrame) -> tuple[float, float, float, dict]:
    """Fit a mixed-effects logistic model
        correct ~ 1 + (1|case_id) + (1|run_id)
    and return (marginal_accuracy, CI_lo, CI_hi, variance_components).

    Falls back to two-source bootstrap if statsmodels is unavailable or
    the fit fails (e.g. all correct / all incorrect, singular gradient).
    """
    sub = sub.dropna(subset=["correct"])
    if sub.empty or sub["correct"].nunique() < 2:
        # Degenerate: all same outcome. Report point with Wilson CI.
        p = float(sub["correct"].mean()) if not sub.empty else float("nan")
        if sub.empty:
            return (p, float("nan"), float("nan"), {})
        k = int(sub["correct"].sum())
        n = len(sub)
        lo, hi = wilson_ci(k, n)
        return (p, lo, hi, {"degenerate": True})

    try:
        sub = sub.copy()
        sub["correct_int"] = sub["correct"].astype(int)
        # BinomialBayesMixedGLM with random intercepts keyed by case_id
        # and run_id. Single-factor models are more stable; we fit two
        # separate GLMMs and combine variance components.
        from statsmodels.genmod.bayes_mixed_glm import (
            BinomialBayesMixedGLM,
        )
        # Case random effect
        vc = {"case_id": "0 + C(case_id)", "run_id": "0 + C(run_id)"}
        model = BinomialBayesMixedGLM.from_formula(
            "correct_int ~ 1", vc_formulas=vc, data=sub,
        )
        fit = model.fit_vb()
        intercept = float(fit.fe_mean[0])
        se = float(fit.fe_sd[0])
        z = 1.959963984540054
        lo_logit = intercept - z * se
        hi_logit = intercept + z * se
        def _invlogit(x): return 1 / (1 + math.exp(-x))
        point = _invlogit(intercept)
        lo = _invlogit(lo_logit)
        hi = _invlogit(hi_logit)
        # Variance components
        vc_sd = fit.vcp_mean  # posterior mean of log(sd) per VC
        var_components = {
            k: float(math.exp(2 * v)) for k, v in zip(fit.model.vc_names, vc_sd, strict=True)
        }
        return (point, lo, hi, var_components)
    except Exception:
        # Bootstrap fallback
        matrix = _to_case_run_matrix(sub)
        res = two_source_bootstrap_ci(matrix)
        return (res.point, res.lo, res.hi, {"fallback": "two_source_bootstrap"})


def _to_case_run_matrix(df: pd.DataFrame) -> np.ndarray:
    """Pivot a (case, run, correct) frame to a 2-D matrix, NaN-padded."""
    pivot = df.pivot_table(index="case_id", columns="run_id",
                           values="correct", aggfunc="mean")
    return pivot.to_numpy(dtype=float)


# --- Per-field CI assembly ---------------------------------------------------

def per_field_ci(df: pd.DataFrame, *, n_boot: int = 2000,
                 random_state: int = 0) -> pd.DataFrame:
    """For each field, emit one row with:
        point_estimate, case_ci_lo/hi, run_ci_lo/hi, total_ci_lo/hi,
        var_case, var_run, n_cases, n_runs, field_kind.
    """
    out_rows: list[dict] = []
    for field, sub in df.groupby("field"):
        sub = sub.dropna(subset=["correct"])
        if sub.empty:
            continue
        kind = sub["field_kind"].iloc[0]
        n_runs = sub["run_id"].nunique()
        n_cases = sub["case_id"].nunique()

        # Per-run accuracies (for run-variance t-CI)
        per_run = sub.groupby("run_id")["correct"].mean().to_numpy(dtype=float)
        point_mean = float(per_run.mean()) if per_run.size else float("nan")
        run_mean, run_lo, run_hi = t_ci(per_run.tolist())

        # Case-bootstrap CI (ignoring run structure, pooled)
        values = sub[["case_id", "correct"]].to_dict(orient="records")
        case_res = bootstrap_ci(
            values,
            lambda xs: float(np.mean([x["correct"] for x in xs])),
            n_boot=n_boot, random_state=random_state,
        )

        # Total (two-source) CI
        if kind == "scalar" and set(np.unique(sub["correct"].astype(float))) <= {0.0, 1.0}:
            point, lo, hi, varcomp = _glmm_marginal_accuracy(sub)
        else:
            # Nested F1 or other float-valued correctness → nested bootstrap.
            matrix = _to_case_run_matrix(sub)
            res = two_source_bootstrap_ci(matrix, n_boot=n_boot,
                                          random_state=random_state)
            point, lo, hi = res.point, res.lo, res.hi
            varcomp = {}

        out_rows.append({
            "field": field,
            "field_kind": kind,
            "n_cases": n_cases,
            "n_runs": n_runs,
            "point_estimate": point,
            "mean_per_run_accuracy": point_mean,
            "case_ci_lo": case_res.lo, "case_ci_hi": case_res.hi,
            "run_ci_lo": run_lo, "run_ci_hi": run_hi,
            "total_ci_lo": lo, "total_ci_hi": hi,
            "var_case": varcomp.get("case_id"),
            "var_run": varcomp.get("run_id"),
            "ci_method": "glmm" if varcomp and "case_id" in varcomp else "two_source_bootstrap",
        })
    return pd.DataFrame(out_rows)


# --- Majority-vote ensemble --------------------------------------------------

def majority_vote_ensemble(
    runs: Sequence[tuple[str, Path]],
    out_dir: Path,
    *,
    case_ids: Sequence[str] | None = None,
) -> Path:
    """Write per-case ensemble predictions by voting across runs.

    - Scalar fields: most common non-None value; ties broken by first-seen.
    - Continuous (int) fields inside cancer_data: median.
    - Nested lists: union of items grouped by their primary key; fields
      within each grouped item resolved by majority vote among runs that
      contributed that item.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect all cases seen across runs
    all_cases: set[str] = set()
    per_run_preds: dict[str, dict[str, dict]] = {}
    for run_id, run_dir in runs:
        run_preds: dict[str, dict] = {}
        for p in run_dir.glob("*.json"):
            if p.name.startswith("_"):
                continue
            try:
                with p.open(encoding="utf-8") as f:
                    run_preds[p.stem] = json.load(f)
            except Exception:
                continue
        per_run_preds[run_id] = run_preds
        all_cases.update(run_preds.keys())

    if case_ids is not None:
        all_cases = set(case_ids) & all_cases

    for cid in sorted(all_cases):
        preds_for_case = [per_run_preds[rid].get(cid) for rid, _ in runs]
        preds_for_case = [p for p in preds_for_case if p and not p.get("_parse_error")]
        if not preds_for_case:
            continue
        ensembled = _ensemble_predictions(preds_for_case)
        with (out_dir / f"{cid}.json").open("w", encoding="utf-8") as f:
            json.dump(ensembled, f, ensure_ascii=False, indent=2)
    return out_dir


def _ensemble_predictions(preds: Sequence[dict]) -> dict:
    """Merge a list of schema-conformant prediction dicts into one."""
    out: dict = {}
    # Collect candidate keys at top level + cancer_data
    top_keys = set()
    cd_keys = set()
    for p in preds:
        top_keys.update(k for k in p if k != "cancer_data")
        cd_keys.update((p.get("cancer_data") or {}).keys())

    for k in top_keys:
        values = [p.get(k) for p in preds if k in p]
        out[k] = _vote(values, k)

    cd_out: dict = {}
    for k in cd_keys:
        values = [(p.get("cancer_data") or {}).get(k)
                  for p in preds if k in (p.get("cancer_data") or {})]
        cd_out[k] = _vote(values, k)
    if cd_out:
        out["cancer_data"] = cd_out
    return out


def _vote(values: Sequence, field: str):
    """Vote resolution. Dispatches on field name and value type."""
    values = [v for v in values if v is not None]
    if not values:
        return None
    # Nested list fields — merge by primary key.
    if field in NESTED_LIST_FIELDS or any(isinstance(v, list) for v in values):
        return _vote_nested_list(values, field)
    # Continuous-looking values — use median.
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
        return int(round(float(np.median(values))))
    # Boolean — majority vote.
    if all(isinstance(v, bool) for v in values):
        return Counter(values).most_common(1)[0][0]
    # Categorical — majority vote on normalised values, return the most
    # common raw value matching that normalisation.
    norm = [normalize(v) for v in values]
    winner = Counter(norm).most_common(1)[0][0]
    for v in values:
        if normalize(v) == winner:
            return v
    return values[0]


_NESTED_KEY = {
    "margins": "margin_category",
    "biomarkers": "biomarker_category",
    "regional_lymph_node": "station_name",
}


def _vote_nested_list(values: Sequence[list], field: str) -> list:
    """Merge list-of-dicts by primary key then vote inner fields."""
    key = _NESTED_KEY.get(field)
    if key is None:
        return values[0] if values else []
    buckets: dict[object, list[dict]] = defaultdict(list)
    for lst in values:
        if not isinstance(lst, list):
            continue
        for item in lst:
            if not isinstance(item, dict):
                continue
            bk = normalize(item.get(key))
            buckets[bk].append(item)
    out: list[dict] = []
    # Require majority support before including an item.
    n_runs = len(values)
    min_support = max(1, (n_runs // 2) + 1)
    for _bk, items in buckets.items():
        if len(items) < min_support:
            continue
        merged: dict = {}
        inner_keys = set().union(*(i.keys() for i in items))
        for ik in inner_keys:
            merged[ik] = _vote([i.get(ik) for i in items], ik)
        out.append(merged)
    return out


# --- Run consistency (self-agreement) ----------------------------------------

def fleiss_kappa(ratings: np.ndarray) -> float:
    """Fleiss' κ for N items × K coders matrix of integer category labels.

    NaN rows are dropped.
    """
    arr = np.asarray(ratings)
    if arr.ndim != 2:
        raise ValueError("ratings must be 2-D")
    # Drop rows with any NaN
    mask = ~np.any(pd.isna(arr), axis=1)
    arr = arr[mask]
    N, k = arr.shape
    if N == 0 or k < 2:
        return float("nan")
    categories = np.unique(arr)
    # Per-item category counts
    n_ij = np.zeros((N, len(categories)), dtype=int)
    for j, cat in enumerate(categories):
        n_ij[:, j] = (arr == cat).sum(axis=1)
    # Agreement per item
    P_i = (np.sum(n_ij * n_ij, axis=1) - k) / (k * (k - 1))
    P_bar = P_i.mean()
    # Overall proportion per category
    p_j = n_ij.sum(axis=0) / (N * k)
    P_e = np.sum(p_j ** 2)
    if P_e >= 1.0:
        return float("nan")
    return float((P_bar - P_e) / (1 - P_e))


def run_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Per-field consistency diagnostics across runs.

    Columns: field, n_cases, fleiss_kappa, flip_rate, min_pairwise_spearman.
    """
    rows = []
    for field, sub in df.groupby("field"):
        sub = sub.dropna(subset=["correct"])
        if sub.empty:
            continue
        pivot = sub.pivot_table(index="case_id", columns="run_id",
                                values="correct", aggfunc="first")
        if pivot.shape[1] < 2:
            continue

        # Fleiss κ on the 0/1 correctness matrix (binary rater agreement
        # on "did the model get this case right").
        fk = fleiss_kappa(pivot.to_numpy(dtype=float))

        # Flip rate: fraction of cases where runs don't all agree.
        matrix = pivot.to_numpy(dtype=float)
        all_same = np.all(matrix == matrix[:, [0]], axis=1)
        flip_rate = float(1.0 - all_same.mean()) if matrix.size else float("nan")

        # Min pairwise Spearman: smallest correlation among run-pairs.
        from scipy.stats import spearmanr
        corrs: list[float] = []
        cols = list(pivot.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                a = pivot[cols[i]].dropna()
                b = pivot[cols[j]].dropna()
                idx = a.index.intersection(b.index)
                if len(idx) < 3:
                    continue
                rho, _ = spearmanr(a.loc[idx], b.loc[idx])
                if not np.isnan(rho):
                    corrs.append(float(rho))
        min_rho = min(corrs) if corrs else float("nan")

        rows.append({
            "field": field,
            "n_cases": int(pivot.shape[0]),
            "n_runs": int(pivot.shape[1]),
            "fleiss_kappa": fk,
            "flip_rate": flip_rate,
            "min_pairwise_spearman": min_rho,
        })
    return pd.DataFrame(rows)


# --- Ensemble vs single comparison ------------------------------------------

def ensemble_vs_single(
    df_single: pd.DataFrame,
    df_ensemble: pd.DataFrame,
    *,
    n_boot: int = 2000,
    random_state: int = 0,
) -> pd.DataFrame:
    """Per-field paired-bootstrap Δ = ensemble_acc - mean_per_run_acc.

    `df_single` is the raw Part-C correctness table (K runs); we take
    its per-case mean across runs as the "single-run baseline" to
    fairly compare with the ensemble (single prediction per case).
    """
    rows = []
    single_by_case = (
        df_single.dropna(subset=["correct"])
        .groupby(["field", "case_id"])["correct"].mean()
        .reset_index()
    )
    ensemble_by_case = df_ensemble.dropna(subset=["correct"])[
        ["field", "case_id", "correct"]
    ]
    for field in set(single_by_case["field"]) & set(ensemble_by_case["field"]):
        s = single_by_case[single_by_case["field"] == field].set_index("case_id")
        e = ensemble_by_case[ensemble_by_case["field"] == field].set_index("case_id")
        shared = s.index.intersection(e.index)
        if len(shared) == 0:
            continue
        a = e.loc[shared, "correct"].astype(float).to_numpy()
        b = s.loc[shared, "correct"].astype(float).to_numpy()
        res = paired_bootstrap_diff(a, b, n_boot=n_boot,
                                    random_state=random_state)
        rows.append({
            "field": field,
            "n_cases": len(shared),
            "ensemble_acc": float(a.mean()),
            "single_mean_acc": float(b.mean()),
            "delta": res.point,
            "delta_ci_lo": res.lo,
            "delta_ci_hi": res.hi,
        })
    return pd.DataFrame(rows)


# --- Stratified summaries ---------------------------------------------------

def per_organ_ci(df: pd.DataFrame, *, n_boot: int = 2000,
                 random_state: int = 0) -> pd.DataFrame:
    """per_field_ci stratified by organ."""
    rows = []
    for (organ, field), sub in df.dropna(subset=["correct"]).groupby(["organ", "field"]):
        if sub.empty:
            continue
        values = sub[["case_id", "correct"]].to_dict(orient="records")
        res = bootstrap_ci(
            values,
            lambda xs: float(np.mean([x["correct"] for x in xs])),
            n_boot=n_boot, random_state=random_state,
        )
        rows.append({
            "organ": organ, "field": field,
            "n_cases": sub["case_id"].nunique(),
            "n_runs": sub["run_id"].nunique(),
            "point_estimate": res.point,
            "ci_lo": res.lo, "ci_hi": res.hi,
        })
    return pd.DataFrame(rows)


def per_fieldtype_ci(df: pd.DataFrame, field_to_type: dict[str, str],
                     *, n_boot: int = 2000,
                     random_state: int = 0) -> pd.DataFrame:
    """Group fields by type (from `field_to_type` mapping) and report a
    type-level accuracy with bootstrap CI."""
    rows = []
    typed = df.copy()
    typed["field_type"] = typed["field"].map(field_to_type).fillna("other")
    for ftype, sub in typed.dropna(subset=["correct"]).groupby("field_type"):
        if sub.empty:
            continue
        values = sub[["case_id", "correct"]].to_dict(orient="records")
        res = bootstrap_ci(
            values,
            lambda xs: float(np.mean([x["correct"] for x in xs])),
            n_boot=n_boot, random_state=random_state,
        )
        rows.append({
            "field_type": ftype,
            "n_fields": sub["field"].nunique(),
            "n_cases": sub["case_id"].nunique(),
            "n_runs": sub["run_id"].nunique(),
            "point_estimate": res.point,
            "ci_lo": res.lo, "ci_hi": res.hi,
        })
    return pd.DataFrame(rows)
