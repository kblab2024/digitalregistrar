"""
Interobserver-agreement (IAA) statistics for paired annotators — Part A
of the CMUH statistical plan.

Discovery / loader
------------------
    discover_cases(annotations_root, annotators)
        Walks a folder tree of the form
            <annotations_root>/<organ_idx>/<case_id>_<annotator>.json
        and groups files into per-case trios keyed by annotator suffix.
        Returns {case_id: {"organ": str, "annotations": {annotator: dict}}}.

Per-field scoring (dispatched by field type)
--------------------------------------------
    score_field_pair(field, pairs, field_type) -> dict
        Returns the type-appropriate statistic(s) for a list of paired
        values from two annotators. Field types: "binary", "nominal",
        "ordinal", "continuous", "nested_list", plus the "coverage"
        pseudo-type (annotated-vs-null indicator).

    classify_field(field, organ) -> str
        Single source of truth for field-type dispatch, built from the
        existing `scope_organs` taxonomy (ORGAN_BOOL -> binary,
        ORGAN_SPAN -> continuous, ORGAN_NESTED_LIST -> nested_list,
        everything else in ORGAN_CATEGORICAL -> ordinal if the field is
        in ORDINAL_FIELDS else nominal).

Pairwise headline
-----------------
    pairwise_iaa(cases, pair=("_nhc", "_kpc"), ...)
        Long-form DataFrame: one row per (organ × field × stat) with
        BCa bootstrap CI, stratified by organ.

    whole_report_stats(cases, pair)
        Case-level exact-match, mean-field-accuracy, per-section mean κ,
        Krippendorff α (per-type and pooled). One row per statistic.

    disagreement_resolution(cases)
        For cases where `_nhc` != `_kpc`, tallies which side `_gold`
        matches. Chi-square test for asymmetry.

Outputs consume the same long-form schema as `aggregate_cases_to_df`:
columns include `organ, section, field, field_type, n, stat_name,
estimate, ci_lo, ci_hi, observed_agreement, n_categories`.

Field-type taxonomy: ``classify_field()`` below maps each (field,
organ) pair to one of binary | ordinal | nominal | continuous |
nested_list, drawing on ``digital_registrar_research.benchmarks.eval.scope``
for the canonical organ-aware groupings.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

from . import ci_gpu
from .ci import (
    bootstrap_ci,
    fisher_z_ci_for_corr,
    mcnemar_test,
    wilson_ci,
)
from .metrics import (
    NUMERIC_TOLERANCE_MM,
    is_attempted,
    match_nested_list,
    normalize,
)
from .nested_metrics import score_lymph_nodes, score_margins
from .scope import (
    BREAST_BIOMARKERS,
    FAIR_SCOPE,
    NESTED_LIST_FIELDS,
    ORGAN_BOOL,
    ORGAN_CATEGORICAL,
    ORGAN_NESTED_LIST,
    ORGAN_SPAN,
    get_bool_fields,
    get_categorical_fields,
    get_field_value,
    get_nested_list_fields,
    get_span_fields,
)

# --- Field-type taxonomy -----------------------------------------------------

# Fields whose categorical values carry natural ordering. We score them
# with quadratic-weighted κ + Kendall τ-b rather than unweighted κ.
ORDINAL_FIELDS: frozenset[str] = frozenset({
    "grade", "nuclear_grade", "tubule_formation", "mitotic_rate",
    "total_score", "dcis_grade",
    "pt_category", "pn_category", "pm_category",
    "stage_group", "overall_stage",
    "pathologic_stage_group", "anatomic_stage_group",
})

# Fields for the top-level section (rest are scalar_pathology unless
# they're nested-list).
TOP_LEVEL_FIELDS: frozenset[str] = frozenset({
    "cancer_category", "cancer_excision_report",
})


def classify_field(field: str, organ: str | None = None) -> str:
    """Map a (field, organ) pair to one of:
        "binary" | "ordinal" | "nominal" | "continuous" | "nested_list".

    Falls back to cross-organ union when `organ` is None.
    """
    if field in NESTED_LIST_FIELDS:
        return "nested_list"
    if organ is not None:
        if field in get_bool_fields(organ):
            return "binary"
        if field in get_span_fields(organ):
            return "continuous"
    else:
        if any(field in v for v in ORGAN_BOOL.values()):
            return "binary"
        if any(field in v for v in ORGAN_SPAN.values()):
            return "continuous"
    if field in ORDINAL_FIELDS:
        return "ordinal"
    return "nominal"


def classify_section(field: str) -> str:
    if field in TOP_LEVEL_FIELDS:
        return "top_level"
    if field in NESTED_LIST_FIELDS:
        return "nested"
    return "scalar_pathology"


# --- Discovery ---------------------------------------------------------------

@dataclass
class CaseEntry:
    organ: str | None
    annotations: dict[str, dict]    # {annotator: annotation dict}
    paths: dict[str, Path]


def discover_cases(
    annotations_root: Path,
    annotators: Sequence[str] = ("_nhc", "_kpc", "_gold"),
) -> dict[str, CaseEntry]:
    """Walk `annotations_root` and group JSON files by case id.

    Files are expected to be named `<case_id><annotator_suffix>.json`
    where the suffix is one of the entries in `annotators` (e.g. `_nhc`).
    The organ is inferred from each annotation's `cancer_category`.
    """
    cases: dict[str, CaseEntry] = {}
    for p in Path(annotations_root).rglob("*.json"):
        stem = p.stem
        suffix = None
        for s in annotators:
            if stem.endswith(s):
                suffix = s
                break
        if suffix is None:
            continue
        case_id = stem[: -len(suffix)]
        try:
            with p.open(encoding="utf-8") as f:
                ann = json.load(f)
        except Exception:
            continue
        organ = normalize(ann.get("cancer_category")) if isinstance(ann, dict) else None
        entry = cases.get(case_id)
        if entry is None:
            entry = CaseEntry(organ=organ, annotations={}, paths={})
            cases[case_id] = entry
        entry.annotations[suffix] = ann
        entry.paths[suffix] = p
        if entry.organ is None and organ is not None:
            entry.organ = organ
    return cases


# --- Pair extraction ---------------------------------------------------------

@dataclass
class Pair:
    case_id: str
    organ: str | None
    a: object         # normalised value from annotator A
    b: object         # normalised value from annotator B
    raw_a: object     # raw (pre-normalisation) value — useful for continuous
    raw_b: object


def extract_pairs(
    cases: dict[str, CaseEntry],
    field: str,
    ann_a: str,
    ann_b: str,
    *,
    require_attempted: bool = True,
) -> list[Pair]:
    """Collect (case, organ, normalised values) triples for `field`.

    A case is included only if both annotators attempted the field
    (explicit key present, possibly with null value) when
    `require_attempted` is True. Nulls remain in the pair stream so
    callers can treat them as a distinct category if appropriate.
    """
    pairs: list[Pair] = []
    for cid, entry in cases.items():
        a_ann = entry.annotations.get(ann_a)
        b_ann = entry.annotations.get(ann_b)
        if a_ann is None or b_ann is None:
            continue
        if require_attempted:
            if not (is_attempted(a_ann, field) and is_attempted(b_ann, field)):
                continue
        raw_a = get_field_value(a_ann, field)
        raw_b = get_field_value(b_ann, field)
        pairs.append(Pair(
            case_id=cid,
            organ=entry.organ,
            a=normalize(raw_a),
            b=normalize(raw_b),
            raw_a=raw_a,
            raw_b=raw_b,
        ))
    return pairs


# --- Categorical agreement statistics ---------------------------------------

def cohen_kappa(
    pairs: Sequence[Pair],
    *,
    weights: str = "unweighted",
    ordinal_order: Sequence | None = None,
) -> float:
    """Cohen's κ. `weights` ∈ {"unweighted", "quadratic"}.

    For quadratic-weighted κ, pass `ordinal_order` (sequence of values in
    rank order). Values not in the order are treated as unknown and the
    pair is dropped.
    """
    if not pairs:
        return float("nan")
    if weights == "quadratic":
        if ordinal_order is None:
            raise ValueError("quadratic kappa needs ordinal_order")
        rank = {v: i for i, v in enumerate(ordinal_order)}
        data = [(rank[p.a], rank[p.b]) for p in pairs
                if p.a in rank and p.b in rank]
        if not data:
            return float("nan")
        k = len(ordinal_order)
        if k <= 1:
            return float("nan")
        a_vals, b_vals = zip(*data, strict=True)
        n = len(data)
        # Observed disagreement (weighted)
        denom = (k - 1) ** 2
        obs_dis = sum(((i - j) ** 2) / denom for i, j in data) / n
        # Expected disagreement from marginals
        a_freq = Counter(a_vals)
        b_freq = Counter(b_vals)
        exp_dis = 0.0
        for i in range(k):
            for j in range(k):
                pa = a_freq.get(i, 0) / n
                pb = b_freq.get(j, 0) / n
                exp_dis += pa * pb * ((i - j) ** 2) / denom
        if exp_dis == 0:
            return float("nan")
        return 1.0 - obs_dis / exp_dis

    # Unweighted
    n = len(pairs)
    a_freq: Counter = Counter(p.a for p in pairs)
    b_freq: Counter = Counter(p.b for p in pairs)
    p_o = sum(1 for p in pairs if p.a == p.b) / n
    categories = set(a_freq) | set(b_freq)
    p_e = sum((a_freq[c] / n) * (b_freq[c] / n) for c in categories)
    if p_e >= 1.0:
        return float("nan")
    return (p_o - p_e) / (1 - p_e)


def observed_agreement(pairs: Sequence[Pair]) -> float:
    if not pairs:
        return float("nan")
    return sum(1 for p in pairs if p.a == p.b) / len(pairs)


def pabak(pairs: Sequence[Pair]) -> float:
    """Prevalence-adjusted bias-adjusted κ (only meaningful for binary)."""
    p_o = observed_agreement(pairs)
    if math.isnan(p_o):
        return p_o
    return 2 * p_o - 1


def mcnemar_on_binary(pairs: Sequence[Pair], positive_value="true") -> dict:
    """Run McNemar on binary pairs with a canonical positive value.

    b = pairs where A=positive, B!=positive.
    c = pairs where A!=positive, B=positive.
    Returns the raw test dict + disagreement counts.
    """
    b = sum(1 for p in pairs if p.a == positive_value and p.b != positive_value)
    c = sum(1 for p in pairs if p.a != positive_value and p.b == positive_value)
    return mcnemar_test(b, c)


def kendall_tau_b(pairs: Sequence[Pair], ordinal_order: Sequence) -> dict:
    rank = {v: i for i, v in enumerate(ordinal_order)}
    data = [(rank[p.a], rank[p.b]) for p in pairs
            if p.a in rank and p.b in rank]
    if len(data) < 4:
        return {"tau": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    a_vals, b_vals = zip(*data, strict=True)
    tau, _p = sstats.kendalltau(a_vals, b_vals, variant="b", nan_policy="omit")
    lo, hi = fisher_z_ci_for_corr(float(tau), len(data))
    return {"tau": float(tau), "ci_lo": lo, "ci_hi": hi, "n": len(data)}


# --- Continuous agreement statistics ----------------------------------------

def lins_ccc(pairs: Sequence[Pair]) -> float:
    xs, ys = _continuous_arrays(pairs)
    if xs.size < 2:
        return float("nan")
    mx, my = xs.mean(), ys.mean()
    sx2, sy2 = xs.var(ddof=0), ys.var(ddof=0)
    sxy = float(((xs - mx) * (ys - my)).mean())
    denom = sx2 + sy2 + (mx - my) ** 2
    if denom == 0:
        return float("nan")
    return float(2 * sxy / denom)


def icc_2_1(pairs: Sequence[Pair]) -> float:
    """Intraclass correlation ICC(2,1): two-way random effects, single rater,
    absolute agreement. Falls back to NaN when degrees of freedom collapse.
    """
    xs, ys = _continuous_arrays(pairs)
    n = xs.size
    if n < 2:
        return float("nan")
    M = np.stack([xs, ys], axis=1)  # (n, 2)
    k = 2
    grand = M.mean()
    row_m = M.mean(axis=1)
    col_m = M.mean(axis=0)
    ss_rows = k * float(((row_m - grand) ** 2).sum())
    ss_cols = n * float(((col_m - grand) ** 2).sum())
    ss_err = float(((M - row_m[:, None] - col_m[None, :] + grand) ** 2).sum())
    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_err = ss_err / ((n - 1) * (k - 1)) if (n - 1) * (k - 1) > 0 else float("nan")
    denom = ms_rows + (k - 1) * ms_err + (k / n) * (ms_cols - ms_err)
    if denom <= 0 or math.isnan(ms_err):
        return float("nan")
    return float((ms_rows - ms_err) / denom)


def bland_altman(pairs: Sequence[Pair]) -> dict:
    xs, ys = _continuous_arrays(pairs)
    if xs.size == 0:
        return {"bias": float("nan"), "sd_diff": float("nan"),
                "loa_lo": float("nan"), "loa_hi": float("nan"), "n": 0}
    diff = xs - ys
    bias = float(diff.mean())
    sd = float(diff.std(ddof=1)) if diff.size > 1 else 0.0
    return {
        "bias": bias,
        "sd_diff": sd,
        "loa_lo": bias - 1.96 * sd,
        "loa_hi": bias + 1.96 * sd,
        "n": int(xs.size),
    }


def _continuous_arrays(pairs: Sequence[Pair]) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for p in pairs:
        if isinstance(p.raw_a, (int, float)) and isinstance(p.raw_b, (int, float)) \
                and not (math.isnan(float(p.raw_a)) or math.isnan(float(p.raw_b))):
            xs.append(float(p.raw_a))
            ys.append(float(p.raw_b))
    return np.array(xs), np.array(ys)


def within_tolerance_rate(pairs: Sequence[Pair], tol: float) -> float:
    xs, ys = _continuous_arrays(pairs)
    if xs.size == 0:
        return float("nan")
    return float(np.mean(np.abs(xs - ys) <= tol))


# --- Nested-list agreement --------------------------------------------------

def nested_list_f1(pairs: Sequence[Pair], field: str) -> float:
    """Treat annotator A as 'gold', B as 'pred' and compute matched F1
    using the existing match_nested_list. Nested list fields live under
    `cancer_data` in the canonical schema; wrap the raw values in that
    container so get_field_value resolves them.
    """
    if not pairs:
        return float("nan")
    tp = fp = fn = 0
    for p in pairs:
        gold = {"cancer_data": {field: p.raw_a}}
        pred = {"cancer_data": {field: p.raw_b}}
        r = match_nested_list(gold, pred, field)
        tp += r["tp"]
        fp += r["fp"]
        fn += r["fn"]
    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


# --- Krippendorff's α -------------------------------------------------------

def krippendorff_alpha(
    units: Sequence[Sequence],
    *,
    level: str = "nominal",
    value_order: Sequence | None = None,
) -> float:
    """Krippendorff's α over a list of units.

    Each unit is a sequence of values (one per coder), possibly with
    missing entries (None). level ∈ {"nominal", "ordinal", "interval"};
    for "ordinal", pass `value_order` (ranked list of possible values).

    Formula: α = 1 - D_o / D_e, with
        D_o = Σ_c (1/(m_c-1)) Σ_{k1≠k2} δ(v_{c,k1}, v_{c,k2})  /  Σ_c m_c
        D_e = (1/(N(N-1))) Σ_{v1,v2} N_v1 · N_v2' · δ(v1, v2)
    where N = total valued observations and N_v2' = N_v2 - [v1==v2].
    """
    # Map values through the type-specific rank where appropriate.
    if level == "ordinal":
        if value_order is None:
            raise ValueError("ordinal α needs value_order")
        rank = {v: i for i, v in enumerate(value_order)}
        mapped: list[list] = [[rank[v] for v in u if v in rank] for u in units]
    else:
        mapped = [[v for v in u if v is not None] for u in units]

    valid = [u for u in mapped if len(u) >= 2]
    if not valid:
        return float("nan")

    def delta(a, b) -> float:
        if level == "nominal":
            return 0.0 if a == b else 1.0
        return float((a - b) ** 2)

    # Observed disagreement: per-unit sum of inter-coder δ, normalised.
    n_total = sum(len(u) for u in valid)
    if n_total < 2:
        return float("nan")
    num_o = 0.0
    for u in valid:
        m = len(u)
        s = 0.0
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                s += delta(u[i], u[j])
        num_o += s / (m - 1)
    D_o = num_o / n_total

    # Expected disagreement: over pooled marginal frequencies.
    freq: Counter = Counter(v for u in valid for v in u)
    num_e = 0.0
    for v1, n1 in freq.items():
        for v2, n2 in freq.items():
            cross = n1 * (n2 - 1) if v1 == v2 else n1 * n2
            num_e += cross * delta(v1, v2)
    D_e = num_e / (n_total * (n_total - 1))
    if D_e == 0:
        return float("nan")
    return float(1.0 - D_o / D_e)


# --- Per-field scoring entry point ------------------------------------------

def score_field_pair(
    field: str,
    organ: str | None,
    pairs: Sequence[Pair],
    *,
    n_boot: int = 2000,
    random_state: int | None = 0,
    device: str = "cpu",
) -> list[dict]:
    """Dispatch on field type. Returns a list of long-form rows suitable
    for the output CSV. Each row is one statistic with point estimate +
    bootstrap CI (where applicable)."""
    rows: list[dict] = []
    if not pairs:
        rows.append({
            "field": field, "field_type": classify_field(field, organ),
            "stat_name": "n", "estimate": 0, "ci_lo": None, "ci_hi": None,
            "observed_agreement": None, "n_categories": 0,
            "note": "no cases",
        })
        return rows

    ftype = classify_field(field, organ)
    strata = [p.organ for p in pairs]

    def _add(stat_name, estimate, lo=None, hi=None, obs=None, n_cat=None, note=None):
        rows.append({
            "field": field, "field_type": ftype,
            "stat_name": stat_name, "estimate": estimate,
            "ci_lo": lo, "ci_hi": hi,
            "observed_agreement": obs,
            "n_categories": n_cat,
            "note": note,
        })

    obs = observed_agreement(pairs)
    _add("observed_agreement", obs, obs=obs, n_cat=None)
    _add("n", len(pairs))

    if ftype in ("binary", "nominal"):
        order = None
        # Unweighted κ → vectorized GPU/CPU path.
        res = ci_gpu.bootstrap_kappa_ci(
            [p.a for p in pairs], [p.b for p in pairs],
            n_boot=n_boot, strata=strata, random_state=random_state,
            device=device,
        )
        _add("cohen_kappa", res.point, res.lo, res.hi, obs=obs,
             n_cat=len({p.a for p in pairs} | {p.b for p in pairs}))
        if ftype == "binary":
            _add("pabak", pabak(pairs), obs=obs)
            mc = mcnemar_on_binary(pairs)
            _add("mcnemar_b", mc["b"])
            _add("mcnemar_c", mc["c"])
            _add("mcnemar_p", mc["p_value"], note=mc["method"])

    elif ftype == "ordinal":
        organ_cats = get_categorical_fields(organ) if organ else {}
        order = organ_cats.get(field) or sorted({p.a for p in pairs if p.a is not None}
                                                | {p.b for p in pairs if p.b is not None})
        order_bst = order  # closed-over in lambdas below
        # Unweighted κ → vectorized.
        res_unw = ci_gpu.bootstrap_kappa_ci(
            [p.a for p in pairs], [p.b for p in pairs],
            n_boot=n_boot, strata=strata, random_state=random_state,
            device=device,
        )
        _add("cohen_kappa_unweighted", res_unw.point, res_unw.lo, res_unw.hi,
             obs=obs, n_cat=len(order_bst))
        # Quadratic-weighted κ stays on the safety-net path: ci_gpu's
        # vectorized κ is unweighted-only, and the cohen_kappa(weights=
        # "quadratic", ordinal_order=...) callable here can't be cleanly
        # closed-formed without expanding the GPU module's surface.
        res_wt = bootstrap_ci(
            pairs,
            lambda xs: cohen_kappa(xs, weights="quadratic",
                                   ordinal_order=order_bst),
            n_boot=n_boot, strata=strata, random_state=random_state,
        )
        _add("cohen_kappa_quadratic", res_wt.point, res_wt.lo, res_wt.hi,
             obs=obs, n_cat=len(order_bst))
        tau = kendall_tau_b(pairs, order_bst)
        _add("kendall_tau_b", tau["tau"], tau["ci_lo"], tau["ci_hi"],
             n_cat=len(order_bst))

    elif ftype == "continuous":
        ccc_res = bootstrap_ci(pairs, lins_ccc,
                               n_boot=n_boot, strata=strata, random_state=random_state)
        _add("lins_ccc", ccc_res.point, ccc_res.lo, ccc_res.hi)
        icc_res = bootstrap_ci(pairs, icc_2_1,
                               n_boot=n_boot, strata=strata, random_state=random_state)
        _add("icc_2_1", icc_res.point, icc_res.lo, icc_res.hi)
        ba = bland_altman(pairs)
        _add("bland_altman_bias", ba["bias"])
        _add("bland_altman_sd_diff", ba["sd_diff"])
        _add("bland_altman_loa_lo", ba["loa_lo"])
        _add("bland_altman_loa_hi", ba["loa_hi"])
        tol = NUMERIC_TOLERANCE_MM if field == "tumor_size" else 1.0
        rate_res = bootstrap_ci(
            pairs, lambda xs: within_tolerance_rate(xs, tol),
            n_boot=n_boot, strata=strata, random_state=random_state,
        )
        _add(f"within_tol_rate_{tol}", rate_res.point, rate_res.lo, rate_res.hi,
             note=f"|Δ| ≤ {tol}")
        def _mae(xs):
            a, b = _continuous_arrays(xs)
            if a.size == 0:
                return float("nan")
            return float(np.mean(np.abs(a - b)))
        mae_res = bootstrap_ci(
            pairs, _mae,
            n_boot=n_boot, strata=strata, random_state=random_state,
        )
        _add("mae", mae_res.point, mae_res.lo, mae_res.hi)

    elif ftype == "nested_list":
        f1_res = bootstrap_ci(
            pairs,
            lambda xs: nested_list_f1(xs, field),
            n_boot=n_boot, strata=strata, random_state=random_state,
        )
        _add("matched_f1", f1_res.point, f1_res.lo, f1_res.hi)
        # Clinical case-level summary where applicable
        if field == "regional_lymph_node":
            _add_nested_case_level(_add, pairs, score_lymph_nodes,
                                   [("ln_examined_total_correct_tol", "examined_tol1_acc"),
                                    ("ln_involved_total_correct_tol", "involved_tol1_acc"),
                                    ("ln_any_positive_correct", "any_positive_acc")],
                                   strata=strata, n_boot=n_boot,
                                   random_state=random_state)
        elif field == "margins":
            _add_nested_case_level(_add, pairs, score_margins,
                                   [("margin_any_involved_correct", "any_involved_acc"),
                                    ("margin_closest_distance_correct_tol", "closest_tol2_acc")],
                                   strata=strata, n_boot=n_boot,
                                   random_state=random_state)

    return rows


def _add_nested_case_level(add_fn, pairs, scorer, metric_to_label,
                           *, strata, n_boot, random_state):
    # For each metric, compute a per-pair scalar 0/1 (or NaN) via the
    # existing scorer, then bootstrap its mean.
    scored = []
    for p in pairs:
        ann_a = {"cancer_data": {}}
        ann_b = {"cancer_data": {}}
        # Re-embed the field so the scorer can pull it
        ann_a["cancer_data"][_field_from_metric(metric_to_label)] = p.raw_a
        ann_b["cancer_data"][_field_from_metric(metric_to_label)] = p.raw_b
        s = scorer(ann_a, ann_b)
        scored.append(s)
    for metric_key, label in metric_to_label:
        values = [s.get(metric_key) for s in scored]
        values = [v for v in values if v is not None]
        if not values:
            add_fn(f"case_level_{label}", float("nan"))
            continue
        res = bootstrap_ci(values, lambda xs: float(np.mean(xs)),
                           n_boot=n_boot, strata=None,
                           random_state=random_state)
        add_fn(f"case_level_{label}", res.point, res.lo, res.hi)


def _field_from_metric(metric_to_label: Sequence[tuple[str, str]]) -> str:
    key = metric_to_label[0][0]
    return "regional_lymph_node" if key.startswith("ln_") else "margins"


# --- Pairwise driver --------------------------------------------------------

def pairwise_iaa(
    cases: dict[str, CaseEntry],
    *,
    ann_a: str = "_nhc",
    ann_b: str = "_kpc",
    fields: Sequence[str] | None = None,
    n_boot: int = 2000,
    random_state: int | None = 0,
    device: str = "cpu",
) -> pd.DataFrame:
    """Long-form per-(organ × field × stat) IAA DataFrame for a specific
    annotator pair.

    When `fields` is None, scores every field reachable from the existing
    scope taxonomy (FAIR_SCOPE + breast biomarkers + nested list fields
    per organ).
    """
    observed_organs = sorted({e.organ for e in cases.values() if e.organ})

    if fields is None:
        fields = _default_field_list()

    rows: list[dict] = []
    for organ in observed_organs + [None]:  # None = pooled across organs
        organ_cases = {cid: e for cid, e in cases.items()
                       if (organ is None or e.organ == organ)}
        for field in fields:
            # Skip fields not scored for this organ (e.g. breast biomarkers
            # outside the breast cohort).
            if organ is not None and not _field_applies_to_organ(field, organ):
                continue
            pairs = extract_pairs(organ_cases, field, ann_a, ann_b)
            if not pairs:
                continue
            sub = score_field_pair(field, organ, pairs,
                                   n_boot=n_boot, random_state=random_state,
                                   device=device)
            for r in sub:
                r["organ"] = organ if organ is not None else "ALL"
                r["section"] = classify_section(field)
                r["pair"] = f"{ann_a.lstrip('_')}_vs_{ann_b.lstrip('_')}"
                rows.append(r)

            # Coverage κ: agreement on "attempted vs null" indicator
            cov_pairs = _coverage_pairs(organ_cases, field, ann_a, ann_b)
            if cov_pairs:
                res = ci_gpu.bootstrap_kappa_ci(
                    [p.a for p in cov_pairs],
                    [p.b for p in cov_pairs],
                    n_boot=n_boot,
                    strata=[p.organ for p in cov_pairs],
                    random_state=random_state,
                    device=device,
                )
                rows.append({
                    "organ": organ if organ is not None else "ALL",
                    "section": classify_section(field),
                    "pair": f"{ann_a.lstrip('_')}_vs_{ann_b.lstrip('_')}",
                    "field": field, "field_type": "coverage",
                    "stat_name": "coverage_kappa",
                    "estimate": res.point, "ci_lo": res.lo, "ci_hi": res.hi,
                    "observed_agreement": observed_agreement(cov_pairs),
                    "n_categories": 2, "note": "annotated-vs-null κ",
                })
    return pd.DataFrame(rows)


def _default_field_list() -> list[str]:
    # Scalar fields that appear somewhere in the schemas, plus nested
    # lists. Dedupe-preserving order: FAIR_SCOPE first, then organ-
    # specific fields, then nested lists.
    seen: list[str] = list(FAIR_SCOPE)
    for organ_map in ORGAN_BOOL.values():
        for f in organ_map:
            if f not in seen:
                seen.append(f)
    for organ_map in ORGAN_CATEGORICAL.values():
        for f in organ_map:
            if f not in seen:
                seen.append(f)
    for organ_map in ORGAN_SPAN.values():
        for f in organ_map:
            if f not in seen:
                seen.append(f)
    for organ_map in ORGAN_NESTED_LIST.values():
        for f in organ_map:
            if f not in seen:
                seen.append(f)
    # Breast biomarker fields (encoded as biomarker_er/pr/her2 in score_case)
    for bm in BREAST_BIOMARKERS:
        name = f"biomarker_{bm}"
        if name not in seen:
            seen.append(name)
    return seen


def _field_applies_to_organ(field: str, organ: str) -> bool:
    if field in TOP_LEVEL_FIELDS or field in FAIR_SCOPE:
        return True
    if field.startswith("biomarker_"):
        return organ == "breast"
    if field in get_bool_fields(organ):
        return True
    if field in get_span_fields(organ):
        return True
    if field in get_nested_list_fields(organ):
        return True
    if field in get_categorical_fields(organ):
        return True
    return False


def _coverage_pairs(
    cases: dict[str, CaseEntry], field: str, ann_a: str, ann_b: str,
) -> list[Pair]:
    out: list[Pair] = []
    for cid, e in cases.items():
        a_ann = e.annotations.get(ann_a)
        b_ann = e.annotations.get(ann_b)
        if a_ann is None or b_ann is None:
            continue
        a_val = _populated(a_ann, field)
        b_val = _populated(b_ann, field)
        out.append(Pair(cid, e.organ, a_val, b_val, a_val, b_val))
    return out


def _populated(ann: dict, field: str) -> str:
    """Binary coverage indicator for IAA on missingness."""
    if not is_attempted(ann, field):
        return "missing"
    v = get_field_value(ann, field)
    if v is None:
        return "null"
    if isinstance(v, list) and len(v) == 0:
        return "null"
    return "populated"


# --- Whole-report & section-level summaries ---------------------------------

def whole_report_stats(
    cases: dict[str, CaseEntry],
    *,
    ann_a: str = "_nhc",
    ann_b: str = "_kpc",
) -> pd.DataFrame:
    """Headline numbers: case-level exact match, mean field κ, per-section
    mean κ, Krippendorff α per type."""
    rows = []

    # Case-level exact match
    fields = _default_field_list()
    n_cases = 0
    n_exact = 0
    for _cid, e in cases.items():
        a_ann, b_ann = e.annotations.get(ann_a), e.annotations.get(ann_b)
        if a_ann is None or b_ann is None:
            continue
        n_cases += 1
        all_match = True
        for f in fields:
            if is_attempted(a_ann, f) and is_attempted(b_ann, f):
                if normalize(get_field_value(a_ann, f)) != normalize(get_field_value(b_ann, f)):
                    all_match = False
                    break
        if all_match:
            n_exact += 1
    lo, hi = wilson_ci(n_exact, n_cases) if n_cases else (float("nan"), float("nan"))
    rows.append({
        "pair": f"{ann_a.lstrip('_')}_vs_{ann_b.lstrip('_')}",
        "stat_name": "case_exact_match_rate",
        "estimate": (n_exact / n_cases) if n_cases else float("nan"),
        "ci_lo": lo, "ci_hi": hi, "n": n_cases,
    })

    # Krippendorff α per field-type bucket
    for level, field_filter in [
        ("nominal", lambda f, o: classify_field(f, o) in ("binary", "nominal")),
        ("ordinal", lambda f, o: classify_field(f, o) == "ordinal"),
        ("interval", lambda f, o: classify_field(f, o) == "continuous"),
    ]:
        units: list[list] = []
        all_values: list = []
        for _cid, e in cases.items():
            a_ann, b_ann = e.annotations.get(ann_a), e.annotations.get(ann_b)
            if a_ann is None or b_ann is None:
                continue
            for f in fields:
                if not field_filter(f, e.organ):
                    continue
                if not (is_attempted(a_ann, f) and is_attempted(b_ann, f)):
                    continue
                va = get_field_value(a_ann, f)
                vb = get_field_value(b_ann, f)
                if level == "interval":
                    if not (isinstance(va, (int, float)) and isinstance(vb, (int, float))):
                        continue
                    units.append([float(va), float(vb)])
                    all_values.extend([float(va), float(vb)])
                else:
                    units.append([normalize(va), normalize(vb)])
        if level == "ordinal":
            # Some ordinal fields mix int and str values across units;
            # sorting with a string key keeps the comparison total.
            distinct = sorted(
                {v for u in units for v in u if v is not None},
                key=lambda x: ("" if x is None else str(x)),
            )
            alpha = krippendorff_alpha(units, level="ordinal",
                                       value_order=distinct)
        else:
            alpha = krippendorff_alpha(units, level=level)
        rows.append({
            "pair": f"{ann_a.lstrip('_')}_vs_{ann_b.lstrip('_')}",
            "stat_name": f"krippendorff_alpha_{level}",
            "estimate": alpha, "ci_lo": None, "ci_hi": None,
            "n": len(units),
        })

    return pd.DataFrame(rows)


# --- Adjudication / disagreement resolution ---------------------------------

def disagreement_resolution(
    cases: dict[str, CaseEntry],
    *,
    ann_a: str = "_nhc",
    ann_b: str = "_kpc",
    gold: str = "_gold",
) -> pd.DataFrame:
    """For each (organ, field), count how _gold resolves _nhc ≠ _kpc disagreements.

    Columns: organ, field, n_disagreements, matches_a, matches_b,
    matches_neither, chi2, p_value.
    """
    fields = _default_field_list()
    rows = []
    for organ in sorted({e.organ for e in cases.values() if e.organ}) + [None]:
        for field in fields:
            if organ is not None and not _field_applies_to_organ(field, organ):
                continue
            ma = mb = mn = 0
            for _cid, e in cases.items():
                if organ is not None and e.organ != organ:
                    continue
                if not all(x in e.annotations for x in (ann_a, ann_b, gold)):
                    continue
                if not (is_attempted(e.annotations[ann_a], field)
                        and is_attempted(e.annotations[ann_b], field)
                        and is_attempted(e.annotations[gold], field)):
                    continue
                va = normalize(get_field_value(e.annotations[ann_a], field))
                vb = normalize(get_field_value(e.annotations[ann_b], field))
                vg = normalize(get_field_value(e.annotations[gold], field))
                if va == vb:
                    continue
                if vg == va:
                    ma += 1
                elif vg == vb:
                    mb += 1
                else:
                    mn += 1
            total = ma + mb + mn
            if total == 0:
                continue
            # Chi-square goodness-of-fit vs uniform {ma, mb, mn} split.
            exp = total / 3
            chi2 = sum((o - exp) ** 2 / exp for o in (ma, mb, mn))
            p = float(sstats.chi2.sf(chi2, df=2))
            rows.append({
                "organ": organ if organ is not None else "ALL",
                "field": field,
                "n_disagreements": total,
                "matches_a": ma, "matches_b": mb, "matches_neither": mn,
                "chi2_uniform": chi2, "p_value": p,
            })
    return pd.DataFrame(rows)


__all__ = [
    "discover_cases",
    "extract_pairs",
    "classify_field",
    "classify_section",
    "score_field_pair",
    "pairwise_iaa",
    "whole_report_stats",
    "disagreement_resolution",
    "cohen_kappa",
    "lins_ccc",
    "icc_2_1",
    "bland_altman",
    "krippendorff_alpha",
    "nested_list_f1",
    "CaseEntry", "Pair",
    "ORDINAL_FIELDS",
]
