"""Pair-focused IAA headline statistics.

The existing :mod:`iaa` module produces per-(organ × field × stat) rows
and a :func:`whole_report_stats` summary that gives Krippendorff α per
type bucket. This module adds the *missing* layer for a chosen pair of
annotators: a single overall Cohen's-κ headline plus per-section /
per-organ roll-ups and confusion matrices.

Four headline κ flavours are computed side-by-side because no single
pooled-κ definition is canonical for heterogeneous fields:

    mean_per_field_kappa            — unweighted mean over fields
    n_weighted_mean_per_field_kappa — n-weighted mean (n = pairs per field)
    pooled_categorical_kappa        — pooled κ over (case, field) tuples,
                                      categorical fields only, missing-on-
                                      one-side encoded as "__missing__"
    agree_disagree_pabak            — PABAK (= 2·p_o − 1) on the
                                      agree(1)/disagree(0) indicator
                                      across all fields. PABAK is the
                                      principled chance-corrected
                                      agreement statistic when one
                                      "rater" is degenerate; tolerance
                                      rule for continuous,
                                      bipartite F1 = 1.0 ↔ agree for
                                      nested lists.

Krippendorff α (nominal/ordinal/interval) and the case-exact-match rate
come from :func:`iaa.whole_report_stats` and are stitched on by
:func:`compute_headline`.

Scope: pair-agnostic. For within-annotator preann pairs the headline κ
is descriptive (magnitude of preann-induced change). The causal
"does preann pull annotators toward gold?" analysis lives in
:mod:`...eval.preann` and the ``iaa`` subcommand's ``preann/`` outputs.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from .ci import bootstrap_ci
from .iaa import (
    CaseEntry,
    Pair,
    classify_field,
    classify_section,
    cohen_kappa,
    extract_pairs,
    observed_agreement,
    pabak,
    pairwise_iaa,
    whole_report_stats,
    _default_field_list,
    _field_applies_to_organ,
)
from .metrics import (
    NUMERIC_TOLERANCE_MM,
    is_attempted,
    match_nested_list,
    normalize,
)
from .scope import get_field_value

MISSING_TOKEN = "__missing__"

DEFAULT_FIELD_TOLERANCES: dict[str, float] = {
    "tumor_size": float(NUMERIC_TOLERANCE_MM),
}
_FALLBACK_CONTINUOUS_TOLERANCE = 1.0

KAPPA_STAT_NAMES = (
    "cohen_kappa",
    "cohen_kappa_unweighted",
    "cohen_kappa_quadratic",
)

HEADLINE_COLUMNS = ("pair", "stat_name", "estimate", "ci_lo", "ci_hi", "n", "note")
PER_SECTION_COLUMNS = (
    "pair", "section", "stat_name", "estimate", "ci_lo", "ci_hi", "n",
)
PER_ORGAN_COLUMNS = (
    "pair", "organ", "stat_name", "estimate", "ci_lo", "ci_hi",
    "n_cases", "n_fields",
)
PER_FIELD_COLUMNS = (
    "pair", "organ", "section", "field", "field_type", "stat_name",
    "estimate", "ci_lo", "ci_hi", "n", "observed_agreement",
)
CONFUSION_COLUMNS = ("value_a", "value_b", "count")


@dataclass(frozen=True)
class HeadlineRow:
    stat_name: str
    estimate: float
    ci_lo: float
    ci_hi: float
    n: int
    note: str | None = None


def _pair_label(ann_a: str, ann_b: str) -> str:
    return f"{ann_a}_vs_{ann_b}"


# --- Per-field κ table (filtered view of pairwise_iaa) ----------------------


def per_field_kappas(
    cases: dict[str, CaseEntry],
    *,
    ann_a: str,
    ann_b: str,
    fields: Sequence[str] | None = None,
    n_boot: int = 2000,
    random_state: int = 0,
    device: str = "cpu",
) -> pd.DataFrame:
    """Long-form per-(organ × field × κ-stat) table for a pair.

    Wraps :func:`iaa.pairwise_iaa` and filters to κ rows only. Empty
    DataFrame (with the expected columns) is returned for pairs with no
    overlapping cases.
    """
    pair_df = pairwise_iaa(
        cases, ann_a=ann_a, ann_b=ann_b, fields=fields,
        n_boot=n_boot, random_state=random_state, device=device,
    )
    if pair_df.empty:
        return pd.DataFrame(columns=list(PER_FIELD_COLUMNS))
    mask = pair_df["stat_name"].isin(KAPPA_STAT_NAMES)
    sub = pair_df.loc[mask].copy()
    if sub.empty:
        return pd.DataFrame(columns=list(PER_FIELD_COLUMNS))

    # Pull n from the matching "n" rows of the same (organ, field) so the
    # κ row carries the count. pairwise_iaa emits one "n" row per field
    # per organ; join on (organ, field).
    n_rows = pair_df.loc[pair_df["stat_name"] == "n",
                         ["organ", "field", "estimate"]].rename(
        columns={"estimate": "n"})
    sub = sub.merge(n_rows, on=["organ", "field"], how="left")
    sub["pair"] = _pair_label(ann_a, ann_b)
    return sub.reindex(columns=list(PER_FIELD_COLUMNS))


# --- Mean / n-weighted-mean of per-field κ ----------------------------------


def mean_per_field_kappa(per_field_df: pd.DataFrame) -> tuple[float, int]:
    """Unweighted mean of the per-field κ point estimates.

    Restricted to ``organ == "ALL"`` rows so each field contributes
    once. Quadratic-weighted κ is preferred over unweighted for ordinal
    fields when both are present.
    """
    if per_field_df.empty:
        return float("nan"), 0
    sub = per_field_df[per_field_df["organ"] == "ALL"].copy()
    if sub.empty:
        return float("nan"), 0
    sub = _prefer_kappa_per_field(sub)
    values = sub["estimate"].astype(float).dropna()
    if values.empty:
        return float("nan"), 0
    return float(values.mean()), int(values.size)


def n_weighted_mean_per_field_kappa(
    per_field_df: pd.DataFrame,
) -> tuple[float, int]:
    """n-weighted mean of per-field κ. Weights are the per-field pair counts."""
    if per_field_df.empty:
        return float("nan"), 0
    sub = per_field_df[per_field_df["organ"] == "ALL"].copy()
    if sub.empty:
        return float("nan"), 0
    sub = _prefer_kappa_per_field(sub)
    sub["estimate"] = pd.to_numeric(sub["estimate"], errors="coerce")
    sub["n"] = pd.to_numeric(sub["n"], errors="coerce")
    sub = sub.dropna(subset=["estimate", "n"])
    if sub.empty or float(sub["n"].sum()) == 0.0:
        return float("nan"), 0
    weighted = float((sub["estimate"] * sub["n"]).sum() / sub["n"].sum())
    return weighted, int(sub["n"].sum())


def _prefer_kappa_per_field(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one κ row per (field, organ): quadratic > unweighted > generic."""
    priority = {"cohen_kappa_quadratic": 0,
                "cohen_kappa_unweighted": 1,
                "cohen_kappa": 1}
    df = df.copy()
    df["_prio"] = df["stat_name"].map(priority).fillna(2).astype(int)
    df = df.sort_values(["field", "organ", "_prio"]).drop_duplicates(
        subset=["field", "organ"], keep="first")
    return df.drop(columns=["_prio"])


# --- Pooled categorical κ ---------------------------------------------------


def pooled_categorical_pairs(
    cases: dict[str, CaseEntry],
    *,
    ann_a: str,
    ann_b: str,
    fields: Sequence[str] | None = None,
    missing_token: str = MISSING_TOKEN,
) -> list[Pair]:
    """Build the (case, field) Pair stream for pooled-categorical κ.

    Each (case, field) where the field is categorical (binary / ordinal
    / nominal) for the case's organ contributes one Pair. When one
    annotator attempted and the other did not, the missing side is
    encoded as ``missing_token`` so coverage disagreements show up.
    Cases where neither annotator attempted are skipped.

    Continuous and nested-list fields are excluded — their value spaces
    don't share a meaningful κ category structure with the categoricals.
    """
    fields = fields or _default_field_list()
    out: list[Pair] = []
    for cid, entry in cases.items():
        a_ann = entry.annotations.get(ann_a)
        b_ann = entry.annotations.get(ann_b)
        if a_ann is None or b_ann is None:
            continue
        organ = entry.organ
        for field in fields:
            if organ is not None and not _field_applies_to_organ(field, organ):
                continue
            ftype = classify_field(field, organ)
            if ftype not in ("binary", "nominal", "ordinal"):
                continue
            a_attempted = is_attempted(a_ann, field)
            b_attempted = is_attempted(b_ann, field)
            if not (a_attempted or b_attempted):
                continue
            va = (
                normalize(get_field_value(a_ann, field))
                if a_attempted else missing_token
            )
            vb = (
                normalize(get_field_value(b_ann, field))
                if b_attempted else missing_token
            )
            # Map None (attempted-but-null) to its own token so it joins
            # the shared category space rather than disappearing.
            if va is None:
                va = "__null__"
            if vb is None:
                vb = "__null__"
            out.append(Pair(
                case_id=f"{cid}::{field}",
                organ=organ,
                a=va, b=vb,
                raw_a=va, raw_b=vb,
            ))
    return out


def pooled_categorical_kappa(pairs: Sequence[Pair]) -> float:
    """Cohen's κ over a pooled (case×field) Pair stream."""
    return cohen_kappa(pairs)


# --- Agree/disagree κ -------------------------------------------------------


def _continuous_within_tol(
    raw_a, raw_b, tol: float,
) -> bool:
    if raw_a is None or raw_b is None:
        return raw_a is None and raw_b is None
    try:
        a = float(raw_a)
        b = float(raw_b)
    except (TypeError, ValueError):
        return normalize(raw_a) == normalize(raw_b)
    return abs(a - b) <= tol


def _nested_list_agree(raw_a, raw_b, field: str) -> bool:
    """Treat nested-list values as agreeing iff bipartite F1 == 1.0."""
    if raw_a is None and raw_b is None:
        return True
    if raw_a is None or raw_b is None:
        return False
    try:
        r = match_nested_list(
            {"cancer_data": {field: raw_a}},
            {"cancer_data": {field: raw_b}},
            field,
        )
    except Exception:
        return False
    return bool(r.get("fp", 1) == 0 and r.get("fn", 1) == 0)


def agree_disagree_pairs(
    cases: dict[str, CaseEntry],
    *,
    ann_a: str,
    ann_b: str,
    fields: Sequence[str] | None = None,
    field_tolerances: dict[str, float] | None = None,
) -> list[Pair]:
    """Build the (case, field) → agree(1)/disagree(0) Pair stream.

    Categoricals: exact equality after :func:`metrics.normalize`.
    Continuous: |a − b| ≤ tolerance (per-field, default
    :data:`DEFAULT_FIELD_TOLERANCES`, fallback 1.0).
    Nested lists: bipartite F1 = 1.0 (no FPs, no FNs).
    Coverage mismatches (one attempted, the other not) count as
    disagreement.
    """
    fields = fields or _default_field_list()
    tolerances = (
        field_tolerances if field_tolerances is not None
        else DEFAULT_FIELD_TOLERANCES
    )
    out: list[Pair] = []
    for cid, entry in cases.items():
        a_ann = entry.annotations.get(ann_a)
        b_ann = entry.annotations.get(ann_b)
        if a_ann is None or b_ann is None:
            continue
        organ = entry.organ
        for field in fields:
            if organ is not None and not _field_applies_to_organ(field, organ):
                continue
            a_attempted = is_attempted(a_ann, field)
            b_attempted = is_attempted(b_ann, field)
            if not (a_attempted or b_attempted):
                continue
            if a_attempted != b_attempted:
                agree = False
            else:
                ftype = classify_field(field, organ)
                raw_a = get_field_value(a_ann, field)
                raw_b = get_field_value(b_ann, field)
                if ftype == "continuous":
                    tol = tolerances.get(field, _FALLBACK_CONTINUOUS_TOLERANCE)
                    agree = _continuous_within_tol(raw_a, raw_b, tol)
                elif ftype == "nested_list":
                    agree = _nested_list_agree(raw_a, raw_b, field)
                else:
                    agree = normalize(raw_a) == normalize(raw_b)
            out.append(Pair(
                case_id=f"{cid}::{field}",
                organ=organ,
                a=int(agree), b=1,
                raw_a=int(agree), raw_b=1,
            ))
    return out


def agree_disagree_pabak(pairs: Sequence[Pair]) -> float:
    """PABAK on the agree(1)/disagree(0) Pair stream.

    PABAK = 2·p_o − 1, where p_o is the observed agreement rate. The
    "B" rater in the Pair stream is constant (always 1) by construction,
    so a direct Cohen's κ is degenerate (p_e == p_o ⇒ κ == 0). PABAK
    sidesteps the degeneracy and yields the canonical
    "chance-corrected agreement assuming uniform marginals."
    """
    if not pairs:
        return float("nan")
    return pabak(pairs)


# --- Per-section roll-up ----------------------------------------------------


def per_section_rollup(
    per_field_df: pd.DataFrame,
    cases: dict[str, CaseEntry],
    *,
    ann_a: str,
    ann_b: str,
) -> pd.DataFrame:
    """One row per section × stat (mean κ, Krippendorff α).

    Mean κ is computed over the per-field κ values (organ='ALL').
    Krippendorff α is recomputed per section by recalling
    :func:`iaa.whole_report_stats` on a pseudo-cases dict that retains
    only fields in the section — but doing that requires field-level
    filtering inside ``whole_report_stats`` which it doesn't expose.
    Instead we approximate α by routing the existing per-type α numbers
    via section: nominal/ordinal α land under ``scalar_pathology`` (the
    bulk), and continuous α (interval) lands under ``scalar_pathology``
    or ``top_level`` depending on which fields contributed. For now we
    compute a per-section mean κ only and document the α omission.
    """
    rows: list[dict] = []
    pair_label = _pair_label(ann_a, ann_b)
    if per_field_df.empty:
        return pd.DataFrame(columns=list(PER_SECTION_COLUMNS))

    sub = per_field_df[per_field_df["organ"] == "ALL"].copy()
    sub = _prefer_kappa_per_field(sub)
    sub["section"] = sub["field"].map(classify_section)
    for section, grp in sub.groupby("section"):
        values = pd.to_numeric(grp["estimate"], errors="coerce").dropna()
        n_total = pd.to_numeric(grp["n"], errors="coerce").fillna(0).sum()
        rows.append({
            "pair": pair_label,
            "section": section,
            "stat_name": "mean_kappa",
            "estimate": float(values.mean()) if not values.empty else float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "n": int(n_total),
        })
    return pd.DataFrame(rows, columns=list(PER_SECTION_COLUMNS))


# --- Per-organ roll-up ------------------------------------------------------


def per_organ_rollup(
    cases: dict[str, CaseEntry],
    *,
    ann_a: str,
    ann_b: str,
    fields: Sequence[str] | None = None,
    field_tolerances: dict[str, float] | None = None,
    n_boot: int = 2000,
    random_state: int = 0,
    device: str = "cpu",
    per_field_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per-organ headline κ summary (4 stats × n_organs rows).

    For each observed organ we compute the same four κ flavours as the
    overall headline, restricted to that organ's cases.
    """
    rows: list[dict] = []
    pair_label = _pair_label(ann_a, ann_b)
    organs = sorted({e.organ for e in cases.values() if e.organ})
    if not organs:
        return pd.DataFrame(columns=list(PER_ORGAN_COLUMNS))

    for organ in organs:
        organ_cases = {cid: e for cid, e in cases.items() if e.organ == organ}
        n_cases = sum(
            1 for e in organ_cases.values()
            if ann_a in e.annotations and ann_b in e.annotations
        )
        if n_cases == 0:
            continue

        organ_field_df = (
            per_field_df[per_field_df["organ"] == organ]
            if per_field_df is not None and not per_field_df.empty
            else per_field_kappas(
                organ_cases, ann_a=ann_a, ann_b=ann_b, fields=fields,
                n_boot=n_boot, random_state=random_state, device=device,
            )
        )
        # Re-tag organ='ALL' for the helpers so they apply within-organ.
        if not organ_field_df.empty:
            organ_field_df = organ_field_df.copy()
            organ_field_df["organ"] = "ALL"
        mean_k, n_fields = mean_per_field_kappa(organ_field_df)
        wmean_k, _ = n_weighted_mean_per_field_kappa(organ_field_df)
        pooled = pooled_categorical_pairs(
            organ_cases, ann_a=ann_a, ann_b=ann_b, fields=fields,
        )
        agree = agree_disagree_pairs(
            organ_cases, ann_a=ann_a, ann_b=ann_b, fields=fields,
            field_tolerances=field_tolerances,
        )
        pooled_k = pooled_categorical_kappa(pooled) if pooled else float("nan")
        agree_pabak = agree_disagree_pabak(agree) if agree else float("nan")

        for stat, est, n_used in (
            ("mean_per_field_kappa", mean_k, n_fields),
            ("n_weighted_mean_per_field_kappa", wmean_k, n_fields),
            ("pooled_categorical_kappa", pooled_k, len(pooled)),
            ("agree_disagree_pabak", agree_pabak, len(agree)),
        ):
            rows.append({
                "pair": pair_label,
                "organ": organ,
                "stat_name": stat,
                "estimate": est,
                "ci_lo": float("nan"),
                "ci_hi": float("nan"),
                "n_cases": n_cases,
                "n_fields": n_used,
            })
    return pd.DataFrame(rows, columns=list(PER_ORGAN_COLUMNS))


# --- Confusion matrices -----------------------------------------------------


def confusion_matrix_for_field(
    cases: dict[str, CaseEntry],
    field: str,
    *,
    ann_a: str,
    ann_b: str,
) -> pd.DataFrame:
    """Long-form (value_a, value_b, count) for a categorical field.

    Includes a ``__missing__`` row when one annotator skipped while the
    other attempted. Cases where neither attempted contribute nothing.
    """
    counter: dict[tuple[str, str], int] = {}
    for _cid, entry in cases.items():
        a_ann = entry.annotations.get(ann_a)
        b_ann = entry.annotations.get(ann_b)
        if a_ann is None or b_ann is None:
            continue
        a_attempted = is_attempted(a_ann, field)
        b_attempted = is_attempted(b_ann, field)
        if not (a_attempted or b_attempted):
            continue
        va = _label(a_ann, field) if a_attempted else MISSING_TOKEN
        vb = _label(b_ann, field) if b_attempted else MISSING_TOKEN
        counter[(va, vb)] = counter.get((va, vb), 0) + 1
    rows = [
        {"value_a": va, "value_b": vb, "count": n}
        for (va, vb), n in sorted(counter.items())
    ]
    return pd.DataFrame(rows, columns=list(CONFUSION_COLUMNS))


def _label(ann: dict, field: str) -> str:
    v = get_field_value(ann, field)
    n = normalize(v)
    if n is None:
        return "__null__"
    return str(n)


def disagreement_count_per_field(
    cases: dict[str, CaseEntry],
    *,
    ann_a: str,
    ann_b: str,
    fields: Sequence[str] | None = None,
) -> dict[str, int]:
    """Count of (case, field) tuples where the two annotators disagree.

    Used to pick the top-N categorical fields for confusion-matrix
    output. Coverage mismatches count as disagreement.
    """
    fields = fields or _default_field_list()
    out: dict[str, int] = {}
    for _cid, entry in cases.items():
        a_ann = entry.annotations.get(ann_a)
        b_ann = entry.annotations.get(ann_b)
        if a_ann is None or b_ann is None:
            continue
        organ = entry.organ
        for field in fields:
            if organ is not None and not _field_applies_to_organ(field, organ):
                continue
            ftype = classify_field(field, organ)
            if ftype not in ("binary", "nominal", "ordinal"):
                continue
            a_attempted = is_attempted(a_ann, field)
            b_attempted = is_attempted(b_ann, field)
            if not (a_attempted or b_attempted):
                continue
            if a_attempted != b_attempted:
                out[field] = out.get(field, 0) + 1
                continue
            if normalize(get_field_value(a_ann, field)) != normalize(
                get_field_value(b_ann, field)
            ):
                out[field] = out.get(field, 0) + 1
    return out


def top_disagreements(per_field_df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """Top-k fields by lowest κ (most disagreement). organ='ALL' rows only."""
    if per_field_df.empty:
        return per_field_df.copy()
    sub = per_field_df[per_field_df["organ"] == "ALL"].copy()
    sub = _prefer_kappa_per_field(sub)
    sub["estimate"] = pd.to_numeric(sub["estimate"], errors="coerce")
    sub = sub.dropna(subset=["estimate"]).sort_values("estimate", ascending=True)
    return sub.head(k).reset_index(drop=True)


# --- Top-level entry point: compute_headline --------------------------------


def compute_headline(
    cases: dict[str, CaseEntry],
    *,
    ann_a: str,
    ann_b: str,
    fields: Sequence[str] | None = None,
    field_tolerances: dict[str, float] | None = None,
    n_boot: int = 2000,
    random_state: int = 0,
    device: str = "cpu",
    per_field_df: pd.DataFrame | None = None,
) -> list[HeadlineRow]:
    """All headline statistics for a single ordered pair.

    Returns a list of :class:`HeadlineRow` in the order they should
    appear in the markdown summary. The two pooled κ stats carry
    case-level bootstrap CIs; the two field-mean stats are point-only
    (CI = NaN); the α / case-exact-match stats are forwarded from
    :func:`iaa.whole_report_stats`.
    """
    rows: list[HeadlineRow] = []

    pf_df = (
        per_field_df if per_field_df is not None
        else per_field_kappas(
            cases, ann_a=ann_a, ann_b=ann_b, fields=fields,
            n_boot=n_boot, random_state=random_state, device=device,
        )
    )

    # Field-level mean κ — point only (bootstrap would be n_fields × n_boot
    # κ recomputations; deferred behind a future opt-in flag).
    mean_k, n_fields = mean_per_field_kappa(pf_df)
    rows.append(HeadlineRow(
        stat_name="mean_per_field_kappa",
        estimate=mean_k, ci_lo=float("nan"), ci_hi=float("nan"),
        n=n_fields, note="field-level mean — CI omitted",
    ))
    wmean_k, n_pairs = n_weighted_mean_per_field_kappa(pf_df)
    rows.append(HeadlineRow(
        stat_name="n_weighted_mean_per_field_kappa",
        estimate=wmean_k, ci_lo=float("nan"), ci_hi=float("nan"),
        n=n_pairs, note="field-level mean — CI omitted",
    ))

    # Pooled categorical κ — case-level bootstrap CI (resample case_ids).
    pooled = pooled_categorical_pairs(
        cases, ann_a=ann_a, ann_b=ann_b, fields=fields,
    )
    if pooled:
        pooled_res = _bootstrap_kappa_by_case(
            pooled, n_boot=n_boot, random_state=random_state,
        )
        rows.append(HeadlineRow(
            stat_name="pooled_categorical_kappa",
            estimate=pooled_res.point,
            ci_lo=pooled_res.lo, ci_hi=pooled_res.hi,
            n=len(pooled), note="categorical fields only",
        ))
    else:
        rows.append(HeadlineRow(
            stat_name="pooled_categorical_kappa",
            estimate=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"),
            n=0, note="no categorical observations",
        ))

    # Agree/disagree PABAK — case-level bootstrap CI.
    agree = agree_disagree_pairs(
        cases, ann_a=ann_a, ann_b=ann_b, fields=fields,
        field_tolerances=field_tolerances,
    )
    if agree:
        agree_res = _bootstrap_pabak_by_case(
            agree, n_boot=n_boot, random_state=random_state,
        )
        rows.append(HeadlineRow(
            stat_name="agree_disagree_pabak",
            estimate=agree_res.point,
            ci_lo=agree_res.lo, ci_hi=agree_res.hi,
            n=len(agree),
            note="PABAK = 2·p_o − 1; tolerance for continuous, F1=1 for nested",
        ))
    else:
        rows.append(HeadlineRow(
            stat_name="agree_disagree_pabak",
            estimate=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"),
            n=0, note="no observations",
        ))

    # Krippendorff α + case-exact-match — borrow from whole_report_stats.
    wr = whole_report_stats(cases, ann_a=ann_a, ann_b=ann_b)
    for _idx, r in wr.iterrows():
        rows.append(HeadlineRow(
            stat_name=str(r["stat_name"]),
            estimate=_to_float(r["estimate"]),
            ci_lo=_to_float(r.get("ci_lo")),
            ci_hi=_to_float(r.get("ci_hi")),
            n=int(r.get("n") or 0),
            note=None,
        ))
    return rows


def _to_float(v) -> float:
    if v is None:
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _bootstrap_kappa_by_case(
    pairs: Sequence[Pair],
    *,
    n_boot: int,
    random_state: int,
):
    """Resample by underlying case_id and recompute Cohen's κ.

    The Pair streams from :func:`pooled_categorical_pairs` and
    :func:`agree_disagree_pairs` use ``case_id == "<cid>::<field>"`` so
    multiple Pairs from the same source case move together under a
    single bootstrap draw.
    """
    return _bootstrap_by_case(
        pairs, statistic=cohen_kappa,
        n_boot=n_boot, random_state=random_state,
    )


def _bootstrap_pabak_by_case(
    pairs: Sequence[Pair],
    *,
    n_boot: int,
    random_state: int,
):
    """Case-level bootstrap of PABAK on the agree/disagree indicator."""
    return _bootstrap_by_case(
        pairs, statistic=pabak,
        n_boot=n_boot, random_state=random_state,
    )


def _bootstrap_by_case(
    pairs: Sequence[Pair],
    *,
    statistic,
    n_boot: int,
    random_state: int,
):
    by_case: dict[str, list[Pair]] = {}
    for p in pairs:
        cid = p.case_id.split("::", 1)[0]
        by_case.setdefault(cid, []).append(p)
    case_ids = list(by_case.keys())
    strata = [by_case[cid][0].organ for cid in case_ids]

    def _stat(sample_cases: Sequence[str]) -> float:
        flat: list[Pair] = []
        for cid in sample_cases:
            flat.extend(by_case[cid])
        return statistic(flat)

    return bootstrap_ci(
        case_ids, _stat,
        n_boot=n_boot, strata=strata, random_state=random_state,
    )


# --- DataFrame conversion ---------------------------------------------------


def headline_rows_to_df(rows: Sequence[HeadlineRow], pair_label: str) -> pd.DataFrame:
    return pd.DataFrame(
        [{
            "pair": pair_label,
            "stat_name": r.stat_name,
            "estimate": r.estimate,
            "ci_lo": r.ci_lo,
            "ci_hi": r.ci_hi,
            "n": r.n,
            "note": r.note,
        } for r in rows],
        columns=list(HEADLINE_COLUMNS),
    )


__all__ = [
    "HeadlineRow",
    "MISSING_TOKEN",
    "DEFAULT_FIELD_TOLERANCES",
    "HEADLINE_COLUMNS",
    "PER_FIELD_COLUMNS",
    "PER_SECTION_COLUMNS",
    "PER_ORGAN_COLUMNS",
    "CONFUSION_COLUMNS",
    "compute_headline",
    "headline_rows_to_df",
    "per_field_kappas",
    "mean_per_field_kappa",
    "n_weighted_mean_per_field_kappa",
    "pooled_categorical_pairs",
    "pooled_categorical_kappa",
    "agree_disagree_pairs",
    "agree_disagree_pabak",
    "per_section_rollup",
    "per_organ_rollup",
    "confusion_matrix_for_field",
    "disagreement_count_per_field",
    "top_disagreements",
]
