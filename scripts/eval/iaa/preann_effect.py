"""Pre-annotation effect: paired analyses on the with/without preann cohort.

Drives the headline "anchoring effect" tables. Walks the paired case
intersection (cases where the same annotator wrote BOTH ``with_preann``
and ``without_preann`` annotations), pulls per-field values from each
mode plus gold and the underlying preann model, and dispatches to the
``preann.py`` library functions.
"""
from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd

from digital_registrar_research.benchmarks.eval.iaa import (
    classify_field, classify_section,
)
from digital_registrar_research.benchmarks.eval.metrics import (
    is_attempted, normalize,
)
from digital_registrar_research.benchmarks.eval.preann import (
    DualPairedRecord, PairedRecord, anchoring_index,
    convergence_to_preann, disagreement_reduction,
    edit_distance_from_preann, paired_delta_kappa,
)
from digital_registrar_research.benchmarks.eval.scope import (
    BREAST_BIOMARKERS, FAIR_SCOPE, get_field_value,
)

from .._common.loaders import load_json, ParseError
from .._common.paths import Paths
from .._common.pairing import discover_paired_cases, PairedCase
from .._common.stratify import organ_name

logger = logging.getLogger(__name__)


SCALAR_FIELDS = list(FAIR_SCOPE) + [f"biomarker_{b}" for b in BREAST_BIOMARKERS]


def collect_paired_records(
    paths: Paths,
    annotator: str,
    *,
    preann_model: str,
    organs: Iterable[int],
) -> list[PairedRecord]:
    """Build the long-form paired record list for one annotator.

    Each (case, field) where gold is non-null contributes one
    PairedRecord with the with-preann human value, without-preann human
    value, preann (model) value, and gold.
    """
    paired = discover_paired_cases(paths, annotator, organs=tuple(organs))
    if not paired:
        return []

    records: list[PairedRecord] = []
    for pc in paired:
        try:
            with_ann = load_json(pc.with_path)
            without_ann = load_json(pc.without_path)
            gold = load_json(paths.gold(pc.organ_idx, pc.case_id))
        except ParseError as e:
            logger.warning("skipping %s: %s", pc.case_id, e)
            continue
        # Preann is optional — read defensively.
        preann = None
        preann_path = paths.preannotation(preann_model, pc.organ_idx, pc.case_id)
        try:
            preann = load_json(preann_path)
        except ParseError:
            preann = None

        organ = normalize(gold.get("cancer_category")) or organ_name(paths.dataset, pc.organ_idx)
        for field in SCALAR_FIELDS:
            if field.startswith("biomarker_") and organ != "breast":
                continue
            g_val = _read_field(gold, field)
            if g_val is None:
                continue
            w_val = _read_field(with_ann, field) if is_attempted(with_ann, _bare_field(field)) else None
            wo_val = _read_field(without_ann, field) if is_attempted(without_ann, _bare_field(field)) else None
            p_val = _read_field(preann, field) if preann else None
            records.append(PairedRecord(
                case_id=pc.case_id, organ=organ, field=field,
                with_value=w_val, without_value=wo_val,
                preann_value=p_val, gold_value=g_val,
            ))
    return records


def collect_dual_paired_records(
    paths: Paths, *, preann_model: str, organs: Iterable[int],
) -> list[DualPairedRecord]:
    """For both annotators (nhc + kpc), collect aligned values across
    all four annotator-mode combinations.

    Used for ``disagreement_reduction``: shared cases where both
    nhc+kpc have both with- and without-preann annotations.
    """
    nhc_paired = {
        (pc.organ_idx, pc.case_id): pc
        for pc in discover_paired_cases(paths, "nhc", organs=tuple(organs))
    }
    kpc_paired = {
        (pc.organ_idx, pc.case_id): pc
        for pc in discover_paired_cases(paths, "kpc", organs=tuple(organs))
    }
    shared = sorted(set(nhc_paired) & set(kpc_paired))
    if not shared:
        return []

    records: list[DualPairedRecord] = []
    for key in shared:
        nhc_pc = nhc_paired[key]
        kpc_pc = kpc_paired[key]
        try:
            nhc_w = load_json(nhc_pc.with_path)
            nhc_wo = load_json(nhc_pc.without_path)
            kpc_w = load_json(kpc_pc.with_path)
            kpc_wo = load_json(kpc_pc.without_path)
            gold = load_json(paths.gold(nhc_pc.organ_idx, nhc_pc.case_id))
        except ParseError as e:
            logger.warning("skipping %s: %s", nhc_pc.case_id, e)
            continue
        organ = normalize(gold.get("cancer_category")) or organ_name(paths.dataset, nhc_pc.organ_idx)
        for field in SCALAR_FIELDS:
            if field.startswith("biomarker_") and organ != "breast":
                continue
            g_val = _read_field(gold, field)
            if g_val is None:
                continue
            records.append(DualPairedRecord(
                case_id=nhc_pc.case_id, organ=organ, field=field,
                a_with=_read_field(nhc_w, field),
                a_without=_read_field(nhc_wo, field),
                b_with=_read_field(kpc_w, field),
                b_without=_read_field(kpc_wo, field),
            ))
    return records


# --- Reductions to long-form CSV -------------------------------------------


def delta_kappa_table(
    records: list[PairedRecord],
    *,
    n_boot: int, seed: int,
    device: str = "cpu",
) -> pd.DataFrame:
    """One row per (organ, field) with κ_with, κ_without, Δκ + paired CI."""
    rows: list[dict] = []
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame([r.__dict__ for r in records])
    grouped = df.groupby(["organ", "field"], dropna=False)
    for (organ, field), sub in grouped:
        recs = [PairedRecord(**r) for r in sub.to_dict(orient="records")]
        result = paired_delta_kappa(recs, n_boot=n_boot, random_state=seed,
                                     device=device)
        rows.append({
            "organ": organ, "field": field,
            "field_kind": classify_field(field, organ),
            "section": classify_section(field),
            **result,
        })
    return pd.DataFrame(rows)


def convergence_table(records: list[PairedRecord]) -> pd.DataFrame:
    """Convergence-to-preann per (organ, field)."""
    rows: list[dict] = []
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame([r.__dict__ for r in records])
    grouped = df.groupby(["organ", "field"], dropna=False)
    for (organ, field), sub in grouped:
        recs = [PairedRecord(**r) for r in sub.to_dict(orient="records")]
        result = convergence_to_preann(recs)
        rows.append({"organ": organ, "field": field, **result})
    return pd.DataFrame(rows)


def anchoring_index_table(records: list[PairedRecord]) -> pd.DataFrame:
    """Anchoring index AI = P(human=preann | with) − P(human=preann | without)."""
    rows: list[dict] = []
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame([r.__dict__ for r in records])
    grouped = df.groupby(["organ", "field"], dropna=False)
    for (organ, field), sub in grouped:
        recs = [PairedRecord(**r) for r in sub.to_dict(orient="records")]
        result = anchoring_index(recs)
        rows.append({"organ": organ, "field": field, **result})
    return pd.DataFrame(rows)


def disagreement_reduction_table(
    records: list[DualPairedRecord], *, n_boot: int, seed: int,
    device: str = "cpu",
) -> pd.DataFrame:
    """Δ-disagreement (with vs without preann) per (organ, field)."""
    rows: list[dict] = []
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame([r.__dict__ for r in records])
    grouped = df.groupby(["organ", "field"], dropna=False)
    for (organ, field), sub in grouped:
        recs = [DualPairedRecord(**r) for r in sub.to_dict(orient="records")]
        result = disagreement_reduction(recs, n_boot=n_boot, random_state=seed,
                                         device=device)
        rows.append({"organ": organ, "field": field, **result})
    return pd.DataFrame(rows)


def edit_distance_summary(records: list[PairedRecord]) -> pd.DataFrame:
    """Per case, count of fields where with-preann human edited away
    from preann. Returns a per-organ summary."""
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame([r.__dict__ for r in records])
    rows: list[dict] = []
    for organ, sub in df.groupby("organ"):
        cases = sub.groupby("case_id")
        case_records = [
            [PairedRecord(**r) for r in g.to_dict(orient="records")]
            for _, g in cases
        ]
        result = edit_distance_from_preann(case_records)
        rows.append({"organ": organ, **result})
    return pd.DataFrame(rows)


# --- Helpers ----------------------------------------------------------------


def _read_field(annotation: dict | None, field: str):
    """Read a possibly-biomarker-synthetic field from an annotation."""
    if annotation is None:
        return None
    if not field.startswith("biomarker_"):
        return get_field_value(annotation, field)
    cat = field.removeprefix("biomarker_")
    bm_list = get_field_value(annotation, "biomarkers") or []
    for b in bm_list:
        if normalize(b.get("biomarker_category")) == cat:
            return b.get("expression")
    return None


def _bare_field(field: str) -> str:
    return "biomarkers" if field.startswith("biomarker_") else field


__all__ = [
    "collect_paired_records",
    "collect_dual_paired_records",
    "delta_kappa_table",
    "convergence_table",
    "anchoring_index_table",
    "disagreement_reduction_table",
    "edit_distance_summary",
]
