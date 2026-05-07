"""'Others' ledger and subtype classifier for the cascade.

When gold ``cancer_category == "others"`` or the prediction does, the
case is diverted out of Stage C scoring (no per-organ schema applies)
and recorded in this ledger. Gives the rebuttal concrete numbers for
"how many dual-primary cases the model handled" and surfaces the
bilateral-breast pattern the manuscript already describes.

The subtype classifier is a regex heuristic, not a clinical decision —
it tags free-text gold descriptions as ``dual_primary`` if they
mention bilateral / synchronous / multi-primary terminology, else
``out_of_scope``. The reviewer-response document quotes this number
with the caveat that 5 spot-checks against gold descriptions agreed
with the heuristic.
"""
from __future__ import annotations

import re
from typing import Literal, TypedDict

import pandas as pd

from digital_registrar_research.benchmarks.eval.metrics import normalize


OthersSubtype = Literal["dual_primary", "out_of_scope", "unknown"]


_DUAL_PRIMARY_PATTERNS = re.compile(
    r"\b(double|dual|synchronous|bilateral|two\s+primaries?|"
    r"second\s+primar|multiple\s+primar|simultaneous\s+primar|"
    r"co[-\s]?existing\s+(tumou?r|cancer|malignan))\b",
    re.IGNORECASE,
)


def classify_others_subtype(description: str | None) -> OthersSubtype:
    """Classify an 'others' description into a subtype bucket.

    Returns:
        ``"dual_primary"`` — description mentions multiple primaries
        or bilateral / synchronous tumors.
        ``"out_of_scope"`` — description present but no dual-primary
        keyword matched (interpreted as a single-primary cancer outside
        the 10-organ schema).
        ``"unknown"`` — description is empty or None.
    """
    if not description or not isinstance(description, str):
        return "unknown"
    return "dual_primary" if _DUAL_PRIMARY_PATTERNS.search(description) else "out_of_scope"


class OthersRow(TypedDict, total=False):
    run_id: str
    model: str
    dataset: str
    case_id: str
    organ_idx: int
    gold_cancer_category: str | None
    pred_cancer_category: str | None
    gold_others_description: str | None
    pred_others_description: str | None
    others_subtype: OthersSubtype
    stage_a_correct: bool | None
    stage_b_disposition: str  # both_others | gold_only | pred_only
    notes: str


def build_others_row(
    *,
    run_id: str,
    model: str,
    dataset: str,
    case_id: str,
    organ_idx: int,
    gold: dict,
    pred: dict | None,
    stage_a_correct: bool | None,
    others_disposition: str,
) -> OthersRow:
    """Build one row for the others ledger.

    ``others_disposition`` comes from ``score_case`` (one of "gold_others",
    "pred_others", "both_others"). The function maps it to the
    ledger's ``stage_b_disposition`` field.
    """
    gold_desc = gold.get("cancer_category_others_description")
    pred_desc = pred.get("cancer_category_others_description") if pred else None
    # Subtype is derived from the gold description when available; falls
    # back to pred description when the case is purely pred-others.
    desc_for_classification = gold_desc or pred_desc
    subtype = classify_others_subtype(desc_for_classification)

    if others_disposition == "both_others":
        stage_b_disposition = "both_others"
    elif others_disposition == "gold_others":
        stage_b_disposition = "gold_only"
    elif others_disposition == "pred_others":
        stage_b_disposition = "pred_only"
    else:
        stage_b_disposition = "none"

    return {
        "run_id": run_id,
        "model": model,
        "dataset": dataset,
        "case_id": case_id,
        "organ_idx": int(organ_idx),
        "gold_cancer_category": gold.get("cancer_category"),
        "pred_cancer_category": pred.get("cancer_category") if pred else None,
        "gold_others_description": gold_desc,
        "pred_others_description": pred_desc,
        "others_subtype": subtype,
        "stage_a_correct": stage_a_correct,
        "stage_b_disposition": stage_b_disposition,
        "notes": str(desc_for_classification or "").strip(),
    }


def others_subtype_breakdown(ledger: pd.DataFrame) -> pd.DataFrame:
    """Counts of (subtype, disposition) per (dataset, model)."""
    if ledger.empty:
        return pd.DataFrame(columns=[
            "dataset", "model", "others_subtype", "stage_b_disposition", "n",
        ])
    grp = (ledger.groupby(
        ["dataset", "model", "others_subtype", "stage_b_disposition"],
        dropna=False,
    ).size().reset_index(name="n"))
    return grp


def others_confusion(ledger: pd.DataFrame) -> pd.DataFrame:
    """Directional 2-way breakdown of gold/pred 'others' cases.

    Two views are stacked:
        gold_others_into_X      gold = others, pred = X
        gold_X_into_others      gold = X (specific organ), pred = others
    Useful for the manuscript's claim "three Others-labeled reports
    were predicted as breast cancer" — a one-row filter.
    """
    if ledger.empty:
        return pd.DataFrame(columns=[
            "dataset", "model", "view",
            "gold_cancer_category", "pred_cancer_category", "n",
        ])
    rows: list[dict] = []
    for (dataset, model), sub in ledger.groupby(
        ["dataset", "model"], dropna=False,
    ):
        gold_others = sub[sub["stage_b_disposition"].isin(
            ["gold_only", "both_others"],
        )]
        for (gold, pred), n in gold_others.groupby(
            ["gold_cancer_category", "pred_cancer_category"], dropna=False,
        ).size().items():
            rows.append({
                "dataset": dataset, "model": model,
                "view": "gold_others_into_X",
                "gold_cancer_category": gold,
                "pred_cancer_category": pred,
                "n": int(n),
            })
        pred_others = sub[sub["stage_b_disposition"].isin(
            ["pred_only", "both_others"],
        )]
        for (gold, pred), n in pred_others.groupby(
            ["gold_cancer_category", "pred_cancer_category"], dropna=False,
        ).size().items():
            rows.append({
                "dataset": dataset, "model": model,
                "view": "gold_X_into_others",
                "gold_cancer_category": gold,
                "pred_cancer_category": pred,
                "n": int(n),
            })
    return pd.DataFrame(rows)


def others_eligibility_audit(ledger: pd.DataFrame) -> pd.DataFrame:
    """For gold-others cases: did Stage A still mark them eligible?

    These are real cancer surgeries that the 10-organ schema cannot
    represent. The model SHOULD still classify them as eligible
    surgery reports — Stage A failure on a gold-others case is a
    signal that triage was confused by the schema mismatch.

    Returns: dataset, model, n_gold_others, n_stage_a_correct,
    rate_stage_a_correct.
    """
    if ledger.empty:
        return pd.DataFrame(columns=[
            "dataset", "model", "n_gold_others",
            "n_stage_a_correct", "rate_stage_a_correct",
        ])
    rows: list[dict] = []
    gold_only = ledger[ledger["stage_b_disposition"].isin(
        ["gold_only", "both_others"],
    )]
    for (dataset, model), sub in gold_only.groupby(
        ["dataset", "model"], dropna=False,
    ):
        n_total = len(sub)
        n_correct = int(sub["stage_a_correct"].fillna(False).astype(bool).sum())
        rate = n_correct / n_total if n_total else float("nan")
        rows.append({
            "dataset": dataset, "model": model,
            "n_gold_others": n_total,
            "n_stage_a_correct": n_correct,
            "rate_stage_a_correct": rate,
        })
    return pd.DataFrame(rows)


__all__ = [
    "OthersSubtype",
    "OthersRow",
    "classify_others_subtype",
    "build_others_row",
    "others_subtype_breakdown",
    "others_confusion",
    "others_eligibility_audit",
]
