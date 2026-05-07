"""Three-way outcome classifier: ``correct / wrong / missing``.

This is the foundation of every model-vs-gold scoring step. Each
``(case, field, run)`` tuple is classified into one of three buckets.
Missing further decomposes into ``parse_error`` (whole-case failure) vs
``field_missing`` (case loaded but specific field null/absent).

The classifier reuses :func:`digital_registrar_research.benchmarks.eval.metrics.field_correct`
for the correct/wrong decision, so existing tolerance rules (±2 mm on
``tumor_size``, etc.) carry over unchanged. The new bit is the explicit
``missing`` flag, which today's binary correctness scoring conflates
with "wrong."

Why this matters: missing fields usually indicate runtime errors —
parse failure, schema-shy model, context-window pressure — and those
are categorically different failure modes from "model tried and got it
wrong." The ablation argument (modular DSPy vs raw_json) hinges on this
distinction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from digital_registrar_research.benchmarks.eval.metrics import (
    field_correct,
    is_attempted,
    normalize,
)
from digital_registrar_research.benchmarks.eval.scope import (
    LIST_OF_LITERALS_FIELDS, get_field_value,
    get_list_of_literals_fields,
)

from .loaders import ErrorMode, LoadOutcome


OutcomeKind = Literal["correct", "wrong", "field_missing", "parse_error", "ineligible"]


def _is_list_of_literals_field(field_name: str, organ: str | None = None) -> bool:
    """Return whether ``field_name`` is list-of-literals for ``organ``.

    The same field name can be a scalar in one organ and a list-of-
    literals in another (e.g. ``tumor_extent`` is a list-of-literals
    only for liver — for esophagus/stomach it's a regular categorical).
    Pass ``organ`` to scope the lookup; if ``None``, falls back to the
    cross-organ union (used at field-discovery time).
    """
    if organ is None:
        return field_name in LIST_OF_LITERALS_FIELDS
    return field_name in get_list_of_literals_fields(organ)


def _normalize_set(v) -> frozenset:
    """Coerce a list-of-literals value to a normalised frozenset.

    ``None`` and the empty list both map to ``frozenset()`` so they
    compare equal — important because the schema's "no involvement"
    canonical form is ``[]`` but some annotators may write ``None``.
    """
    if v is None:
        return frozenset()
    if isinstance(v, list):
        return frozenset(normalize(x) for x in v if x is not None)
    # Fallback: treat scalar as a single-element set so unequal lengths
    # show up as wrong rather than crashing.
    return frozenset({normalize(v)})


def list_of_literals_match(gold_value, pred_value) -> bool:
    """Exact unordered-set match for list-of-literals fields."""
    return _normalize_set(gold_value) == _normalize_set(pred_value)


def list_of_literals_set_metrics(gold_value, pred_value) -> dict:
    """Item-level TP / FP / FN + F1 for list-of-literals fields.

    Useful for partial-credit reporting (analogous to nested-list F1
    but on plain string items rather than dicts).
    """
    g = _normalize_set(gold_value)
    p = _normalize_set(pred_value)
    tp = len(g & p)
    fp = len(p - g)
    fn = len(g - p)
    if tp + fp == 0 or tp + fn == 0:
        f1 = 0.0
    else:
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "f1": f1,
            "exact_match": g == p}


@dataclass(frozen=True)
class Outcome:
    """Per-(case, field) outcome.

    Flags are mutually exclusive and exhaustive within ``eligible``
    cases. ``ineligible`` is a separate bucket for cases the field
    doesn't apply to (e.g. breast biomarkers when cancer is colorectal).

    Conventions:
        - ``parse_error``: case-level load failed — every field of this
          case is parse_error. ``error_mode`` is set on the case load
          and propagates here.
        - ``field_missing``: case loaded but this field is absent or
          null on the prediction side, while gold has a non-null value.
        - ``attempted``: the model produced a value (even if wrong).
        - ``correct``: ``attempted AND value matches gold``.
        - ``wrong``: ``attempted AND not correct``.

    Sum invariant: across an eligible cohort,
        ``correct + wrong + field_missing + parse_error == n_eligible``.
    """

    kind: OutcomeKind
    gold_present: bool
    parse_error: bool
    field_missing: bool
    attempted: bool
    correct: bool
    wrong: bool
    error_mode: ErrorMode | None = None
    pred_value: Any = None
    gold_value: Any = None

    def as_row(self) -> dict[str, Any]:
        """Flat dict for DataFrame ingestion."""
        return {
            "outcome": self.kind,
            "gold_present": self.gold_present,
            "parse_error": self.parse_error,
            "field_missing": self.field_missing,
            "attempted": self.attempted,
            "correct": self.correct,
            "wrong": self.wrong,
            "error_mode": self.error_mode,
        }


@dataclass(frozen=True)
class CaseLoad:
    """Captures how a single case loaded.

    Used by :func:`classify_outcome` to short-circuit field-level
    classification when the case-level load failed.
    """

    ok: bool
    pred: dict[str, Any] | None
    error_mode: ErrorMode | None
    error_message: str | None = None

    @classmethod
    def from_load_outcome(cls, lo: LoadOutcome) -> "CaseLoad":
        return cls(
            ok=lo.ok,
            pred=lo.data if lo.ok else None,
            error_mode=lo.error_mode,
            error_message=lo.error_message,
        )

    @classmethod
    def ok_load(cls, pred: dict[str, Any]) -> "CaseLoad":
        return cls(ok=True, pred=pred, error_mode=None, error_message=None)

    @classmethod
    def parse_failed(cls, mode: ErrorMode, message: str | None = None) -> "CaseLoad":
        return cls(ok=False, pred=None, error_mode=mode, error_message=message)


# --- Eligibility -------------------------------------------------------------

def is_field_eligible(gold: dict[str, Any], field_name: str,
                      *, organ: str | None = None) -> bool:
    """Return whether a field is eligible for scoring on a given gold case.

    Today's policy: a field is eligible if gold has the key (value or
    null). Breast biomarkers are eligible only when ``cancer_category ==
    "breast"``. Other organ-specific fields are filtered upstream by
    the caller selecting which fields to score per organ.
    """
    if field_name.startswith("biomarker_"):
        # Breast biomarkers are conditional.
        from digital_registrar_research.benchmarks.eval.metrics import normalize
        return normalize(gold.get("cancer_category")) == "breast"
    # Default: any field declared by the gold annotation is eligible.
    # Returns True even for null gold values — callers may further
    # restrict to ``gold_present`` based on their metric definition.
    if field_name in gold:
        return True
    cd = gold.get("cancer_data") or {}
    return field_name in cd


# --- Classifier --------------------------------------------------------------

def classify_outcome(
    gold: dict[str, Any],
    case_load: CaseLoad,
    field_name: str,
    *,
    organ: str | None = None,
) -> Outcome:
    """Classify one ``(case, field)`` outcome.

    ``case_load`` summarises how the prediction file loaded — if it
    failed at the case level, every field becomes ``parse_error``.

    Dispatches by field type:
        - List-of-literals fields (per ``ORGAN_LIST_OF_LITERALS``) score
          by unordered set equality. Pass ``organ`` for organ-aware
          dispatch — the same field name can be scalar in one organ and
          list-of-literals in another (e.g. ``tumor_extent``).
        - Everything else scores via :func:`metrics.field_correct`.
    """
    if organ is None:
        # Best effort — fall back to gold's cancer_category.
        from digital_registrar_research.benchmarks.eval.metrics import normalize as _norm
        organ = _norm(gold.get("cancer_category"))
    g_value = get_field_value(gold, field_name)
    is_list_literals = _is_list_of_literals_field(field_name, organ=organ)
    g_value_is_list = isinstance(g_value, list)
    if is_list_literals:
        # An empty list is *not* the same as null for set-equality
        # scoring — empty list means "no items present" (a definite
        # answer); null means "not assessed".
        gold_present = g_value_is_list  # only count actual lists
    else:
        gold_present = g_value is not None

    if not case_load.ok:
        return Outcome(
            kind="parse_error",
            gold_present=gold_present,
            parse_error=True,
            field_missing=False,
            attempted=False,
            correct=False,
            wrong=False,
            error_mode=case_load.error_mode,
            pred_value=None,
            gold_value=g_value,
        )

    pred = case_load.pred or {}
    attempted_flag = is_attempted(pred, field_name)
    if not attempted_flag:
        return Outcome(
            kind="field_missing",
            gold_present=gold_present,
            parse_error=False,
            field_missing=True,
            attempted=False,
            correct=False,
            wrong=False,
            error_mode=None,
            pred_value=None,
            gold_value=g_value,
        )

    p_value = get_field_value(pred, field_name)

    if is_list_literals:
        # Cascade redesign: when gold has no list to score against, the
        # field is ``ineligible`` for accuracy. The legacy "empty pred =
        # correct refusal" branch was dropped because it conflated
        # silence with correct silence and inflated Stage-C accuracy.
        if not g_value_is_list:
            return Outcome(
                kind="ineligible",
                gold_present=False,
                parse_error=False,
                field_missing=False,
                attempted=True,
                correct=False,
                wrong=False,
                error_mode=None,
                pred_value=p_value,
                gold_value=g_value,
            )
        is_correct = list_of_literals_match(g_value, p_value)
        return Outcome(
            kind="correct" if is_correct else "wrong",
            gold_present=True,
            parse_error=False,
            field_missing=False,
            attempted=True,
            correct=is_correct,
            wrong=not is_correct,
            error_mode=None,
            pred_value=p_value,
            gold_value=g_value,
        )

    correct_flag = field_correct(gold, pred, field_name)
    # ``field_correct`` returns None when not attempted; we already
    # handled that branch.
    if correct_flag is True:
        return Outcome(
            kind="correct",
            gold_present=gold_present,
            parse_error=False,
            field_missing=False,
            attempted=True,
            correct=True,
            wrong=False,
            error_mode=None,
            pred_value=p_value,
            gold_value=g_value,
        )
    return Outcome(
        kind="wrong",
        gold_present=gold_present,
        parse_error=False,
        field_missing=False,
        attempted=True,
        correct=False,
        wrong=True,
        error_mode=None,
        pred_value=p_value,
        gold_value=g_value,
    )


__all__ = [
    "Outcome",
    "OutcomeKind",
    "CaseLoad",
    "is_field_eligible",
    "classify_outcome",
    "list_of_literals_match",
    "list_of_literals_set_metrics",
]
