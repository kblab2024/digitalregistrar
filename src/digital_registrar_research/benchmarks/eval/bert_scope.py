"""BERT-eligible field scope for the head-to-head comparison harness.

ClinicalBERT can only emit non-nested, non-list-of-literals, non-free-text
fields. Scoring it against the standard ``FAIR_SCOPE`` (9 fields) under-
covers what the model can actually do; scoring against the full schema
unfairly counts unattempted nested fields against it. This module
constructs the right middle ground:

    BERT-eligible(organ) =
        categorical_enums(organ)
        | boolean_fields(organ)
        | numeric_span_fields(organ)
        | {"cancer_category", "cancer_excision_report"}

Used by:
- ``eval/metrics.py:score_case(scope=bert_scope_for_organ)``
- ``eval/run_all.py --scope bert``
- ``eval/pairwise_compare.py``
"""
from __future__ import annotations

from .scope import (
    IMPLEMENTED_ORGANS,
    get_bool_fields,
    get_categorical_fields,
    get_span_fields,
)

TOP_LEVEL_FIELDS: set[str] = {"cancer_category", "cancer_excision_report"}


def bert_scope_for_organ(organ: str | None) -> set[str]:
    """Fields ClinicalBERT can legitimately produce for a case of this organ.

    Falls back to the cross-organ union when ``organ`` is None / unknown,
    so cases with ``cancer_category == "others"`` or ``null`` still get a
    sensible set of columns scored.
    """
    if organ is None or organ not in IMPLEMENTED_ORGANS:
        return BERT_SCOPE
    return (
        set(get_categorical_fields(organ).keys())
        | get_bool_fields(organ)
        | get_span_fields(organ)
        | TOP_LEVEL_FIELDS
    )


BERT_SCOPE: set[str] = set(TOP_LEVEL_FIELDS)
for _o in IMPLEMENTED_ORGANS:
    BERT_SCOPE |= (
        set(get_categorical_fields(_o).keys())
        | get_bool_fields(_o)
        | get_span_fields(_o)
    )


__all__ = ["bert_scope_for_organ", "BERT_SCOPE", "TOP_LEVEL_FIELDS"]
