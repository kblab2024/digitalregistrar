"""Curated semantic-neighbor lists per categorical field.

Some confusions matter less than others. ``pt2 ↔ pt2a`` is a sub-stage
nuance often clinically irrelevant; ``pn0 ↔ pn3`` is a major staging
error. The reviewer feedback (R2.1) called out one specific pattern —
``anatomic_stage_group`` ↔ ``pathologic_stage_group`` — where the model
sometimes substitutes one for the other.

This module is the single source of truth for "near-miss" pairs. Each
entry has:
    field       — the field where the confusion is plausible
    a, b        — the two values that can be conflated
    reason      — short string explaining why they're near-misses

Used by:
    - ``scripts/eval/non_nested/metrics_non_nested.py`` to compute
      ``accuracy_collapsing_neighbors`` alongside the strict accuracy.
    - ``docs/eval/confusion_pairs.md`` to render the curated-neighbor
      table for the writeup.

Neighbors are *symmetric*: ``(a, b)`` implies ``(b, a)``. Lookups via
:func:`is_neighbor`.

Lists are intentionally conservative — adding a pair here loosens the
strict accuracy definition, so only add when the clinical literature
agrees the values are interchangeable in this dataset's context.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .metrics import normalize


@dataclass(frozen=True)
class NeighborPair:
    field: str
    a: str
    b: str
    reason: str


# Initial curated list. EXTEND CAREFULLY — every addition makes the
# corresponding ``accuracy_collapsing_neighbors`` higher.
_CURATED: list[NeighborPair] = [
    # ---- Stage-group system substitutions (R2.1) ------------------------
    # Reviewer 2.1: model occasionally swapped anatomic for pathologic
    # staging on early-stage cases. The numeric stage values themselves
    # match — only the system label differs. Treated as near-miss when
    # the underlying stage value is the same.
    NeighborPair(
        field="tnm_descriptor", a="anatomic", b="pathologic",
        reason="anatomic vs pathologic staging system label",
    ),
    # ---- T-substages: a/b/c are sub-resolutions of the same major
    # stage. Many clinical workflows treat the major stage as the
    # actionable level.
    NeighborPair(
        field="pt_category", a="t1", b="t1a",
        reason="t1 major vs t1a substage",
    ),
    NeighborPair(
        field="pt_category", a="t1", b="t1b",
        reason="t1 major vs t1b substage",
    ),
    NeighborPair(
        field="pt_category", a="t1", b="t1c",
        reason="t1 major vs t1c substage",
    ),
    NeighborPair(
        field="pt_category", a="t2", b="t2a",
        reason="t2 major vs t2a substage",
    ),
    NeighborPair(
        field="pt_category", a="t2", b="t2b",
        reason="t2 major vs t2b substage",
    ),
    NeighborPair(
        field="pt_category", a="t3", b="t3a",
        reason="t3 major vs t3a substage",
    ),
    NeighborPair(
        field="pt_category", a="t3", b="t3b",
        reason="t3 major vs t3b substage",
    ),
    NeighborPair(
        field="pt_category", a="t4", b="t4a",
        reason="t4 major vs t4a substage",
    ),
    NeighborPair(
        field="pt_category", a="t4", b="t4b",
        reason="t4 major vs t4b substage",
    ),
    # ---- N-substages
    NeighborPair(
        field="pn_category", a="n1", b="n1a",
        reason="n1 major vs n1a substage",
    ),
    NeighborPair(
        field="pn_category", a="n1", b="n1b",
        reason="n1 major vs n1b substage",
    ),
    NeighborPair(
        field="pn_category", a="n2", b="n2a",
        reason="n2 major vs n2a substage",
    ),
    NeighborPair(
        field="pn_category", a="n2", b="n2b",
        reason="n2 major vs n2b substage",
    ),
    # ---- M staging — m0 vs mx are common clinically equivalent in
    # absence of distant-metastasis workup.
    NeighborPair(
        field="pm_category", a="m0", b="mx",
        reason="m0 (clinically negative) vs mx (not assessed) — often equivalent",
    ),
    # ---- Lymph node category (sentinel ↔ axillary level I overlap in
    # breast cases where the sentinel happens to be in level I).
    NeighborPair(
        field="lymph_node_category", a="sentinel", b="axillary_level_i",
        reason="sentinel node from level I axilla — overlap when only one node sampled",
    ),
    # ---- Histology subtype simplifications. Add as the data demands.
    NeighborPair(
        field="histology", a="invasive_carcinoma_no_special_type",
        b="invasive_ductal_carcinoma",
        reason="legacy 'invasive ductal carcinoma' renamed to 'NST' in 2012 WHO",
    ),
]


def _key(field: str, a: str, b: str) -> tuple[str, frozenset[str]]:
    return (field, frozenset((normalize(a) or "", normalize(b) or "")))


_INDEX: set[tuple[str, frozenset[str]]] = {_key(p.field, p.a, p.b) for p in _CURATED}


def is_neighbor(field: str, a: object, b: object) -> bool:
    """Return whether ``a`` and ``b`` are a curated near-miss pair for
    ``field``. Order-insensitive; uses normalised string comparison."""
    if a is None or b is None:
        return False
    return (field, frozenset((normalize(a) or "", normalize(b) or ""))) in _INDEX


def neighbors_for_field(field: str) -> list[NeighborPair]:
    """All curated pairs for a given field (in declaration order)."""
    return [p for p in _CURATED if p.field == field]


def all_pairs() -> Iterable[NeighborPair]:
    """Iterator over every curated pair."""
    return iter(_CURATED)


__all__ = [
    "NeighborPair",
    "is_neighbor",
    "neighbors_for_field",
    "all_pairs",
]
