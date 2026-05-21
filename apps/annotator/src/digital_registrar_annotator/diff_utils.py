"""Pure helpers for comparing two annotations. No Streamlit imports."""

from __future__ import annotations

from dataclasses import dataclass

from .io import NA_SENTINEL

# ── Scalar equality ────────────────────────────────────────────────────────────

def values_differ(a, b) -> bool:
    """None, '', missing, and NA_SENTINEL are treated as equivalent.

    Mirrors the _values_differ helper in app.py: we normalise empties to
    None on save, so a freshly-typed "" must compare equal to a saved None.
    An NA_SENTINEL in session state also serialises to null, so for diff
    purposes it counts as empty too.
    """
    if _is_empty(a) and _is_empty(b):
        return False
    return a != b


def _is_empty(v) -> bool:
    if v is None:
        return True
    if isinstance(v, str) and (v == "" or v == NA_SENTINEL):
        return True
    if isinstance(v, list) and not v:
        return True
    return False


# ── Array alignment ────────────────────────────────────────────────────────────

# Map array_field_name → the per-item key used to align items between A and B.
ARRAY_KEY_FIELDS: dict[str, str] = {
    "margins": "margin_category",
    "biomarkers": "biomarker_category",
    "regional_lymph_node": "lymph_node_category",
}


def align_arrays_by_key(
    list_a: list[dict] | None,
    list_b: list[dict] | None,
    key_field: str,
) -> list[tuple[dict | None, dict | None, object]]:
    """Return aligned [(item_a, item_b, key), ...].

    Order: A's items first in their original order; B-only items appended after.
    Items missing a key are paired only by position within the "missing-key"
    bucket (defensive — real data should always have the category set).
    """
    a = list(list_a or [])
    b = list(list_b or [])

    b_remaining = list(enumerate(b))
    aligned: list[tuple[dict | None, dict | None, object]] = []

    for item_a in a:
        key = item_a.get(key_field)
        match_pos = None
        if key is not None:
            for pos, (_orig, item_b) in enumerate(b_remaining):
                if item_b.get(key_field) == key:
                    match_pos = pos
                    break
        if match_pos is not None:
            _orig, item_b = b_remaining.pop(match_pos)
            aligned.append((item_a, item_b, key))
        else:
            aligned.append((item_a, None, key))

    for _orig, item_b in b_remaining:
        aligned.append((None, item_b, item_b.get(key_field)))

    return aligned


# ── Field-level diffs ──────────────────────────────────────────────────────────

@dataclass
class FieldDiff:
    name: str
    title: str
    value_a: object
    value_b: object
    differs: bool


def diff_flat_fields(a: dict, b: dict, fields) -> list[FieldDiff]:
    """`fields` is a list of FieldSpec-like objects (duck-typed on .name/.title)."""
    out: list[FieldDiff] = []
    a = a or {}
    b = b or {}
    for f in fields:
        va = a.get(f.name)
        vb = b.get(f.name)
        out.append(FieldDiff(
            name=f.name,
            title=getattr(f, "title", f.name),
            value_a=va,
            value_b=vb,
            differs=values_differ(va, vb),
        ))
    return out


def aggregate_stats(field_diffs: list[FieldDiff]) -> dict:
    total = len(field_diffs)
    agree = sum(1 for d in field_diffs if not d.differs)
    pct = (agree / total) if total else 1.0
    return {"total": total, "agree": agree, "disagree": total - agree, "pct": pct}


# ── Flat-format section extraction ─────────────────────────────────────────────

def section_container(annotation: dict, section_name: str) -> dict:
    """Return the dict holding the fields for a given section name.

    IsCancer fields live at the top level; every other section's fields
    live under `cancer_data`. Mirrors the get/set helpers in app.py.
    """
    if section_name == "IsCancer":
        return annotation or {}
    return (annotation or {}).get("cancer_data") or {}
