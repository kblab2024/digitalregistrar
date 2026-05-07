"""Re-nesting helper for the F3 ablation (flat schema variant).

F3 asks the model to emit nested-list fields (``margins``, ``biomarkers``,
``regional_lymph_node``) as flat lists of strings. To grade against the
existing nested-bipartite metric in
:func:`digital_registrar_research.benchmarks.eval.metrics.match_nested_list`
we have to re-parse those strings into the dict-of-fields shape.

This re-parser is intentionally lossy — that loss is precisely what F3
is meant to quantify: how much accuracy is left on the table when the
list structure is removed before the model writes its answer.
"""
from __future__ import annotations

import re

from ...benchmarks.eval.metrics import NESTED_KEY


def _split_string(s: str) -> list[str]:
    """Split a field string into ``key: value`` chunks."""
    parts = re.split(r"[;,|]", s)
    return [p.strip() for p in parts if p.strip()]


def _parse_kv(chunk: str) -> tuple[str, str] | None:
    if ":" in chunk:
        k, v = chunk.split(":", 1)
        return k.strip().lower().replace(" ", "_"), v.strip()
    if "=" in chunk:
        k, v = chunk.split("=", 1)
        return k.strip().lower().replace(" ", "_"), v.strip()
    return None


def _parse_one_item(text: str, primary_key: str) -> dict:
    """Parse a single line/string into a dict.

    The first chunk (split on the first comma/semicolon) becomes the
    primary key value; remaining chunks may be ``key: value`` pairs.
    """
    item: dict = {}
    chunks = _split_string(text)
    if not chunks:
        return item
    # First chunk → primary key value; falls back to whole string.
    first = chunks[0]
    kv = _parse_kv(first)
    if kv:
        k, v = kv
        item[k] = v
    else:
        item[primary_key] = first
    for chunk in chunks[1:]:
        kv = _parse_kv(chunk)
        if kv:
            item[kv[0]] = kv[1]
    return item


def flat_to_nested(flat_value, field: str) -> list[dict]:
    """Re-parse a flat list-of-strings into a list-of-dicts.

    ``flat_value`` may be a list of strings, a dict (already structured —
    return as-is), or a single string (treated as a one-element list).
    Unknown ``field`` falls back to a generic ``{value: ...}`` shape.
    """
    if flat_value is None:
        return []
    primary_key = NESTED_KEY.get(field, "value")
    if isinstance(flat_value, list):
        items: list[dict] = []
        for entry in flat_value:
            if isinstance(entry, dict):
                items.append(entry)
            elif isinstance(entry, str) and entry.strip():
                items.append(_parse_one_item(entry, primary_key))
        return items
    if isinstance(flat_value, dict):
        return [flat_value]
    if isinstance(flat_value, str) and flat_value.strip():
        return [_parse_one_item(flat_value, primary_key)]
    return []


def renest_cancer_data(cancer_data: dict) -> dict:
    """Re-nest every known nested-list field in a cancer_data dict."""
    if not isinstance(cancer_data, dict):
        return cancer_data
    out = dict(cancer_data)
    for field in NESTED_KEY:
        if field in out and not _is_nested(out[field]):
            out[field] = flat_to_nested(out[field], field)
    return out


def _is_nested(value) -> bool:
    return (
        isinstance(value, list)
        and value
        and all(isinstance(item, dict) for item in value)
    )


__all__ = ["flat_to_nested", "renest_cancer_data"]
