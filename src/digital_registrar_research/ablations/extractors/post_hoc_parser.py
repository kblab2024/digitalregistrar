"""Post-hoc parser for the B2 ablation (DSPy with `str` outputs).

Cell B2 keeps the DSPy framework but strips every ``Literal[...]``
annotation down to ``str``. The model writes free-text values; we then
project each value back into the typed enum / boolean / numeric space
expected by the grader.

The parsing rules are intentionally LIBERAL — they're not part of the
prompt contract the model is meant to satisfy. The whole point of B2 is
to measure how much accuracy is gained by letting DSPy enforce types at
generation time vs. recovering them after the fact.
"""
from __future__ import annotations

import re

from ...benchmarks.eval.scope import (
    get_allowed_values,
    get_bool_fields,
    get_organ_scoreable_fields,
    get_span_fields,
)

# --- Field-kind classifiers --------------------------------------------------

_TRUE_TOKENS = frozenset({
    "yes", "y", "true", "t", "1",
    "positive", "+", "present", "identified", "involved",
    "detected", "seen",
})
_FALSE_TOKENS = frozenset({
    "no", "n", "false", "f", "0",
    "negative", "-", "absent", "not identified", "not involved",
    "not detected", "not seen", "none", "uninvolved", "free",
})

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")


def _tokenise(s: str) -> list[str]:
    return [t for t in _TOKEN_SPLIT_RE.split(s.lower()) if t]


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _parse_bool(value: str) -> bool | None:
    v = value.strip().lower()
    if not v or v in {"null", "none", "n/a", "na", "unknown", "?", "-"}:
        return None
    # Sub-string match against multi-word tokens first.
    for token in _FALSE_TOKENS:
        if token in v:
            return False
    for token in _TRUE_TOKENS:
        if token in v:
            return True
    return None


def _parse_continuous(value: str) -> float | None:
    """Extract the first numeric token. Returns float; caller may cast."""
    m = _NUMBER_RE.search(value)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _parse_enum(value: str, allowed: list[str]) -> str | None:
    """Match free text to an allowed enum value via fuzzy token Jaccard.

    Returns the canonical (lowercased) allowed value if the best match
    exceeds J >= 0.5, else the literal lowercased value if it exact-matches
    any allowed option, else None.
    """
    v = value.strip().lower()
    if not v:
        return None
    if v in {"null", "none", "n/a", "na", "unknown"}:
        return None
    allowed_lower = [str(a).lower() for a in allowed]
    if v in allowed_lower:
        return v
    # Substring containment — ``"poorly differentiated, grade 3"`` should
    # match enum value ``"poorly_differentiated"``.
    v_norm = v.replace("_", " ")
    for opt in allowed_lower:
        if opt.replace("_", " ") in v_norm or v_norm in opt.replace("_", " "):
            return opt
    # Token Jaccard fallback.
    v_tokens = _tokenise(v)
    best_score = 0.0
    best_opt: str | None = None
    for opt in allowed_lower:
        score = _jaccard(v_tokens, _tokenise(opt))
        if score > best_score:
            best_score = score
            best_opt = opt
    if best_opt is not None and best_score >= 0.5:
        return best_opt
    return None


# --- Top-level dispatch ------------------------------------------------------

def parse_field(value, field: str, organ: str):
    """Coerce a raw value (typically a string) into the typed expectation.

    Returns ``None`` for unparseable / unknown values. Already-typed
    inputs (bool, numbers, dicts/lists) pass through untouched so this
    function is safe to apply to any cancer_data dict.
    """
    if value is None:
        return None
    if isinstance(value, (bool, int, float, dict, list)):
        return value
    if not isinstance(value, str):
        return value

    bool_fields = get_bool_fields(organ)
    span_fields = get_span_fields(organ)
    if field in bool_fields:
        return _parse_bool(value)
    if field in span_fields:
        return _parse_continuous(value)
    allowed = get_allowed_values(field, organ)
    if allowed:
        return _parse_enum(value, allowed)

    # Unknown field kind — return cleaned string.
    cleaned = value.strip()
    return cleaned or None


def parse_cancer_data(cancer_data: dict, organ: str
                      ) -> tuple[dict, dict[str, str]]:
    """Coerce every field in ``cancer_data`` for the given organ.

    Returns ``(parsed_dict, parse_errors)`` where ``parse_errors`` maps
    field name → raw string that could not be parsed (only populated for
    fields that have a defined type and produced ``None`` from a non-None
    raw value — explicit-null inputs are NOT errors).
    """
    if not isinstance(cancer_data, dict):
        return cancer_data, {}

    scoreable = get_organ_scoreable_fields(organ)
    parsed: dict = {}
    errors: dict[str, str] = {}

    for field, raw in cancer_data.items():
        coerced = parse_field(raw, field, organ)
        parsed[field] = coerced
        # Flag a parse error only if input was a non-empty string AND
        # parsing produced None AND the field has a known type.
        if (
            field in scoreable
            and isinstance(raw, str)
            and raw.strip()
            and coerced is None
        ):
            errors[field] = raw[:200]

    return parsed, errors


__all__ = ["parse_field", "parse_cancer_data"]
