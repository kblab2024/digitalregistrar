"""
Defines the "fair scope" — which fields each method can legitimately
produce, and the allowed-value lists for the categorical ones.

This is the single source of truth for both the benchmark code and the
results-table column definitions. The `FAIR_SCOPE` list is the
head-to-head comparison set (fields that ALL four methods populate).

Per-organ option lists live in `scope_organs.py` (`ORGAN_CATEGORICAL`,
`ORGAN_BOOL`, `ORGAN_SPAN`, `ORGAN_NESTED_LIST`), derived from the
canonical schemas under `digital_registrar_research.schemas.data`.

Accessors:
    get_allowed_values(field, organ=None)  — enum list for a categorical field
    get_categorical_fields(organ)          — all enum fields for an organ
    get_bool_fields(organ)                 — all bool fields for an organ
    get_span_fields(organ)                 — all int span fields for an organ
    get_nested_list_fields(organ)          — all array fields for an organ
    get_field_value(annotation, field)     — read a field from a gold record
"""
from __future__ import annotations

from functools import cache

from .scope_organs import (
    ORGAN_BOOL,
    ORGAN_CATEGORICAL,
    ORGAN_LIST_OF_LITERALS,
    ORGAN_NESTED_LIST,
    ORGAN_SPAN,
)

# --- Top-level cancer category (drives organ-gated field scoring) -------------

CANCER_CATEGORIES: list[str] = [
    "breast", "lung", "colorectal", "prostate", "esophagus",
    "pancreas", "thyroid", "cervix", "liver", "stomach", "others",
]

# Organs that have a populated schema (excludes "others" and the
# placeholder "bladder" entry in schemas/data).
IMPLEMENTED_ORGANS: list[str] = sorted(ORGAN_CATEGORICAL.keys())


# --- Shared categorical option lists -----------------------------------------
#
# Options are lowercased to match the gold-annotation normalisation rule
# applied by `eval/metrics.py:normalize`. Where a field's enum differs by
# organ (`pt_category`, `pn_category`, `pm_category`, `grade`, `histology`,
# `procedure`, stage-group fields), the value below is the UNION across
# all organs — a superset useful for a single cross-organ classifier head.
# For organ-specific scoring use `get_allowed_values(field, organ)`.

def _union_field_values(field: str) -> list[str]:
    seen: list[str] = []
    for organ_map in ORGAN_CATEGORICAL.values():
        for v in organ_map.get(field, []):
            if v not in seen:
                seen.append(v)
    return seen


def _union_bool_fields() -> list[str]:
    seen: list[str] = []
    for fields in ORGAN_BOOL.values():
        for f in sorted(fields):
            if f not in seen:
                seen.append(f)
    return seen


CATEGORICAL_FIELDS: dict[str, list[str]] = {
    "tnm_descriptor":          _union_field_values("tnm_descriptor"),
    "pt_category":             _union_field_values("pt_category"),
    "pn_category":             _union_field_values("pn_category"),
    "pm_category":             _union_field_values("pm_category"),
    "stage_group":             _union_field_values("stage_group"),
    "overall_stage":           _union_field_values("overall_stage"),
    "pathologic_stage_group":  _union_field_values("pathologic_stage_group"),
    "anatomic_stage_group":    _union_field_values("anatomic_stage_group"),
    "grade":                   _union_field_values("grade"),
    "nuclear_grade":           ["1", "2", "3"],
    "tubule_formation":        ["1", "2", "3"],
    "mitotic_rate":            ["1", "2", "3"],
    "total_score":             ["3", "4", "5", "6", "7", "8", "9"],
    "histology":               _union_field_values("histology"),
    "procedure":               _union_field_values("procedure"),
    "surgical_technique":      _union_field_values("surgical_technique"),
    "cancer_primary_site":     _union_field_values("cancer_primary_site"),
    "tumor_focality":          _union_field_values("tumor_focality"),
    "tumor_site":              _union_field_values("tumor_site"),
    "tumor_extent":            _union_field_values("tumor_extent"),
    # Boolean / 3-way {true, false, null} fields kept in CATEGORICAL_FIELDS
    # for the flat-head classifier API (clinicalbert_cls.py) that treats
    # everything in this dict as a head.
    **{f: ["true", "false"] for f in _union_bool_fields()},
}


# --- Span / numeric fields (go to ClinicalBERT-QA head) ----------------------
#
# Cross-organ union of integer fields. Grading sub-scores (`grade`,
# `nuclear_grade`, `tubule_formation`, `mitotic_rate`, `total_score`) are
# listed here as well because in some organs (colorectal, lung) `grade` is
# stored as a free integer rather than a Literal, so the QA head is
# responsible for it.

SPAN_FIELDS: set[str] = set()
for _s in ORGAN_SPAN.values():
    SPAN_FIELDS |= _s
SPAN_FIELDS |= {
    "grade", "nuclear_grade", "tubule_formation",
    "mitotic_rate", "total_score",
}


# --- Nested-list fields — reported N/A for ClinicalBERT and rules. -----------

NESTED_LIST_FIELDS: set[str] = set()
for _n in ORGAN_NESTED_LIST.values():
    NESTED_LIST_FIELDS |= _n

# --- List-of-literals fields (cross-organ union) -----------------------------
#
# Distinct from NESTED_LIST_FIELDS — these are arrays of enum strings,
# not arrays of dicts. Scored as unordered sets in the non_nested
# subcommand (exact-match for headline accuracy, set-F1 for partial
# credit).

LIST_OF_LITERALS_FIELDS: set[str] = set()
for _organ_map in ORGAN_LIST_OF_LITERALS.values():
    LIST_OF_LITERALS_FIELDS |= set(_organ_map.keys())


# --- Fair-scope whitelist (head-to-head comparison table) --------------------
#
# Fields that EVERY method can legitimately populate. Nested-list fields
# (margins, biomarkers, regional_lymph_node) are NOT in this list — they
# appear in the supplementary coverage table instead.

FAIR_SCOPE: list[str] = [
    "cancer_category",
    "cancer_excision_report",
    "pt_category", "pn_category", "pm_category",
    "grade",
    "lymphovascular_invasion",
    "perineural_invasion",
    "tumor_size",
    # Plus breast-specific biomarkers when cancer_category == "breast":
    # these are scored conditionally in metrics.py via BREAST_BIOMARKERS.
]

BREAST_BIOMARKERS: list[str] = ["er", "pr", "her2", "ki67"]


# --- Cascade-wide eval exclusions -------------------------------------------
#
# Fields removed from EVERY metric in the cascade redesign — not just
# stat-test layer. Replaces the old `STATS_EXCLUDED_FIELDS` which only
# filtered at the test layer.

# Top-level scalar fields inside `cancer_data` that are never scored.
# These are version metadata or unstable free-text that should not appear
# in any per-field accuracy table.
EVAL_EXCLUDED_FIELDS: frozenset[str] = frozenset({
    "ajcc_version",      # version metadata
    "treatment_effect",  # free-text in practice; not a categorical endpoint
})

# Inner keys of nested list-of-dicts fields. Applied PER LIST ELEMENT,
# i.e. for each item in `cancer_data.<field>[*]` the listed inner keys
# are stripped before per-attribute comparison.
#
# Format: { "cancer_data.<field>[*].<inner_key>": "<reason>" }
# The matcher only cares about the (field, inner_key) tuples — the path
# notation is for human readability and grep-ability.
EVAL_EXCLUDED_NESTED_INNER_KEYS: dict[str, str] = {
    "cancer_data.margins[*].description":
        "free-text — annotators wrote 'medial', 'Negative', etc.; not a categorical endpoint",
    "cancer_data.regional_lymph_node[*].station_name":
        "free-text identifier; aggregated-by-category scoring replaces bipartite-on-station_name",
}


def _build_inner_key_index() -> dict[str, frozenset[str]]:
    """Index `EVAL_EXCLUDED_NESTED_INNER_KEYS` by field for fast lookup."""
    out: dict[str, set[str]] = {}
    prefix = "cancer_data."
    for path in EVAL_EXCLUDED_NESTED_INNER_KEYS:
        if not path.startswith(prefix):
            continue
        rest = path[len(prefix):]
        # rest looks like "<field>[*].<inner>"
        if "[*]." not in rest:
            continue
        field, inner = rest.split("[*].", 1)
        out.setdefault(field, set()).add(inner)
    return {k: frozenset(v) for k, v in out.items()}


EXCLUDED_INNER_KEYS_BY_FIELD: dict[str, frozenset[str]] = _build_inner_key_index()


# Biomarker whitelist by organ. Any biomarker entry whose
# `biomarker_category` (after `normalize`) is not in the organ's whitelist
# is dropped from gold AND prediction before scoring, on both sides.
# This keeps the metric focused on the eight clinically-acted-on
# biomarkers and ignores spurious or out-of-scope categories.
BIOMARKER_WHITELIST: dict[str, frozenset[str]] = {
    "breast":     frozenset({"er", "pr", "her2", "ki67"}),
    "colorectal": frozenset({"msh2", "msh6", "pms2", "mlh1"}),
}


def biomarkers_for_organ(organ: str | None) -> frozenset[str]:
    """Return the whitelisted biomarker categories for ``organ``.

    Returns an empty frozenset for organs with no biomarker scoring
    (e.g. lung, liver, thyroid) or for ``None`` / "others".
    """
    if not organ:
        return frozenset()
    return BIOMARKER_WHITELIST.get(organ, frozenset())


# --- Deprecated: backward-compat alias for STATS_EXCLUDED_FIELDS -------------
#
# The cascade redesign replaces this single flat set with the more
# expressive `EVAL_EXCLUDED_FIELDS` (top-level scalars) +
# `EVAL_EXCLUDED_NESTED_INNER_KEYS` (per-list-element inner keys). The
# alias below is the union of both sources so legacy callers that
# filter a flat `field` column keep working until they migrate.
#
# DEPRECATED: do not introduce new uses. Migrate to the two new
# structures.

STATS_EXCLUDED_FIELDS: frozenset[str] = frozenset(
    EVAL_EXCLUDED_FIELDS
    | {inner for inners in EXCLUDED_INNER_KEYS_BY_FIELD.values() for inner in inners}
)


# --- Accessors ---------------------------------------------------------------

@cache
def get_allowed_values(field: str, organ: str | None = None) -> list[str] | None:
    """Return the allowed option list for a categorical field.

    If `organ` is given and the field has an organ-specific enum, return
    that. Otherwise fall back to the cross-organ union in
    `CATEGORICAL_FIELDS`. Returns `None` if the field is not categorical.
    """
    if organ is not None:
        organ_map = ORGAN_CATEGORICAL.get(organ, {})
        if field in organ_map:
            return list(organ_map[field])
        bools = ORGAN_BOOL.get(organ, set())
        if field in bools:
            return ["true", "false"]
    return list(CATEGORICAL_FIELDS.get(field, [])) or None


def get_categorical_fields(organ: str) -> dict[str, list[str]]:
    """All categorical (enum) fields for `organ`, with their option lists."""
    return dict(ORGAN_CATEGORICAL.get(organ, {}))


def get_bool_fields(organ: str) -> set[str]:
    """All boolean fields for `organ` (scored 3-way: true/false/null)."""
    return set(ORGAN_BOOL.get(organ, set()))


def get_span_fields(organ: str) -> set[str]:
    """All integer span fields for `organ` (ClinicalBERT-QA head)."""
    return set(ORGAN_SPAN.get(organ, set()))


def get_nested_list_fields(organ: str) -> set[str]:
    """All nested-list (list-of-dicts) fields for `organ`."""
    return set(ORGAN_NESTED_LIST.get(organ, set()))


def get_list_of_literals_fields(organ: str) -> dict[str, list[str]]:
    """All list-of-literals fields for `organ` with their allowed value lists.

    Distinct from nested-list (list-of-dicts) fields.
    """
    return dict(ORGAN_LIST_OF_LITERALS.get(organ, {}))


def get_organ_scoreable_fields(organ: str) -> dict[str, str]:
    """Return all per-organ scalar / list-of-literals fields with their kind.

    Output keys are field names; values are field-kind strings
    consistent with :func:`iaa.classify_field`:
        ``"binary"``, ``"nominal"``, ``"ordinal"``, ``"continuous"``,
        ``"list_of_literals"``.

    Excludes nested-list (list-of-dicts) fields — those go through the
    nested subcommand. The ``non_nested`` subcommand scores everything
    in this dict.
    """
    out: dict[str, str] = {}
    # Categoricals — distinguish ordinal vs nominal at the iaa layer;
    # default to "nominal" here, callers refine via classify_field.
    for f in get_categorical_fields(organ).keys():
        out[f] = "nominal"
    for f in get_bool_fields(organ):
        out[f] = "binary"
    for f in get_span_fields(organ):
        out[f] = "continuous"
    for f in get_list_of_literals_fields(organ).keys():
        out[f] = "list_of_literals"
    return out


def get_field_value(annotation: dict, field: str):
    """Read a field from a flat gold annotation, handling the nested
    `cancer_data` container the project uses."""
    if field in annotation:
        return annotation[field]
    data = annotation.get("cancer_data") or {}
    return data.get(field)


__all__ = [
    "CANCER_CATEGORIES",
    "IMPLEMENTED_ORGANS",
    "CATEGORICAL_FIELDS",
    "ORGAN_CATEGORICAL",
    "ORGAN_BOOL",
    "ORGAN_SPAN",
    "ORGAN_NESTED_LIST",
    "ORGAN_LIST_OF_LITERALS",
    "SPAN_FIELDS",
    "NESTED_LIST_FIELDS",
    "LIST_OF_LITERALS_FIELDS",
    "FAIR_SCOPE",
    "BREAST_BIOMARKERS",
    "EVAL_EXCLUDED_FIELDS",
    "EVAL_EXCLUDED_NESTED_INNER_KEYS",
    "EXCLUDED_INNER_KEYS_BY_FIELD",
    "BIOMARKER_WHITELIST",
    "biomarkers_for_organ",
    "STATS_EXCLUDED_FIELDS",  # deprecated, see definition for migration target
    "get_allowed_values",
    "get_categorical_fields",
    "get_bool_fields",
    "get_span_fields",
    "get_nested_list_fields",
    "get_list_of_literals_fields",
    "get_organ_scoreable_fields",
    "get_field_value",
]
