"""Per-section DSPy signatures — the A5 ablation.

Splits the monolithic signature into one signature per report section
(header / gross / micro / dx / comments). Each per-section signature
declares only the subset of output fields whose values are most likely
to be found in that section. The runner runs each predictor on the
matching slice of the report, then merges results with first-wins —
mirroring the existing :class:`MonolithicPipeline.update()` pattern.

Field-to-section mapping is heuristic and conservative — we err toward
exposing a field in MULTIPLE sections rather than risk a section never
seeing a field at all (the merger makes the redundancy harmless).
"""
from __future__ import annotations

from functools import cache

import dspy

from .monolithic import (
    INPUT_FIELD_NAMES,
    get_monolithic_signature,
)

# --- Field → section preference ---------------------------------------------
#
# Each set lists the SECTIONS in which a field is likely to be findable.
# Fields not listed default to {"dx"} — the most defensive bucket.

_FIELD_TO_SECTIONS: dict[str, set[str]] = {
    # Header / specimen — procedure, laterality, site identifiers
    "procedure":            {"header", "gross"},
    "surgical_technique":   {"header", "gross"},
    "cancer_laterality":    {"header", "gross", "dx"},
    "cancer_quadrant":      {"header", "gross", "dx"},
    "cancer_clock":         {"header", "gross", "dx"},
    "cancer_primary_site":  {"header", "gross", "dx"},
    "tumor_focality":       {"gross", "dx"},
    "tumor_site":           {"gross", "dx"},
    "tumor_extent":         {"gross", "dx"},

    # Gross / size
    "tumor_size":           {"gross", "dx"},

    # Microscopic / dx — the bulk of structured findings live here.
    "histology":            {"micro", "dx"},
    "grade":                {"micro", "dx"},
    "nuclear_grade":        {"micro", "dx"},
    "tubule_formation":     {"micro", "dx"},
    "mitotic_rate":         {"micro", "dx"},
    "total_score":          {"micro", "dx"},

    "lymphovascular_invasion":  {"micro", "dx"},
    "perineural_invasion":      {"micro", "dx"},

    # Staging — usually in a synoptic checklist (dx) but sometimes
    # carried into comments.
    "pt_category":          {"dx", "comments"},
    "pn_category":          {"dx", "comments"},
    "pm_category":          {"dx", "comments"},
    "tnm_descriptor":       {"dx", "comments"},
    "stage_group":          {"dx", "comments"},
    "overall_stage":        {"dx", "comments"},
    "pathologic_stage_group":   {"dx", "comments"},
    "anatomic_stage_group":     {"dx", "comments"},
    "ajcc_version":         {"dx", "comments"},

    # Margins / lymph node — synoptic
    "margins":              {"dx"},
    "regional_lymph_node":  {"dx"},

    # Biomarkers / IHC live in the comments section as a rule.
    "biomarkers":           {"comments", "dx"},
    "dcis_grade":           {"micro", "dx"},
    "dcis_present":         {"micro", "dx"},
    "dcis_size":            {"micro", "dx"},
    "dcis_comedo_necrosis": {"micro", "dx"},
}

_DEFAULT_SECTIONS: set[str] = {"dx"}

SECTION_DOCSTRING_TEMPLATE = (
    "You are a cancer registrar. From the {section} section of a "
    "{organ} cancer pathology report, extract the structured fields "
    "listed below. Do NOT guess from outside this section. Return null "
    "for fields that are not present in this slice — they will be "
    "filled in by signatures specialised for other sections."
)


@cache
def get_section_signature(organ: str, section: str) -> type[dspy.Signature]:
    """Return a DSPy signature for ``organ`` × ``section``.

    Outputs are the subset of monolithic fields whose
    :data:`_FIELD_TO_SECTIONS` mapping includes ``section``. Fields not
    in the mapping default to ``{"dx"}`` so they appear on the dx
    signature only.
    """
    base = get_monolithic_signature(organ)

    merged_annotations: dict[str, object] = {}
    merged_attrs: dict[str, object] = {}

    for name, type_hint in getattr(base, "__annotations__", {}).items():
        descriptor = base.__dict__.get(name)
        if descriptor is None:
            continue
        if name in INPUT_FIELD_NAMES:
            # Replace ``report`` with the section-specific input.
            if name == "report":
                merged_annotations["report"] = str
                merged_attrs["report"] = dspy.InputField(
                    desc=f"The '{section}' section of a {organ} cancer "
                         f"pathology report (one heuristically-extracted slice).")
            else:
                # Drop ``report_jsonized`` — per-section is run without
                # the upstream ReportJsonize step.
                continue
        else:
            sections = _FIELD_TO_SECTIONS.get(name, _DEFAULT_SECTIONS)
            if section not in sections:
                continue
            merged_annotations[name] = type_hint
            merged_attrs[name] = descriptor

    merged_attrs["__annotations__"] = merged_annotations
    merged_attrs["__doc__"] = SECTION_DOCSTRING_TEMPLATE.format(
        section=section, organ=organ)

    cls_name = f"{organ.title()}Cancer{section.title()}Section"
    sig_cls = type(cls_name, (dspy.Signature,), merged_attrs)
    return sig_cls


def list_section_fields(organ: str, section: str) -> list[str]:
    sig = get_section_signature(organ, section)
    return [name for name in sig.__annotations__ if name not in INPUT_FIELD_NAMES]


__all__ = ["get_section_signature", "list_section_fields", "_FIELD_TO_SECTIONS"]
