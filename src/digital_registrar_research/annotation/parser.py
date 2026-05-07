"""Parse JSON Schema 2020-12 files into flat Python data structures for Streamlit rendering."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from ..paths import SCHEMAS_DATA as SCHEMAS_DIR

_PRECOMPUTED_GROUPS_PATH = Path(__file__).resolve().parent / "_section_groups.json"
_PRECOMPUTED_GROUPS: dict | None = None

DISPLAY_NAMES = {
    "Nonnested": "General",
    "Staging": "Staging",
    "Margins": "Margins",
    "LN": "Lymph Nodes",
    "Biomarkers": "Biomarkers",
    "Grading": "Grading",
    "Othernested": "Patterns",
    "DCIS": "DCIS",
}

CANCER_TO_FILE = {
    "stomach": "stomach.json",
    "colorectal": "colorectal.json",
    "breast": "breast.json",
    "esophagus": "esophagus.json",
    "lung": "lung.json",
    "prostate": "prostate.json",
    "thyroid": "thyroid.json",
    "pancreas": "pancreas.json",
    "cervix": "cervix.json",
    "liver": "liver.json",
    "bladder": "bladder.json",
    "others": None,
}

CANCER_CATEGORIES = [
    None,
    "stomach",
    "colorectal",
    "breast",
    "esophagus",
    "lung",
    "prostate",
    "thyroid",
    "pancreas",
    "cervix",
    "liver",
    "bladder",
    "others",
]


@dataclass
class FieldSpec:
    name: str
    title: str
    description: str
    field_type: str  # enum|int_enum|bool|int|string|array_of_objects|array_of_strings_enum
    enum_values: list = field(default_factory=list)
    item_fields: list = field(default_factory=list)  # list[FieldSpec] for array_of_objects
    required: bool = False


@dataclass
class SectionSpec:
    name: str
    display_name: str
    flat_fields: list  # list[FieldSpec]
    array_field_name: str | None = None
    array_item_fields: list = field(default_factory=list)  # list[FieldSpec]
    array_item_enum_values: list = field(default_factory=list)  # for array_of_strings_enum
    array_field_type: str | None = None  # array_of_objects | array_of_strings_enum


def _detect_field_type(prop: dict) -> tuple[str, list, list]:
    """Returns (field_type, enum_values, item_fields)."""
    ao = prop.get("anyOf")
    if ao:
        non_null = [s for s in ao if s.get("type") != "null"]
        if not non_null:
            return "string", [], []
        core = non_null[0]
    elif "type" in prop:
        # Non-nullable field (e.g. required boolean in common.json)
        core = prop
    else:
        return "string", [], []

    if "enum" in core:
        if core.get("type") == "integer":
            return "int_enum", core["enum"], []
        return "enum", core["enum"], []
    t = core.get("type")
    if t == "boolean":
        return "bool", [], []
    if t == "integer":
        return "int", [], []
    if t == "array":
        items = core.get("items", {})
        if "$ref" in items:
            return "array_of_objects", [], []  # item_fields resolved separately
        if "enum" in items:
            return "array_of_strings_enum", items["enum"], []
        return "array_of_strings_enum", [], []
    return "string", [], []


def _parse_properties(properties: dict, defs: dict, required_fields: list) -> list[FieldSpec]:
    result = []
    for fname, fprop in properties.items():
        ftype, enum_vals, _ = _detect_field_type(fprop)
        item_fields: list[FieldSpec] = []
        if ftype == "array_of_objects":
            ao = fprop.get("anyOf", [])
            non_null = [s for s in ao if s.get("type") != "null"]
            if non_null:
                ref = non_null[0].get("items", {}).get("$ref", "")
                ref_name = ref.split("/")[-1] if ref else ""
                if ref_name and ref_name in defs:
                    item_def = defs[ref_name]
                    item_required = item_def.get("required", [])
                    item_fields = _parse_properties(
                        item_def.get("properties", {}), defs, item_required
                    )
        result.append(
            FieldSpec(
                name=fname,
                title=fprop.get("title", fname.replace("_", " ").title()),
                description=fprop.get("description", ""),
                field_type=ftype,
                enum_values=enum_vals,
                item_fields=item_fields,
                required=fname in required_fields,
            )
        )
    return result


def _derive_display_name(section_name: str) -> str:
    # Strip leading cancer-type prefix, e.g. "ColonCancer" from "ColonCancerLN"
    # Pattern: one or more Title-cased words followed by "Cancer" (possibly none) then the suffix
    suffix = re.sub(r"^[A-Za-z]+Cancer", "", section_name)
    if not suffix:
        suffix = section_name
    return DISPLAY_NAMES.get(suffix, suffix)


def _build_section_from_props(
    section_name: str,
    field_names: list[str],
    props: dict,
    defs: dict,
    required: list,
) -> SectionSpec:
    """Build one SectionSpec from a subset of top-level schema properties."""
    flat_fields: list[FieldSpec] = []
    array_field_name: str | None = None
    array_item_fields: list[FieldSpec] = []
    array_item_enum_values: list = []
    array_field_type: str | None = None

    for fname in field_names:
        fprop = props.get(fname)
        if not isinstance(fprop, dict):
            continue
        ftype, enum_vals, _ = _detect_field_type(fprop)
        if ftype in ("array_of_objects", "array_of_strings_enum"):
            array_field_name = fname
            array_field_type = ftype
            if ftype == "array_of_objects":
                ao = fprop.get("anyOf", [])
                non_null = [s for s in ao if s.get("type") != "null"]
                if non_null:
                    ref = non_null[0].get("items", {}).get("$ref", "")
                    ref_name = ref.split("/")[-1] if ref else ""
                    if ref_name and ref_name in defs:
                        item_def = defs[ref_name]
                        item_required = item_def.get("required", [])
                        array_item_fields = _parse_properties(
                            item_def.get("properties", {}), defs, item_required
                        )
            else:
                array_item_enum_values = enum_vals
        else:
            flat_fields.append(
                FieldSpec(
                    name=fname,
                    title=fprop.get("title", fname.replace("_", " ").title()),
                    description=fprop.get("description", ""),
                    field_type=ftype,
                    enum_values=enum_vals,
                    item_fields=[],
                    required=fname in required,
                )
            )

    return SectionSpec(
        name=section_name,
        display_name=_derive_display_name(section_name),
        flat_fields=flat_fields,
        array_field_name=array_field_name,
        array_item_fields=array_item_fields,
        array_item_enum_values=array_item_enum_values,
        array_field_type=array_field_type,
    )


def _is_nested_layout(schema_props: dict) -> bool:
    """Old layout: each top-level property is a section object with its own `properties`."""
    if not schema_props:
        return False
    first = next(iter(schema_props.values()))
    return isinstance(first, dict) and "properties" in first and first.get("type") == "object"


def _load_precomputed_groups() -> dict:
    """Load the build-time-generated {organ: [[section_name, [field_names]]]} map.

    Shipped in the standalone Windows bundle so the annotation UI does not
    need to import DSPy/pydantic-derived case-models at runtime. Returns
    an empty dict if the file is absent (dev source tree) — callers fall
    back to the dynamic lookup via `..models.modellist`.
    """
    global _PRECOMPUTED_GROUPS
    if _PRECOMPUTED_GROUPS is None:
        if _PRECOMPUTED_GROUPS_PATH.exists():
            try:
                _PRECOMPUTED_GROUPS = json.loads(
                    _PRECOMPUTED_GROUPS_PATH.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                _PRECOMPUTED_GROUPS = {}
        else:
            _PRECOMPUTED_GROUPS = {}
    return _PRECOMPUTED_GROUPS


def _section_field_groups(cancer_type: str, props: dict) -> list[tuple[str, list[str]]]:
    """Return [(section_name, [field_names])] for the flat-schema layout.

    Each DSPy signature contributes its OutputField names as one section.
    Fields are assigned first-wins across signatures — matching the
    canonical merge order in `schemas.pydantic._builder._merge_fields_for_organ`,
    so the UI tabs line up with how the Pydantic case-model was built.
    Any leftover schema property (unlikely, but possible if a field exists
    in the schema without a sibling DSPy signature) lands in a trailing
    "Other" section so it still renders.

    Prefers a precomputed map bundled alongside this module (used by the
    standalone Windows package so the runtime stays DSPy-free); falls back
    to introspecting the live signature classes in-process.
    """
    precomputed = _load_precomputed_groups().get(cancer_type)
    if precomputed is not None:
        groups: list[tuple[str, list[str]]] = []
        assigned: set[str] = set()
        for sig_name, field_names in precomputed:
            filtered = [n for n in field_names if n not in assigned and n in props]
            assigned.update(filtered)
            if filtered:
                groups.append((sig_name, filtered))
        leftover = [n for n in props if n not in assigned]
        if leftover:
            groups.append((f"{cancer_type.capitalize()}CancerOther", leftover))
        return groups

    try:
        from ..models.modellist import organmodels
        from ..schemas.pydantic import _builder
    except Exception:
        return []

    sig_names = organmodels.get(cancer_type, [])
    module_globals = sys.modules[_builder.__name__].__dict__

    groups = []
    assigned = set()
    for sig_name in sig_names:
        cls = module_globals.get(sig_name)
        if cls is None:
            continue
        field_names = []
        for name, _type, _fi in _builder._iter_signature_output_fields(cls):
            if name in assigned or name not in props:
                continue
            assigned.add(name)
            field_names.append(name)
        if field_names:
            groups.append((sig_name, field_names))

    leftover = [n for n in props if n not in assigned]
    if leftover:
        groups.append((f"{cancer_type.capitalize()}CancerOther", leftover))
    return groups


def parse_cancer_schema(cancer_type: str) -> list[SectionSpec]:
    """Parse a cancer schema file and return a list of SectionSpecs.

    Supports both schema layouts:
    - Nested (legacy): top-level properties are section objects, each with
      its own `properties`. Used by `bladder.json`.
    - Flat (canonical): top-level properties are the individual fields of
      `<Organ>CancerCase`. Sections are reconstructed by grouping fields
      per DSPy signature in `organmodels[cancer_type]`.
    """
    schema_file = CANCER_TO_FILE.get(cancer_type)
    if not schema_file:
        return []
    path = SCHEMAS_DIR / schema_file
    if not path.exists():
        return []
    with open(path) as f:
        schema = json.load(f)

    defs = schema.get("$defs", {})
    props = schema.get("properties", {})

    if _is_nested_layout(props):
        sections: list[SectionSpec] = []
        for section_name, section_schema in props.items():
            if not isinstance(section_schema, dict):
                continue
            sub_props = section_schema.get("properties", {})
            required = section_schema.get("required", [])
            sections.append(
                _build_section_from_props(
                    section_name, list(sub_props.keys()), sub_props, defs, required
                )
            )
        return sections

    groups = _section_field_groups(cancer_type, props)
    if not groups:
        # Fallback: render everything as one section so the form is still usable.
        groups = [(f"{cancer_type.capitalize()}CancerAll", list(props.keys()))]
    required = schema.get("required", [])
    return [
        _build_section_from_props(name, fields, props, defs, required)
        for name, fields in groups
    ]


if __name__ == "__main__":
    for cancer in CANCER_TO_FILE:
        if cancer == "others":
            continue
        sections = parse_cancer_schema(cancer)
        print(f"\n=== {cancer.upper()} ===")
        for s in sections:
            flat = [f.name for f in s.flat_fields]
            arr = f"{s.array_field_name}({s.array_field_type})" if s.array_field_name else "-"
            print(f"  {s.display_name:15} flat={flat}  array={arr}")
