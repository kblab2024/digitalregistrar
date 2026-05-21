"""
Loads the per-organ JSON schemas shipped in
`../digitalregistrar-annotation/schemas/*.json` and stitches them into
the prompt for the raw-JSON runner (Cell C).

Two-step approach:
    1. `load_organ_schema(organ)` returns the full nested JSON-Schema
       dict for one organ (the original file already merges all
       per-subsection signatures — e.g. breast.json includes
       BreastCancerNonnested, DCIS, BreastCancerStaging, ... under its
       top-level `properties`).
    2. `flatten_schema_for_prompt(schema)` strips it to the form the
       parent pipeline actually emits at runtime: a flat `cancer_data`
       dict produced by `.update()`-ing each subsection's output into
       a single top-level object. This is what we ask the raw LLM to
       produce so the comparison is apples-to-apples.

We also expose `validate_cancer_data(organ, d)` for Pydantic-style
post-validation on the LLM's raw output.
"""
from __future__ import annotations

import json
from functools import cache

try:
    import jsonschema
except ImportError:
    jsonschema = None  # optional; only needed for validate_cancer_data

from ..paths import SCHEMAS_DATA as SCHEMA_ROOT

# Top-level JSON schema keys we expect inside a per-organ schema file.
# These are the DSPy signature class names.
SUBSECTION_KEYS_FALLBACK = {
    "breast": ["BreastCancerNonnested", "DCIS", "BreastCancerGrading",
               "BreastCancerStaging", "BreastCancerMargins",
               "BreastCancerLN", "BreastCancerBiomarkers"],
    "lung": ["LungCancerNonnested", "LungCancerStaging",
             "LungCancerMargins", "LungCancerLN",
             "LungCancerBiomarkers", "LungCancerOthernested"],
    # The loader auto-detects from the file when possible; this dict is
    # only used for error reporting and prompt field ordering.
}


@cache
def load_organ_schema(organ: str) -> dict:
    """Return the raw JSON-Schema dict for the organ."""
    path = SCHEMA_ROOT / f"{organ}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No JSON schema at {path}. Available schemas: "
            f"{sorted(p.stem for p in SCHEMA_ROOT.glob('*.json'))}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def flatten_schema_for_prompt(schema: dict) -> dict:
    """Return a schema describing the flat cancer_data dict the parent
    pipeline emits at runtime.

    Two input shapes are supported:

    1. Already-flat (current canonical organ schemas under
       ``schemas/data/{organ}.json``)::

           { properties: { field_name: <field_schema>, ... } }

       Returned as-is (with ``additionalProperties=False`` set).

    2. Nested-by-subsection (legacy)::

           { properties: { BreastCancerNonnested: {properties: {...}},
                           BreastCancerStaging:   {properties: {...}},
                           ... } }

       Subsection groups are merged into one flat properties dict
       (first-wins, matching ``CancerPipeline.forward``'s
       ``.update()`` semantics).
    """
    top_props = schema.get("properties", {})
    # Detect already-flat: at least one property whose value lacks a
    # nested ``properties`` key (i.e. is a field spec, not a subsection
    # group). Field specs use ``anyOf`` / ``type`` / ``enum``.
    already_flat = any(
        isinstance(v, dict) and "properties" not in v
        for v in top_props.values()
    )
    if already_flat:
        return {
            "type": "object",
            "title": schema.get("title", "cancer_data"),
            "properties": dict(top_props),
            "$defs": schema.get("$defs", {}),
            "additionalProperties": False,
        }

    merged_props: dict = {}
    for _subsection_name, subsection_schema in top_props.items():
        if not isinstance(subsection_schema, dict):
            continue
        fields = subsection_schema.get("properties", {})
        for field_name, field_schema in fields.items():
            if field_name in merged_props:
                continue
            merged_props[field_name] = field_schema

    return {
        "type": "object",
        "title": schema.get("title", "cancer_data"),
        "properties": merged_props,
        "$defs": schema.get("$defs", {}),
        "additionalProperties": False,
    }


def describe_field_list(flat_schema: dict) -> str:
    """Return a human-readable field checklist for the prompt. We keep
    it concise: `  - field_name (type, optional): description`."""
    lines: list[str] = []
    defs = flat_schema.get("$defs", {})
    for name, spec in flat_schema.get("properties", {}).items():
        type_desc = _spec_type_label(spec, defs)
        desc = spec.get("description", "")
        lines.append(f"  - {name} ({type_desc}): {desc}")
    return "\n".join(lines)


def _resolve_ref(spec: dict, defs: dict) -> dict:
    """Dereference a ``$ref`` once against the supplied ``$defs`` map.

    Returns ``spec`` unchanged when there's no ``$ref`` or the referenced
    name isn't in ``defs``. Only handles intra-schema refs of the form
    ``"#/$defs/<Name>"`` — sufficient for what Pydantic emits.
    """
    ref = spec.get("$ref")
    if not isinstance(ref, str) or not ref.startswith("#/$defs/"):
        return spec
    name = ref.removeprefix("#/$defs/")
    target = defs.get(name)
    return target if isinstance(target, dict) else spec


def _enum_values(spec: dict, defs: dict | None = None) -> list:
    """Return the inline enum value list for a field spec, recursing
    through ``anyOf`` arms (e.g. ``Literal[...] | None``) and
    dereferencing ``$ref`` entries against ``defs`` (a ``$defs`` map).
    Empty list if no enum is found.

    Pydantic emits StrEnum-typed fields as ``$ref`` entries, so callers
    that produce an enum-allowed-values prompt line (see
    :func:`describe_field_list_strict`) must thread ``$defs`` through.
    """
    if "enum" in spec:
        return list(spec["enum"])
    if "$ref" in spec and defs is not None:
        resolved = _resolve_ref(spec, defs)
        if resolved is not spec:
            return _enum_values(resolved, defs)
    for arm in spec.get("anyOf", []):
        vals = _enum_values(arm, defs)
        if vals:
            return vals
    if spec.get("type") == "array":
        return _enum_values(spec.get("items", {}), defs)
    return []


def describe_field_list_strict(flat_schema: dict) -> str:
    """Like :func:`describe_field_list` but renders one block per field
    so the model has more guidance: a name + type, then the description
    on its own line, then (for enum-typed fields) an explicit
    ``Allowed:`` line listing every permitted value. Pushes the model
    harder to comply with the schema instead of inventing values."""
    blocks: list[str] = []
    defs = flat_schema.get("$defs", {})
    for name, spec in flat_schema.get("properties", {}).items():
        type_desc = _spec_type_label(spec, defs)
        desc = spec.get("description", "")
        block = [f"- {name} ({type_desc}):"]
        if desc:
            block.append(f"    {desc}")
        enum_vals = _enum_values(spec, defs)
        if enum_vals:
            rendered = ", ".join(json.dumps(v) for v in enum_vals)
            block.append(f"    Allowed: [{rendered}] or null")
        blocks.append("\n".join(block))
    return "\n".join(blocks)


def describe_skeleton(flat_schema: dict) -> str:
    """Return a literal JSON skeleton (every field set to ``null``) so
    the model can fill in the blanks rather than synthesise the shape
    from scratch."""
    skeleton = {name: None for name in flat_schema.get("properties", {})}
    return json.dumps(skeleton, indent=2, ensure_ascii=False)


def _spec_type_label(spec: dict, defs: dict | None = None) -> str:
    if "anyOf" in spec:
        inner = [_spec_type_label(x, defs) for x in spec["anyOf"]]
        return " | ".join(inner)
    if "enum" in spec:
        return f"enum{tuple(spec['enum'])}"
    if "$ref" in spec:
        if defs is not None:
            resolved = _resolve_ref(spec, defs)
            if resolved is not spec and "enum" in resolved:
                return f"enum{tuple(resolved['enum'])}"
        return f"ref:{spec['$ref'].split('/')[-1]}"
    if "type" in spec:
        t = spec["type"]
        if t == "array":
            return f"array<{_spec_type_label(spec.get('items', {}), defs)}>"
        return str(t)
    return "any"


def validate_cancer_data(organ: str, cancer_data: dict) -> list[str]:
    """Return a list of validation-error strings; empty if valid.
    Requires `jsonschema` to be installed."""
    if jsonschema is None:
        return ["jsonschema not installed — skipping validation"]
    schema = flatten_schema_for_prompt(load_organ_schema(organ))
    errors: list[str] = []
    validator = jsonschema.Draft202012Validator(schema)
    for err in sorted(validator.iter_errors(cancer_data), key=lambda e: e.path):
        errors.append(f"{list(err.path)}: {err.message}")
    return errors


if __name__ == "__main__":
    # Smoke test: print per-organ field counts.
    for path in sorted(SCHEMA_ROOT.glob("*.json")):
        if path.stem == "common":
            continue
        schema = load_organ_schema(path.stem)
        flat = flatten_schema_for_prompt(schema)
        n = len(flat["properties"])
        print(f"{path.stem:12s}  {n:3d} flat fields")
