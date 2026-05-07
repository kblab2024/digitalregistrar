"""Layer 0: schema-driven random sampling of a 'canonical' cancer_data dict.

We don't depend on jsf — the schemas are simple enough to walk directly.
"""
from __future__ import annotations

import random
from typing import Any

from .. import profiles
from ..schemas import load_schema


def sample_canonical(organ: str, rng: random.Random) -> dict[str, Any]:
    """Return a schema-conformant cancer_data dict for the organ."""
    schema = load_schema(organ)
    return _sample_object(schema, rng, root_schema=schema)


def _resolve_ref(ref: str, root_schema: dict) -> dict:
    # JSON Pointer: '#/$defs/Name'
    if not ref.startswith("#/"):
        raise ValueError(f"unsupported $ref: {ref}")
    parts = ref[2:].split("/")
    cur: Any = root_schema
    for p in parts:
        cur = cur[p]
    return cur


def _sample_object(schema: dict, rng: random.Random, *, root_schema: dict,
                   field_path: tuple = ()) -> dict[str, Any]:
    out: dict[str, Any] = {}
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    for name, spec in properties.items():
        out[name] = _sample_value(spec, rng, root_schema=root_schema,
                                  field_path=field_path + (name,),
                                  is_required=(name in required))
    return out


def _sample_value(spec: dict, rng: random.Random, *, root_schema: dict,
                  field_path: tuple, is_required: bool) -> Any:
    if "$ref" in spec:
        return _sample_object(_resolve_ref(spec["$ref"], root_schema), rng,
                              root_schema=root_schema, field_path=field_path)
    if "anyOf" in spec:
        return _sample_anyof(spec["anyOf"], rng, root_schema=root_schema,
                             field_path=field_path, is_required=is_required)
    if "enum" in spec:
        return rng.choice(list(spec["enum"]))
    t = spec.get("type")
    if t == "object":
        return _sample_object(spec, rng, root_schema=root_schema, field_path=field_path)
    if t == "array":
        return _sample_array(spec, rng, root_schema=root_schema, field_path=field_path)
    if t == "string":
        return _sample_string(field_path)
    if t == "integer":
        return _sample_int(field_path[-1] if field_path else "", rng)
    if t == "number":
        rng_pf = profiles.numeric_range(field_path[-1] if field_path else "") or {}
        lo, hi = float(rng_pf.get("lo", 0)), float(rng_pf.get("hi", 100))
        return round(rng.uniform(lo, hi), 2)
    if t == "boolean":
        return rng.random() < 0.5
    if t == "null":
        return None
    return None


def _sample_anyof(options: list[dict], rng: random.Random, *,
                  root_schema: dict, field_path: tuple, is_required: bool) -> Any:
    # Common pattern in these schemas: [{enum/type/...}, {type: null}]
    null_idx = next((i for i, o in enumerate(options) if o.get("type") == "null"), None)
    non_null = [o for i, o in enumerate(options) if i != null_idx]
    if null_idx is not None and not is_required:
        # ~30% null for optional fields by default — gives realistic sparsity
        if rng.random() < 0.30:
            return None
    if non_null:
        chosen = rng.choice(non_null)
        return _sample_value(chosen, rng, root_schema=root_schema,
                             field_path=field_path, is_required=True)
    return None


def _sample_array(spec: dict, rng: random.Random, *,
                  root_schema: dict, field_path: tuple) -> list:
    items_spec = spec.get("items", {})
    field_name = field_path[-1] if field_path else ""
    # Heuristic per-array length defaults
    if field_name == "biomarkers":
        n = 4
    elif field_name == "margins":
        n = rng.randint(1, 5)
    elif field_name == "regional_lymph_node":
        n = rng.randint(1, 4)
    else:
        n = rng.randint(0, 3)
    return [_sample_value(items_spec, rng, root_schema=root_schema,
                          field_path=field_path + (str(i),), is_required=True)
            for i in range(n)]


def _sample_string(field_path: tuple) -> str | None:
    """Free-text fields default to None — eval doesn't score them and synth strings risk leaking patterns."""
    return None


def _sample_int(field_name: str, rng: random.Random) -> int:
    rng_pf = profiles.numeric_range(field_name) or {}
    lo = int(rng_pf.get("lo", 1))
    hi = int(rng_pf.get("hi", 50))
    return rng.randint(lo, hi)
