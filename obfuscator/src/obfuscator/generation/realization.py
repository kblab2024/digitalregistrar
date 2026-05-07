"""Layer 1: derive report_realization from canonical_state by applying omission noise.

The 'realization' is what the report TEXT actually says — some fields aren't mentioned.
Schema-aware: only drops OPTIONAL fields (those whose schema includes a null branch in
anyOf). Required fields must stay populated so downstream slot annotations remain
schema-conformant.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from .. import profiles
from ..schemas import load_schema
from ..seeding import derive


def derive_realization(canonical: dict[str, Any], organ: str,
                       master_seed: int,
                       case_seed_parts: tuple) -> dict[str, Any]:
    """Return a new dict with some optional fields dropped to None."""
    out = deepcopy(canonical)
    cfg = profiles.realization()
    drop_rate = cfg["drop_rate"]
    rng = derive(master_seed, *case_seed_parts, "realization")
    schema = load_schema(organ)
    _walk_drop_object(out, schema, schema, rng, drop_rate)
    return out


def _walk_drop_object(obj: dict, spec: dict, root_schema: dict, rng, drop_rate: float) -> None:
    properties = spec.get("properties", {})
    required = set(spec.get("required", []))
    for k, v in list(obj.items()):
        field_spec = properties.get(k, {})
        is_required = k in required
        is_optional = _has_null_branch(field_spec)
        if isinstance(v, dict):
            sub_spec = _resolve(field_spec, root_schema, want="object")
            if sub_spec is not None:
                _walk_drop_object(v, sub_spec, root_schema, rng, drop_rate)
        elif isinstance(v, list):
            items_spec = field_spec.get("items") or _resolve(field_spec, root_schema, want="array")
            for item in v:
                if isinstance(item, dict):
                    item_spec = _resolve(items_spec or {}, root_schema, want="object") or items_spec or {}
                    _walk_drop_object(item, item_spec, root_schema, rng, drop_rate)
        elif (v is not None and not isinstance(v, bool) and is_optional
              and not is_required and rng.random() < drop_rate):
            obj[k] = None


def _has_null_branch(spec: dict) -> bool:
    if "anyOf" in spec:
        return any(o.get("type") == "null" for o in spec["anyOf"])
    return False


def _resolve(spec: dict, root_schema: dict, *, want: str) -> dict | None:
    """Resolve $ref / anyOf to a concrete object/array sub-schema, if any."""
    if not spec:
        return None
    if "$ref" in spec:
        ref = spec["$ref"]
        if ref.startswith("#/"):
            cur: Any = root_schema
            for p in ref[2:].split("/"):
                cur = cur.get(p, {}) if isinstance(cur, dict) else {}
            return cur if isinstance(cur, dict) and cur.get("type") == want else cur
    if "anyOf" in spec:
        for o in spec["anyOf"]:
            inner = _resolve(o, root_schema, want=want)
            if inner is not None:
                return inner
    if spec.get("type") == want:
        return spec
    return None
