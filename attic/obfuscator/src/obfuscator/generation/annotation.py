"""Layer 2b: derive a slot-specific annotation from report_realization."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from .. import noise, profiles
from ..schemas import load_schema
from ..seeding import derive


def derive_annotation(realization: dict[str, Any], organ: str,
                      slot: str, master_seed: int,
                      case_seed_parts: tuple,
                      preann: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a noisy annotation for one slot.

    For *_with_preann slots: when our perturbed value disagrees with preann,
    follow preann with probability `preann_follow_prob`.
    """
    profile = profiles.annotator(slot)
    schema = load_schema(organ)
    out = deepcopy(realization)
    rng = derive(master_seed, *case_seed_parts, "annotation", slot)
    _perturb_object(out, schema, schema, master_seed,
                    case_seed_parts + ("annotation", slot),
                    profile.rate)
    if profile.bias == "preann_anchor" and preann is not None:
        _anchor_to_preann(out, preann, rng, profile.preann_follow_prob)
    return out


def _perturb_object(obj: dict, spec: dict, root_schema: dict,
                    master_seed: int, parts: tuple, rate: float) -> None:
    properties = spec.get("properties", {})
    for name, field_spec in properties.items():
        if name not in obj:
            continue
        obj[name] = _perturb_value(obj[name], field_spec, root_schema,
                                   master_seed, parts + (name,), rate)


def _perturb_value(value: Any, spec: dict, root_schema: dict,
                   master_seed: int, parts: tuple, rate: float) -> Any:
    if value is None:
        return value
    if "$ref" in spec:
        from .canonical import _resolve_ref
        sub_spec = _resolve_ref(spec["$ref"], root_schema)
        if isinstance(value, dict):
            _perturb_object(value, sub_spec, root_schema, master_seed, parts, rate)
        return value
    if "anyOf" in spec:
        non_null = [o for o in spec["anyOf"] if o.get("type") != "null"]
        if non_null:
            return _perturb_value(value, non_null[0], root_schema, master_seed, parts, rate)
        return value
    if "enum" in spec:
        nn = profiles.near_neighbors(parts[-1] if parts else "")
        return noise.perturb_enum(value, list(spec["enum"]), master_seed,
                                  parts, rate, near_neighbors=nn or None)
    t = spec.get("type")
    if t == "object":
        if isinstance(value, dict):
            _perturb_object(value, spec, root_schema, master_seed, parts, rate)
        return value
    if t == "array":
        items_spec = spec.get("items", {})
        return [_perturb_value(item, items_spec, root_schema, master_seed,
                               parts + (str(i),), rate)
                for i, item in enumerate(value)]
    if t == "boolean":
        return noise.perturb_bool(value, master_seed, parts, rate)
    if t == "integer":
        rng_pf = profiles.numeric_range(parts[-1] if parts else "") or {}
        return noise.perturb_int(value, master_seed, parts, rate,
                                 sigma_pct=float(rng_pf.get("sigma_pct", 0.15)),
                                 lo=int(rng_pf["lo"]) if "lo" in rng_pf else None,
                                 hi=int(rng_pf["hi"]) if "hi" in rng_pf else None)
    return value


def _anchor_to_preann(obj: Any, preann: Any, rng, follow_prob: float) -> None:
    if isinstance(obj, dict) and isinstance(preann, dict):
        for k in obj.keys() & preann.keys():
            v_obj = obj[k]
            v_pre = preann[k]
            if isinstance(v_obj, dict) and isinstance(v_pre, dict):
                _anchor_to_preann(v_obj, v_pre, rng, follow_prob)
            elif isinstance(v_obj, list) and isinstance(v_pre, list):
                # Element-wise anchor for lists of equal length
                for i in range(min(len(v_obj), len(v_pre))):
                    if isinstance(v_obj[i], dict) and isinstance(v_pre[i], dict):
                        _anchor_to_preann(v_obj[i], v_pre[i], rng, follow_prob)
                    elif v_obj[i] != v_pre[i] and rng.random() < follow_prob:
                        v_obj[i] = v_pre[i]
            elif v_obj != v_pre and rng.random() < follow_prob:
                obj[k] = v_pre
