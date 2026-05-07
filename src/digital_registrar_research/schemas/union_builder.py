"""Union schema across all organs — for the F2 ablation.

The default per-organ schema lets the model focus on the (5–7) sub-
sections relevant to the organ in hand. F2 collapses ALL ten organs
into a single mega-schema and lets the model both pick the organ and
extract every field in one pass. The dimensionality difference is the
point of the ablation: how much does narrow schema scope buy?

Design choice: when two organs both expose a field with the same name
(e.g. ``grade``), the cross-organ enum union from
:func:`scope.get_allowed_values` already collapses the value space.
We do the same here at the JSON-schema level — first-wins on field
definition, with the alphabetical organ order as the tie-break for
reproducibility.
"""
from __future__ import annotations

from functools import cache

from ..benchmarks.eval.scope import IMPLEMENTED_ORGANS
from .builder import flatten_schema_for_prompt, load_organ_schema


@cache
def build_union_schema(organs: tuple[str, ...] | None = None) -> dict:
    """Return a flat JSON-Schema union over the requested organs.

    ``organs`` defaults to all implemented organs. The result is suitable
    for ``response_format={"type": "json_schema", "schema": <this>}``
    or for inlining into a prompt via :func:`describe_field_list`.
    """
    organ_list = list(organs) if organs is not None else sorted(IMPLEMENTED_ORGANS)

    merged: dict = {}
    for organ in organ_list:
        try:
            flat = flatten_schema_for_prompt(load_organ_schema(organ))
        except FileNotFoundError:
            continue
        for field, spec in flat.get("properties", {}).items():
            if field in merged:
                continue
            merged[field] = spec

    # Surface the routing decision as a top-level enum so the model can
    # populate cancer_excision_report + cancer_category in the same call.
    merged.setdefault("cancer_excision_report", {
        "type": "boolean",
        "description": "True if the report describes a primary cancer "
                       "excision eligible for cancer registry.",
    })
    merged.setdefault("cancer_category", {
        "type": "string",
        "enum": organ_list + ["others"],
        "description": "Which organ the primary cancer arises from.",
    })

    return {
        "type": "object",
        "title": "cancer_data_union",
        "properties": merged,
        "additionalProperties": False,
    }


__all__ = ["build_union_schema"]
