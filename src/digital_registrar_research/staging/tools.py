"""`dspy.Tool` wrappers around the staging adapter for agentic use."""
from __future__ import annotations

import dspy

from .adapter import observable_schema, stage_from_observations

stage_from_observations_tool = dspy.Tool(
    func=stage_from_observations,
    name="cancer_stage_from_observations",
    desc=(
        "Compute the AJCC TNM stage from extracted observations. "
        "Args: organ (str, e.g. 'lung'), edition (str like 'AJCC 9' or 'AJCC 8'), "
        "observations (dict mapping observable name to value; strings for enums, "
        "numbers for numeric observables, bools for booleans), and pass-through "
        "kwargs for non-derived columns (Classification='c'|'p'|'yc'|'yp', "
        "DescY/DescR/DescM='Yes'|'No', plus any biomarkers required by the organ). "
        "Returns a DerivedStage with fields T, N, M, stage, source, state."
    ),
)

observable_schema_tool = dspy.Tool(
    func=observable_schema,
    name="cancer_observable_schema",
    desc=(
        "Look up the input schema for an (organ, edition) pair before extracting. "
        "Returns a dict mapping observable name to ObservableSpec(type, unit, values, "
        "default, label, choices). Use this to decide which observations to extract "
        "from a clinical report and what their valid values are."
    ),
)

STAGING_TOOLS = [stage_from_observations_tool, observable_schema_tool]

__all__ = [
    "STAGING_TOOLS",
    "observable_schema_tool",
    "stage_from_observations_tool",
]
