"""Staging service: wraps tnmhelper as an AJCC TNM staging engine for DSPy.

Two surfaces:

- Plain callables (`organs`, `editions_for`, `observable_schema`,
  `derive_tnm`, `stage_from_observations`, `stage`) for direct
  post-processing in extraction pipelines.
- `STAGING_TOOLS`: a list of `dspy.Tool` instances suitable for
  agentic / ReAct-style use.

Data is loaded from a zip bundle packaged inside this module
(`staging/data/tnmhelper_data.zip`); the uncompressed `tnmhelper_data/`
tree is never read at runtime.
"""
from __future__ import annotations

from .adapter import (
    PASSTHROUGH_DEFAULTS,
    active_data_source,
    derive_tnm,
    editions_for,
    model_spec,
    observable_schema,
    organs,
    set_data_source_override,
    stage,
    stage_from_observations,
)
from .tools import (
    STAGING_TOOLS,
    observable_schema_tool,
    stage_from_observations_tool,
)

__all__ = [
    "PASSTHROUGH_DEFAULTS",
    "STAGING_TOOLS",
    "active_data_source",
    "derive_tnm",
    "editions_for",
    "model_spec",
    "observable_schema",
    "observable_schema_tool",
    "organs",
    "set_data_source_override",
    "stage",
    "stage_from_observations",
    "stage_from_observations_tool",
]
