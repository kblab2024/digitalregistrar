# AJCC TNM staging service

> Last updated: 2026-05-20 · Reflects: `a4552f6`

`digital_registrar_research.staging` wraps the vendored
[`tnmhelper`](../../vendor/) wheel to make deterministic AJCC TNM
staging available inside this repo. Two surfaces are exposed: a set of
plain callables for direct invocation, and a list of `dspy.Tool`
wrappers (`STAGING_TOOLS`) for `dspy.ReAct`-style use.

## Why

Extraction produces *observations* — T category, N category, M
category, tumor size, lymph-node counts, biomarkers. **Staging** is the
deterministic step that maps those observations to AJCC TNM stage
groups per edition. Doing it in code (not in the LM) means:

- stages are reproducible across runs and models,
- the AJCC edition is explicit and traceable (per-organ, per-edition
  tables live in the `tnmhelper` data bundle),
- there is one place to fix a staging-table error, instead of N model
  prompts.

The module exists so that a future pipeline change, or a downstream
consumer (e.g. a registry-export script), does not have to re-vendor
or re-wrap `tnmhelper`.

## Current status

The staging module is **shipped but not imported by any pipeline,
runner, or eval script in this repo** (verified by grep across `src/`,
`scripts/`, `tests/`). It is a building block, not a wired step. If
you go searching for the call site that turns extraction output into
stages, you will not find one — yet. See [worked examples](#worked-examples)
for runnable demonstrations of both surfaces.

## Layers

```
tnmhelper (vendored wheel, no DSPy awareness)
     │  Layer-3 public API:
     │  organs() · editions_for() · observable_schema() ·
     │  derive_tnm() · stage_from_observations() · stage()
     ▼
digital_registrar_research.staging.adapter
     │  - configures tnmhelper to read the packaged zip on first call
     │  - re-exports the Layer-3 API verbatim
     │  - adds set_data_source_override() / active_data_source()
     │    introspection hooks for tests
     ▼
digital_registrar_research.staging.tools
     │  - dspy.Tool wrappers: stage_from_observations_tool,
     │    observable_schema_tool
     ▼
digital_registrar_research.staging.STAGING_TOOLS   (list)
```

`staging.adapter` is the *only* place that calls
`tnmhelper.set_data_source(...)`; all other modules import the
re-exported callables. This keeps the data-source plumbing in one place.

## Two surfaces, not two patterns

The module exposes two *surfaces*. It does not prescribe how they
should be wired into a pipeline — that is a downstream decision.

### Plain-callable surface — direct invocation, no LM in the loop

```python
from digital_registrar_research.staging import (
    observable_schema, stage_from_observations,
)

schema = observable_schema("breast", "AJCC 8")
# {'tumor_size_cm': ObservableSpec(type='number', ...), ...}

derived = stage_from_observations(
    "breast", "AJCC 8",
    observations={"tumor_size_cm": 2.3, "ln_positive_count": 0, ...},
    Classification="p",
)
# DerivedStage(T='T2', N='N0', M='M0', stage='IIA', ...)
```

Suitable for: a deterministic post-processing pass after extraction;
a standalone staging CLI; batch re-staging of historical extractions
under a new AJCC edition.

### DSPy-tool surface — `STAGING_TOOLS` plus `dspy.ReAct`

```python
import dspy
from digital_registrar_research.staging import STAGING_TOOLS

class StageFromReport(dspy.Signature):
    """Derive the AJCC TNM stage from a pathology report.

    Use cancer_observable_schema first to learn which observations the
    engine expects, then call cancer_stage_from_observations once you
    have values for them.
    """
    report: str = dspy.InputField()
    organ: str = dspy.InputField()
    edition: str = dspy.InputField()
    stage: str = dspy.OutputField()

agent = dspy.ReAct(StageFromReport, tools=STAGING_TOOLS, max_iters=6)
```

Suitable for: agentic experiments where the LM plans which observations
to extract; one-shot "stage this report" tasks without a pre-built
extraction schema. **No module in drr-next currently constructs such
an agent** — the surface exists; the application does not.

## Data lifecycle

| Concern | Where |
|---|---|
| Wheel | [`vendor/tnmhelper-0.1.0-py3-none-any.whl`](../../vendor/tnmhelper-0.1.0-py3-none-any.whl) |
| Data bundle (ships in this package) | [`src/digital_registrar_research/staging/data/tnmhelper_data.zip`](../../src/digital_registrar_research/staging/data/tnmhelper_data.zip) |
| Package-data declaration | `[tool.setuptools.package-data]` in [pyproject.toml](../../pyproject.toml) (line 64) |
| Data-source pointer | `tnmhelper.set_data_source(...)`, called once by `_ensure_configured` in [adapter.py](../../src/digital_registrar_research/staging/adapter.py) |
| Test override | `set_data_source_override(path)` accepts a zip or directory |
| Introspection | `active_data_source()` returns the resolved path, or `None` until first use |

Rebuild instructions for both the wheel and the bundle live in
[vendor/README.md](../../vendor/README.md) — link rather than duplicate.

## Editions and organs

Coverage tracks `tnmhelper` upstream. Inspect what is currently
available with:

```python
from digital_registrar_research.staging import organs, editions_for

organs()
for organ in organs():
    print(organ, editions_for(organ))
```

Hard-coded lists in docs drift fast; defer to the runtime API.

## Where to make changes

- **Add a new organ or AJCC edition**: a `tnmhelper`-upstream change.
  Edit the upstream Layer-3 tables, rebuild the data bundle (see
  `vendor/README.md`), commit the new zip. No drr-next code change.
- **Add a new derived field on `DerivedStage`**: same — `tnmhelper`
  owns the schema. Re-export is a no-op (`staging.adapter` returns
  whatever `tnmhelper` returns).
- **Change the LM-facing tool description**: edit
  [tools.py](../../src/digital_registrar_research/staging/tools.py).
  The string in `dspy.Tool(desc=...)` is what the ReAct planner reads;
  the function signature it wraps is unchanged.

## Possible future wirings

A short, non-binding list of integration shapes this module enables.
None of these is on a schedule; they are listed so a reader can
evaluate which (if any) fits a downstream need.

- **Post-extraction hook on the factory pipeline.** After
  `CancerPipelineV2.forward()` produces `cancer_data`, an optional
  step could translate the relevant fields into `tnmhelper`
  observations and attach a `stage` block to the output dict.
- **Standalone `registrar-stage` console script.** Reads
  `<stem>_output.json` files produced by `registrar-pipeline` and
  emits a sidecar `<stem>_staging.json` with the derived TNM stage.
  Useful for re-staging historical runs under a new AJCC edition.
- **ReAct-style staging module.** A `dspy.Module` that takes a report
  and uses `STAGING_TOOLS` instead of the per-group extraction
  decomposition. Interesting as a baseline against the current
  factory pipeline; meaningful only with eval harness support.

## Worked examples

Two runnable scripts live at [examples/staging_demo/](../../examples/staging_demo/):

| Script | Surface | What it shows |
|---|---|---|
| [`post_process.py`](../../examples/staging_demo/post_process.py) | Plain callable | Run the factory pipeline on a hard-coded report, inspect `observable_schema(...)`, build the observations dict, call `stage_from_observations(...)`. Highlights the adapter friction between drr-next extraction output and `tnmhelper`'s observable names. |
| [`react_agent.py`](../../examples/staging_demo/react_agent.py) | DSPy-tool | Wraps `STAGING_TOOLS` in a `dspy.ReAct` module against a hard-coded report; prints the ReAct trajectory so the tool calls are visible. |

Neither script is part of the shipped pipeline; both live outside
`src/` deliberately. They are the canonical "see it work" entry
points for this module.
