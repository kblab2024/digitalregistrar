# registrar-staging

AJCC TNM staging service. Wraps the vendored
[`tnmhelper`](../../../vendor/) (Layer-3 API) and ships its data as a
packaged zip, so DSPy-side code never reads from an uncompressed
`tnmhelper_data/` tree.

> **Status:** shipped but not yet wired into any pipeline / runner /
> eval script in this repo. The module exists to make staging available
> as a building block. Downstream integration (post-processing step on
> the factory pipeline, or a separate stage-only runner) is future work.
> See [worked examples](../../../examples/staging_demo/).

## Two surfaces

### Plain callables (direct invocation; no LM in the loop)

```python
from digital_registrar.staging import (
    organs,
    editions_for,
    observable_schema,
    derive_tnm,
    stage_from_observations,
    stage,
)

organs()                                   # -> ['breast', 'lung', ...]
editions_for("breast")                     # -> [<Edition: AJCC 8>, ...]
schema = observable_schema("breast", "AJCC 8")
# {'tumor_size_cm': ObservableSpec(type='number', ...), ...}

derived = stage_from_observations(
    "breast", "AJCC 8",
    observations={"tumor_size_cm": 2.3, "ln_positive_count": 0, ...},
    Classification="p",
)
# DerivedStage(T='T2', N='N0', M='M0', stage='IIA', source=..., state=...)
```

`stage(...)` is a column-level variant: supply pre-decided
`pt_category`, `pn_category`, `pm_category`, … strings and get back the
resulting stage. Use `stage_from_observations(...)` when you have raw
observations and want `tnmhelper` to do the derivation.

`PASSTHROUGH_DEFAULTS` (re-exported from `tnmhelper`) holds the default
`Classification` / `Desc*` values; override only what differs.

### DSPy tool surface (`STAGING_TOOLS`)

```python
import dspy
from digital_registrar.staging import STAGING_TOOLS

class StageFromReport(dspy.Signature):
    """Derive the AJCC TNM stage from a pathology report."""
    report: str = dspy.InputField()
    organ: str = dspy.InputField()
    edition: str = dspy.InputField()
    stage: str = dspy.OutputField()

agent = dspy.ReAct(StageFromReport, tools=STAGING_TOOLS, max_iters=6)
agent(report=..., organ="breast", edition="AJCC 8")
```

`STAGING_TOOLS` is a list of two `dspy.Tool` instances:

| Tool name | Wraps | When to call |
|---|---|---|
| `cancer_observable_schema` | `observable_schema(organ, edition)` | First — learn what observations the engine expects |
| `cancer_stage_from_observations` | `stage_from_observations(organ, edition, observations, **pass_through)` | Last — pass the values found in the report |

Per-tool descriptions are authored in [tools.py](tools.py) and are what
the ReAct planner actually reads.

## Data bundle

Ships at [staging/data/tnmhelper_data.zip](data/tnmhelper_data.zip),
declared as `package-data` in [pyproject.toml](../../../pyproject.toml)
(line 64). Loaded lazily on first call via `_ensure_configured`
([adapter.py:30-42](adapter.py)).

Testing hooks:

```python
from digital_registrar.staging import (
    active_data_source,
    set_data_source_override,
)

set_data_source_override("/path/to/custom/tnmhelper_data")  # zip or dir
active_data_source()  # introspect the resolved source
```

## Rebuilding the bundle

See [../../../vendor/README.md](../../../vendor/README.md). Short
version:

```bash
python -m tnmhelper.bundle export \
  --out src/digital_registrar/staging/data/tnmhelper_data.zip \
  --source /path/to/tnmhelper/tnmhelper_data \
  --verify
```

A rebuild is required whenever the upstream `tnmhelper_data/` tree
changes (new organ, new edition, fixed table).

## Limitations

- Requires the `tnmhelper` wheel installed. Handled automatically via
  `[tool.uv.sources]` in this repo's pyproject; for pip-only installs
  follow `vendor/README.md`.
- `tnmhelper` is not on PyPI as of this writing — the vendored wheel
  is the only distribution.
- Field names in `observable_schema(...)` are `tnmhelper`-native; they
  do not match drr-next's extraction-output field names verbatim. A
  small adapter step is required when piping `cancer_data` (from the
  factory pipeline) into `stage_from_observations(...)`. See
  [examples/staging_demo/post_process.py](../../../examples/staging_demo/post_process.py)
  for the friction this implies.

## Where the surfaces live

- Plain callables → [adapter.py](adapter.py)
- DSPy tools → [tools.py](tools.py)
- Public API → [`__init__.py`](__init__.py)
