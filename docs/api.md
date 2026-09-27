# Public Python API — `digital_registrar`

The `digital_registrar` package re-exports the most useful entry points at the top level. Everything documented here is part of the stable surface; submodules also remain importable for power users.

## Pipeline

```python
from digital_registrar import run_pipeline, setup_pipeline

# v2 factory engine (schema-driven, recommended)
result, duration = run_pipeline(
    report="The breast pathology report text here…",
    fname="case_0001",
    decomposition="per_group",        # or "monolithic", "auto"
    jsonize_enabled=True,
    validate_output=True,
)
```

- `run_pipeline(report, fname, decomposition='auto', jsonize_enabled=True, validate_output=False)` → `(dict, float)` — v2 factory engine. Primary entry point.
- `setup_pipeline(model_name)` → DSPy LM context. Call once per process; pipeline functions read from `dspy.context`.
- `run_pipeline_legacy(report, fname)` → `(dict, float)` — v1 hard-coded engine. Kept for parity testing.

## Schemas (CAP-aligned clinical ontology)

```python
from digital_registrar import (
    list_organs, load_pydantic_model, load_json_schema,
    CASE_MODELS, build_case_model,
)

list_organs()
# → ['breast', 'cervix', 'colorectal', 'esophagus', 'liver', 'lung',
#    'pancreas', 'prostate', 'stomach', 'thyroid']

BreastCase = load_pydantic_model("breast")  # type[pydantic.BaseModel]
breast_schema = load_json_schema("breast")   # dict — JSON Schema 2020-12
```

- `list_organs()` → `list[str]` — canonical organ codes.
- `load_pydantic_model(organ)` → `type[BaseModel]` — case-model for one organ.
- `load_json_schema(organ)` → `dict` — pre-generated JSON schema (regenerate with `registrar-schemas`).
- `CASE_MODELS: dict[str, type[BaseModel]]` — registry of all case-models.
- `build_case_model(organ)` → `type[BaseModel]` — dynamic construction.

## Signatures (DSPy)

```python
from digital_registrar import (
    ExtractionStep, build_extraction_signatures,
    build_router_signature, build_jsonize_signature,
)

case_model = load_pydantic_model("breast")
steps: list[ExtractionStep] = build_extraction_signatures(
    case_model, decomposition="per_group",
)
```

- `ExtractionStep` — dataclass: `(name, signature, output_field_names, group)`.
- `build_extraction_signatures(case_model, decomposition='per_group', model_profile=None)` → `list[ExtractionStep]`.
- `build_router_signature(case_models)` → `type[dspy.Signature]` — selects organ from text.
- `build_jsonize_signature()` → `type[dspy.Signature]` — legacy raw-JSON pass.

## Evaluation (prediction vs annotation)

```python
from digital_registrar import (
    field_metrics, nested_field_metrics,           # submodule aliases
    pairwise_compare, completeness,                # submodules
    score_case, score_lymph_nodes, score_margins,  # functions
)

per_case = score_case(gold_dict, pred_dict)  # cascade scoring; see docstring for scope=
```

- `field_metrics` — alias for `digital_registrar.eval.metrics` (exact-match accuracy, nested-list F1, folder scoring via `score_pairs` / `summarize_scores`).
- `nested_field_metrics` — alias for `digital_registrar.eval.nested_metrics` (LN, margins).
- `pairwise_compare` — two-run comparison: `compare_runs(atomic, method_a, method_b)` → per-field Δ, paired bootstrap CI, McNemar.
- `completeness` — missingness / refusal / out-of-vocab analysis (`completeness_atomic`, `aggregate_missingness`, `out_of_vocab_table`).
- `score_case(gold, pred, scope=None)` → `dict` — per-case scoring. `scope=None` runs the eligibility → organ → fields cascade; pass a field list for flat scoring.
- `digital_registrar.eval.load_pairs(pred_dir, gold_dir)` → `(pairs, stats)` — pairs `<case>_output.json` with `<case>_annotation.json` / `<case>.json`.
- `score_lymph_nodes(gold, pred)`, `score_margins(gold, pred)` → `dict` — nested-field scoring.

CLI access: `registrar-eval {metrics,compare,completeness}` — see `registrar-eval --help` and [eval/index.md](eval/index.md).

## Paths

```python
from digital_registrar import WORKSPACE_ROOT, workspace_root, results_root, SCHEMAS_DATA
```

- `WORKSPACE_ROOT: Path` — default runtime root (gitignored). Override via `$DIGITAL_REGISTRAR_WORKSPACE`.
- `workspace_root(name=None)` → `Path` — env-configurable workspace lookup.
- `results_root(name=None)` → `Path` — `$WORKSPACE_ROOT/results`.
- `SCHEMAS_DATA: Path` — packaged JSON schemas directory.

Filesystem details (`REPO_ROOT`, `DATA_ROOT`, `RUNS_ROOT`, etc.) are available from `digital_registrar.paths` if you need them, but they are **not** part of the top-level API because they're meaningless in non-editable (PyPI-installed) deployments.

## Environment variables

| Variable | What it does |
|---|---|
| `DIGITAL_REGISTRAR_WORKSPACE` | Override the workspace dir name (default: `workspace`). |
| `DIGITAL_REGISTRAR_DATA_ROOT` | Override the data root if your dataset lives outside the workspace tree. |

## Console scripts

| Script | Provided by | What it does |
|---|---|---|
| `registrar-pipeline` | digital-registrar | Batch extraction CLI. |
| `registrar-schemas` | digital-registrar | Regenerate `schemas/data/*.json`; `--check` for CI drift. |
| `registrar-eval` | digital-registrar | Prediction-vs-annotation eval CLI. |
| `registrar-infer-gui` | digital-registrar-gui | Streamlit inference GUI. |
| `registrar-schema-gui` | digital-registrar-schema-editor | Streamlit schema editor. |
| `registrar-annotate*` | digital-registrar-annotator | Streamlit annotation UIs. |
