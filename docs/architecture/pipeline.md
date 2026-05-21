# Extraction pipeline

> Last updated: 2026-05-11 · Reflects: `d11d072`

**The Digital Registrar** extracts CAP-aligned cancer fields from pathology
reports. The pipeline is model-agnostic, locally deployable via Ollama, and
covers 193+ fields across 10 cancer types.

## Engines

Two engines coexist on the active import path. `runner.py` wires both —
`--engine {legacy,factory}` selects, both produce the same `cancer_data`
shape.

| Engine | Module | Source-of-truth | Selection |
|---|---|---|---|
| `legacy`  | [`pipeline.py`](../../src/digital_registrar_research/pipeline.py) | Hand-written `dspy.Signature` per subsection in [`models/<organ>.py`](../../src/digital_registrar_research/models/) | Default |
| `factory` | [`pipeline_factory.py`](../../src/digital_registrar_research/pipeline_factory.py) | Pydantic case-models in [`schemas/pydantic/<organ>.py`](../../src/digital_registrar_research/schemas/pydantic/) compiled via the [signature factory](../../src/digital_registrar_research/signatures/factory.py) | `--engine factory` |

The factory engine adds a new cancer type from **one** Pydantic file — the
router's `cancer_category` Literal is auto-derived from `CASE_MODELS`. No
manual edits to the router.

Two earlier variants (`pipeline_dspy_strict.py`, `pipeline_structured.py`)
are preserved at [`src/digital_registrar_research/_legacy/`](../../src/digital_registrar_research/_legacy/),
off the active import path. They are not reachable through `runner.py` or
any console script. See `_legacy/README.md` for the rationale.

## Layout

```
src/digital_registrar_research/
├── pipeline.py                 # legacy: CancerPipeline (dspy.Module)
├── pipeline_factory.py         # v2: CancerPipelineV2 (schema-driven)
├── runner.py                   # batch entry point — `registrar-pipeline` CLI
├── signatures/
│   ├── __init__.py
│   └── factory.py              # build_signature, build_extraction_signatures, …
├── schemas/                    # see architecture/schemas.md
│   ├── pydantic/               # source of truth (v2)
│   ├── extraction/             # per-field desc + group (Layer 2)
│   ├── aliases/                # canonical → surface forms (Layer 3, TOML)
│   ├── data/<organ>.json       # auto-generated; never edit
│   └── generate.py             # `registrar-schemas` CLI
├── models/                     # legacy per-organ DSPy signatures
├── util/                       # logger, prediction dump
├── annotation/, benchmarks/, ablations/   # research subpackages
└── _legacy/                    # archived modules, not on import path
    ├── pipeline_dspy_strict.py
    └── pipeline_structured.py
```

## Schema-driven flow (factory engine)

```
report (str | list[str])
        │
        ▼
  ┌──────────────┐
  │ Router       │   ← build_router_signature(CASE_MODELS)
  │ is_cancer    │     cancer_category Literal AUTO-DERIVED
  └─────┬────────┘
        │ cancer_excision_report=True, cancer_category="breast"
        ▼
  ┌──────────────┐ (optional; gate with --jsonize)
  │ ReportJsonize│
  └─────┬────────┘
        │ json_report (rough dict)
        ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ Per-group extraction (decomposition="per_group")            │
  │   build_extraction_signatures(BreastCancerCase, ...) →      │
  │     [Step("nonnested", BreastCancer__nonnested, …),         │
  │      Step("dcis",      BreastCancer__dcis,      …),         │
  │      Step("grading",   BreastCancer__grading,   …),         │
  │      … one signature per _GROUP_INSTRUCTIONS entry … ]      │
  │   first-wins on duplicate field names                       │
  └─────┬────────────────────────────────────────────────────────┘
        │ merged dict
        ▼
  ┌──────────────┐
  │ Validate     │   ← CASE_MODELS["breast"].model_validate(data)
  └─────┬────────┘     (degraded mode: warn on errors, don't crash)
        │
        ▼
  {"cancer_excision_report": True, "cancer_category": "breast",
   "cancer_category_others_description": null, "cancer_data": {...}}
```

## Decomposition modes (factory engine)

`--decomposition`:

- **`per_group`** — one signature per group tag. Small focused schemas per
  LM call. Best for ~20–30B local models.
- **`monolithic`** — one signature with all fields, one LM call. Best for
  hosted frontier models with reliable large-schema decoding.
- **`auto`** *(default)* — picks per call based on:
  - `model_profile` argument (e.g. `"small"`, `"large"`, the dspy/litellm model id),
  - the live LM's `supports_response_schema` capability flag (DSPy 3.2),
  - schema size: `per_group` when >25 fields or >4 groups; `monolithic` otherwise.

`auto` consults `dspy.settings.lm` at signature build time, which is why
extractors are built lazily on first `forward()`. `setup_pipeline_v2()`
must run first.

## How a report flows through (legacy engine)

1. `is_cancer` — routing signature decides whether the report describes a
   primary cancer excision and, if so, which of ten organs.
2. `ReportJsonize` — first-pass conversion of the raw report into a
   roughly-structured JSON.
3. Per-organ subsection signatures (5–7 per organ) extract `cancer_data`
   fields. Each targets a CAP-checklist slice (Nonnested / Staging /
   Margins / LN / Biomarkers / Othernested) so it fits the LM context.
4. Outputs are merged via `cancer_data.update(...)` into a single flat
   dict matching the canonical `<organ>.json` schema.

## Running

Programmatically (factory engine):

```python
from digital_registrar_research.pipeline_factory import (
    setup_pipeline_v2, run_cancer_pipeline_v2,
)
setup_pipeline_v2("gpt")
output, elapsed = run_cancer_pipeline_v2(
    report=open("report.txt").read(),
    decomposition="auto",     # "per_group" | "monolithic" | "auto"
    jsonize_enabled=False,
)
```

CLI (batch over a folder of `*.txt`):

```bash
# Legacy (default)
registrar-pipeline --input data/tcga_dataset_20251117/tcga1 --model gpt

# Factory v2
registrar-pipeline --engine factory \
    --input data/tcga_dataset_20251117/tcga1 \
    --model gpt --decomposition auto --no-jsonize
```

Interactively (single report or folder, with sidebar controls for engine /
model / decomposition):

```bash
registrar-infer-gui              # Streamlit GUI, defaults to port 8502
```

See [../../src/digital_registrar_research/inference_gui/README.md](../../src/digital_registrar_research/inference_gui/README.md)
for sidebar semantics and output paths. The GUI uses the same
`run_cancer_pipeline_v2` / `run_cancer_pipeline` entry points as the CLI,
so its `runs/run_<timestamp>/<stem>_output.json` outputs are interchangeable
with batch runs.

Pipeline wrappers under `scripts/pipeline/` are all factory-engine:
`run_factory_ollama_single.py`, `run_factory_ollama_smoke.py`,
`run_factory_openai_single.py`, `run_factory_inference_smoke.py`. The
historical legacy wrappers are at `scripts/_legacy/`.

## Backbones supported

[`models.common.model_list`](../../src/digital_registrar_research/models/common.py)
ships with Ollama and OpenAI-compatible entries:

```
gemma1b, gemma4b, gemma12b, gemma27b      → ollama_chat/gemma3:*
gemma4e2b                                  → ollama_chat/gemma4:e2b
gpt, gptoss                                → ollama_chat/gpt-oss:20b
phi4, qwen30b, med8b, med70b               → ollama_chat/*
gpt5_4_mini                                → openai/gpt-5.4-mini
```

For OpenAI / Azure backbones the entry is dispatched through DSPy's
LiteLLM bridge automatically.

## Where to make changes (factory engine)

- **Add a new cancer type** — write `schemas/pydantic/<organ>.py` (a
  `BaseModel` with `_GROUP_INSTRUCTIONS: ClassVar[dict]` and one
  `GroupedField(group=, desc=)` per output field; copy an existing
  organ). Register in `CASE_MODELS`. Run `registrar-schemas`. No DSPy or
  router edits.
- **Tweak an existing field's vocabulary** — edit the `Literal[...]` in
  `schemas/pydantic/<organ>.py` and run `registrar-schemas --check`.
  [Concordance test](schemas.md#concordance) catches drift.
- **Change a per-group LM instruction** — edit the entry in
  `_GROUP_INSTRUCTIONS`. Insertion order matters: it controls extraction
  order AND first-wins on duplicate field names.
- **Add a surface-form alias** — edit `schemas/aliases/<organ>.toml`.
  No Python touch required; injected into prompts via `value_hints`.
- **Replace the LM backbone** — extend `model_list` in
  `models/common.py`. Both engines share this.
- **Build a custom decomposition strategy** — extend
  [`_choose_decomposition`](../../src/digital_registrar_research/signatures/factory.py)
  in the factory.

## Where to make changes (legacy engine)

Legacy is preserved so existing benchmarks and reproducibility runs keep
working. To add a cancer type, prefer the v2 Pydantic path; the legacy
engine picks up new types via `CASE_MODELS` once promoted there.

## Available: staging service

`digital_registrar_research.staging` exposes the vendored `tnmhelper`
engine through two surfaces — plain callables (`stage_from_observations`
et al.) and a `dspy.ReAct`-compatible tool list (`STAGING_TOOLS`). It is
**not invoked by the pipeline today**; the module ships as a building
block for downstream consumers and future wirings. See
[staging.md](staging.md) for the API, the data-bundle lifecycle, and
worked examples in [`examples/staging_demo/`](../../examples/staging_demo/).
