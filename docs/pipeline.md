# Extraction pipeline

**The Digital Registrar** is a model-agnostic, resource-efficient AI
framework for comprehensive cancer surveillance from pathology reports.
The pipeline is privacy-first (locally deployable via Ollama) and
currently extracts 193+ CAP-aligned fields across 10 cancer types.

## Engines

Two engines coexist. The CLI flag `--engine {legacy,factory}` selects
between them; both produce the same `cancer_data` shape.

- **`legacy`** (default): the original
  [`CancerPipeline`](../src/digital_registrar_research/pipeline.py)
  with hand-written `dspy.Signature` subclasses in
  [`models/<organ>.py`](../src/digital_registrar_research/models/).
- **`factory`** (v2): the schema-driven
  [`CancerPipelineV2`](../src/digital_registrar_research/pipeline_factory.py),
  which builds DSPy signatures dynamically from
  [Pydantic case-models](../src/digital_registrar_research/schemas/pydantic/)
  via the [signature factory](../src/digital_registrar_research/signatures/factory.py).
  Adding a new cancer type = write one Pydantic file. The router's
  `cancer_category` Literal is auto-derived from `CASE_MODELS`; no
  manual edits to the router are ever needed.

The factory engine becomes the default once it has validated against
legacy on a TCGA fixture. Until then, legacy is the default and `--engine
factory` is opt-in.

## Layout

```
src/digital_registrar_research/
├── pipeline.py                 # legacy: CancerPipeline (dspy.Module)
├── pipeline_factory.py         # v2: CancerPipelineV2 (schema-driven)
├── runner.py                   # batch entry point — `registrar-pipeline` CLI
├── signatures/
│   ├── __init__.py
│   └── factory.py              # build_signature, build_extraction_signatures, etc.
├── schemas/
│   ├── pydantic/               # ← v2 SOURCE OF TRUTH
│   │   ├── _factory_helpers.py # GroupedField + group iteration helpers
│   │   ├── _common_types.py    # nested BaseModels (BreastMargin, ColonLN, …)
│   │   └── <organ>.py × 10     # hand-authored case-models per organ
│   ├── data/<organ>.json       # auto-generated JSON schemas
│   ├── generate.py             # `registrar-schemas` CLI
│   └── pydantic/_builder.py    # DEPRECATED — used only by bootstrap script
├── models/                     # legacy: per-organ DSPy signatures (kept until v2 validates)
│   ├── common.py               # legacy is_cancer + ReportJsonize + model_list
│   ├── modellist.py            # legacy organmodels routing dict
│   └── <organ>.py × 10         # legacy per-organ DSPy signatures
└── util/
    ├── logger.py
    └── predictiondump.py       # DSPy Prediction → flat JSON-safe dict
```

## Schema-driven flow (factory engine)

```
report (str | list[str])
        │
        ▼
  ┌──────────────┐
  │ Router       │   ← build_router_signature(CASE_MODELS)
  │ is_cancer    │     cancer_category Literal AUTO-DERIVED from CASE_MODELS
  └─────┬────────┘
        │ cancer_excision_report=True, cancer_category="breast"
        ▼
  ┌──────────────┐ (optional, default OFF; gate with --jsonize)
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
  │      … one signature per `_GROUP_INSTRUCTIONS` entry … ]    │
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

Factory decomposes the flat case-model into LM calls. Choose with
`--decomposition`:

- **`per_group`** — one signature per group tag (small focused schemas
  per LM call). Good for ~20-30B local models.
- **`monolithic`** — one signature with all fields (one LM call). Good
  for hosted frontier models that handle big response schemas reliably.
- **`auto`** *(default)* — picks based on
  - `model_profile` argument (e.g. `"small"`, `"large"`, the dspy/litellm
    model id),
  - the live LM's `supports_response_schema` capability flag (DSPy 3.2),
  - schema size: per-group when >25 fields or >4 groups; monolithic
    otherwise.

`auto` consults `dspy.settings.lm` at signature build time, which is why
extractors are built lazily on first `forward()` (`setup_pipeline_v2()`
must run first).

## How a report flows through (legacy engine)

1. `is_cancer` — routing signature decides whether the report describes
   a primary cancer excision and, if so, which of ten organs.
2. `ReportJsonize` — first-pass conversion of the raw report into a
   roughly-structured JSON.
3. Per-organ subsection signatures (5–7 per organ) extract `cancer_data`
   fields. Each signature targets a slice of the CAP checklist (Nonnested /
   Staging / Margins / LN / Biomarkers / Othernested) so it fits the LM's
   context window comfortably.
4. Outputs are merged via `cancer_data.update(...)` into a single flat
   dict that matches the canonical `<organ>.json` schema.

## Running

Programmatically (factory engine):

```python
from digital_registrar_research.pipeline_factory import (
    setup_pipeline_v2, run_cancer_pipeline_v2,
)
setup_pipeline_v2("gpt")
output, elapsed = run_cancer_pipeline_v2(
    report=open("report.txt").read(),
    decomposition="auto",     # or "per_group" / "monolithic"
    jsonize_enabled=False,    # default OFF
)
```

CLI (batch over a folder of `*.txt`):

```bash
# Legacy (default).
registrar-pipeline --input data/tcga_dataset_20251117/tcga1 --model gpt

# Factory v2.
registrar-pipeline --engine factory \
    --input data/tcga_dataset_20251117/tcga1 \
    --model gpt --decomposition auto --no-jsonize
```

Wrappers under `scripts/pipeline/` come in two flavors:

- `scripts/pipeline/run_factory_*.py` — factory-engine wrappers (Ollama
  single, Ollama smoke, OpenAI single, inference smoke).
- `scripts/pipeline/legacy/run_*.py` — original wrappers, untouched.

## Backbones supported

[`models.common.model_list`](../src/digital_registrar_research/models/common.py#L18-L46)
ships with Ollama and OpenAI-compatible entries:

```
gemma1b, gemma4b, gemma12b, gemma27b      → ollama_chat/gemma3:*
gemma4e2b                                  → ollama_chat/gemma4:e2b
gpt, gptoss                               → ollama_chat/gpt-oss:20b
phi4, qwen30b, med8b, med70b              → ollama_chat/*
gpt5_4_mini                                → openai/gpt-5.4-mini
```

For OpenAI / Azure backbones, the entry is dispatched through DSPy's
LiteLLM bridge automatically.

## Where to make changes (factory engine)

- **Add a new cancer type** — write `schemas/pydantic/<organ>.py`
  (a `BaseModel` with `_GROUP_INSTRUCTIONS: ClassVar[dict]` and one
  `GroupedField(group=, desc=)` per output field — copy an existing
  organ as a template). Register in `CASE_MODELS`. Run
  `registrar-schemas`. Done — no DSPy edits, no router edits.
- **Tweak an existing field's vocabulary** — edit the `Literal[...]` in
  the appropriate `schemas/pydantic/<organ>.py` and run
  `registrar-schemas --check`. The
  [concordance test](schemas.md#concordance) catches drift.
- **Change a per-group LM instruction** — edit the corresponding entry
  in `_GROUP_INSTRUCTIONS` in the same file. Insertion order matters:
  it controls extraction order AND first-wins on duplicate field names.
- **Replace the LM backbone** — extend `model_list` in
  `models/common.py`. Both engines share this.
- **Build a custom decomposition strategy** — extend
  [`_choose_decomposition`](../src/digital_registrar_research/signatures/factory.py)
  in the factory.

## Where to make changes (legacy engine)

The legacy stack is preserved so existing ablation runs and the
strict/structured pipelines (`pipeline_structured.py`,
`pipeline_dspy_strict.py`) keep working. To add a cancer type to the
legacy engine, see the `models/<organ>.py` layout — but new types should
be authored on the v2 schema side; the legacy engine will pick them up
through `CASE_MODELS` once the factory engine is the default.
