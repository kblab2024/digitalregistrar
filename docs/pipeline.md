# Extraction pipeline

**The Digital Registrar** is a model-agnostic, resource-efficient AI framework for comprehensive cancer surveillance from pathology reports. The pipeline is privacy-first (locally deployable via Ollama) and currently extracts 193+ CAP-aligned fields across 10 cancer types.

## Layout

```
src/digital_registrar_research/
├── pipeline.py                 # CancerPipeline (dspy.Module)
├── experiment.py               # batch entry-point — `registrar-pipeline` CLI
├── models/
│   ├── common.py               # is_cancer router, ReportJsonize, model_list, autoconf_dspy
│   ├── modellist.py            # organmodels: organ → list of subsection signature class names
│   └── <organ>.py × 11         # per-organ DSPy signatures + nested Pydantic types
└── util/
    ├── logger.py               # structured / json logging
    └── predictiondump.py       # DSPy Prediction → flat JSON-safe dict
```

## How a report flows through

1. `is_cancer` — top-level routing signature decides whether the report describes a primary cancer excision and, if so, which of ten organs.
2. `ReportJsonize` — first-pass conversion of the raw report into a roughly-structured JSON.
3. Per-organ subsection signatures (5–7 per organ) extract `cancer_data` fields. Each signature targets a slice of the CAP checklist (Nonnested / Staging / Margins / LN / Biomarkers / Othernested) so it fits the LM's context window comfortably.
4. Outputs are merged via `cancer_data.update(...)` into a single flat dict that matches the canonical `<organ>.json` schema.

## Running

Programmatically:

```python
from digital_registrar_research.pipeline import setup_pipeline, run_cancer_pipeline

setup_pipeline("gpt")                           # backbone from models.common.model_list
output, elapsed = run_cancer_pipeline(report=open("report.txt").read())
```

CLI (batch over a folder of `*.txt`):

```bash
registrar-pipeline --input data/tcga_dataset_20251117/tcga1 --model gpt
```

## Backbones supported

`models.common.model_list` ships with Ollama and OpenAI-compatible entries:

```
gemma1b, gemma4b, gemma12b, gemma27b      → ollama_chat/gemma3:*
gemma4e2b                                  → ollama_chat/gemma4:e2b
gpt                                       → ollama_chat/gpt-oss:20b
phi4, qwen30b, med8b, med70b              → ollama_chat/*
```

For OpenAI / Azure backbones, swap `dspy.LM(model="openai/gpt-4-turbo", ...)` and re-run `dspy.configure(lm=...)`.

## Where to make changes

- **Add a new organ** — write `models/<organ>.py` (mirror an existing organ's structure), register subsection class names in `models.modellist.organmodels`, add the organ key to `cancer_category`'s `Literal` in `models.common.is_cancer`. Then create `schemas/pydantic/<organ>.py` (one-liner) and run `registrar-schemas` to generate the JSON.
- **Tweak an existing field's vocabulary** — edit the `Literal[...]` in the appropriate `models/<organ>.py` signature. Run `registrar-schemas --check` and re-generate; the [concordance test](schemas.md#concordance) ensures Pydantic and JSON schemas stay aligned.
- **Replace the LM backbone** — extend `model_list` in `models/common.py`.
