# Digital Registrar — Research

> Last updated: 2026-05-20 · Reflects: `a4552f6`

Research stack for **The Digital Registrar** — a schema-driven, model-agnostic
pipeline for cancer-registry extraction from pathology reports. One import
root, one `pyproject.toml`, four research concerns around the slim production
extractor (vendored under `src/digital_registrar_research/`).

## Repo layout

```
drr-next/
├── src/digital_registrar_research/
│   ├── pipeline.py + pipeline_factory.py + runner.py   # extraction (v1 + v2)
│   ├── signatures/                                     # DSPy signature factory
│   ├── schemas/                                        # 3-layer source-of-truth
│   ├── annotation/                                     # Streamlit review UI
│   ├── schema_gui/                                     # Streamlit 3-layer schema editor
│   ├── inference_gui/                                  # Streamlit pipeline-inference GUI
│   ├── staging/                                        # tnmhelper-backed AJCC TNM staging service
│   ├── benchmarks/                                     # baselines + harness
│   ├── ablations/                                      # ablation grid runners
│   ├── models/, util/, paths.py                        # support modules
│   └── _legacy/                                        # archived, off import path
├── scripts/                                            # CLIs + run-time helpers
│   ├── _helpers/, _legacy/
│   ├── ablations/, annotation/, baselines/, data/
│   ├── eval/    ── unified cascade eval CLI
│   ├── pipeline/, repo/
├── tests/                                              # mirrors src/ tree
├── docs/                                               # this directory
├── configs/, data/, results/                           # canonical layout
├── examples/dummy/                                     # runnable skeleton
├── obfuscator/                                         # standalone synthetic-data subpkg
├── packaging/                                          # build & launch dispatchers
└── vendor/                                             # vendored wheels (tnmhelper)
```

## Where to read what

| Concern | Doc |
|---|---|
| Extraction engines (v1 legacy + v2 factory + runner) | [architecture/pipeline.md](architecture/pipeline.md) |
| Canonical schema architecture (pydantic / extraction / aliases) | [architecture/schemas.md](architecture/schemas.md) |
| Why DSPy + gpt-oss:20b works; roadmap status | [architecture/dspy_deep_dive.md](architecture/dspy_deep_dive.md) |
| DSPy compatibility across local models | [architecture/dspy_ollama_model_compatibility.md](architecture/dspy_ollama_model_compatibility.md) |
| Annotation UI workflows | [workflows/annotation.md](workflows/annotation.md) |
| Schema editor GUI (`registrar-schema-gui`) | [../src/digital_registrar_research/schema_gui/README.md](../src/digital_registrar_research/schema_gui/README.md) |
| Pipeline inference GUI (`registrar-infer-gui`) | [../src/digital_registrar_research/inference_gui/README.md](../src/digital_registrar_research/inference_gui/README.md) |
| AJCC TNM staging service (`tnmhelper` wrap) | [architecture/staging.md](architecture/staging.md) |
| 2026-04 experiment protocol & status | [workflows/experiment_protocol.md](workflows/experiment_protocol.md) |
| 12-branch working model | [workflows/branching_strategy.md](workflows/branching_strategy.md) |
| Obfuscated workspace (PHI-free debug copy) | [workflows/obfuscation.md](workflows/obfuscation.md) |
| Comparison benchmarks (LLM / ClinicalBERT / rules) | [benchmarks/00_overview.md](benchmarks/00_overview.md) |
| Ablation suite & design rationale | [ablations/index.md](ablations/index.md) |
| Cascade evaluation methodology | [eval/index.md](eval/index.md) |
| Datasets, layout, naming conventions | [reference/data.md](reference/data.md) |
| Statistical methods (cascade + ablation) | [reference/stat_methods.md](reference/stat_methods.md) |
| Schema-editor GUI — original blueprint (design history) | [reference/schema_gui_blueprint.md](reference/schema_gui_blueprint.md) |
| Literature review | [reference/literature_review.md](reference/literature_review.md) |

## Quick start

```bash
git clone <this-repo> drr-next
cd drr-next
pip install -e .[all]
pytest -q                                  # 458 tests collected
registrar-schemas --check                  # Pydantic ↔ JSON parity check
registrar-annotate                         # launch the annotation UI
```

## Why this exists

The slim [`digitalregistrar`](https://github.com/kblab2024/digitalregistrar)
package is the pip-installable extractor most non-academic users want. This
research package **vendors** that pipeline at `src/digital_registrar_research/`
and adds the surrounding research apparatus — schemas as a canonical
source-of-truth, an annotation UI, comparison baselines, an ablation grid,
and a cascade-style evaluation pipeline — under one import root.

Concretely the consolidation:

1. One `pyproject.toml` with extras (`[annotation]`, `[benchmarks]`, `[ablations]`, `[dev]`, `[all]`) — no more sibling-repo `sys.path.insert` glue.
2. Pydantic case-models are the single source of truth; JSON schemas regenerate via `registrar-schemas`; concordance enforced in CI.
3. TCGA gold annotations are co-located (`data/tcga_annotation_20251117/`) so benchmarks and ablations run out of the box.
4. Archived material lives at `src/digital_registrar_research/_legacy/` and `scripts/_legacy/`, off the active import path and out of CI.
