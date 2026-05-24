# Digital Registrar

> **A schema-first framework for multi-cancer, privacy-preserving pathology abstraction via local LLMs.**

[![Preprint](https://img.shields.io/badge/medRxiv-10.1101%2F2025.10.21.25338475-blue)](https://www.medrxiv.org/content/10.1101/2025.10.21.25338475v8) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Python 3.11+](https://img.shields.io/badge/python-3.11+-brightgreen)](https://www.python.org/downloads/)

Digital Registrar transforms free-text surgical pathology reports into machine-readable registry records using a College of American Pathologists (CAP)-aligned clinical ontology, encoded as strictly-typed DSPy signatures. The system covers **10 major cancer types across 193 registry fields** — including complex variable-length structures like lymph-node groups and surgical margins — and is **model-agnostic**: any local LLM can serve as the inference engine. Designed for on-premise deployment on a single 48 GB GPU, it keeps sensitive clinical text inside the institution.

## Highlights

- **Schema-first architecture** — the clinical ontology is the durable contribution; LLMs are interchangeable engines.
- **CAP-aligned, registry-grade** — 10 cancer types, 193 fields, validated against gold-standard human annotations.
- **Privacy-preserving by design** — local LLMs only, single 48 GB GPU, no cloud round-trip required.
- **Validated generalizability** — **94.3 %** mean exact-match on 893 internal reports; **92.4 %** on the external TCGA cohort of 150 reports ([preprint](https://www.medrxiv.org/content/10.1101/2025.10.21.25338475v8)).

## Quickstart (end users)

The toolkit ships as four pip-installable packages. Pick the apps you need:

```bash
# Inference GUI — paste a report, see the structured extraction
pip install digital-registrar-gui
registrar-infer-gui                 # opens http://localhost:8502

# Annotation tool — review pipeline output against gold
pip install digital-registrar-annotator
registrar-annotate-workspace

# Schema editor — curate the CAP-aligned per-organ schema
pip install digital-registrar-schema-editor
registrar-schema-gui

# Core only (CLI + Python API) — for pipelines, scripts, and downstream tools
pip install digital-registrar
registrar-pipeline --input <folder>
```

Each app depends on `digital-registrar` (the core), so installing any of the apps brings the pipeline along automatically.

## Audience

Built for **cancer registrars, pathology informatics teams, and clinical researchers** who need registry-grade structured extraction from narrative pathology reports without sending PHI off-premise.

## Repository layout

```
drr-next/
├── src/digital_registrar/      ← THE core (pipeline, schemas, signatures, eval, paths)
├── apps/
│   ├── infer-gui/              ← digital-registrar-gui (Streamlit inference)
│   ├── schema-editor/          ← digital-registrar-schema-editor
│   └── annotator/              ← digital-registrar-annotator
├── attic/                      ← research scaffolding (benchmarks, ablations, baselines, obfuscator)
├── packaging/                  ← release pipeline (PyInstaller, Docker, hosted demo)
├── workspace/                  ← gitignored runtime data (data, results, runs)
├── examples/                   ← small read-only fixtures
├── tests/                      ← core tests
└── docs/                       ← architecture, API, eval, release
```

## Dev install (cloners)

```bash
git clone https://github.com/kblab2024/digitalregistrar.git digitalregistrar 
cd digitalregistrar 
 make install-dev      # installs core + 3 apps + dev tooling
make test             # core + app test suites
make lint             # ruff
```

`make install-dev` installs the vendored `tnmhelper` wheel first, then `pip install -e .` (core), then `pip install -e apps/<each>` for the three downstream apps. Anyone with `pip` can clone and install in one command — no `uv` required.

## Public Python API

```python
from digital_registrar import (
    run_pipeline, setup_pipeline,                  # extraction
    load_pydantic_model, load_json_schema,         # schemas
    list_organs, CASE_MODELS, build_case_model,
    build_extraction_signatures, ExtractionStep,   # signatures
    field_metrics, nested_field_metrics,           # eval
    pairwise_compare, completeness, score_case,
    WORKSPACE_ROOT, workspace_root, results_root,  # paths
)
```

See [docs/api.md](docs/api.md) for the full reference.

## Documentation

| Topic | Where |
|---|---|
| Pipeline architecture (v1 legacy, v2 factory) | [docs/architecture/pipeline.md](docs/architecture/pipeline.md) |
| Three-layer schema architecture | [docs/architecture/schemas.md](docs/architecture/schemas.md) |
| AJCC TNM staging via `tnmhelper` | [docs/architecture/staging.md](docs/architecture/staging.md) |
| DSPy deep dive | [docs/architecture/dspy_deep_dive.md](docs/architecture/dspy_deep_dive.md) |
| Annotation workflow | [docs/workflows/annotation.md](docs/workflows/annotation.md) |
| Eval (prediction vs annotation) | [docs/eval/index.md](docs/eval/index.md) |
| Public Python API | [docs/api.md](docs/api.md) |
| Release pipeline (PyPI / hosted demo / bundles / Docker) | [docs/release.md](docs/release.md) |
| Research scaffolding (benchmarks, ablations, obfuscator) | [attic/README.md](attic/README.md) |

## Releasing

The project supports three distribution paths for layman users (see [docs/release.md](docs/release.md)):

- **PyPI** — `pip install digital-registrar-gui` for Python users.
- **Hosted Streamlit demo** — public URL for paper reviewers / casual visitors. Safety checklist in [docs/release.md](docs/release.md).
- **Native bundles + Docker** — `.dmg` / `.exe` / Docker images for non-technical end users, built via `make bundle` and `make docker-build`.

## Citation

If you use the Digital Registrar in your research, please cite:

> Chow N-H, Chang H, Chen H-K, et al. *Digital Registrar: A Schema-First Framework for Multi-Cancer Privacy-Preserving Pathology Abstraction via Local LLMs.* medRxiv 2026. doi: [10.1101/2025.10.21.25338475](https://doi.org/10.1101/2025.10.21.25338475)

(Preprint; the citation will be updated to the published-journal version when available. Machine-readable metadata in [CITATION.cff](CITATION.cff).)

## License

MIT — see [LICENSE](LICENSE).
