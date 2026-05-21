# Digital Registrar — documentation

> Software companion to: Chow N-H, Chang H, Chen H-K, et al. *Digital Registrar: A Schema-First Framework for Multi-Cancer Privacy-Preserving Pathology Abstraction via Local LLMs.* medRxiv 2026. [doi: 10.1101/2025.10.21.25338475](https://doi.org/10.1101/2025.10.21.25338475)

The toolkit ships as **four pip-installable packages** plus an `attic/` of research scaffolding kept for reproducibility but not maintained.

## Repo layout

```
drr-next/
├── src/digital_registrar/      ← THE core (pipeline, schemas, signatures, eval, paths)
├── apps/
│   ├── infer-gui/              ← digital-registrar-gui
│   ├── schema-editor/          ← digital-registrar-schema-editor
│   └── annotator/              ← digital-registrar-annotator
├── attic/                      ← class III/IV: benchmarks, ablations, baselines, obfuscator
├── packaging/                  ← PyPI / hosted demo / PyInstaller / Docker
├── workspace/                  ← gitignored runtime data
├── examples/                   ← small read-only fixtures
├── tests/                      ← core tests
├── vendor/                     ← tnmhelper wheel
└── docs/                       ← this directory
```

## Where to read what

| You want to … | Read |
|---|---|
| Install and try the inference GUI | [../README.md](../README.md) (Quickstart) |
| Understand the public Python API | [api.md](api.md) |
| Understand the v1 / v2 pipeline | [architecture/pipeline.md](architecture/pipeline.md) |
| Understand the 3-layer schema | [architecture/schemas.md](architecture/schemas.md) |
| Understand AJCC TNM staging via `tnmhelper` | [architecture/staging.md](architecture/staging.md) |
| Run regular eval (prediction vs annotation) | [eval/index.md](eval/index.md) |
| Use the annotation tool | [workflows/annotation.md](workflows/annotation.md) |
| Configure the schema editor | [reference/schema_gui_blueprint.md](reference/schema_gui_blueprint.md) |
| Release to PyPI / hosted demo / Docker / bundle | [release.md](release.md) |
| Find paper-time scaffolding (IAA, ablations, baselines) | [../attic/README.md](../attic/README.md) and [../attic/docs/](../attic/docs/) |

## Class hierarchy

The codebase is organised in four classes of importance, mirrored in the directory layout:

- **Class I — main**: pipeline (CLI + Python API), inference GUI, supporting infrastructure (schemas, signatures, staging, eval).
- **Class II — nice-to-have**: schema editor, annotation tool.
- **Class III — keep but tucked away**: benchmarks (in `attic/` but installable separately).
- **Class IV — near-retirement**: ablations, baselines, obfuscator, paper-specific eval scripts, legacy modules (all in `attic/`).

Class I + II are the four published PyPI packages. Class III + IV live under `attic/` as a single `drr-attic` package, installed only when you need to reproduce paper-time experiments.

## Quick links

- Paper: [medRxiv preprint](https://www.medrxiv.org/content/10.1101/2025.10.21.25338475v8)
- Repo: [github.com/kblab2024/digitalregistrar](https://github.com/kblab2024/digitalregistrar)
- Citation: [../CITATION.cff](../CITATION.cff)
