# Digital Registrar — Research

> Research stack for **The Digital Registrar**: the extraction pipeline, the annotation UI, the comparison benchmarks, and the ablation study. One `pyproject.toml`, one import root.

For the slim, production-facing extractor (what non-academic users typically install), see the standalone [`digitalregistrar`](../digitalregistrar) repo. This package **vendors** that pipeline and adds the research apparatus around it.

## Install

```bash
git clone <this-repo> drr-next
cd drr-next
pip install -e .[all]
```

Extras are split by concern — install only what you need:

| Extra | What you get |
|---|---|
| `[annotation]` | Streamlit UI (`streamlit`) |
| `[benchmarks]` | GPT-4 / ClinicalBERT / rule-based baselines (`torch`, `transformers`, `openai`, `scikit-learn`, …) |
| `[ablations]` | Raw-JSON baseline (`jsonschema`) |
| `[dev]`        | `pytest`, `ruff`, `mypy` |
| `[all]`        | All of the above |

The core install (no extras) gives you the DSPy extraction pipeline plus the canonical Pydantic schemas.

After cloning, install the local pre-commit hook so lint errors are caught before they hit CI:

```bash
bash scripts/repo/install_git_hooks.sh
```

## Repo layout

```
drr-next/
├── src/digital_registrar_research/
│   ├── pipeline.py, pipeline_factory.py, runner.py   # DSPy extraction (v1 + v2 factory + batch runner)
│   ├── signatures/factory.py                         # dspy.Signature built from Pydantic case-models
│   ├── schemas/                                      # 3-layer source-of-truth (see docs/architecture/schemas.md)
│   │   ├── pydantic/   ← shape (Layer 1)
│   │   ├── extraction/ ← per-field desc + group (Layer 2)
│   │   ├── aliases/    ← canonical → surface forms TOML (Layer 3)
│   │   └── data/*.json ← auto-generated; never edit
│   ├── annotation/                                   # Streamlit doctor-review UI
│   ├── benchmarks/                                   # rule / ClinicalBERT / LLM baselines + eval harness
│   ├── ablations/                                    # modular vs monolithic × DSPy vs raw-JSON grid
│   ├── models/, util/, paths.py                      # support modules
│   └── _legacy/                                      # archived modules, off active import path
├── scripts/                                          # CLIs and run-time helpers
│   ├── _helpers/   ← _config_loader.py, _run_id.py
│   ├── _legacy/    ← archived scripts (eval_*, run_dspy_*, bootstrap_schema_v2.py, pathhelper*)
│   ├── ablations/, annotation/, baselines/, data/
│   ├── eval/       ← unified cascade eval CLI
│   ├── pipeline/, repo/
├── tests/                                            # mirrors src/
│   ├── _legacy/    ← collected only on opt-in
│   ├── ablations/, annotation/, baselines/, benchmarks/, eval/, pipeline/, schemas/
│   └── fixtures/reference/                           # TCGA reference set used by --folder reference
├── docs/                                             # see Documentation below
├── configs/                                          # eval endpoints, organ codes, per-model decoding
├── data/tcga_annotation_20251117/                    # tracked ground-truth annotations
├── results/                                          # benchmark output skeleton (.parquet fixtures + .gitkeep)
├── examples/dummy/                                   # runnable skeleton; data/results gitignored
├── obfuscator/                                       # standalone synthetic-data subpkg (own pyproject.toml)
└── packaging/                                        # build dispatcher + end-user run launchers
```

## Console scripts

```bash
registrar-pipeline   --input data/tcga_dataset_20251117/tcga1     # batch extraction
registrar-annotate                                                # launches Streamlit UI (legacy layout)
registrar-annotate-workspace                                      # launches against workspace/
registrar-annotate-dummy                                          # launches against examples/dummy/
registrar-benchmark                                               # aggregates baseline comparisons
registrar-ablate                                                  # runs ablation grid
registrar-schemas                                                 # regenerates JSON from Pydantic (use --check in CI)
```

## Documentation

| Topic | Doc |
|---|---|
| Repo overview & quick links | [docs/index.md](docs/index.md) |
| Extraction pipeline (v1 legacy + v2 factory) | [docs/architecture/pipeline.md](docs/architecture/pipeline.md) |
| Schema architecture (3-layer source-of-truth) | [docs/architecture/schemas.md](docs/architecture/schemas.md) |
| DSPy deep dive — strict-schema protocol, roadmap | [docs/architecture/dspy_deep_dive.md](docs/architecture/dspy_deep_dive.md) |
| DSPy ↔ Ollama model compatibility (frozen audit) | [docs/architecture/dspy_ollama_model_compatibility.md](docs/architecture/dspy_ollama_model_compatibility.md) |
| Annotation UI | [docs/workflows/annotation.md](docs/workflows/annotation.md) |
| 2026-04 experiment protocol | [docs/workflows/experiment_protocol.md](docs/workflows/experiment_protocol.md) |
| Branching strategy (12-branch model) | [docs/workflows/branching_strategy.md](docs/workflows/branching_strategy.md) |
| Obfuscated workspace (PHI-free debug copy) | [docs/workflows/obfuscation.md](docs/workflows/obfuscation.md) |
| Benchmarks — quickstart + 6-chapter tutorial | [docs/benchmarks/00_overview.md](docs/benchmarks/00_overview.md) |
| Ablation suite — full reference + rationale | [docs/ablations/index.md](docs/ablations/index.md) |
| Cascade evaluation pipeline | [docs/eval/index.md](docs/eval/index.md) |
| Datasets, layout, naming conventions | [docs/reference/data.md](docs/reference/data.md) |
| Statistical methods (cascade + ablation) | [docs/reference/stat_methods.md](docs/reference/stat_methods.md) |
| Schema-editor GUI blueprint | [docs/reference/schema_gui_blueprint.md](docs/reference/schema_gui_blueprint.md) |
| Literature review | [docs/reference/literature_review.md](docs/reference/literature_review.md) |

## Evaluation pipeline

The `scripts/eval/` tree exposes a unified subcommand CLI:

```bash
python -m scripts.eval.cli cascade       --root examples/dummy --dataset cmuh --model gpt_oss_20b --annotator gold --out <out>
python -m scripts.eval.cli iaa           --root examples/dummy --dataset cmuh --annotators gold nhc_with_preann nhc_without_preann kpc_with_preann kpc_without_preann --out <out>
python -m scripts.eval.cli completeness  --root examples/dummy --dataset cmuh --methods llm:gpt_oss_20b clinicalbert:v2_finetuned rule_based: --annotator gold --out <out>
python -m scripts.eval.cli diagnostics   --cascade-out <...> --iaa-out <...> --out <out>
python -m scripts.eval.cli cross_dataset --left <cmuh_out> --right <tcga_out> --out <out>
python -m scripts.eval.cli headline      --cascade-out <...> --iaa-out <...> --out <out>
```

All eval subcommands also accept `--obfustrated` as an alternative to `--root` — points reads/writes at the schema-conformant synthetic copy `workspace_obfustrated/` produced by `python scripts/data/obfuscate_workspace.py`. Useful for debugging eval logic without touching PHI; see [docs/workflows/obfuscation.md](docs/workflows/obfuscation.md). Existing `--root examples/dummy` / `--root workspace` invocations are unchanged.

Every subcommand also accepts `--device {auto,cpu,cuda,mps}` (default `cpu`) to route the bootstrap-CI / McNemar / Cohen's-κ / Fleiss-κ machinery onto a GPU. Use `--device mps` on Apple Silicon, `--device cuda` on a CUDA workstation, or `--device auto` for cross-machine scripts. The original CPU implementation in `ci.py` is preserved as the safety net (default behavior). See [docs/eval/gpu_acceleration.md](docs/eval/gpu_acceleration.md).

See [docs/eval/recipes.md](docs/eval/recipes.md) for the full recipe book and [docs/eval/methods_citations.md](docs/eval/methods_citations.md) for paper-ready statistical-method citations.

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

MIT. See [`LICENSE`](LICENSE).
