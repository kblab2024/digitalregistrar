# Digital Registrar — Research

> Research stack for **The Digital Registrar**: the pipeline, the annotation UI, the comparison benchmarks, and the ablation study — one pyproject, one import root.

For the slim, production-facing extraction package (what non-academic users typically install), see the standalone [`digitalregistrar`](../digitalregistrar) repo. This package **vendors** that pipeline and adds the research apparatus around it.

## Install

```bash
git clone <this-repo> digital-registrar-research
cd digital-registrar-research
pip install -e .[all]
```

Extras are split by concern — install only what you need:

| Extra | What you get |
|---|---|
| `[annotation]` | Streamlit UI (`streamlit`) |
| `[benchmarks]` | GPT-4 / ClinicalBERT / rule-based baselines (`torch`, `transformers`, `openai`, `scikit-learn`, …) |
| `[ablations]` | Raw-JSON baseline (`jsonschema`) |
| `[dev]` | `pytest`, `ruff`, `mypy` |
| `[all]` | all of the above |

The core install (no extras) gives you the DSPy extraction pipeline plus the canonical Pydantic schemas.

After cloning, install the local pre-commit hook so lint errors are caught before they hit CI:

```bash
bash scripts/repo/install_git_hooks.sh
```

## What's in the box

```
src/digital_registrar_research/
├── pipeline.py, experiment.py    # DSPy extraction pipeline (vendored from digitalregistrar)
├── models/                        # per-organ DSPy signatures + nested Pydantic types
├── util/                          # logging, prediction dump
├── schemas/                       # canonical Pydantic case-models → generated JSON schemas
│   ├── pydantic/                  # ← the source of truth
│   └── data/*.json                # ← generated artifacts used by downstream tools
├── annotation/                    # Streamlit UI for doctors to review pre-annotations
├── benchmarks/                    # baselines (GPT-4, ClinicalBERT, rules) + eval harness
└── ablations/                     # modular-vs-monolithic DSPy + raw-JSON grid
```

Data and results use a flat, convention-driven layout covering
`data/{dataset}/reports|preannotation|annotations/` and
`results/predictions/{dataset}/{llm,clinicalbert,rule_based}/...` —
see [docs/data.md](docs/data.md) for the full tree, and
[scripts/data/gen_dummy_skeleton.py](scripts/data/gen_dummy_skeleton.py) to
generate a runnable dummy under `dummy/`.

## Console scripts

```bash
registrar-pipeline   --input data/tcga_20251117/dataset/tcga1     # batch extraction
registrar-annotate                                                # launches Streamlit UI
registrar-benchmark                                               # aggregates baseline comparisons
registrar-ablate                                                  # runs ablation grid
registrar-schemas                                                 # regenerates JSON from Pydantic (use --check in CI)
```

## Documentation

- [docs/index.md](docs/index.md) — overview
- [docs/pipeline.md](docs/pipeline.md) — how the extraction pipeline works
- [docs/annotation.md](docs/annotation.md) — using the Streamlit app
- [docs/schemas.md](docs/schemas.md) — canonical Pydantic pattern; regenerating JSON; concordance test
- [docs/benchmarks.md](docs/benchmarks.md) — comparison methodology + literature review
- [docs/ablations.md](docs/ablations.md) — ablation design rationale
- [docs/data.md](docs/data.md) — datasets, layout, and naming conventions
- [docs/experiment_protocol.md](docs/experiment_protocol.md) — the 2026-04 experiment cross-product, evaluation questions, and invariants
- [docs/branching_strategy.md](docs/branching_strategy.md) — the 12-branch working model (testing / refactor / experiment state)
- [docs/eval/index.md](docs/eval/index.md) — evaluation pipeline (paper-grade metric explanations + citations)

## Evaluation pipeline

The `scripts/eval/` tree exposes a unified subcommand CLI for all evaluation work:

```bash
python -m scripts.eval.cli non_nested    --root dummy --dataset cmuh --model gpt_oss_20b --annotator gold --out <out>
python -m scripts.eval.cli nested        --root dummy --dataset cmuh --model gpt_oss_20b --annotator gold --field regional_lymph_node --out <out>
python -m scripts.eval.cli iaa           --root dummy --dataset cmuh --annotators gold nhc_with_preann nhc_without_preann kpc_with_preann kpc_without_preann --out <out>
python -m scripts.eval.cli completeness  --root dummy --dataset cmuh --methods llm:gpt_oss_20b clinicalbert:v2_finetuned rule_based: --annotator gold --out <out>
python -m scripts.eval.cli diagnostics   --non-nested-out <...> --iaa-out <...> --out <out>
python -m scripts.eval.cli cross_dataset --left <cmuh_out> --right <tcga_out> --out <out>
python -m scripts.eval.cli headline      --non-nested-out <...> --iaa-out <...> --out <out>
```

All eval subcommands also accept `--obfustrated` as an alternative to `--root` — points reads/writes at the schema-conformant synthetic copy `workspace_obfustrated/` produced by `python scripts/obfuscate_workspace.py`. Useful for debugging eval logic without touching PHI; see [docs/obfuscation.md](docs/obfuscation.md). The existing `--root dummy` / `--root workspace` invocations are unchanged.

Every subcommand also accepts `--device {auto,cpu,cuda,mps}` (default `cpu`) to route the bootstrap-CI / McNemar / Cohen's-κ / Fleiss-κ machinery onto a GPU. Use `--device mps` on Apple Silicon, `--device cuda` on a CUDA workstation, or `--device auto` for cross-machine scripts. The original CPU implementation in `ci.py` is preserved as the safety net (default behavior). See [docs/eval/gpu_acceleration.md](docs/eval/gpu_acceleration.md) for the full guide, expected speedups, and per-primitive coverage.

See [docs/eval/recipes.md](docs/eval/recipes.md) for the full recipe book and [docs/eval/methods_citations.md](docs/eval/methods_citations.md) for paper-ready statistical-method citations. Legacy scripts are archived under `scripts/legacy/` for one transition release.

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

MIT. See [`LICENSE`](LICENSE).
