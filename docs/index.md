# Digital Registrar — Research

Research stack for **The Digital Registrar**, a model-agnostic, resource-efficient AI framework for comprehensive cancer surveillance from pathology reports.

This package wraps four research concerns around the slim production extractor:

| Subpackage | Purpose |
|---|---|
| [`digital_registrar_research.pipeline` + `models/` + `util/`](pipeline.md) | Core DSPy extraction (vendored from the slim `digitalregistrar` release) |
| [`digital_registrar_research.schemas`](schemas.md) | Canonical Pydantic case-models → generated JSON schemas |
| [`digital_registrar_research.annotation`](annotation.md) | Streamlit UI for doctors to review GPT-OSS pre-annotations |
| [`digital_registrar_research.benchmarks`](benchmarks/README.md) | Comparison vs LLM / ClinicalBERT / rule-based — canonical run + eval workflow |
| [`digital_registrar_research.ablations`](ablations.md) | Modular vs monolithic DSPy × DSPy vs raw-JSON grid |
| [example data](data.md) | Datasets (cmuh, tcga), layout conventions, dummy skeleton |
| [experiment protocol](experiment_protocol.md) | 2026-04 experiment cross-product, evaluation questions, invariants |
| [branching strategy](branching_strategy.md) | 12-branch working model (testing / refactor / experiment state) |
| [evaluation pipeline](eval/index.md) | Paper-grade metric explanations, recipes, and citations |
| [DSPy deep dive](dspy_deep_dive.md) | Why the strict-schema protocol works on `gpt-oss:20b`, deeper DSPy concepts, phased dev roadmap |
| [obfuscated workspace](obfuscation.md) | Schema-conformant synthetic copy of `workspace/` for PHI-free debugging |

## Why this exists

The original `digitalregistrar` package is the slim, production-facing extractor — most non-academic users install just that and never touch the surrounding research apparatus. This research package vendors the pipeline and adds the four research concerns under one import root, so academic users don't have to juggle four sibling repos with `sys.path.insert` glue.

Concretely, this consolidation:

1. Replaces four `requirements.txt` files with one `pyproject.toml` + extras.
2. Eliminates every `sys.path.insert` — clean `from digital_registrar_research.…` imports throughout.
3. Establishes one canonical Pydantic schema layer that both the JSON schemas (consumed by the annotation UI and the raw-JSON ablation runner) and the DSPy signatures (used by the extraction pipeline) are pinned against in CI.
4. Co-locates the TCGA gold set so benchmarks and ablations work out of the box.

## Quick start

```bash
git clone <this-repo> digital-registrar-research
cd digital-registrar-research
pip install -e .[all]
pytest                                    # 46 tests
registrar-schemas --check                 # confirm Pydantic ↔ JSON parity
registrar-annotate                        # launch the annotation UI
```

## Relationship to the slim release

The `digitalregistrar/` repo (sibling folder, untouched) is still the pip-installable home for production users. The vendored copy under `src/digital_registrar_research/` is the **research tip-of-tree**. When research-side improvements stabilise, backport them to the slim release manually; `scripts/repo/diff_against_slim.py` shows what has diverged.
