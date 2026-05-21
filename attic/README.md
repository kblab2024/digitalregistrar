# attic — research scaffolding

This directory holds code that supported the paper but is **not part of the maintained Digital Registrar toolkit**. It's kept in-tree for reproducibility, not active use.

## What lives here

| Path | What it is | Status |
|---|---|---|
| `src/drr_attic/benchmarks/` | Baseline comparisons (rule, ClinicalBERT, LLM), IAA, preann, multirun, cross-dataset eval helpers. | Class III: kept for future testing |
| `src/drr_attic/ablations/` | Ablation grid runners (modular vs monolithic × DSPy vs raw-JSON). | Class IV: near-retirement |
| `src/drr_attic/legacy/` | Archived early-iteration pipeline code. | Class IV |
| `obfuscator/` | PHI-safe workspace obfuscator (own pyproject). | Class IV |
| `eval_scripts/` | Paper-specific eval scripts: canonical tables, cascade, joint, cross-dataset, diagnostics, IAA, completeness. | Class IV |
| `ablations_scripts/`, `baselines_scripts/`, `legacy_scripts/` | Companion scripts. | Class IV |
| `docs/` | Retired documentation: benchmarks, ablations, IAA/preann/multirun eval docs, obfuscation workflow, etc. | Class III/IV |
| `configs/` | Eval endpoints, ablation grids, organ-code map (only used by attic'd code). | Class IV |
| `tests/` | Tests for the above. | Class III/IV |

## Install (only if you need it)

```bash
pip install -e attic/
```

This pulls in heavy deps (torch, transformers, scikit-learn, scipy, statsmodels, etc.) that the core `digital-registrar` does **not** require.

Console scripts provided:
- `registrar-benchmark` — `drr_attic.benchmarks.eval.run_all:main`
- `registrar-ablate` — `drr_attic.ablations.eval.run_ablations:main`
- `obfuscate-workspace` — PHI-safe synthetic workspace generator

## Regular eval is in the core

If you just want **prediction-vs-annotation metrics** (field-level P/R/F1, nested-field metrics, pairwise run comparison, completeness, bootstrap CIs), they live in the **core** package at `digital_registrar.eval` — install only `digital-registrar` and run `registrar-eval`. The attic holds only the paper-specific evaluation scaffolding (IAA, preann effect, multirun statistical analysis, BERT-specific eval, GPU-accelerated CIs).

## Caveats

- **DSPy compiled programs** saved with the old `digital_registrar_research.*` class names will not load. Regenerate via `scripts/ablations/compile_dspy.py` (now under `attic/ablations_scripts/`).
- The `examples/dummy/configs/models/rule_based.yaml` config points at `drr_attic.benchmarks.baselines.rules` — only resolves if attic is installed.
- The obfuscator now reads schemas via `importlib.resources` from the installed `digital_registrar` package, not a filesystem path. Make sure `digital-registrar` is installed before running it.
