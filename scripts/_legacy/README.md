# `_legacy/` — archived scripts

Scripts here are kept for historical reference. They are **not** invoked by any
console entry point or active workflow. The underscore prefix keeps the
directory at the bottom of file listings.

## What lives here

### Eval scripts (replaced by `scripts/eval/cli.py`)

| File | Replaced by |
|---|---|
| `eval_gpt_oss_multirun.py`    | `python -m scripts.eval.cli non_nested --root <dummy\|workspace> --dataset <cmuh\|tcga> --model <name> --annotator gold --out <out>` |
| `eval_iaa.py`                 | `python -m scripts.eval.cli iaa --root <...> --dataset <...> --annotators ... --out <out>` |
| `eval_lymph_nodes.py`         | `python -m scripts.eval.cli nested --field regional_lymph_node ...` |
| `eval_margins.py`             | `python -m scripts.eval.cli nested --field margins ...` |
| `iaa_and_accuracy_report.py`  | `python -m scripts.eval.cli headline --non-nested-out <...> --iaa-out <...>` |

The new pipeline adds three-way outcome (correct / wrong / missing),
multi-method × multi-model × multi-run, pre-annotation effect, schema
conformance, source-of-error decomposition, multi-primary stratification,
semantic-neighbor analysis, cross-dataset generalisation, multiple-comparisons
correction, and paper-grade docs in `docs/eval/`. See
[docs/eval/index.md](../../docs/eval/index.md).

### Pipeline runners (replaced by `runner.py` + `pipeline_factory.py`)

| File | Notes |
|---|---|
| `run_dspy_ollama_*.py` (4 files, single/multirun × smoke/full) | Pre-v2 schema runs against Ollama; superseded by `registrar-pipeline` |
| `run_dspy_strict_ollama_*.py` (2 files) | Pair to `_legacy/pipeline_dspy_strict.py` |
| `run_structured_ollama_*.py` (2 files) | Pair to `_legacy/pipeline_structured.py` |
| `run_pipeline_openai_*.py` (2 files)    | Pre-v2 OpenAI runs |
| `run_gpt_oss_multirun.py` | Replaced by current ablation grid |
| `run_inference_smoke.py`  | Replaced by `tests/pipeline/test_v2_parity.py` |

### One-shot scripts

| File | Status |
|---|---|
| `bootstrap_schema_v2.py` | Marked `DEPRECATED: remove after v2 ships` in its own docstring. v2 shipped at commit `259cf27` |
| `pathhelper.py`          | Zero references anywhere in the tree |
| `pathhelper2.py`         | Zero references anywhere in the tree |

## When to look here

Reproducing a result from a paper or experiment log that cites one of these
scripts by name. For anything else, the current canonical entry points are:

- `registrar-pipeline` (extraction)
- `python -m scripts.eval.cli <subcommand>` (evaluation)
- `python -m scripts.ablations.run_ablations` (ablation grid)

## When to delete

Drop the whole directory once no external citation, branch, or experiment log
depends on it. `git log -- scripts/_legacy/` recovers anything needed.
