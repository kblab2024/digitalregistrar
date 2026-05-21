# Benchmarks — overview

> Last updated: 2026-05-11 · Reflects: `d11d072`

The benchmark workflow compares three extraction methods against the same
gold annotations:

| Method | Predictor | Output path |
|---|---|---|
| Rule-based | `src/.../benchmarks/baselines/rules.py` | `{root}/results/predictions/{dataset}/rule_based/...` |
| ClinicalBERT (v1 baseline + v2 finetuned) | `src/.../benchmarks/baselines/clinicalbert/` | `{root}/results/predictions/{dataset}/clinicalbert/...` |
| LLM via DSPy + factory engine | `pipeline_factory.py` through `runner.py` | `{root}/results/predictions/{dataset}/llm/{model}/run{NN}/...` |

All three feed the same `cascade` evaluation subcommand at
[`scripts/eval/cli.py`](../../scripts/eval/cli.py).

## Read in order

1. [01_data_layout.md](01_data_layout.md) — canonical input + output paths
2. [02_train_bert.md](02_train_bert.md) — training the ClinicalBERT heads
3. [03_run_baselines.md](03_run_baselines.md) — predicting with rule, BERT, LLM
4. [04_evaluate.md](04_evaluate.md) — per-method evaluation via the `cascade` subcommand (replaces legacy `non_nested` / `nested`)
5. [05_compare.md](05_compare.md) — side-by-side comparison via `run_compare` and convenience wrappers
6. [06_methods.md](06_methods.md) — descriptions, scope, and limitations of each method

## Deprecated entry points

- The old `registrar-benchmark` console script (which ran
  `benchmarks.eval.run_all:main`) is a deprecation stub that prints the
  new commands.
- The old `pairwise_compare` module is retired in favour of
  `scripts.eval.cascade.compare_runs`.
- The legacy per-method eval scripts (`eval_gpt_oss_multirun.py`,
  `eval_iaa.py`, `eval_lymph_nodes.py`, `eval_margins.py`,
  `iaa_and_accuracy_report.py`) live at [`scripts/_legacy/`](../../scripts/_legacy/)
  for historical reference.
