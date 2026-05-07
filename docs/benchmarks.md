# Benchmarks → moved

This single-file overview has been replaced by a multi-file guide under [`benchmarks/`](benchmarks/README.md).

Why: the benchmark workflow now spans rule-based, ClinicalBERT (CLS / QA / merged), and LLM baselines, all sharing the canonical predictions tree at `{folder}/results/predictions/{dataset}/{method}/...`. The full surface (data layout, training, prediction, evaluation, side-by-side comparison) is too much for one page; it now lives as:

- [`benchmarks/README.md`](benchmarks/README.md) — quickstart + index
- [`benchmarks/01_data_layout.md`](benchmarks/01_data_layout.md) — canonical input + output paths
- [`benchmarks/02_train_bert.md`](benchmarks/02_train_bert.md) — training the ClinicalBERT heads
- [`benchmarks/03_run_baselines.md`](benchmarks/03_run_baselines.md) — predicting with rule, BERT, LLM
- [`benchmarks/04_evaluate.md`](benchmarks/04_evaluate.md) — per-method evaluation via the `cascade` subcommand (replaces legacy `non_nested` / `nested`)
- [`benchmarks/05_compare.md`](benchmarks/05_compare.md) — side-by-side comparison via `run_compare` and convenience wrappers
- [`benchmarks/06_methods.md`](benchmarks/06_methods.md) — descriptions, scope, and limitations of each method

The old `registrar-benchmark` console entry point (which ran `benchmarks.eval.run_all:main`) is now a deprecation stub that prints the new commands. The old `pairwise_compare` module is similarly retired.
