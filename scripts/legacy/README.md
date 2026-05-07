# Legacy evaluation scripts

These scripts are **deprecated** and retained only for one transition release. Use the new `scripts/eval/cli.py` subcommand-based pipeline instead — it covers everything these scripts did plus a much wider statistics surface, paper-grade documentation, and proper missingness / preann-effect / semantic-neighbor analyses.

## Migration map

| Legacy script | New equivalent |
|---|---|
| `eval_gpt_oss_multirun.py` | `python -m scripts.eval.cli non_nested --root <dummy\|workspace> --dataset <cmuh\|tcga> --model <name> --annotator gold --out <out>` |
| `eval_iaa.py` | `python -m scripts.eval.cli iaa --root <...> --dataset <...> --annotators gold nhc_with_preann nhc_without_preann kpc_with_preann kpc_without_preann --out <out>` |
| `eval_lymph_nodes.py` | `python -m scripts.eval.cli nested --field regional_lymph_node --root <...> --dataset <...> --model <name> --annotator gold --out <out>` |
| `eval_margins.py` | `python -m scripts.eval.cli nested --field margins --root <...> --dataset <...> --model <name> --annotator gold --out <out>` |
| `iaa_and_accuracy_report.py` | `python -m scripts.eval.cli headline --non-nested-out <...> --iaa-out <...> --out <out>` |

## What's new in the rewrite

- **Three-way outcome classification** (correct / wrong / missing) — missing is no longer conflated with wrong.
- **Multi-method, multi-model, multi-run** as a first-class concept — no more model-name-baked scripts.
- **Pre-annotation effect analysis** — Δκ, anchoring index, convergence-to-preann, disagreement reduction.
- **Schema conformance / out-of-vocabulary rate** — direct measurement of the modularity advantage.
- **Source-of-error decomposition** — model_error vs report_ambiguity vs report_silent.
- **Multi-primary case stratification** — bilateral / multifocal subgroup column on every metric.
- **Curated semantic-neighbor analysis** — `accuracy_collapsing_neighbors` for clinically-equivalent confusion pairs.
- **Cross-dataset generalisation** — KL / JS / Wasserstein distribution-shift indicators.
- **Multiple-comparisons correction** — Holm-Bonferroni for primary, Benjamini-Hochberg for secondary endpoints.
- **Paper-grade documentation** in `docs/eval/` — every metric carries its package + function + version + original-paper citation.
- **Standardised folder layout** — `--root dummy|workspace`, `--dataset cmuh|tcga`, `--annotator <full_subdir_name>`.

See [docs/eval/index.md](../../docs/eval/index.md) for the full documentation tree.

## Removal timeline

These files will be deleted in a follow-up release once the new pipeline has been validated against the same input data. Until then, keep them available so we can run side-by-side regression diffs.
