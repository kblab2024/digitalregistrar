# Reading the eval outputs

How to interpret column names, CI bands, and the long-form vs wide-form conventions used across `scripts/eval/`.

> **Cascade-redesign note (2026-05).** The output tree changed top-to-bottom. The legacy `non_nested/` and `nested/` directories were replaced by:
>
> - `cascade_atomic.parquet` — long-form atomic table with new columns `cascade_stage` (A/B/C), `gate_pass` (bool), `others_disposition` (none / gold_others / pred_others / both_others).
> - `chapter1_eligibility/` — Stage-A `cancer_excision_report` metrics: overall.csv, confusion.csv, multirun_consistency.csv (when ≥ 2 runs).
> - `chapter2_organ_classification/` — Stage-B organ-classification metrics: overall.csv (with macro-F1 and Cohen's κ), confusion_per_class.csv, plus `others/` subdir holding `others_ledger.csv`, `others_subtype_breakdown.csv`, `others_confusion.csv`, `others_eligibility_audit.csv`.
> - `chapter3_field_extraction/` — Stage-C field accuracy: per_field_overall.csv, per_field_by_organ.csv, per_organ_overall.csv, plus the cascade-specific `cascade_funnel.csv` and `conditional_accuracy_grid.csv`.
> - `model_pair_tests/` — only when scoring multiple models in one run; pairwise McNemar / Cochran-Q / Stuart-Maxwell with Holm + BH adjustments.
>
> See [CHANGELOG.md](CHANGELOG.md) and [../stat_methods.md](../stat_methods.md).

## Output structure

Every subcommand writes to `--out <dir>` and stamps a `manifest.json` capturing:

- `args` — the full CLI invocation.
- `git_sha` — the repo state when the run was produced.
- `utc_timestamp` — when it ran.
- `n_cases_per_organ` — the case-count breakdown.
- `extra` — subcommand-specific metadata (run IDs, atomic-table size, ...).

**Always cite the manifest's `git_sha` next to any number quoted in the paper.** It's the canonical reproducibility key — combined with the input data, it uniquely identifies the run.

## Column conventions

Numeric columns follow consistent naming patterns:

### Point estimate columns
- `attempted_accuracy`, `effective_accuracy`, `f1_micro`, `cohen_kappa`, `delta`, etc.

### CI columns
- `<metric>_ci_lo`, `<metric>_ci_hi` — generic 95% CI lo/hi.
- `<metric>_wilson_lo`, `<metric>_wilson_hi` — Wilson CI for proportions.
- `<metric>_boot_lo`, `<metric>_boot_hi` — BCa bootstrap CI.
- `<metric>_t_ci_lo`, `<metric>_t_ci_hi` — Student-t CI for run-level means.

CIs are **always 95%** (α = 0.05) unless `--alpha` was overridden. Width and method are documented per metric in [ci_methods.md](ci_methods.md).

### Count columns
- `n_total` — full eligible cohort.
- `n_eligible` — cases with gold present.
- `n_attempted` — cases the model produced a value for.
- `n_correct`, `n_wrong`, `n_field_missing`, `n_parse_error` — three-way outcome counts.
- `n_cases`, `n_runs` — distinct cases / runs in the slice.

### p-value columns
- `p_value` — raw.
- `p_holm` — Holm-Bonferroni adjusted (within endpoint tier).
- `p_bh` — Benjamini-Hochberg adjusted (within endpoint tier).

See [multiple_comparisons.md](multiple_comparisons.md).

### Effect-size columns
- `effect_size` — Cohen's d / odds ratio / Cliff's δ as appropriate.
- `effect_size_ci_lo`, `effect_size_ci_hi`.

## Long-form vs wide-form

CSVs are written in **long form**: one row per (entity × statistic) tuple. Pivot in pandas / Excel for wide-form views.

Example: `non_nested/per_field_overall.csv` is long-form with one row per **field**. To get a (field × method) matrix:

```python
import pandas as pd
dfs = {m: pd.read_csv(f"{out}/non_nested/{m}/per_field_overall.csv")
       for m in ["gpt_oss_20b", "qwen3_30b", "clinicalbert"]}
wide = pd.concat({m: df.set_index("field")["attempted_accuracy"]
                  for m, df in dfs.items()}, axis=1)
```

Atomic tables (`*_atomic.parquet`) are also long-form — one row per `(run, case, field)` tuple, with all six outcome flags. **Every CSV is a deterministic reduce from a parquet**, so reruns with new aggregation parameters don't require re-scoring.

## Per-organ vs ALL rows

Per-organ tables also include an `"ALL"` row (or rows with `organ = "ALL"`) that aggregates across all organs. The "ALL" row treats organs equally weighted by case count — not equally weighted per-organ.

When citing in the paper:

- Headline numbers: use `organ = "ALL"`.
- Stratified breakdown: use per-organ rows.
- Don't mix the two without saying which.

## Subgroup column

Many CSVs include a `subgroup` column with values `single_primary` / `multi_primary` / `all`. Use:

- `subgroup = "all"` for the headline number.
- `subgroup = "single_primary"` and `subgroup = "multi_primary"` when stratification matters (multi-primary cases are clinically distinct and tend to have systematically different accuracy).

See [glossary.md](glossary.md).

## NaN conventions

- **NaN in CI columns** typically means the bootstrap couldn't produce a useful estimate (e.g. degenerate sample where every case is correct).
- **NaN in correlation / effect-size columns** typically means n is too small to compute (< 4 for Fisher-z, < 2 for SD-based metrics).
- **NaN in `cohen_kappa` etc.** means classes were degenerate (only one class observed).

NaN ≠ zero. Don't impute when consuming.

## Inf / -Inf

Should never appear. If you see them, it's a bug — please report.

## Schema-conformance and refusal calibration

These are *separate from* accuracy/missingness. Don't sum across categories — they answer different questions:

- **Accuracy** — quality on attempted answers.
- **Missingness** — what fraction was skipped or failed.
- **Schema conformance** — were attempted answers in the allowed enum?
- **Refusal calibration** — when the model said null, was that justified?

A field can be high on accuracy AND high on lazy-missingness simultaneously — the latter doesn't pull down the former.

## Reading the manifest

```json
{
  "subcommand": "non_nested",
  "args": {...},                  # exact CLI invocation
  "git_sha": "97c15d8...",        # commit to reproduce against
  "package_version": "0.1.0",
  "utc_timestamp": "2026-04-26T16:24:49.010290+00:00",
  "n_cases_per_organ": {
    "1": 100, "2": 100, "3": 100, ...
  },
  "extra": {
    "n_runs": 3,
    "run_ids": ["run01", "run02", "run03"],
    "n_atomic_rows": 21600,
    "n_unique_cases": 1000,
    "n_unique_fields": 12
  }
}
```

The `extra` block is subcommand-specific. Common keys:

- `run_ids` — exact runs included in this output.
- `n_atomic_rows` — size of the underlying parquet.
- `n_unique_cases` / `n_unique_fields` — cohort dimensions.

Citation pattern in the paper:

> "All evaluation results were generated with the `non_nested` subcommand at git SHA 97c15d8 on 2026-04-26 (UTC), using runs `run01–run03` of `gpt_oss_20b` against the `gold` annotator, n = 1000 unique cases across 10 organs."

## Reproducing a run

Given a manifest:

```bash
git checkout <git_sha>
python -m scripts.eval.cli <subcommand> <args>
```

Bootstrap CIs use `--seed` for determinism (default 0). The exact same input data + git SHA + seed should produce bit-identical numerics — if not, that's a regression to investigate.
