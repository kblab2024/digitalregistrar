# Diagnostics — source-of-error decomposition, difficulty tiers, worst cases

When a model gets a field wrong, the cause matters: a model that misreads a clearly-stated value is fixable by better prompting; a model that "fails" on a field annotators themselves disagree about is bounded by the source reports, not by the model. Diagnostics decomposes errors into model-bound vs report-bound buckets so the actionable signal isn't hidden by an aggregate accuracy number.

## ELI5

Suppose a model gets 70% accuracy on `tumor_size`. Is the 30% gap because:

1. The model is bad at extracting numbers? (model error)
2. The reports themselves are vague — annotators also disagree? (report ambiguity)
3. The field isn't actually mentioned in the report — model has to refuse rather than guess? (report silent)

These three buckets call for **different remediation**: better prompts (1), better reports / better guidelines (2), better refusal training (3). Lumping them as "30% wrong" hides the actionable signal.

## Source-of-error decomposition

For each (organ, field) we classify each model error into one of three buckets using IAA-derived field difficulty:

| Bucket | Heuristic |
|---|---|
| `parse_error` | Whole-case load failed. |
| `report_silent` | `field_missing` AND gold null. (Both annotators left it blank.) |
| `model_error` | `wrong` AND κ_humans ≥ 0.8 (humans agree, so the model is the limit). Plus all `field_missing` cases where gold is non-null. |
| `report_ambiguity` | `wrong` AND κ_humans < 0.5 (humans disagreed too). |
| `borderline` | `wrong` AND 0.5 ≤ κ_humans < 0.8. |

Where `κ_humans` is sourced from `iaa/pair_*.csv` — preferring the cross-human pair (`nhc_with_preann_vs_kpc_with_preann`) and falling back to gold-vs-human if absent.

The headline interpretation:

```
model-bound accuracy ceiling = 1 − model_error_rate
```

This is the accuracy you could plausibly reach with a perfect model — anything above is fundamentally limited by the source reports.

**Implementation:** `scripts/eval/diagnostics/run_diagnostics.py:_error_source_decomposition`. Reads `non_nested/correctness_table.parquet` and `iaa/pair_*.csv`.
**Output:** `diagnostics/error_source_decomposition.csv`.

**How to lift this into the writeup:**

> "Of the model's errors on `tumor_size` in lung cases, 60% occurred on cases where the human annotators also disagreed (κ_humans < 0.5), suggesting that the field's lower accuracy primarily reflects source-report ambiguity rather than model deficiency. The model-bound accuracy ceiling for this field is 0.91, vs. an observed accuracy of 0.83 — a 8 pp gap attributable to inherent ambiguity rather than further model improvement."

## Difficulty-tier stratification

Stratify model accuracy by human-IAA tier:

| Tier | κ_humans threshold |
|---|---|
| `easy` | ≥ 0.8 |
| `medium` | 0.5 – 0.8 |
| `hard` | < 0.5 |
| `unknown` | no IAA data |

If the model's accuracy degrades across tiers (high on `easy`, low on `hard`), performance is **source-bound** — even humans struggle on the hard cases. If accuracy is uniform across tiers, the model is the limit.

Thresholds are configurable via `--difficulty-thresholds hard,easy` (default `0.5,0.8`).

**Implementation:** `_difficulty_tier`. **Output:** `diagnostics/accuracy_by_difficulty_tier.csv`.

## Worst-cases catalog

Per field, the top-N (default 20) cases ranked by a "badness" score:

```
score = (1 − mean_correct_across_runs) + std_correct_across_runs
```

The first term flags consistently-wrong cases; the second flags **brittle** cases (some runs right, others wrong). High score = either consistently failed OR run-to-run unstable — both are interesting for manual review.

**Output:** `diagnostics/worst_cases.csv`. One row per (case, field) with: `organ`, `mean_correct`, `std_correct`, `attempted_rate`. Manual-review aid for the Discussion section.

## Running the subcommand

```
python -m scripts.eval.cli diagnostics \
    --non-nested-out /path/to/non_nested_output \
    --iaa-out /path/to/iaa_output \
    --out /path/to/diagnostics_output \
    --top-n-worst 20 \
    --difficulty-thresholds 0.5,0.8
```

Inputs are pre-computed outputs from the `non_nested` and `iaa` subcommands — diagnostics doesn't re-score; it joins.

## What to do if your numbers are bad

- **High `report_ambiguity` share for a specific field:** the field's annotation guideline is fuzzy. Time to write better guidelines or accept the inherent uncertainty.
- **High `report_silent` share:** the field isn't actually documented in many reports. Don't penalise the model for refusing; consider whether the field belongs in the schema at all.
- **Uniform low accuracy across difficulty tiers:** the model is the bottleneck, not the data. Prompt / training is the lever.
- **High accuracy on `easy` tier but collapse on `hard`:** the model is "easy enough" — invest in pre-annotation rather than retraining, since the hard cases are inherently disagreeable anyway.
- **Many cases in `worst_cases.csv` from a single organ:** organ-specific issue. Check that organ's prompt / schema.

## Implementation references

Foundational citations are in [methods_citations.md](methods_citations.md). The diagnostics-specific concepts are not novel: the source-of-error decomposition philosophy follows standard IE evaluation practice (e.g. Sang & De Meulder 2003 on CoNLL annotation difficulty stratification), and difficulty-tier stratification by IAA is the standard way to characterise gold-standard quality so that model accuracy can be read against the human ceiling.

- Sang, E. F. T. K., & De Meulder, F. (2003). "Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition." *CoNLL 2003*.
