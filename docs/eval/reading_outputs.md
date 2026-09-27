# Reading the eval outputs

Column dictionary for every CSV `registrar-eval` writes, plus the conventions they share. For how the scores are computed, see [index.md](index.md#how-scoring-works---scope-cascade-the-default).

All files are plain CSV (no parquet, no pyarrow needed), are written to `--out` (default `eval_out/`), and are overwritten on re-run.

## `metrics` / `compare`

### `atomic.csv`

Long form: one row per (run, case, scored field). Every other table is a reduction of this one, so any custom breakdown starts here.

| Column | Meaning |
|---|---|
| `method` | run label (`--label`, or the prediction folder name) |
| `dataset` | gold file's subfolder relative to `--gold` (empty for a flat folder) |
| `organ` | gold `cancer_category` |
| `case_id` | file stem without `_output` / `_annotation` |
| `stage` | `A` = `cancer_excision_report`, `B` = `cancer_category`, `C` = everything in `cancer_data` |
| `field` | field name; see the field-name table below |
| `metric` | `accuracy` → `correct` is True / False; `f1` → `correct` is a per-case F1 in [0, 1] |
| `correct` | True / False / F1 value; empty when not attempted |
| `attempted` | the prediction has the key (a `null` value counts as attempted) |

Field names beyond the schema's own fields:

| `field` | `metric` | Scored as |
|---|---|---|
| `biomarker_<name>` (e.g. `biomarker_er`) | accuracy | expression value of a whitelisted biomarker. Breast: ER / PR / HER2 / Ki-67. Colorectal: MLH1 / MSH2 / MSH6 / PMS2. Skipped when neither side lists it |
| `margins`, `biomarkers` | f1 | per-case F1 over items matched on category. Skipped when both lists are empty |
| `regional_lymph_node.examined_total` | accuracy | total nodes examined within ±1 |
| `regional_lymph_node.involved_total` | accuracy | total nodes involved within ±1 |
| `regional_lymph_node.any_positive` | accuracy | both sides agree on node-positive vs node-negative |
| `regional_lymph_node.group_f1` | f1 | F1 over (side, category) node groups. Skipped when both lists are empty |

### `summary.csv`

One row per (method, stage, field, metric):

| Column | Meaning |
|---|---|
| `method`, `stage`, `field`, `metric` | as in `atomic.csv` |
| `total` | cases where the field was in scope. For stage C, that means cases that passed both gates |
| `attempted` | of those, how many the prediction attempted |
| `coverage` | `attempted / total` |
| `accuracy_attempted` | mean of `correct` over attempted rows: accuracy for `accuracy` rows, mean F1 for `f1` rows |
| `ci_lo`, `ci_hi` | 95% CI on `accuracy_attempted`. Wilson for accuracy rows, percentile bootstrap (2000 resamples) for F1 rows. See [ci_methods.md](ci_methods.md) |

To penalise non-attempts as errors, use effective accuracy = `accuracy_attempted × coverage`.

### `compare.csv`

Described column by column in [comparing_runs.md](comparing_runs.md#reading-comparecsv).

## `completeness`

These tables are **not** gated by the cascade. Fields come from the gold record: the two gate fields for every case, plus every scalar / list-of-literals field of the gold organ's schema when the gold case is an eligible cancer report. A value counts as present unless it is `null` or `[]`, which is stricter than "attempted" in `metrics`, where a `null` value counts.

### `completeness.csv`

One row per (method, field, organ), where organ is the gold organ.

| Column | Meaning |
|---|---|
| `n_total` | case × field rows in the group |
| `n_eligible` | rows where gold has a value |
| `n_parse_error` | prediction file missing or not a JSON object |
| `n_field_missing` | prediction loaded, but the field is absent / null / `[]` |
| `n_attempted` | prediction has a value |
| `n_correct`, `n_wrong` | attempted and right / wrong (ungated `field_correct`) |
| `attempted_rate` (+ `_ci_lo` / `_ci_hi`) | `n_attempted / n_total`, Wilson CI |
| `parse_error_rate` (+ CI) | `n_parse_error / n_total` |
| `field_missing_rate` (+ CI) | `n_field_missing / n_total` |
| `total_missing_rate` | parse-error rate + field-missing rate |
| `attempted_accuracy` | `n_correct / n_attempted` |
| `effective_accuracy` | `n_correct / n_total`. The gap from `attempted_accuracy` is the completeness penalty |

### `refusal_calibration.csv`

Every time the prediction had no value, was that justified?

| Column | Meaning |
|---|---|
| `n_pred_null` | rows with no predicted value |
| `n_correct_refusal` | … and gold is also empty (justified) |
| `n_lazy_missing` | … but gold has a value (the model gave up) |
| `correct_refusal_rate`, `lazy_missing_rate` (+ CIs) | shares of `n_pred_null` |
| `justified_missingness_share` | same as `correct_refusal_rate` |

### `out_of_vocab.csv`

For each categorical field, this checks whether attempted predictions stayed inside the schema's allowed values. Predictions are grouped by **their own** `cancer_category`, which is the schema the model filled in.

| Column | Meaning |
|---|---|
| `organ`, `field` | predicted organ and categorical field |
| `n_attempted` | non-null predictions of the field |
| `n_oov` | of those, values outside the allowed enum |
| `oov_rate` (+ `ci_lo` / `ci_hi`) | `n_oov / n_attempted`, Wilson CI |

## Conventions

- **CIs are 95%.** Wilson for proportions, bootstrap for everything else, fixed seed (`random_state=0`), so re-runs are identical.
- **NaN is not zero.**
  - `accuracy_attempted` / `attempted_accuracy` is NaN when nothing was attempted.
  - A CI is NaN when there are no observations.
  - `mcnemar_p_value` is NaN for F1 rows.
  - Don't impute NaNs when aggregating.
- **The four views answer different questions.** Accuracy measures quality on attempted answers, missingness measures what was skipped, out-of-vocab measures schema conformance, and refusal calibration asks whether nulls were justified. A field can score high on accuracy and high on lazy missingness at the same time.
- **Record provenance yourself.** `registrar-eval` does not write a manifest. When quoting a number, note the git SHA of the `digital-registrar` checkout, or the package version, next to the command you ran.
