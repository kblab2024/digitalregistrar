# Evaluation documentation

How to score `registrar-pipeline` output against gold annotations with the `registrar-eval` CLI (installed with the core `digital-registrar` package), and what each metric means.

## Quickstart

```bash
# 1. Run the pipeline over a folder of reports → runs/model_x/<stem>_output.json
registrar-pipeline --input reports/ --output runs/model_x --model gpt

# 2. Score it against the gold folder → eval_out/model_x/{atomic,summary}.csv
registrar-eval metrics --pred runs/model_x --gold gold/ --out eval_out/model_x

# 3. Compare two runs on the same gold
registrar-eval compare --pred-a runs/model_x --pred-b runs/model_y \
    --gold gold/ --out eval_out/x_vs_y

# 4. Missingness / refusal / out-of-vocab report
registrar-eval completeness --pred runs/model_x --gold gold/ --out eval_out/model_x_completeness
```

Run `registrar-eval --help` or `registrar-eval <subcommand> --help` for every flag.

## How files are paired

Predictions and gold are paired by **case id**. The case id is the file stem with one trailing `_output` or `_annotation` removed, plus a leftover `.txt` if there is one:

| File | Case id |
|---|---|
| `runs/model_x/tcga4_1_output.json` (registrar-pipeline) | `tcga4_1` |
| `runs/model_x/tcga4_1.txt_output.json` (pipeline random-report mode) | `tcga4_1` |
| `gold/tcga4_1_annotation.json` | `tcga4_1` |
| `gold/tcga4_1.json` | `tcga4_1` |

- Both folders are searched **recursively**. The per-dataset subfolders the pipeline writes when run without `--input` (`<run>/<dataset>/<stem>_output.json`) work as-is. The gold file's subfolder is reported in the `dataset` column.
- Case ids must be unique within each folder. A duplicate is an error.
- Every gold file is a case:
  - A gold case with no prediction file, or with an unreadable one, counts as *not attempted*.
  - Predictions with no gold file are ignored. The CLI prints how many.
- Each prediction and gold file is one case record, shaped like `registrar-pipeline` output: top-level `cancer_excision_report`, `cancer_category`, `cancer_data`.

## Outputs

| Subcommand | File | What it is | Doc |
|---|---|---|---|
| `metrics` | `atomic.csv` | one row per (case, field): `correct`, `attempted`, `stage`, `metric` | [reading_outputs.md](reading_outputs.md#atomiccsv) |
| `metrics` | `summary.csv` | per-field accuracy (or mean F1), coverage, 95% CI | [reading_outputs.md](reading_outputs.md#summarycsv) |
| `compare` | `atomic.csv`, `summary.csv` | as above, for both runs (`method` column) | [reading_outputs.md](reading_outputs.md#summarycsv) |
| `compare` | `compare.csv` | per-field paired Δ (A − B), bootstrap CI, McNemar | [comparing_runs.md](comparing_runs.md) |
| `completeness` | `completeness.csv` | parse-error / field-missing / attempted rates with Wilson CIs | [reading_outputs.md](reading_outputs.md#completenesscsv) |
| `completeness` | `refusal_calibration.csv` | justified vs lazy nulls | [reading_outputs.md](reading_outputs.md#refusal_calibrationcsv) |
| `completeness` | `out_of_vocab.csv` | predicted values outside the schema enum | [reading_outputs.md](reading_outputs.md#out_of_vocabcsv) |

## How scoring works (`--scope cascade`, the default)

Scoring is a three-stage cascade, so a mistake early on doesn't produce a flood of meaningless field errors later:

1. **Stage A: eligibility.** Checks `cancer_excision_report` against gold. If it's wrong or missing, the case stops here.
2. **Stage B: organ.** For gold-eligible cases, checks `cancer_category`. If it's wrong, the case stops here. It also stops when both sides say `others`, since there's no schema to score.
3. **Stage C: fields.** Scores every field of the gold organ's schema:
   - **Scalar and categorical fields** use exact match after normalisation. Size fields allow ±2 mm; list-of-literals fields use set equality. See [non_nested_metrics.md](non_nested_metrics.md).
   - **Whitelisted biomarkers** (`biomarker_er`, …) compare the expression value.
   - **`margins` and `biomarkers`** are scored as per-case F1 over matched items. See [nested_metrics.md](nested_metrics.md).
   - **Lymph nodes** are scored on examined and involved totals (±1 node), any-positive agreement, and F1 over (side, category) groups. See [nested_metrics.md](nested_metrics.md).
   - `ajcc_version` and `treatment_effect` are never scored.

Stage-C accuracy is therefore *conditional* on passing Stages A and B. Read `summary.csv`'s `total` column to see how many cases reached each field.

A value counts as **attempted** when the prediction has the key, even if the value is `null`. A missing key counts as not attempted, and so does a missing prediction file. `accuracy_attempted` is computed over attempted rows only; `coverage` is attempted / total.

`--scope fair` skips the cascade. It scores the fixed head-to-head field list `FAIR_SCOPE` (the gate fields, pT/pN/pM, grade, LVI, PNI and tumor size) on every case. It has no nested-list rows.

## Python API

The same functions the CLI uses:

```python
from digital_registrar.eval import load_pairs, score_pairs, summarize_scores, compare_runs

pairs, stats = load_pairs("runs/model_x", "gold/")
atomic = score_pairs(pairs, method="model_x")   # long-form DataFrame
summary = summarize_scores(atomic)              # per-field accuracy + CI
```

Per case: `digital_registrar.eval.score_case(gold, pred)` and `digital_registrar.eval.score_lymph_nodes(gold, pred)`. More in [recipes.md](recipes.md).

## Which doc do I read?

| Question | File |
|---|---|
| "Show me the commands." | [recipes.md](recipes.md) |
| "What does this CSV column mean?" | [reading_outputs.md](reading_outputs.md) |
| "Which of two runs is better?" | [comparing_runs.md](comparing_runs.md) |
| "How are scalar / categorical fields scored? What's κ / MCC?" | [non_nested_metrics.md](non_nested_metrics.md) |
| "How are lymph nodes / margins / biomarkers scored?" | [nested_metrics.md](nested_metrics.md) |
| "Wilson vs bootstrap CI?" | [ci_methods.md](ci_methods.md) |
| "What does field type / scope / attempted mean?" | [glossary.md](glossary.md) |

## Paper-only analyses

The paper also reported inter-annotator agreement, the pre-annotation effect, multirun statistics, cross-dataset shift, and error-source diagnostics. Those came from research scripts that are **not** part of `registrar-eval`. Their documentation is kept for reference in [`attic/docs/eval/`](../../attic/docs/eval/), and the scripts are in [`attic/eval_scripts/`](../../attic/eval_scripts/); see [attic/README.md](../../attic/README.md). The old `python -m scripts.eval.cli …` commands no longer exist.
