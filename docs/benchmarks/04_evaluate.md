# Per-method evaluation

The eval pipeline lives at `scripts/eval/cli.py` with subcommand dispatch. For benchmark-style accuracy / coverage on scalar **and** nested-list fields, use the `cascade` subcommand. Other subcommands (`iaa`, `completeness`, `diagnostics`, `cross_dataset`, `headline`) are documented separately under [`docs/eval/`](../eval/).

The whole eval surface is **method-agnostic** — pass `--method {rule_based|clinicalbert|llm}` and it consumes the canonical predictions tree. So the rule, BERT, and LLM baselines all produce comparable metrics CSVs out of the box.

> **Cascade-redesign note (2026-05).** This page is being rewritten incrementally. The legacy `non_nested` and `nested` subcommands are gone — `cascade` produces strictly more output (three chapter folders, an others ledger, multirun reliability, paired-test CSVs) and applies cascade gating so that field-level accuracy is reported on the cohort that legitimately reached Stage C. See [eval/CHANGELOG.md](../eval/CHANGELOG.md) for the migration map and [stat_methods.md](../stat_methods.md) for the new statistical-methods inventory.

## Per-method `cascade`

```bash
python -m scripts.eval.cli cascade \
    --root workspace --dataset tcga \
    --method clinicalbert --model merged \
    --annotator gold \
    --out workspace/results/eval/cascade_bert_merged_tcga
```

| Flag | Required | Notes |
|---|---|---|
| `--root` | yes | Experiment root — same `dummy` / `workspace` / abs-path shorthand as the runners. |
| `--dataset` | yes | `cmuh` or `tcga`. |
| `--method` | yes | `rule_based`, `clinicalbert`, or `llm`. |
| `--model` | required for `clinicalbert` and `llm` | For `clinicalbert`: `cls`, `qa`, or `merged`. For `llm`: the model slug, e.g. `gpt_oss_20b`. Not used for `rule_based`. |
| `--run-ids` | optional (LLM and ClinicalBERT) | Specific run IDs to score. Default: auto-discover every `run*` subdir under the model. Applies to LLM (Ollama and OpenAI alike — same namespace) and to ClinicalBERT when the K-seed multirun trainer was used. Ignored for `rule_based` (deterministic, no run slots). |
| `--annotator` | default `gold` | Annotator subdir to score against. |
| `--organs` | default all | Restrict to organ indices (1..10) or names. |
| `--cases` | optional | Allowlist of case IDs (inline or `@path/to/list.txt`). |
| `--out` | default `workspace/results/eval/cascade` | Output directory. |
| `--n-boot` | 2000 | Bootstrap replicates for CIs. |
| `--alpha` | 0.05 | CI coverage. |

## Outputs

```
{--out}/
├── manifest.json                     CLI args + git sha + UTC timestamp
├── correctness_table.parquet         atomic per-(case, organ, field, run) outcome
├── per_field_overall.csv             accuracy + Wilson CI per field, across all organs / runs
├── per_field_by_organ.csv            per-(field, organ) cell
├── per_field_by_subgroup.csv         stratified by multi-primary subgroup label
├── per_organ_overall.csv             across-field aggregate per organ
├── headline_classification.csv       precision / recall / F1 per (field, organ)
├── per_class_prf1.csv                per-class P/R/F1
├── confusion/<field>__<organ>.csv    confusion matrices
├── confusion_pairs.csv               most-confused class pairs
├── accuracy_collapsing_neighbors.csv accuracy after collapsing semantic-neighbor classes
├── rank_distance.csv                 ordinal rank distance
├── top_k_ordinal.csv                 top-k accuracy for ordinal fields
├── schema_conformance.csv            does the prediction respect the field's enum?
├── refusal_calibration.csv           is the method's "I don't know" rate calibrated?
├── chapter*/multirun_consistency.csv per-chapter cross-run variance — Fleiss κ across runs, flip rate, stability accuracy, brittle-case rate, per-run accuracy CI (Student-t over the K per-run accuracy vector). Emitted only when `--run-ids` resolved >1 run, i.e. for LLM and for ClinicalBERT-multirun. Not produced for `rule_based`.
├── section_rollup.csv                accuracy rolled up by report section
└── missingness_summary.csv           where each method drops out
```

The atomic `correctness_table.parquet` is what the side-by-side comparison consumes — see [`05_compare.md`](05_compare.md).

## Per-field schema

`correctness_table.parquet` columns:

| Column | Type | Meaning |
|---|---|---|
| `run_id` | str | Empty `""` for `rule_based`. Populated as `run01`..`runK` for `llm` and for `clinicalbert` when the K-seed multirun trainer was used. Empty for single-seed `clinicalbert` (legacy flat layout). |
| `method` | str | `rule_based` / `clinicalbert` / `llm`. |
| `model` | str | Model name (head for BERT, slug for LLM, empty for rule_based). |
| `annotator` | str | `gold` etc. |
| `case_id` | str | e.g. `tcga1_17`. |
| `organ_idx` | int | 1..10. |
| `organ` | str | `breast`, `lung`, ... — derived from `organ_idx` and the gold's `cancer_category` (gold takes precedence when not `others`). |
| `subgroup` | str | Multi-primary subgroup label. |
| `field` | str | Schema field, e.g. `pt_category`, `tumor_size`, `lymphovascular_invasion`. |
| `field_kind` | str | `binary` / `nominal` / `ordinal` / `continuous` / `list_of_literals`. |
| `gold_present` | bool | Did the gold record have a non-null value? |
| `attempted` | bool | Did the method emit a non-null value? |
| `correct` | bool | Was the attempted value correct? (None when `attempted=False`.) |
| `wrong` | bool | Inverse of `correct` for attempted cells. |
| `field_missing` | bool | Method's prediction had no key for this field (vs `attempted` with explicit `null`). |
| `parse_error` | bool | Prediction file was malformed / sentinel. |
| `error_mode` | str | One of `json_parse`, `schema_invalid`, `timeout`, `refusal`, `file_missing`, `other`, or null. |
| `gold_value`, `pred_value` | Any | Raw values for human inspection. |

## Per-organ scope

`non_nested` pulls the field list **per organ** from `digital_registrar_research.benchmarks.eval.scope.get_organ_scoreable_fields(organ)` plus `cancer_category` and `cancer_excision_report`. That gives ~25–30 fields per organ rather than the 12 in the legacy `FAIR_SCOPE`. Methods are scored against the right scope automatically — there's no need to pass an explicit `--scope`.

## Filtering tricks

The cross-corpus baseline scores **every gold case** under the configured datasets — there's no train/test split inside a corpus. To restrict scope, pass `--organs` or an explicit case-id allowlist:

- **One organ**: `--organs breast` or `--organs 1`.
- **A specific case set**: `--cases tcga1_17 tcga1_22 cmuh3_5`.
- **From a file**: `--cases @/tmp/case_ids.txt` (one id per line).

## What to read first

Headlines for human eyes: `per_field_overall.csv` and `per_field_by_organ.csv`. Everything else is downstream of the atomic table — useful for deeper questions like "where does the model refuse?", "what's the top confusion pair?", "is the schema conformance high?".
