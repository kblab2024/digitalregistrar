# Glossary

## Field types (`classify_field`)

| Type | Examples | Scoring approach |
|---|---|---|
| `binary` | `lymphovascular_invasion`, `cancer_excision_report` | Cohen's κ, MCC, McNemar |
| `nominal` | `cancer_category`, `procedure`, `histology` | Unweighted κ, balanced accuracy |
| `ordinal` | `grade`, `pt_category`, `nuclear_grade` | Quadratic-weighted κ, Kendall's τ-b, top-k, rank distance |
| `continuous` | `tumor_size`, `dcis_size`, `maximal_ln_size` | MAE, RMSE, CCC, ICC, Bland-Altman |
| `list_of_literals` | `tumor_extent` (liver), `vascular_invasion` (liver), `involved_margin_list` (prostate) | Unordered set equality + set-F1 (TP/FP/FN on items) |
| `nested_list` | `regional_lymph_node`, `margins`, `biomarkers` | Bipartite F1, hallucination/miss rate, count MAE |

Single source of truth: `drr_attic.benchmarks.eval.iaa.classify_field(field, organ)`. Per-organ taxonomy in `scope_organs.py`. The `list_of_literals` registry is in `scope_organs.ORGAN_LIST_OF_LITERALS` — distinct from `ORGAN_NESTED_LIST` (which is list-of-dicts).

**Note on organ-aware classification.** The same field name can be a different type in different organs. `tumor_extent` is `list_of_literals` for liver but `nominal` for esophagus and stomach (it's a single-string enum there). The classifier dispatches by `(field, organ)`, not field alone. Pass `organ=` to `classify_outcome` to get the right scoring path.

## Per-organ scoreable fields

Under cascade scoring (`registrar-eval --scope cascade`, the default), Stage C iterates every per-organ scoreable field — not the (much smaller) `FAIR_SCOPE`. Field counts per organ in the canonical schemas (as of 2026-04):

| Organ | Categorical | Boolean | Continuous | List-of-literals | Total |
|---|---:|---:|---:|---:|---:|
| breast | 17 | 6 | 4 | 0 | 27 (+ 3 biomarker_* synthetic) |
| colorectal | 18 | 4 | 1 | 0 | 23 |
| lung | 17 | 5 | 1 | 0 | 23 |
| prostate | 13 | 5 | 4 | 1 | 23 |
| liver | 9 | 6 | 3 | 2 | 20 |
| stomach | 16 | 4 | 1 | 0 | 21 |
| esophagus | 14 | 4 | 1 | 0 | 19 |
| cervix | 13 | 5 | 1 | 0 | 19 |
| pancreas | 13 | 4 | 1 | 0 | 18 |
| thyroid | 17 | 4 | 1 | 0 | 22 |

The two top-level gate fields (`cancer_excision_report`, `cancer_category`) are scored as Stages A and B. `biomarker_<name>` rows are added for each whitelisted biomarker (`BIOMARKER_WHITELIST`: breast `er`/`pr`/`her2`/`ki67`, colorectal `mlh1`/`msh2`/`msh6`/`pms2`). `ajcc_version` and `treatment_effect` are never scored (`EVAL_EXCLUDED_FIELDS`). Cross-organ union: ~69 distinct fields.

## Sections (`classify_section`)

| Section | Examples |
|---|---|
| `top_level` | `cancer_category`, `cancer_excision_report` |
| `staging` | `pt_category`, `pn_category`, `pm_category`, `tnm_descriptor`, stage groups |
| `grading` | `grade`, `nuclear_grade`, `tubule_formation`, `mitotic_rate`, `total_score`, `dcis_grade` |
| `invasion` | `lymphovascular_invasion`, `perineural_invasion` |
| `size` | `tumor_size`, `dcis_size`, `maximal_ln_size` |
| `biomarker` | `biomarker_er`, `biomarker_pr`, `biomarker_her2` |
| `other` | everything else |

Used in the paper pipeline's `non_nested/section_rollup.csv` to aggregate field-level metrics into clinically-meaningful sections.

## Outcome flags (three-way model)

These are the columns `registrar-eval completeness` builds (`completeness.completeness_atomic`). Background: [completeness.md](../../attic/docs/eval/completeness.md).

| Flag | Definition |
|---|---|
| `gold_present` | Gold has the field non-null (and not `[]`). |
| `attempted` | Model produced a non-null value. |
| `correct` | `attempted AND value matches gold`. |
| `wrong` | `attempted AND not correct`. |
| `field_missing` | Case loaded but this field absent / null. |
| `parse_error` | Prediction file missing or not a JSON object. |

`registrar-eval metrics` uses a looser **attempted**: the prediction has the key, even with a `null` value. So "said null when gold is null" scores as correct there.

## Scope terms

- **`FAIR_SCOPE`** — fields all four method families (DSPy, GPT-4, ClinicalBERT, rules) can produce. The head-to-head comparison set.
- **`NESTED_LIST_FIELDS`** — list-of-dict fields (margins, biomarkers, regional_lymph_node, plus organ-specific extras).
- **`BIOMARKER_WHITELIST`** — per-organ biomarkers scored under the cascade: breast `er`/`pr`/`her2`/`ki67`, colorectal `mlh1`/`msh2`/`msh6`/`pms2`. Other entries are dropped from both sides.
- **`ORDINAL_FIELDS`** — fields with natural ranking (grade, T/N/M categories, stage groups).
- **`SPAN_FIELDS`** — integer span fields (the ClinicalBERT-QA head's domain).

Defined in `src/digital_registrar/eval/scope.py` and `scope_organs.py`.

## Paper-pipeline terms

The sections below (annotators, run IDs, multi-primary, endpoint tiers) describe the paper's research pipeline in [`attic/eval_scripts/`](../../attic/eval_scripts/). They are not used by `registrar-eval`.

## Annotators

Disk subdirectories under `<root>/data/<dataset>/annotations/`:

- `gold` — adjudicated reference. No `_meta` block.
- `nhc_with_preann`, `nhc_without_preann` — annotator NHC, with or without LLM pre-annotation visible.
- `kpc_with_preann`, `kpc_without_preann` — annotator KPC, same modes.

Human annotations carry a `_meta` block: `{annotator, mode, annotated_at}`.

## Run IDs

String format: `runNN[-slug]` where `NN` is zero-padded slot number and the optional slug supports multi-machine parallelism. Examples:

- `run01` — single-machine canonical.
- `run02-alpha` — slot 2 from machine "alpha".
- `run01-beta` — slot 1 from machine "beta".

Regex: `^run(\d+)(?:-([a-z0-9][a-z0-9-]*))?$`. See `scripts/_run_id.py` for parsing.

## Multi-primary

A case is `multi_primary` if it describes more than one distinct tumor (bilateral breast, double primary lung, multifocal disease). Detection heuristics in `drr_attic.benchmarks.eval.multi_primary`:

- `cancer_laterality == "bilateral"` → `multi_primary`.
- `tumor_focality` ∈ {multifocal, multicentric, ...} → `multi_primary`.
- Multi-clock / multi-quadrant strings (e.g. `"12 and 3"`) → `multi_primary`.

Subgroup column: `single_primary` / `multi_primary` / `unknown`. Used to stratify all metrics so the writeup can say "in the multi-primary subgroup, accuracy was X" — multi-primary cases are clinically distinct and a single aggregate accuracy hides large per-subgroup differences.

## Endpoint tiers

Defined in `configs/eval_endpoints.yaml`:

- **Primary** (12 fields) — Holm-Bonferroni p-value adjustment. Headline claims.
- **Secondary** (the rest) — Benjamini-Hochberg FDR adjustment. Exploratory.

See [multiple_comparisons.md](../../attic/docs/eval/multiple_comparisons.md).

## Time conventions

All timestamps in manifests are UTC ISO 8601 (e.g. `2026-04-26T16:24:49.010290+00:00`). `_meta.annotated_at` from human annotations is also UTC.

Heuristic per-case duration estimation (in `iaa/preann/edit_distance__*.csv`): order an annotator's saves chronologically, take diffs, drop diffs > 30 min as session breaks. Approximate — documented as such in the CSV.

## File extensions

`registrar-eval` writes only `.csv`. The long-form `atomic.csv` is the source table, and `summary.csv` / `compare.csv` / `completeness.csv` are reductions of it. To re-aggregate differently (e.g. by `dataset` or `organ`), load `atomic.csv` and call `digital_registrar.eval.summary_table(df, by=[...])`. There's no need to re-score.

The retired paper pipeline also wrote `.parquet` atomic tables, `.json` manifests and a `.yaml` endpoint pre-registration.
