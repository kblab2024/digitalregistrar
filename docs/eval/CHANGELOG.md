# `docs/eval/` changelog

## `iaa_pair` subcommand (2026-05)

Added a new `iaa_pair` subcommand for pair-focused IAA reports. The
existing `iaa` subcommand emits per-field κ + Krippendorff α per type
bucket but no single overall Cohen's κ for an arbitrary pair;
`iaa_pair` fills that gap.

- **CLI**: `python -m scripts.eval.cli iaa_pair --pair <a>:<b> ...`
  (`--pair` is repeatable; one output subdirectory per pair).
- **Headline**: 4 κ flavours side-by-side — `mean_per_field_kappa`,
  `n_weighted_mean_per_field_kappa`, `pooled_categorical_kappa`,
  `agree_disagree_pabak` — plus Krippendorff α (nominal/ordinal/interval)
  and `case_exact_match_rate` forwarded from
  `iaa.whole_report_stats`.
- **Outputs per pair**: `headline.csv`, `per_field_kappa.csv`,
  `per_section.csv`, `per_organ.csv`, `confusion/<field>.csv` (top-N
  categorical fields by disagreement), `summary.md`.
- **Scope**: pair-agnostic. For within-annotator preann pairs the
  summary banner points readers to the existing `iaa/preann/`
  outputs for the causal Δκ-vs-gold and anchoring analyses.
- **Library**:
  `digital_registrar_research.benchmarks.eval.iaa_headline`.
  Discovery helper (`_discover_cases_dir_layout`) lifted from
  `run_iaa.py` to `scripts/eval/iaa/_discovery.py` and shared.

See [iaa_basics.md §Pair-focused headline](iaa_basics.md#pair-focused-headline-iaa_pair)
for definitions and [recipes.md](recipes.md#pair-focused-iaa-overall-κ--roll-ups-for-one-pair)
for canonical invocations.

## Cascade redesign (2026-05)

The `non_nested` and `nested` subcommands of `scripts.eval.cli` were
removed. All scoring is now done by the single `cascade` subcommand,
which gates evaluation as three sequential stages (Stage A:
`cancer_excision_report`; Stage B: `cancer_category`; Stage C: field
extraction). Cascade attrition is recorded explicitly in
`chapter3_field_extraction/cascade_funnel.csv`.

### What changed

- **CLI**: `python -m scripts.eval.cli non_nested ...` and
  `python -m scripts.eval.cli nested ...` no longer exist. Use
  `python -m scripts.eval.cli cascade ...` instead.
- **Output layout**: `correctness_table.parquet` →
  `cascade_atomic.parquet` (strict superset of columns; adds
  `cascade_stage`, `gate_pass`, `others_disposition`).
  `per_field_overall.csv` and the per-organ tables now live under
  `chapter3_field_extraction/`. Eligibility metrics are in
  `chapter1_eligibility/`. Organ classification is in
  `chapter2_organ_classification/`.
- **Lymph-node scoring**: bipartite-on-`station_name` was replaced by
  category-aggregation. `examined` and `involved` are summed per
  `(lymph_node_side, lymph_node_category)` group, then matched on the
  group key. See [nested_metrics.md](nested_metrics.md) and
  [../stat_methods.md](../stat_methods.md) §1.
- **Field exclusions**: `ajcc_version`, `treatment_effect`,
  `margins[*].description`, `regional_lymph_node[*].station_name` are
  excluded from every metric (not just stat tests). The deprecated
  alias `STATS_EXCLUDED_FIELDS` is the union of these for backwards
  compat; new code should use `EVAL_EXCLUDED_FIELDS` and
  `EVAL_EXCLUDED_NESTED_INNER_KEYS`.
- **Biomarker whitelist**: only `{er, pr, her2, ki67}` (breast) and
  `{msh2, msh6, pms2, mlh1}` (colorectal) are scored. All other
  biomarker entries are filtered from gold and prediction.
- **Stat package**: a new module surface lives at
  `digital_registrar_research.benchmarks.eval.stats` with cohens
  kappa, weighted kappa, Krippendorff alpha, ICC(2,1), ICC(3,k),
  Cronbach alpha, Stuart-Maxwell, Cochran's Q, Lin's CCC, Bland-Altman,
  cascade-funnel and conditional-accuracy diagnostics. See
  [../stat_methods.md](../stat_methods.md).
- **Outputs gained**: each chapter has Cohen's-kappa CSVs, paired-test
  CSVs (when multiple models scored), `multirun_consistency.csv` with
  ICC and flip-rate columns, and the cascade funnel.
- **Outputs lost**: nothing — every column the legacy CSVs produced
  has a home in the new layout, modulo renamed paths.

### Migration cheat sheet

| Legacy invocation | Cascade equivalent |
|---|---|
| `cli non_nested --root X --dataset cmuh --model gpt_oss_20b --out Y` | `cli cascade --root X --dataset cmuh --model gpt_oss_20b --out Y` |
| `Y/per_field_overall.csv` | `Y/chapter3_field_extraction/per_field_overall.csv` |
| `Y/per_field_by_organ.csv` | `Y/chapter3_field_extraction/per_field_by_organ.csv` |
| `Y/correctness_table.parquet` | `Y/cascade_atomic.parquet` |
| `cli nested --field margins ...` | (folded into cascade; see chapter3 outputs) |

### Doc status

Each existing doc was tagged with one of: `needs-rewrite`, `needs-edit`,
`obsolete`, or `unchanged` during the redesign. Pages that retain
historical command names carry a "Cascade-redesign note" banner at the
top pointing readers here.
