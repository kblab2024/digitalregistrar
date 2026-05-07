# Evaluation documentation

ELI5-style explanations of every metric the `scripts/eval/` pipeline produces, plus paper-ready citations.

> **Cascade redesign (2026-05).** The `non_nested` and `nested` subcommands have been replaced by a single `cascade` subcommand that gates evaluation as three sequential stages. Output paths have moved into `chapter1_eligibility/`, `chapter2_organ_classification/`, `chapter3_field_extraction/`. See [CHANGELOG.md](CHANGELOG.md) for the migration map and [../stat_methods.md](../stat_methods.md) for the new statistical-methods inventory.

## Decision tree — which doc do I read?

| Question | File |
|---|---|
| "What does this CSV column mean?" | [reading_outputs.md](reading_outputs.md) |
| "What's accuracy / κ / MCC / balanced accuracy?" | [non_nested_metrics.md](non_nested_metrics.md) |
| "What does MAE / RMSE / Bland-Altman LoA / CCC mean?" | [continuous_metrics.md](continuous_metrics.md) |
| "How are lymph nodes / margins / biomarkers scored?" | [nested_metrics.md](nested_metrics.md) |
| "Why is missing field different from wrong field?" | [completeness.md](completeness.md) |
| "What's Fleiss κ / vote calibration / ensemble Δ?" | [multirun.md](multirun.md) |
| "Cohen κ vs Krippendorff α — when do I use which?" | [iaa_basics.md](iaa_basics.md) |
| "I just want one overall κ for a specific annotator pair." | [iaa_basics.md](iaa_basics.md#pair-focused-headline-iaa_pair) |
| "What's the anchoring index? Δκ? Convergence to preann?" | [preann_effect.md](preann_effect.md) |
| "Wilson vs BCa vs two-source bootstrap CI?" | [ci_methods.md](ci_methods.md) |
| "Source-of-error decomposition / difficulty tiers / worst cases?" | [diagnostics.md](diagnostics.md) |
| "Cross-dataset Δ / KL / JS / Wasserstein?" | [cross_dataset.md](cross_dataset.md) |
| "Holm-Bonferroni vs BH? Effect sizes?" | [multiple_comparisons.md](multiple_comparisons.md) |
| "Anatomic vs pathologic stage / curated semantic neighbors?" | [confusion_pairs.md](confusion_pairs.md) |
| "What does field type / section / scope mean?" | [glossary.md](glossary.md) |
| "Show me the canonical CLI invocations." | [recipes.md](recipes.md) |
| "How do I compare two runs / models / methodologies?" | [comparing_runs.md](comparing_runs.md) |
| "Which run is better?" | [comparing_runs.md](comparing_runs.md) (Recipe 1) |
| "How reproducible is this model across reruns?" | [comparing_runs.md](comparing_runs.md) (Recipe 2) |
| "How do I run eval on GPU (CUDA / MPS)?" | [gpu_acceleration.md](gpu_acceleration.md) |
| "How do I cite this in the paper Methods section?" | [methods_citations.md](methods_citations.md) |
| "I want to debug eval without exposing PHI." | [../obfuscation.md](../obfuscation.md) |

## CSV → metric crosswalk

| Output file (cascade layout) | Headline metric | Doc |
|---|---|---|
| `cascade_atomic.parquet` | every (run, case, field) row with `cascade_stage`, `gate_pass`, `others_disposition` | [reading_outputs.md](reading_outputs.md) |
| `chapter1_eligibility/overall.csv` | sensitivity, specificity, MCC, Cohen's κ for `cancer_excision_report` | [non_nested_metrics.md](non_nested_metrics.md) |
| `chapter1_eligibility/confusion.csv` | TP/FP/FN/TN for the eligibility decision | [confusion_pairs.md](confusion_pairs.md) |
| `chapter2_organ_classification/overall.csv` | per-organ accuracy, macro-F1, Cohen's κ | [non_nested_metrics.md](non_nested_metrics.md) |
| `chapter2_organ_classification/confusion_per_class.csv` | per-class P/R/F1/support over the 11-class organ classifier | [confusion_pairs.md](confusion_pairs.md) |
| `chapter2_organ_classification/others/others_ledger.csv` | per-case dual-primary / out-of-scope ledger | [reading_outputs.md](reading_outputs.md) |
| `chapter3_field_extraction/per_field_overall.csv` | accuracy, coverage, Cohen's κ per Stage-C field | [non_nested_metrics.md](non_nested_metrics.md) |
| `chapter3_field_extraction/per_field_by_organ.csv` | same, stratified by organ | [non_nested_metrics.md](non_nested_metrics.md) |
| `chapter3_field_extraction/per_organ_overall.csv` | mean accuracy across fields per organ | [non_nested_metrics.md](non_nested_metrics.md) |
| `chapter3_field_extraction/cascade_funnel.csv` | n_total / n_passed / n_dropped per stage | [diagnostics.md](diagnostics.md) |
| `chapter3_field_extraction/conditional_accuracy_grid.csv` | `P(field_correct \| stage_b ∧ stage_a)` per field | [diagnostics.md](diagnostics.md) |
| `chapter*/multirun_consistency.csv` | ICC(2,1), ICC(3,k), Cronbach α, accuracy-flip-rate | [multirun.md](multirun.md) |
| `model_pair_tests/*.csv` | McNemar / Cochran-Q / Stuart-Maxwell + Holm/BH adjusted p | [multiple_comparisons.md](multiple_comparisons.md), [../stat_methods.md](../stat_methods.md) §2.2 |
| `iaa/pair_*.csv` | Cohen's κ (un/weighted), CCC, ICC, BA LoA, F1, Krippendorff α | [iaa_basics.md](iaa_basics.md) |
| `iaa_pair/pair_<a>_vs_<b>/headline.csv` | overall pair κ — mean per-field κ, n-weighted mean, pooled categorical κ, agree/disagree PABAK, Krippendorff α | [iaa_basics.md](iaa_basics.md#pair-focused-headline-iaa_pair) |
| `iaa_pair/pair_<a>_vs_<b>/per_section.csv` | section roll-up (top_level / scalar_pathology / nested) | [iaa_basics.md](iaa_basics.md#pair-focused-headline-iaa_pair) |
| `iaa_pair/pair_<a>_vs_<b>/per_organ.csv` | per-organ κ summary (4 stats × n_organs) | [iaa_basics.md](iaa_basics.md#pair-focused-headline-iaa_pair) |
| `iaa_pair/pair_<a>_vs_<b>/confusion/<field>.csv` | top-N categorical fields' confusion matrices | [confusion_pairs.md](confusion_pairs.md) |
| `iaa_pair/pair_<a>_vs_<b>/summary.md` | human-readable headline + top-K most disagreed fields | [iaa_basics.md](iaa_basics.md#pair-focused-headline-iaa_pair) |
| `iaa/preann/delta_kappa_per_field__*.csv` | Δκ with vs without preann + paired bootstrap CI | [preann_effect.md](preann_effect.md) |
| `iaa/preann/anchoring_index__*.csv` | AI = P(human=preann \| with) − P(human=preann \| without) | [preann_effect.md](preann_effect.md) |
| `completeness/modularity_advantage.csv` | sorted method-pair Δ on attempted_rate (ablation headline) | [completeness.md](completeness.md) |
| `diagnostics/error_source_decomposition.csv` | model_error / report_ambiguity / report_silent buckets | [diagnostics.md](diagnostics.md) |
| `diagnostics/accuracy_by_difficulty_tier.csv` | accuracy stratified by IAA-derived difficulty | [diagnostics.md](diagnostics.md) |
| `cross_dataset/per_field_delta.csv` | CMUH vs TCGA Δ accuracy with bootstrap CI | [cross_dataset.md](cross_dataset.md) |
| `headline/headline_forest.csv` | unified long-form for forest-plot rendering | [reading_outputs.md](reading_outputs.md) |

## Conventions

- **Units of analysis** are typically `(case, field, run)`; a run is one full re-execution of a model on the same case set.
- Three-way outcome model: every model-vs-gold scoring distinguishes **correct / wrong / missing**. Missing further splits into `parse_error` (whole case failed) vs `field_missing` (case loaded, this field absent). See [completeness.md](completeness.md).
- All accuracy is reported in two flavors: **attempted_accuracy** (correct / attempted) and **effective_accuracy** (correct / eligible). The gap is the *completeness penalty* — primary signal for the modularity ablation.
- All proportions are reported with Wilson 95% CI by default; bootstrap CI for non-binary statistics (BCa, n_boot=2000 unless overridden).
- Multi-run statistics use case-stratified bootstrap and / or GLMM with random intercepts for `(case_id, run_id)` (see [ci_methods.md](ci_methods.md)).
- Endpoint pre-registration in `configs/eval_endpoints.yaml` separates **primary** (Holm-Bonferroni) from **secondary** (Benjamini-Hochberg) p-value adjustment families.

Every numerical claim in the paper traces back to a column in one of these CSVs, with provenance captured in the corresponding `manifest.json` (git SHA + UTC timestamp + full args).
