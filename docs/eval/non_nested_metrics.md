# Non-nested (scalar) field metrics

> **Cascade-redesign note (2026-05).** The standalone `non_nested` subcommand was removed; scalar-field accuracy is now produced by the `cascade` subcommand and lives under `chapter3_field_extraction/`. Output filenames in this doc that reference `non_nested/` should be read as `chapter3_field_extraction/`. The metric definitions below are unchanged. See [CHANGELOG.md](CHANGELOG.md) and [../stat_methods.md](../stat_methods.md).

For single-value fields (categorical, boolean, ordinal, continuous numeric) **plus list-of-literals fields** (set-valued enums like `tumor_extent`, `vascular_invasion`, `involved_margin_list`). Reported per-field, per-organ, and per-subgroup (single vs multi-primary). For nested-list (list-of-dicts) fields see [nested_metrics.md](nested_metrics.md).

## Field coverage

The atomic table iterates **every per-organ scoreable field** declared in `scope_organs` — categorical + boolean + continuous + list-of-literals. Not just the 12 fields in `FAIR_SCOPE`. Roughly 18–32 fields per organ; ~69 distinct field names cross-organ. See [glossary.md](glossary.md) for the per-organ counts.

The two top-level fields (`cancer_category`, `cancer_excision_report`) and the three breast biomarker synthetic fields (`biomarker_<er|pr|her2>`) are added explicitly on top of the per-organ list.

## List-of-literals scoring

For set-valued enum fields (e.g. `tumor_extent: ["hepatic_vein", "small_vessel"]`):

- **Headline scoring**: unordered set equality. The model's set must exactly match gold's set to count as correct.
- **Partial-credit scoring** (separate output): item-level TP / FP / FN with set-F1, analogous to nested-list scoring but on plain string items.
- **Empty list vs null**: an empty list (`[]`) is a definite "no items present" answer, distinct from `null` (not assessed). Both score paths handle this distinction.

**Implementation:** `scripts/eval/_common/outcome.py:list_of_literals_match` (set equality) and `list_of_literals_set_metrics` (TP/FP/FN/F1).
**Reference:** Set-F1 follows standard IR practice. See van Rijsbergen (1979).

The `list_of_literals` field type is **organ-aware** — the same field name can be a list-of-literals in one organ and a regular categorical in another. `tumor_extent` is `list_of_literals` for liver but `nominal` for esophagus and stomach. The classifier dispatches by `(field, organ)`. Registry: `scope_organs.ORGAN_LIST_OF_LITERALS`.

## Overall accuracy CSVs

Two complementary "overall" views beyond per-field detail:

- **`per_organ_overall.csv`** — one row per organ + a cross-organ `ALL` row, aggregating across all that organ's fields. Reports `attempted_accuracy_micro` (correct / attempted, weighted by sample), `effective_accuracy_micro` (correct / total, penalises missingness), and `macro_field_accuracy` (per-field mean, weights every field equally).
- **`section_rollup.csv`** — same idea but grouped by section (`top_level`, `staging`, `grading`, `invasion`, `size`, `biomarker`, `other`).

Use the `ALL` row of `per_organ_overall.csv` for the single headline accuracy number; per-organ rows for the stratified breakdown.

## Two accuracy flavors

For every (field, organ):

- **Attempted accuracy** = correct / attempted. Quality on what the model tried.
- **Effective accuracy** = correct / eligible. Quality across the full cohort, treating missing as wrong.

Both flavors are reported with multiple CIs side by side. The gap is the **completeness penalty** — see [completeness.md](completeness.md).

**Wilson 95% CI** on both via `statsmodels.stats.proportion.proportion_confint(method="wilson")` (Pedregosa et al., 2011 documents the equivalent in scikit-learn; we use statsmodels). Reference: Wilson, E. B. (1927). "Probable inference, the law of succession, and statistical inference." *JASA* 22 (158): 209–212.

**BCa bootstrap CI** on attempted accuracy via the in-house `digital_registrar_research.benchmarks.eval.ci.bootstrap_ci(method="bca")`, equivalent to `scipy.stats.bootstrap(method="BCa")` (Virtanen et al., 2020). Reference: Efron, B. (1987). "Better bootstrap confidence intervals." *JASA* 82 (397): 171–185.

**Student-t CI on the run-level mean** (when ≥ 2 runs) via `scipy.stats.t.interval`. Reference: Student (1908). "The probable error of a mean." *Biometrika* 6 (1): 1–25.

**Output:** `non_nested/per_field_overall.csv`, `non_nested/per_field_by_organ.csv`.

## Cohen's κ — agreement beyond chance

Plain accuracy can be misleadingly high when one class dominates (e.g. 95% of cases are `lymphovascular_invasion = false`). Cohen's κ corrects for the chance level of agreement.

- **Unweighted κ** for nominal categoricals: counts off-diagonal cells uniformly.
- **Quadratic-weighted κ** for ordinal categoricals: penalises far-apart errors more than near-misses.

For ordinal fields (`grade`, `pt_category`, `pn_category`, etc. — see `iaa.ORDINAL_FIELDS`) we report **both** weighted and unweighted variants.

**Implementation:** `sklearn.metrics.cohen_kappa_score(weights=None|"quadratic", labels=...)`.
**References:** Cohen, J. (1960). "A coefficient of agreement for nominal scales." *Educational and Psychological Measurement* 20 (1): 37–46. Weighted variant: Cohen, J. (1968). "Weighted kappa: nominal scale agreement with provision for scaled disagreement or partial credit." *Psychological Bulletin* 70 (4): 213–220.
**Output:** `non_nested/headline_classification.csv` (`cohen_kappa`, `cohen_kappa_quadratic`).

## Matthews correlation coefficient (MCC)

Robust to class imbalance, ranges in [-1, 1]. Particularly useful as a binary-field headline (`lymphovascular_invasion`, `perineural_invasion`, `cancer_excision_report`).

**Implementation:** `sklearn.metrics.matthews_corrcoef`.
**Reference:** Matthews, B. W. (1975). "Comparison of the predicted and observed secondary structure of T4 phage lysozyme." *BBA - Protein Structure* 405 (2): 442–451. Multi-class generalisation: Gorodkin, J. (2004). "Comparing two K-category assignments by a K-category correlation coefficient." *Computational Biology and Chemistry* 28 (5–6): 367–374.
**Output:** `non_nested/headline_classification.csv` (`matthews_corrcoef`).

## Balanced accuracy

Mean per-class recall. Equivalent to plain accuracy when classes are balanced; fairer than plain accuracy when they're not.

**Implementation:** `sklearn.metrics.balanced_accuracy_score`.
**Reference:** Brodersen, K. H., Ong, C. S., Stephan, K. E., & Buhmann, J. M. (2010). "The balanced accuracy and its posterior distribution." *Proceedings of the 20th International Conference on Pattern Recognition*: 3121–3124.
**Output:** `non_nested/headline_classification.csv` (`balanced_accuracy`).

## Confusion matrix and per-class P/R/F1

Per (field, organ), the confusion matrix is written to `non_nested/confusion/<field>__<organ>.csv` in long form (`gold_value`, `pred_value`, `count`). Per-class precision / recall / F1 plus macro / micro / weighted averages go to `non_nested/per_class_prf1.csv`.

**Implementation:** `sklearn.metrics.confusion_matrix`, `sklearn.metrics.precision_recall_fscore_support` (Pedregosa et al., 2011).

## Confusion pairs / semantic neighbors

Top-N most-frequent confusion pairs per (field, organ), with a curated `is_semantic_neighbor` flag for known clinically-equivalent pairs (anatomic vs pathologic stage, t1 vs t1a substages, m0 vs mx, sentinel vs axillary level I, etc.). See [confusion_pairs.md](confusion_pairs.md) for the curated list.

`accuracy_collapsing_neighbors` re-computes accuracy treating curated neighbor errors as correct. Useful for the writeup paragraph that quantifies clinically-equivalent confusions (anatomic vs pathologic staging, T-substages within the same major stage).

**Output:** `non_nested/confusion_pairs.csv`, `non_nested/accuracy_collapsing_neighbors.csv`.

## Top-k ordinal accuracy and rank-distance distribution

For ordinal fields, an "off-by-one" error is materially less bad than "off-by-three." We report:

- `top-1 accuracy` = strict accuracy.
- `top-k accuracy` = `|rank(pred) − rank(gold)| ≤ k`.
- Mean rank distance of wrong predictions, plus a histogram of rank distances.

**Implementation:** in-house — `scripts/eval/_common/stats_extra.top_k_ordinal_accuracy` and `rank_distance_distribution`. (`sklearn.metrics.top_k_accuracy_score` requires probability scores we don't have.)
**Output:** `non_nested/top_k_ordinal.csv`, `non_nested/rank_distance.csv`.

## Multi-primary subgroup column

Every metric is also broken down by `subgroup ∈ {single_primary, multi_primary, all}`. Multi-primary detection lives in `digital_registrar_research.benchmarks.eval.multi_primary` (heuristics: `cancer_laterality == "bilateral"`, multi-clock / multi-quadrant strings, multifocal flags, etc.). Multi-primary cases are clinically distinct and per-organ schemas designed for single-primary disease can systematically misidentify them; reporting per-subgroup accuracy makes this visible instead of hiding it in the aggregate.

`per_field_by_subgroup.csv` reports the same metric set per subgroup so the writeup can cite "in the multi-primary subgroup, accuracy was X with 95% CI [...]".

## Schema-conformance / out-of-vocabulary rate

Categorical predictions outside the allowed enum aren't "wrong" in the usual sense — they're not even valid options. See [completeness.md](completeness.md) for full treatment. Output: `non_nested/schema_conformance.csv`.

## Refusal calibration

When pred is null, is gold also null? Distinguishes correct refusal from lazy missingness. See [completeness.md](completeness.md). Output: `non_nested/refusal_calibration.csv`.

## Run consistency (multi-run only)

When ≥ 2 runs exist, we additionally report:

- Fleiss' κ on per-case correctness across runs.
- Fleiss' κ on per-case prediction values across runs.
- Flip rate, missing-flip rate.
- Stability accuracy (accuracy on cases where all runs agreed).
- Brittle case rate (cases where ≥ 1 run wrong AND ≥ 1 run right).

See [multirun.md](multirun.md) for full treatment. Output: `non_nested/run_consistency.csv`.

## Section rollup

Fields are grouped into sections (`top_level`, `staging`, `grading`, `invasion`, `size`, `biomarker`, `other`) and we report the mean per-field accuracy per section with bootstrap CI over fields. Useful for the high-level writeup.

**Output:** `non_nested/section_rollup.csv`.

## What to do if your numbers are bad

- **High plain accuracy but low κ:** class imbalance. Report κ alongside accuracy in the writeup.
- **High accuracy but low MCC on binary fields:** the model is predicting the majority class blindly. Inspect the confusion matrix.
- **Wide bootstrap CI:** small n. Either gather more cases or report the strict Wilson CI (which doesn't go negative for small counts).
- **Large `accuracy_collapsing_neighbors − accuracy_strict`:** the model is making "near-miss" errors that are clinically benign. Useful detail to lift into Discussion.
- **High off-diagonal mass on a specific cell in the confusion matrix:** systematic confusion. If the pair is in `confusion_pairs.md`'s curated list, it's a known issue; if not, consider adding it (or fixing the prompt).
