# Continuous (numeric) field metrics

For integer / float-valued fields like `tumor_size`, `dcis_size`, `maximal_ln_size`. Categorical metrics don't apply directly — we use regression-style measures of agreement.

## Mean Absolute Error (MAE) and RMSE

- `MAE = mean(|pred − gold|)` — robust, in original units.
- `RMSE = √mean((pred − gold)²)` — penalises large errors more.

**Implementation:** `sklearn.metrics.mean_absolute_error`, `sklearn.metrics.root_mean_squared_error` (sklearn ≥ 1.4) or `mean_squared_error(squared=False)`.
**Reference:** standard regression metrics; see Pedregosa et al. (2011) for the scikit-learn reference.

## Bias

`bias = mean(pred − gold)` *with sign*. Tells you whether the model systematically over- or under-predicts. Distinct from MAE which is unsigned.

**Output:** Continuous-field rows in `iaa/pair_*.csv` (`bland_altman_bias`).

## Bland-Altman limits of agreement

`LoA = bias ± 1.96 · SD(differences)`. The range within which 95% of inter-rater (or inter-method) differences fall. The reader can quote, "the model is within ± 4 mm of the human reading 95% of the time."

**Implementation:** `digital_registrar_research.benchmarks.eval.iaa.bland_altman` (in-house).
**Reference:** Bland, J. M., & Altman, D. G. (1986). "Statistical methods for assessing agreement between two methods of clinical measurement." *The Lancet* 327 (8476): 307–310.

## Lin's CCC and ICC(2,1)

See [iaa_basics.md](iaa_basics.md) — both apply to continuous agreement; CCC combines correlation and mean shift, ICC(2,1) is its variance-component twin.

## Within-tolerance rates

For clinically meaningful tolerances (e.g. ±2 mm for `tumor_size`), the binary "is this prediction within tolerance?" rate is more interpretable than MAE for a clinician. We report multiple tolerance levels (±1, ±2, ±5 mm) where appropriate.

**Implementation:** in-house one-liner. Tolerances in `digital_registrar_research.benchmarks.eval.metrics.NUMERIC_TOLERANCE_MM` (default ±2 mm for `tumor_size`) and `nested_metrics.LN_COUNT_TOLERANCE` (±1 node for examined / involved counts).

## Pearson and Spearman correlation

- **Pearson r** — linear association; sensitive to outliers.
- **Spearman ρ** — rank-order association; robust to outliers, non-parametric.

Both reported with **Fisher-z CI** for transparency.

**Implementation:** `scipy.stats.pearsonr` (which since SciPy ≥ 1.9 also provides `.confidence_interval()`), `scipy.stats.spearmanr`. Fisher-z via `digital_registrar_research.benchmarks.eval.ci.fisher_z_ci_for_corr`.
**References:** Pearson, K. (1895). "Notes on regression and inheritance in the case of two parents." *Proc. Royal Society* 58: 240–242. Spearman, C. (1904). "The proof and measurement of association between two things." *American Journal of Psychology* 15 (1): 72–101. Fisher (1915) for the z-transform.

## What to do if your numbers are bad

- **High MAE with low CCC:** the model is systematically off (large bias). Check Bland-Altman bias for direction.
- **Low MAE but high RMSE:** small typical errors, occasional huge ones. Inspect the worst cases.
- **Within-tolerance rate high but CCC low:** the model is in the right ballpark but with a systematic offset that LoA bands will reveal.
- **Pearson r high but Spearman ρ low:** the relationship is non-monotonic — likely a bug, since pathology measurements should be monotonic in their gold reference.
