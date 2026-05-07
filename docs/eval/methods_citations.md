# Methods citations — paper-ready bibliography

This file is the **single source of truth** for statistical-method citations in the paper. Every metric reported by `scripts/eval/` has both a library-implementation citation and an original-paper citation here. The Methods section of the paper is assembled from this file.

For each metric we record:
- **Implementation** — exact `package.function` and version pin.
- **Reference** — original paper(s) defining the metric.
- **Used in** — which evaluation output(s) report this metric.

---

## Confidence intervals

### Wilson score interval
- **Implementation:** `digital_registrar_research.benchmarks.eval.ci.wilson_ci` (in-house, ~10 LOC matching the canonical formula). Library equivalent: `statsmodels.stats.proportion.proportion_confint(method="wilson")` (Seabold & Perktold, 2010).
- **Reference:** Wilson, E. B. (1927). "Probable inference, the law of succession, and statistical inference." *JASA* 22 (158): 209–212.
- **Used in:** every binary-rate column in `non_nested/per_field_overall.csv`, `non_nested/per_field_by_organ.csv`, `nested/nested_missingness.csv`, `completeness/per_method_*.csv`, `iaa/preann/convergence_to_preann*.csv`.

### Clopper-Pearson exact CI
- **Implementation:** `ci.clopper_pearson_ci`. Library equivalent: `statsmodels.stats.proportion.proportion_confint(method="beta")`.
- **Reference:** Clopper, C., & Pearson, E. S. (1934). "The use of confidence or fiducial limits illustrated in the case of the binomial." *Biometrika* 26 (4): 404–413.

### BCa bootstrap CI
- **Implementation:** `ci.bootstrap_ci(method="bca")` (in-house). Library equivalent: `scipy.stats.bootstrap(method="BCa")` (SciPy ≥ 1.7; Virtanen et al., 2020).
- **Reference:** Efron, B. (1987). "Better bootstrap confidence intervals." *JASA* 82 (397): 171–185.
- **Used in:** every accuracy / F1 CI column in `non_nested/per_field_*.csv`, `nested/per_field_per_organ.csv`.

### Student-t CI for the run-level mean
- **Implementation:** `ci.t_ci`, wrapping `scipy.stats.t.interval`.
- **Reference:** Student (1908). "The probable error of a mean." *Biometrika* 6 (1): 1–25.
- **Used in:** `mean_per_run_t_ci_lo/hi` columns in `non_nested/per_field_*.csv` (only meaningful with ≥ 2 runs).

### Two-source nested bootstrap
- **Implementation:** `ci.two_source_bootstrap_ci` (in-house; nested case × run resampling for repeated-measures designs).
- **Reference:** Adapted from Owen, A. B. (2007). "The pigeonhole bootstrap." *Annals of Applied Statistics* 1 (2): 386–411.
- **Used in:** total-CI columns when GLMM falls back.

### Fisher-z CI for correlation coefficients
- **Implementation:** `ci.fisher_z_ci_for_corr`.
- **Reference:** Fisher, R. A. (1915). "Frequency distribution of the values of the correlation coefficient in samples from an indefinitely large population." *Biometrika* 10 (4): 507–521.

### GLMM marginal accuracy
- **Implementation:** `multirun._glmm_marginal_accuracy` via `statsmodels.genmod.bayes_mixed_glm.BinomialBayesMixedGLM` (Seabold & Perktold, 2010). Variance components for `case_id` and `run_id` random intercepts.
- **Reference:** Bayesian variational fit. Foundational mixed-model reference: Laird, N. M., & Ware, J. H. (1982). "Random-effects models for longitudinal data." *Biometrics* 38 (4): 963–974.

---

## Agreement / IAA metrics

### Cohen's κ (unweighted)
- **Implementation:** `iaa.cohen_kappa(weights="unweighted")` (in-house) and, in the new code path, `sklearn.metrics.cohen_kappa_score` (Pedregosa et al., 2011). Both should produce identical values.
- **Reference:** Cohen, J. (1960). "A coefficient of agreement for nominal scales." *Educational and Psychological Measurement* 20 (1): 37–46.
- **Used in:** `iaa/pair_*.csv` (`stat_name = cohen_kappa_unweighted`), `non_nested/headline_classification.csv` (`cohen_kappa`), `iaa/preann/delta_kappa_per_field*.csv`.

### Cohen's quadratic-weighted κ
- **Implementation:** `iaa.cohen_kappa(weights="quadratic", ordinal_order=...)` and `sklearn.metrics.cohen_kappa_score(weights="quadratic", labels=...)`.
- **Reference:** Cohen, J. (1968). "Weighted kappa: nominal scale agreement with provision for scaled disagreement or partial credit." *Psychological Bulletin* 70 (4): 213–220.
- **Used in:** `non_nested/headline_classification.csv` (`cohen_kappa_quadratic`), every ordinal field in `iaa/pair_*.csv`.

### Fleiss' κ
- **Implementation:** `multirun.fleiss_kappa` (in-house). Library equivalent: `statsmodels.stats.inter_rater.fleiss_kappa`.
- **Reference:** Fleiss, J. L. (1971). "Measuring nominal scale agreement among many raters." *Psychological Bulletin* 76 (5): 378–382.
- **Used in:** `non_nested/run_consistency.csv` (`fleiss_kappa_correctness`, `fleiss_kappa_values`).

### Krippendorff's α
- **Implementation:** `iaa.krippendorff_alpha` (in-house). Library equivalent: `krippendorff.alpha` from the `krippendorff` PyPI package.
- **Reference:** Krippendorff, K. (2004). *Content Analysis: An Introduction to Its Methodology* (2nd ed.). Sage. Ch. 11.
- **Used in:** `iaa/whole_report.csv` (`stat_name = krippendorff_alpha_{nominal,ordinal,interval}`).

### Lin's concordance correlation coefficient
- **Implementation:** `iaa.lins_ccc` (in-house — no canonical scipy/sklearn function). `pingouin.concordance_corr` is an alternative package.
- **Reference:** Lin, L. I.-K. (1989). "A concordance correlation coefficient to evaluate reproducibility." *Biometrics* 45 (1): 255–268.
- **Used in:** continuous-field rows in `iaa/pair_*.csv` (`stat_name = lins_ccc`).

### ICC(2,1)
- **Implementation:** `iaa.icc_2_1` (in-house). Library equivalent: `pingouin.intraclass_corr` filtered to the `ICC2` row.
- **Reference:** Shrout, P. E., & Fleiss, J. L. (1979). "Intraclass correlations: uses in assessing rater reliability." *Psychological Bulletin* 86 (2): 420–428.

### Bland-Altman bias and limits of agreement
- **Implementation:** `iaa.bland_altman` (in-house).
- **Reference:** Bland, J. M., & Altman, D. G. (1986). "Statistical methods for assessing agreement between two methods of clinical measurement." *The Lancet* 327 (8476): 307–310.
- **Used in:** continuous-field rows in `iaa/pair_*.csv` (`stat_name = bland_altman_*`).

### Kendall's τ-b
- **Implementation:** `iaa.kendall_tau_b` (in-house wrapper over `scipy.stats.kendalltau(variant="b")`).
- **Reference:** Kendall, M. G. (1948). *Rank Correlation Methods*. Griffin.

### Prevalence- and bias-adjusted κ (PABAK)
- **Implementation:** `iaa.pabak` (one-line `2·p_observed − 1`).
- **Reference:** Byrt, T., Bishop, J., & Carlin, J. B. (1993). "Bias, prevalence and kappa." *Journal of Clinical Epidemiology* 46 (5): 423–429.

### McNemar's test
- **Implementation:** `ci.mcnemar_test` (in-house, with continuity correction and exact-binomial fallback for small n). Library equivalent: `statsmodels.stats.contingency_tables.mcnemar`.
- **Reference:** McNemar, Q. (1947). "Note on the sampling error of the difference between correlated proportions or percentages." *Psychometrika* 12 (2): 153–157.
- **Used in:** `completeness/method_pairwise_deltas.csv` (`mcnemar_p_value`).

---

## Classification metrics

### Confusion matrix
- **Implementation:** `sklearn.metrics.confusion_matrix` (Pedregosa et al., 2011), wrapped by `scripts/eval/_common/stats_extra.py:confusion_matrix_long`.
- **Used in:** `non_nested/confusion/<field>__<organ>.csv`.

### Per-class precision, recall, F1
- **Implementation:** `sklearn.metrics.precision_recall_fscore_support`.
- **Used in:** `non_nested/per_class_prf1.csv`.

### Matthews correlation coefficient
- **Implementation:** `sklearn.metrics.matthews_corrcoef`.
- **Reference:** Matthews, B. W. (1975). "Comparison of the predicted and observed secondary structure of T4 phage lysozyme." *BBA - Protein Structure* 405 (2): 442–451. Multi-class generalisation: Gorodkin (2004).
- **Used in:** `non_nested/headline_classification.csv` (`matthews_corrcoef`).

### Balanced accuracy
- **Implementation:** `sklearn.metrics.balanced_accuracy_score`.
- **Reference:** Brodersen, K. H., et al. (2010). "The balanced accuracy and its posterior distribution." *ICPR 2010*: 3121–3124.
- **Used in:** `non_nested/headline_classification.csv` (`balanced_accuracy`).

### Top-k ordinal accuracy
- **Implementation:** in-house — `_common/stats_extra.top_k_ordinal_accuracy`. (No direct sklearn equivalent because `top_k_accuracy_score` requires probability scores.)
- **Used in:** `non_nested/top_k_ordinal.csv`.

### Mean rank distance (ordinal-aware error severity)
- **Implementation:** in-house — `_common/stats_extra.rank_distance_distribution`.
- **Used in:** `non_nested/rank_distance.csv`.

---

## Continuous / regression metrics

### MAE, RMSE
- **Implementation:** `sklearn.metrics.mean_absolute_error`, `sklearn.metrics.root_mean_squared_error` (sklearn ≥ 1.4) or `mean_squared_error(squared=False)`.

### Pearson and Spearman correlations
- **Implementation:** `scipy.stats.pearsonr` (returns `.statistic`, `.pvalue`, `.confidence_interval()` since SciPy ≥ 1.9), `scipy.stats.spearmanr`.
- **Used in:** `completeness/position_in_schema_correlation.csv` (Spearman ρ vs schema position).

### Within-tolerance rate
- **Implementation:** in-house one-liner. Tolerances: ±2 mm for `tumor_size`, ±1 node for LN counts. See `metrics.NUMERIC_TOLERANCE_MM`, `nested_metrics.LN_COUNT_TOLERANCE`.

---

## Statistical tests

### Chi-square test of independence
- **Implementation:** `scipy.stats.chi2_contingency`.
- **Used in:** `iaa/disagreement_resolution.csv` (`chi2_p`), `cross_dataset/distribution_shift.csv` (`chi2_p`).

### Paired bootstrap Δ
- **Implementation:** `ci.paired_bootstrap_diff` (in-house — paired case-level resampling).
- **Used in:** `non_nested/ensemble_vs_single.csv`, `iaa/preann/delta_kappa_per_field*.csv`, `iaa/preann/disagreement_reduction.csv`.

---

## Multiple-comparisons correction

### Holm-Bonferroni
- **Implementation:** `statsmodels.stats.multitest.multipletests(method="holm")`, wrapped by `_common/stats_extra.adjust_pvalues`.
- **Reference:** Holm, S. (1979). "A simple sequentially rejective multiple test procedure." *Scandinavian Journal of Statistics* 6 (2): 65–70.
- **Used in:** primary-endpoint p-values across the eval suite (added by orchestrators when `endpoint_tier == "primary"`).

### Benjamini-Hochberg (FDR)
- **Implementation:** `statsmodels.stats.multitest.multipletests(method="fdr_bh")`.
- **Reference:** Benjamini, Y., & Hochberg, Y. (1995). "Controlling the false discovery rate: a practical and powerful approach to multiple testing." *JRSS B* 57 (1): 289–300.
- **Used in:** secondary-endpoint p-values.

---

## Effect sizes

### Cohen's d
- **Implementation:** `_common/stats_extra.cohens_d` (in-house pooled-SD formula). Library equivalent: `pingouin.compute_effsize(eftype="cohen")`.
- **Reference:** Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum.

### Cliff's δ
- **Implementation:** `_common/stats_extra.cliffs_delta` (in-house O(n_a · n_b) dominance count).
- **Reference:** Cliff, N. (1993). "Dominance statistics: ordinal analyses to answer ordinal questions." *Psychological Bulletin* 114 (3): 494–509.

### Odds ratio with Wald CI
- **Implementation:** `_common/stats_extra.odds_ratio_with_ci` (in-house with Haldane-Anscombe correction). Library equivalent: `scipy.stats.contingency.odds_ratio` (SciPy ≥ 1.10).

---

## Nested-list bipartite matching

### Greedy bipartite assignment with composite similarity
- **Implementation:** `nested_metrics._greedy_match` plus field-specific similarity functions (`_ln_similarity`, `_margin_similarity`, `nested.biomarkers._bm_similarity`). Library alternative for optimal matching: `scipy.optimize.linear_sum_assignment` (Hungarian algorithm). The greedy approximation is faster and adequate for the ~1–10 items per case observed.
- **Reference:** Kuhn, H. W. (1955). "The Hungarian method for the assignment problem." *Naval Research Logistics Quarterly* 2 (1–2): 83–97. (For the optimal alternative; we use a greedy approximation.)

### Jaccard similarity (description-text matching)
- **Implementation:** in-house `_jaccard` in `nested_metrics`.
- **Reference:** Jaccard, P. (1912). "The distribution of the flora in the alpine zone." *New Phytologist* 11 (2): 37–50.

---

## Distribution-shift metrics (cross-dataset)

### KL divergence
- **Implementation:** `_common/stats_extra.kl_divergence` via `scipy.special.rel_entr`.
- **Reference:** Kullback, S., & Leibler, R. A. (1951). "On information and sufficiency." *Annals of Mathematical Statistics* 22 (1): 79–86.

### Jensen-Shannon distance
- **Implementation:** `_common/stats_extra.jensen_shannon` via `scipy.spatial.distance.jensenshannon` (base 2).
- **Reference:** Lin, J. (1991). "Divergence measures based on the Shannon entropy." *IEEE Transactions on Information Theory* 37 (1): 145–151.

### Wasserstein-1 distance (Earth Mover's)
- **Implementation:** `_common/stats_extra.wasserstein` via `scipy.stats.wasserstein_distance`.
- **Reference:** Vaserstein, L. N. (1969). "Markov processes over denumerable products of spaces describing large systems of automata." *Problems in Information Transmission* 5 (3): 64–72.

---

## Software / library citations

When citing the libraries themselves in the paper:

- **scikit-learn:** Pedregosa, F., et al. (2011). "Scikit-learn: Machine Learning in Python." *Journal of Machine Learning Research* 12: 2825–2830.
- **SciPy:** Virtanen, P., et al. (2020). "SciPy 1.0: fundamental algorithms for scientific computing in Python." *Nature Methods* 17: 261–272.
- **statsmodels:** Seabold, S., & Perktold, J. (2010). "statsmodels: Econometric and statistical modeling with Python." *Proceedings of the 9th Python in Science Conference*: 92–96.
- **NumPy:** Harris, C. R., et al. (2020). "Array programming with NumPy." *Nature* 585: 357–362.
- **pandas:** McKinney, W. (2010). "Data structures for statistical computing in Python." *Proceedings of the 9th Python in Science Conference*: 56–61.
- **pingouin:** Vallat, R. (2018). "Pingouin: statistics in Python." *Journal of Open Source Software* 3 (31): 1026.
- **krippendorff (Python package):** Castro, S. "Fast Krippendorff: Fast computation of Krippendorff's alpha agreement measure." GitHub: `pln-fing-udelar/fast-krippendorff`.

---

## How to use this file in the paper

The Methods section should cite both the original methodological reference (e.g. Cohen 1960 for κ) AND the implementation library (e.g. scikit-learn). Example sentence:

> Inter-annotator agreement was quantified with Cohen's κ (Cohen, 1960; weighted variant Cohen, 1968) computed via `sklearn.metrics.cohen_kappa_score` (Pedregosa et al., 2011). Bootstrap confidence intervals (n = 2000 replicates) used the bias-corrected accelerated method (Efron, 1987) implemented in `digital_registrar_research.benchmarks.eval.ci.bootstrap_ci`. Multiple-comparisons correction used the Holm-Bonferroni procedure (Holm, 1979) for primary endpoints (12 fields pre-registered in `configs/eval_endpoints.yaml`) and the Benjamini-Hochberg FDR (Benjamini & Hochberg, 1995) for secondary endpoints, both via `statsmodels.stats.multitest.multipletests` (Seabold & Perktold, 2010).

This pattern — original-paper citation followed by library-implementation citation — should appear for every metric reported.
