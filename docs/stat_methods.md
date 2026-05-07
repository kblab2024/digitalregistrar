# Statistical methods of the cascade evaluation

This document is the canonical reference for every statistic the
cascade pipeline reports. It maps each method to (a) its file in the
`benchmarks/eval/stats/` package, (b) the chapter CSV it produces, and
(c) the reviewer comment it addresses. The manuscript supplement
should cite this document when referencing methods.

## 1. Cascade architecture

The evaluation is gated as three sequential stages. Each stage's
denominator is the prior stage's "passed" cohort. Cohort attrition is
recorded explicitly in `chapter3_field_extraction/cascade_funnel.csv`.

```
   All cases (n_total)
        │
        ▼  Stage A: cancer_excision_report
   ┌────────────┐
   │  passed_A  │   → Chapter 1 (eligibility) reports here.
   └────────────┘
        │
        ▼  Stage B: cancer_category
   ┌────────────┐
   │  passed_B  │   → Chapter 2 (organ classification) reports here.
   │  except    │      "Others" disposition diverts to others_ledger.
   │  others    │
   └────────────┘
        │
        ▼  Stage C: cancer_data fields (after exclusions / whitelist)
   ┌────────────┐
   │ scored_C   │   → Chapter 3 (field extraction) reports here.
   └────────────┘
```

A case that fails Stage A contributes to Chapter 1 but is silent in
Chapters 2 and 3. A case where gold or prediction is `"others"` is
recorded in `chapter2_organ_classification/others/others_ledger.csv`
but not in Chapter 3 — the 10-organ schema does not apply.

Field exclusions are applied uniformly at Stage C:

| Exclusion | Reason |
|---|---|
| `cancer_data.ajcc_version` | version metadata, not a categorical endpoint |
| `cancer_data.treatment_effect` | free-text in practice |
| `cancer_data.margins[*].description` | free-text annotators wrote inconsistently |
| `cancer_data.regional_lymph_node[*].station_name` | unstable free-text identifier; replaced by category-aggregation scoring |

Biomarker scope is restricted to the eight clinically-acted-on
categories: `{er, pr, her2, ki67}` for breast and `{msh2, msh6, pms2,
mlh1}` for colorectal. All other biomarker entries are dropped from
both gold and prediction before scoring.

## 2. Method registry

Each entry below names the public function (importable from
`digital_registrar_research.benchmarks.eval.stats`), the formula it
implements, when to use it, and the chapter CSV it lives in.

### 2.1 Confidence intervals (`stats.ci`)

| Function | Formula | When | Output column |
|---|---|---|---|
| `wilson_ci(k, n)` | Wilson score interval | proportions, `n ≥ 30` | `ci_lo`, `ci_hi` in chapter1/2/3 overall.csv |
| `clopper_pearson_ci(k, n)` | Beta-distribution exact | proportions, small `n` or extreme `p` | same as above |
| `agresti_coull_ci(k, n)` | Adjusted-Wald | sanity cross-check on Wilson | not yet in CSV |
| `bca_bootstrap_ci(values, stat)` | Bias-corrected accelerated bootstrap (Efron 1987) | F1, kappa, Lin's CCC; non-symmetric sampling distributions | per-row in nested CSVs |
| `jackknife_ci(values, stat)` | Leave-one-out, normal CI | small `n`, sanity check on bootstrap | not yet in CSV |
| `paired_bootstrap_diff(a, b)` | Paired-case resampling on `mean(a) - mean(b)` | accuracy/F1 deltas between two methods | `effect_ci_lo`, `effect_ci_hi` in `model_pair_tests/*.csv` |
| `pick_ci(kind, n, p)` | Selector | runtime CI choice for a metric cell | internal |

References: Wilson (1927); Clopper & Pearson (1934); Agresti & Coull
(1998); Efron (1982, 1987).

### 2.2 Paired hypothesis tests (`stats.paired`)

All return a unified `TestResult` dataclass.

| Function | When | Test statistic | Reviewer link |
|---|---|---|---|
| `mcnemar(a, b)` | Two paired binary outcomes (Stage A correctness, Stage C scalar) | χ²₁ with continuity, exact-binomial when `b+c<25` | R1(d) "statistical tests" |
| `paired_bootstrap_delta(a, b)` | CI on accuracy delta | non-parametric percentile | R1(d) |
| `cochran_q(matrix)` | k ≥ 3 paired binary outcomes | χ²ₖ₋₁ | R1(d) — guard against pairwise McNemar inflation |
| `stuart_maxwell(a, b)` | Two paired multi-class outcomes (Stage B 11 organs) | χ²ₖ₋₁ marginal-homogeneity | R2 §1 (anatomic vs pathologic staging) |
| `bhapkar(a, b)` | More powerful Stuart-Maxwell variant for `df > 1` | χ²ₖ₋₁ | same |
| `friedman(matrix)` | k ≥ 3 paired continuous (per-case F1 across 3 models) | χ²ₖ₋₁ | R1(d) |
| `nemenyi_posthoc(matrix)` | Pairwise post-hoc after Friedman | studentized-range CD | R1(d) |

References: McNemar (1947); Stuart (1955); Bhapkar (1966); Cochran
(1950); Friedman (1937); Nemenyi (1963); Demšar (2006).

### 2.3 Effect sizes (`stats.effect_size`)

| Function | Use | Reviewer link |
|---|---|---|
| `cohens_d` | Continuous mean difference | R1(d) |
| `cliffs_delta` | Non-parametric ordinal effect size | R1(d) |
| `cohens_kappa` | Two-rater nominal agreement, model-vs-gold | R1(b), R1(d) |
| `weighted_kappa(weights="quadratic")` | Ordinal staging concordance (T, N, M, grade) | R2 §1 |
| `krippendorff_alpha(raters, level)` | k ≥ 2 raters, missing data, mixed scale types | R1(b) — direct response |
| `matthews_corrcoef` | Class-imbalance-robust binary endpoint | R1(d) |
| `balanced_accuracy` | Mean per-class recall | R1(d) |
| `brier_score` | Probabilistic calibration | scaffolded for §4 future-work multimodal extension |

References: Cohen (1960); Fleiss & Cohen (1973); Krippendorff (2004,
2011); Brier (1950); Landis & Koch (1977) for kappa interpretation
bands; Cohen (1988) for `d`.

### 2.4 Multiple-comparisons corrections (`stats.multiple_comparisons`)

| Function | When | Reviewer link |
|---|---|---|
| `holm(p_values)` | Family-wise error within a chapter primary-endpoint family | R1(d) |
| `sidak(p_values)` | FWER cross-check; reported alongside Holm | — |
| `bh_fdr(p_values)` | FDR for the per-field × per-organ exploratory grid | — |
| `by_fdr(p_values)` | FDR for dependent tests (rows share denominators) | — |
| `permutation_omnibus(a, b)` | "Is *any* field-level delta real?" guard against cherry-picking | — |

References: Holm (1979); Šidák (1967); Benjamini & Hochberg (1995);
Benjamini & Yekutieli (2001).

### 2.5 Multi-run reliability (`stats.reliability`)

| Function | Use | Output |
|---|---|---|
| `icc_2_1(matrix)` | Single-run reliability | `chapter*_*/multirun_consistency.csv` |
| `icc_3_k(matrix)` | k-run-average reliability | same |
| `cronbach_alpha(matrix)` | Internal consistency of k runs | same |
| `spearman_brown(rho, n)` | Predicted reliability of n-run average | utility for "do we need more runs?" |
| `accuracy_flip_rate(matrix)` | Fraction of cases flipping correct↔wrong across runs | same |
| `missing_flip_rate(matrix)` | Fraction of cases with mixed attempted/missing across runs | same |
| `per_case_run_sd(matrix)` | Mean / max / 90th-percentile of per-case SD | same |

References: Shrout & Fleiss (1979); McGraw & Wong (1996); Cronbach
(1951); Spearman (1910); Brown (1910).

### 2.6 Concordance on continuous outputs (`stats.concordance`)

For LN counts (per-category aggregated `examined`/`involved`) and
`tumor_size`.

| Function | Use |
|---|---|
| `lin_ccc(gold, pred)` | Method-comparison standard for paired continuous |
| `bland_altman(gold, pred)` | Mean diff + 95% LoA |
| `mape(gold, pred)` | Mean absolute percentage error (skips zero gold) |
| `spearman_rho(gold, pred)` | Outlier-resistant rank correlation |
| `count_mae(gold, pred)` | Plain MAE on counts |

References: Lin (1989); Bland & Altman (1986); Spearman (1904).

### 2.7 Cross-cohort heterogeneity (`stats.heterogeneity`)

| Function | Use |
|---|---|
| `q_test_heterogeneity(estimates, ses)` | Cochran's Q (between-organ heterogeneity) |
| `i_squared(estimates, ses)` | Higgins-Thompson I² as a fraction in [0, 1] |
| `forest_plot_csv(rows)` | Long-form CSV with weights + pooled estimate |
| `interaction_test(long_df)` | Logistic GLMM organ × method LR test |

References: Cochran (1954); Higgins & Thompson (2002).

### 2.8 Cascade-specific diagnostics (`stats.cascade_diag`)

| Function | Output |
|---|---|
| `cascade_funnel(atomic)` | `chapter3/cascade_funnel.csv` — n_total / n_passed / n_dropped per stage |
| `conditional_accuracy_grid(atomic)` | `chapter3/conditional_accuracy_grid.csv` — `P(field_correct \| stage_b ∧ stage_a)` per field |
| `attrition_propensity(atomic, features)` | Logit per stage of "case dropped" on case-level features |

## 3. Pre-registered thresholds (`stats.thresholds`)

These constants are fixed before any cascade run is interpreted, so
the analysis cannot be steered by the data.

| Constant | Value | Meaning |
|---|---|---|
| `HEADLINE_ACCURACY_DELTA_THRESHOLD` | 0.02 | Below this, an accuracy delta is reported with CI but not claimed as a difference |
| `HEADLINE_F1_DELTA_THRESHOLD` | 0.03 | Same for F1 |
| `KAPPA_SUBSTANTIAL` | 0.80 | Landis-Koch threshold for "substantial" agreement |
| `KAPPA_NEAR_PERFECT` | 0.90 | Internal threshold for "near-perfect" claim |
| `FAMILYWISE_ALPHA` | 0.05 | Holm correction within each chapter primary-endpoint family |
| `FDR_ALPHA` | 0.10 | FDR control on the per-field × per-organ exploratory grid |
| `POWER_FLAG_THRESHOLD` | 0.30 | Tests with post-hoc power below this are flagged `low_power=True` |

## 4. Reproducibility manifest

To regenerate every CSV in the cascade output:

```bash
# Single-model run on cmuh.
python -m scripts.eval.cli cascade \
    --root workspace --dataset cmuh --annotator gold \
    --method llm --model gpt_oss_20b --run-ids run01 run02 \
    --out workspace/results/eval/cascade/cmuh

# External validation on TCGA.
python -m scripts.eval.cli cascade \
    --root workspace --dataset tcga --annotator gold \
    --method llm --model gpt_oss_20b --run-ids run01 \
    --out workspace/results/eval/cascade/tcga

# Three-model paired tests (cross-model significance).
# Run cascade three times (one per model), then run the ablation
# canonical_stats with --modular-method gpt_oss_20b for paired comparisons.
```

Seed: pass `--seed 0` (default) for deterministic bootstrap.
Library versions: pinned via `pyproject.toml` (statsmodels ≥ 0.14,
scipy ≥ 1.10, scikit-learn ≥ 1.5).
Hardware: bootstrap CIs run on CPU by default; pass `--device cuda`
for the GPU-accelerated path.

## 5. Mapping of methods to reviewer comments

| Reviewer comment (verbatim excerpt) | Method that addresses it | Output CSV |
|---|---|---|
| R1(b) "incorporating independently annotated samples or reporting inter-annotator agreement metrics (e.g., Cohen's kappa)" | `cohens_kappa`, `krippendorff_alpha` | `chapter1_eligibility/agreement_kappas.csv`, IAA subcommand outputs |
| R1(c) "contribution of individual components ... ablation studies" | In-tree ablation harness now consumes the cascade atomic; `canonical_stats.modularity_advantage` | `results/ablations/<dataset>/canonical_stats/modularity_advantage.csv` |
| R1(d) "confidence intervals" | `wilson_ci`, `clopper_pearson_ci`, `bca_bootstrap_ci`, `bland_altman` | every chapter overall.csv |
| R1(d) "statistical tests" | `mcnemar`, `cochran_q`, `stuart_maxwell`, `friedman` | `model_pair_tests/*.csv` |
| R1(d) "robustness analyses across multiple runs" | `icc_2_1`, `icc_3_k`, `cronbach_alpha`, `accuracy_flip_rate` | each chapter `multirun_consistency.csv` |
| R2 §1 "anatomic vs pathologic staging" | `weighted_kappa` (quadratic) on T/N/M; Stuart-Maxwell on stage_group classifications | `chapter3/staging_weighted_kappa.csv` |
| R2 §2 "double primary malignancies" | "Others" ledger with `dual_primary` regex classifier | `chapter2/others/others_ledger.csv` and `others_subtype_breakdown.csv` |

## 6. References

- Agresti, A., & Coull, B. A. (1998). *The American Statistician*.
- Benjamini, Y., & Hochberg, Y. (1995). *JRSS-B*.
- Benjamini, Y., & Yekutieli, D. (2001). *Annals of Statistics*.
- Bhapkar, V. P. (1966). *JASA*.
- Bland, J. M., & Altman, D. G. (1986). *Lancet*.
- Brier, G. W. (1950). *Monthly Weather Review*.
- Brown, W. (1910). *British Journal of Psychology*.
- Cliff, N. (1993). *Psychological Bulletin*.
- Clopper, C. J., & Pearson, E. S. (1934). *Biometrika*.
- Cochran, W. G. (1950). *Biometrika*.
- Cochran, W. G. (1954). *Biometrics*.
- Cohen, J. (1960). *Educational and Psychological Measurement*.
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*, 2nd ed.
- Cronbach, L. J. (1951). *Psychometrika*.
- Demšar, J. (2006). *JMLR*.
- Efron, B. (1982). *The Jackknife, the Bootstrap, and Other Resampling Plans*. CBMS-NSF.
- Efron, B. (1987). *JASA*.
- Fleiss, J. L., & Cohen, J. (1973). *Educational and Psychological Measurement*.
- Friedman, M. (1937). *JASA*.
- Higgins, J. P. T., & Thompson, S. G. (2002). *Statistics in Medicine*.
- Holm, S. (1979). *Scandinavian Journal of Statistics*.
- Krippendorff, K. (2004). *Content Analysis*. Sage.
- Krippendorff, K. (2011). "Computing Krippendorff's α-reliability."
- Landis, J. R., & Koch, G. G. (1977). *Biometrics*.
- Lin, L. I.-K. (1989). *Biometrics*.
- Matthews, B. W. (1975). *BBA — Protein Structure*.
- McGraw, K. O., & Wong, S. P. (1996). *Psychological Methods*.
- McNemar, Q. (1947). *Psychometrika*.
- Nemenyi, P. (1963). PhD thesis, Princeton.
- Šidák, Z. (1967). *JASA*.
- Shrout, P. E., & Fleiss, J. L. (1979). *Psychological Bulletin*.
- Spearman, C. (1904). *American Journal of Psychology*.
- Spearman, C. (1910). *British Journal of Psychology*.
- Stuart, A. (1955). *Biometrika*.
- Wilson, E. B. (1927). *JASA*.
