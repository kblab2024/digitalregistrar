# Confidence interval methods

Why we report CIs the way we do, when each variant is appropriate, and where they live in the code.

## ELI5

A point estimate ("accuracy = 0.83") tells you nothing about how reliable that number is. A confidence interval ("accuracy = 0.83, 95% CI [0.78, 0.87]") tells you the range of plausible true values consistent with what you observed. Different statistics need different CI methods because the underlying sampling distribution is different.

## Wilson score interval — for proportions

The default for any binary rate (accuracy, parse_error_rate, attempted_rate, ...). Better than the textbook normal-approximation CI especially at small n or near 0/1. Doesn't go below 0 or above 1.

**Use when:** you have count `k` correct out of `n` total, and want a CI on the proportion.

**Implementation:** `digital_registrar_research.benchmarks.eval.ci.wilson_ci(k, n, alpha)` (in-house, ~10 LOC). Library equivalent: `statsmodels.stats.proportion.proportion_confint(method="wilson")`.
**Reference:** Wilson, E. B. (1927). "Probable inference, the law of succession, and statistical inference." *JASA* 22 (158): 209–212.

## Clopper-Pearson exact CI — for proportions, conservative

Exact binomial CI. Strict coverage guarantee but wider than Wilson. Use when n is small AND coverage matters more than width (e.g. safety-relevant rare-event rates).

**Implementation:** `ci.clopper_pearson_ci`. Library: `statsmodels.stats.proportion.proportion_confint(method="beta")`.
**Reference:** Clopper, C., & Pearson, E. S. (1934). "The use of confidence or fiducial limits illustrated in the case of the binomial." *Biometrika* 26 (4): 404–413.

## BCa bootstrap — for any statistic on case-level data

The general-purpose CI. Resample cases with replacement, recompute the statistic on each resample, take the (bias-corrected, accelerated) percentile interval. Asymmetric — the lower and upper bounds aren't equidistant from the point estimate, which correctly reflects skewed distributions.

**Use when:** the statistic isn't a simple proportion (e.g. F1, mean rank distance, paired Δ, anything compound).

**Implementation:** `ci.bootstrap_ci(method="bca", n_boot=2000, ...)` (in-house). Library equivalent: `scipy.stats.bootstrap(method="BCa")` (SciPy ≥ 1.7).
**Reference:** Efron, B. (1987). "Better bootstrap confidence intervals." *JASA* 82 (397): 171–185.

The BCa version uses **bias correction** (`z0`, the proportion of bootstrap samples below the point estimate) and **acceleration** (`a_hat`, jackknife-derived skewness adjustment). When acceleration blows up (rare, degenerate), we degrade to **percentile bootstrap** silently.

## Paired bootstrap Δ — for paired comparisons

When comparing two methods (or two annotators) **on the same cases**, the paired structure must be preserved during resampling. Otherwise the CI is too wide because it ignores the case-level correlation between the two arms.

Each bootstrap draw picks indices `i ∈ [0, n)` with replacement, and uses the same indices for both arms before computing the difference of means.

**Implementation:** `ci.paired_bootstrap_diff` (in-house — `scipy.stats.bootstrap` doesn't directly support paired Δ).
**Used in:** ensemble-vs-single Δ, preann Δκ, disagreement reduction Δ, cross-method McNemar deltas.

## Two-source nested bootstrap — for case × run designs

Multi-run designs have **two sources of variability**: cases and runs. A simple bootstrap over cases (or over runs) underestimates the total CI because it ignores one source.

The nested approach: outer loop resamples cases with replacement, inner loop resamples runs with replacement, then compute the statistic on the (n_cases × n_runs) resampled matrix. The CI on the bootstrap distribution captures both sources simultaneously.

**Use when:** the GLMM mixed-effects model fails (degenerate, all-correct, all-incorrect, singular gradient).

**Implementation:** `ci.two_source_bootstrap_ci`.
**Reference:** Adapted from Owen, A. B. (2007). "The pigeonhole bootstrap." *Annals of Applied Statistics* 1 (2): 386–411.

## GLMM marginal accuracy — primary multi-run CI

For multi-run accuracy (binary correctness across cases × runs), fit a Bayesian binomial mixed-effects model with random intercepts for `case_id` and `run_id`:

```
correct_int ~ 1  with  vc = {case_id, run_id}
```

The fitted intercept (after inverse-logit) is the marginal accuracy; ± 1.96 · SE on the intercept gives the total CI. Variance components for case and run are extracted from the posterior.

**Use when:** you have ≥ 3 runs, ≥ 30 cases, and the outcome isn't all 0 or all 1 (the model needs variance to fit).

**Implementation:** `multirun._glmm_marginal_accuracy` via `statsmodels.genmod.bayes_mixed_glm.BinomialBayesMixedGLM` with variational inference. Falls back to two-source bootstrap on convergence failures.
**Reference:** Laird, N. M., & Ware, J. H. (1982). "Random-effects models for longitudinal data." *Biometrics* 38 (4): 963–974.

## Student-t CI — for the run-level mean

When you have K per-run accuracy values (one per run), the Student-t CI on their mean tells you "if I drew another run, what's the plausible accuracy I'd see?" This is the **run-level reliability** answer, complementary to the case-level answer.

**Use when:** K ≥ 2 runs. Width scales as `t(α/2, K-1) · SD / √K`.

**Implementation:** `ci.t_ci`, wrapping `scipy.stats.t.interval`.
**Reference:** Student (1908). "The probable error of a mean." *Biometrika* 6 (1): 1–25.

## Fisher-z CI for correlations

For Pearson, Spearman, Kendall correlations. The Fisher-z transform `0.5 · log((1+r)/(1−r))` makes the sampling distribution approximately normal so you can apply a `±z · 1/√(n−3)` CI.

**Implementation:** `ci.fisher_z_ci_for_corr`.
**Reference:** Fisher, R. A. (1915). "Frequency distribution of the values of the correlation coefficient in samples from an indefinitely large population." *Biometrika* 10 (4): 507–521.

## Choosing between methods — quick reference

| Statistic | Best CI |
|---|---|
| Binary proportion (accuracy, attempted_rate) | Wilson |
| Binary proportion, very small n, conservative | Clopper-Pearson |
| F1, MAE, rank distance, Lin's CCC, anything else | BCa bootstrap |
| Paired Δ (method A vs method B on same cases) | Paired bootstrap |
| Multi-run accuracy across cases AND runs | GLMM (fallback: two-source bootstrap) |
| Run-level mean accuracy across K runs | Student-t |
| Pearson / Spearman / Kendall correlation | Fisher-z |

## Why do we report multiple CIs?

The same accuracy point estimate can have very different CIs depending on which source of variance you account for. **A single CI is a half-truth.** The eval pipeline reports up to three flavors side by side:

- **case_ci** — across patients, holding runs constant.
- **run_ci** — across runs, averaged over patients.
- **total_ci** — both sources.

The total CI is the headline. The other two are decomposition that helps you **diagnose** whether width is driven by case difficulty, run noise, or both.

## What to do if your numbers are bad

- **CI ≈ point estimate (zero width):** degenerate sample (e.g. all correct). The Wilson CI handles this gracefully (returns [k/n, k/n]); BCa returns the point.
- **Wide CI (> 0.2):** small n. Either get more cases or accept the uncertainty honestly.
- **CI lo < 0 or hi > 1:** check that you used Wilson / Clopper-Pearson rather than the normal approximation. Wilson clamps to [0, 1].
- **GLMM doesn't converge:** the fallback two-source bootstrap kicks in automatically. Look for `ci_method = two_source_bootstrap` in the output to confirm.

## Determinism

All CI methods accept a `random_state` (default 0) so reruns are bit-identical. The manifest captures the seed alongside `n_boot` so paper claims are reproducible.
