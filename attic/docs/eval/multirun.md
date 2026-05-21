# Multi-run statistics

Multiple runs of the same model on the same case set let us measure both **point accuracy** and **run-to-run reliability**. The latter is critical for the writeup: a model that scores 88% on a single run might swing between 82% and 94% across runs, or it might be stable to within ± 1pp. The reader needs to know which.

## ELI5

If you flip a coin three times and it comes up heads, heads, heads, you can't conclude the coin is biased — three is too few flips. The same caution applies to LLM evaluation: **a single run is one sample of a stochastic process**, and conclusions should be drawn from a distribution of runs, not from one realisation. We default to 3 runs per model with different seeds; the eval pipeline aggregates across them.

## Three-tier confidence interval

For each (field, organ) we report **three** confidence intervals on accuracy:

1. **Case CI** — bootstrap resampling cases (within a run / pooled). Captures variance from "different patients give different difficulties."
2. **Run CI** — Student-t over the per-run accuracy vector. Captures variance from "different stochastic realisations of the model give different answers."
3. **Total CI** — GLMM with random intercepts for both `case_id` and `run_id`, or two-source bootstrap as fallback. Captures **both** sources of variance simultaneously — the most honest CI to cite in the headline.

In the writeup, the *total CI* is what answers "if I redeploy this model on a new patient cohort, what range of accuracy should I expect?" — the run CI alone underestimates the answer because it ignores case-level noise.

**Implementation:**
- Case CI via `digital_registrar_research.benchmarks.eval.ci.bootstrap_ci(method="bca")` (Efron 1987).
- Run CI via `digital_registrar_research.benchmarks.eval.ci.t_ci`, wrapping `scipy.stats.t.interval`.
- Total CI via `digital_registrar_research.benchmarks.eval.multirun._glmm_marginal_accuracy` using `statsmodels.genmod.bayes_mixed_glm.BinomialBayesMixedGLM` (Seabold & Perktold, 2010); falls back to `ci.two_source_bootstrap_ci` when the GLMM fails.

**Output columns** in `non_nested/per_field_overall.csv`:
- `attempted_acc_boot_lo/hi` — case CI on attempted accuracy.
- `mean_per_run_t_ci_lo/hi` — run CI on the mean-per-run vector.
- (Total CI is added to the multi-run consistency table when applicable; see [ci_methods.md](ci_methods.md).)

## Fleiss' κ on correctness

Treat each run as a "rater" producing a 0/1 correctness label per case. Fleiss' κ measures whether the runs agree more often than chance.

- **κ ≈ 1**: deterministic — every run gets the same answer.
- **κ ≈ 0**: random — runs are uncorrelated.
- **κ < 0**: anti-correlated — possible but rare in practice (suggests the runs are *systematically* disagreeing, e.g. seed-dependent prompt sensitivity).

**Implementation:** `multirun.fleiss_kappa` (in-house). Reference: Fleiss (1971).
**Output:** `non_nested/run_consistency.csv` (`fleiss_kappa_correctness`).

## Fleiss' κ on prediction values

Same as above, but on the *raw predicted value* rather than the binary correctness flag. High value-κ + low correctness-κ would mean "the runs all give the same answer, and they're all consistently wrong" — useful diagnostic. Categorical fields with too many distinct values (> 50) skip this metric.

**Output:** `non_nested/run_consistency.csv` (`fleiss_kappa_values`).

## Flip rate

Fraction of cases where not all runs agreed on correctness. Complementary to Fleiss' κ — flip rate is interpretable on its own (a number you can quote: "20% of cases had at least one run disagree").

**Output:** `non_nested/run_consistency.csv` (`flip_rate`).

## Missing-flip rate

Fraction of cases where ≥ 1 run had the field missing AND ≥ 1 run had it attempted. Distinct from value-flip rate — captures **whether the model decides to skip** consistently across runs.

**Output:** `non_nested/run_consistency.csv` (`missing_flip_rate`).

## Stability accuracy

Accuracy on cases where **all runs agreed** (didn't flip). Typically substantially higher than overall accuracy. The gap is informative: if stability accuracy is 95% but overall accuracy is 80%, then the 15pp shortfall is entirely on the brittle cases.

**Output:** `non_nested/run_consistency.csv` (`stability_accuracy`).

## Brittle case rate

Fraction of cases where ≥ 1 run got it right AND ≥ 1 run got it wrong. The model "knows" but is unreliable on these. High brittle-case rate is a deployment-readiness concern even if mean accuracy is acceptable.

**Output:** `non_nested/run_consistency.csv` (`brittle_case_rate`).

## Vote-concentration calibration

For ensemble-style consumption: when 3 of 3 runs agree on an answer, what's the accuracy of that consensus? When 2 of 3 agree? Reports accuracy stratified by vote concentration. Equivalent to a calibration curve for "majority-vote confidence."

**Implementation:** `multirun.run_consistency` extension. (Currently a planned addition; the foundational columns are already in the atomic table.)

## Majority-vote ensemble

For each case, take the most-common prediction across runs. Ensemble outputs are written to `<out>/ensemble_predictions/<organ_idx>/<case>.json` and re-scored as a synthetic single run.

- **Scalars:** plurality vote; ties broken by first-seen.
- **Continuous (int) fields:** median.
- **Nested lists:** union by primary key, requiring majority support (≥ ⌈K/2⌉ runs) before including an item.

**Implementation:** `digital_registrar_research.benchmarks.eval.multirun.majority_vote_ensemble`.
**Output:** `ensemble_predictions/`, `ensemble_by_case.csv`.

## Ensemble-vs-single Δ

Paired-bootstrap Δ between ensemble accuracy and the per-case mean-across-runs single accuracy. Positive Δ means the ensemble is doing real work (recovering majority-correct answers from minority-wrong runs).

**Implementation:** `multirun.ensemble_vs_single` using `ci.paired_bootstrap_diff`.
**Output:** `ensemble_vs_single.csv` (per-field Δ + 95% CI).

## What to do if your numbers are bad

- **High flip_rate (> 30%):** the model is non-deterministic in a way the temperature setting probably hides. Lower temperature, fix seed, or accept that this is a stochastic system and lean on the ensemble.
- **Stability accuracy ≫ overall accuracy:** the brittle cases are dragging the headline down. Inspect them — they may be inherently ambiguous (in which case `accuracy_by_difficulty_tier.csv` will show low human-IAA), or systematically hard for the model.
- **High missing_flip_rate:** the model "decides" inconsistently whether to fill a field. Different runs see the same prompt but produce different completion-vs-skip choices. Suggests prompt is borderline; clarify what should be skipped.
- **Negative Fleiss κ:** runs are systematically *disagreeing* — almost certainly a bug (different prompts, different seeds in unexpected ways, decoding mode flapping). Investigate before reporting other multi-run numbers.

## References

Foundational citations are in [methods_citations.md](methods_citations.md). The most directly relevant:

- Fleiss' κ: Fleiss (1971).
- BCa bootstrap: Efron (1987).
- McNemar: McNemar (1947).
- GLMM with random intercepts: Laird & Ware (1982); Bayesian variational fit per `BinomialBayesMixedGLM`.
- Two-source bootstrap: Owen, A. B. (2007). "The pigeonhole bootstrap." *Annals of Applied Statistics* 1 (2): 386–411.
