# Inter-annotator agreement basics

## ELI5

Two doctors annotate the same 100 reports. They agree on 85. Is that good? Bad?

If both doctors picked "no cancer" 80% of the time *by chance*, you'd expect them to agree on `0.8² + 0.2² = 68%` of cases just by coincidence. So 85% agreement is only **17 percentage points better than chance**. A *chance-corrected* agreement metric like Cohen's κ tells you that signal directly.

## Cohen's κ — paired raters

Two raters, possibly with different label distributions. Subtracts chance agreement.

- **Unweighted κ** — for nominal categoricals (e.g. `cancer_category`). Treats off-diagonal cells equally.
- **Quadratic-weighted κ** — for ordinals (`grade`, `pt_category`). "Off by one" gets less penalty than "off by three." Use this whenever the field has a natural ordering.

**Implementation:** `sklearn.metrics.cohen_kappa_score(weights=None|"quadratic", labels=ordinal_order)` (Pedregosa et al., 2011); also `digital_registrar_research.benchmarks.eval.iaa.cohen_kappa`.
**References:** Cohen, J. (1960, 1968). *(see [methods_citations.md](methods_citations.md))*
**Output:** `iaa/pair_*.csv` rows with `stat_name = cohen_kappa_unweighted` or `cohen_kappa_quadratic`.

### Interpretation guide (Landis & Koch 1977)

| κ | Interpretation |
|---|---|
| < 0 | Worse than chance |
| 0–0.2 | Slight |
| 0.21–0.4 | Fair |
| 0.41–0.6 | Moderate |
| 0.61–0.8 | Substantial |
| 0.81–1.0 | Almost perfect |

Reference: Landis, J. R., & Koch, G. G. (1977). "The measurement of observer agreement for categorical data." *Biometrics* 33 (1): 159–174.

## Fleiss' κ — three or more raters

Generalisation of Cohen's κ to N raters per item. Used internally for multi-run consistency analysis (treating each run as a "rater").

**Implementation:** `digital_registrar_research.benchmarks.eval.multirun.fleiss_kappa` (in-house). Library equivalent: `statsmodels.stats.inter_rater.fleiss_kappa`.
**Reference:** Fleiss, J. L. (1971). "Measuring nominal scale agreement among many raters." *Psychological Bulletin* 76 (5): 378–382.
**Output:** `non_nested/run_consistency.csv` (`fleiss_kappa_correctness`, `fleiss_kappa_values`).

## Krippendorff's α — flexible across types

Handles nominal / ordinal / interval data, missing values, any number of raters. The most flexible agreement coefficient. We report:

- `krippendorff_alpha_nominal` — for unordered categoricals.
- `krippendorff_alpha_ordinal` — for ordinals.
- `krippendorff_alpha_interval` — for continuous values.

**Implementation:** `digital_registrar_research.benchmarks.eval.iaa.krippendorff_alpha` (in-house). Library equivalent: `krippendorff.alpha` from the `krippendorff` PyPI package.
**Reference:** Krippendorff, K. (2004). *Content Analysis: An Introduction to Its Methodology* (2nd ed.). Sage. Ch. 11.
**Output:** `iaa/whole_report.csv`.

## Lin's CCC — continuous agreement

For numeric fields like `tumor_size`. Combines correlation (do the values track each other?) with mean shift (are they on the same scale?). Strictly stricter than Pearson r — two raters can correlate perfectly but disagree by a constant offset, in which case Pearson = 1 but CCC < 1.

**Implementation:** `iaa.lins_ccc` (in-house — no canonical library equivalent in scipy/sklearn). `pingouin.concordance_corr` is an alternative.
**Reference:** Lin, L. I.-K. (1989). "A concordance correlation coefficient to evaluate reproducibility." *Biometrics* 45 (1): 255–268.
**Output:** Continuous-field rows in `iaa/pair_*.csv` (`stat_name = lins_ccc`).

## ICC(2,1) — random-effects intraclass correlation

For continuous fields with two raters, "absolute agreement" formulation. Equivalent to CCC under standard assumptions but motivated from a variance-components perspective.

**Implementation:** `iaa.icc_2_1` (in-house). Library: `pingouin.intraclass_corr` filtered to `ICC2`.
**Reference:** Shrout, P. E., & Fleiss, J. L. (1979). "Intraclass correlations: uses in assessing rater reliability." *Psychological Bulletin* 86 (2): 420–428.

## Bland-Altman — bias and limits of agreement

For continuous fields. Reports:

- **Bias** = mean(rater_a − rater_b) — systematic difference.
- **Limits of agreement** = bias ± 1.96 · SD(differences) — the range within which 95% of inter-rater differences fall.

Useful complement to CCC because it gives a **directional** signal (is rater A systematically higher than rater B?).

**Implementation:** `iaa.bland_altman`.
**Reference:** Bland, J. M., & Altman, D. G. (1986). "Statistical methods for assessing agreement between two methods of clinical measurement." *The Lancet* 327 (8476): 307–310.

## Observed agreement (raw)

Just `n_agree / n_total`. Useful as a sanity check alongside κ — if observed agreement is high but κ is low, you have class-imbalance-driven inflation. Always report both.

**Implementation:** `iaa.observed_agreement` (one-line formula).
**Output:** Every `iaa/pair_*.csv` row carries `observed_agreement` alongside `estimate`.

## PABAK — Prevalence- and bias-adjusted κ

For binary fields. PABAK = `2 · p_observed − 1`. Useful when class imbalance is so severe that κ becomes pathologically low despite high agreement (the "kappa paradox"). Don't replace κ with PABAK — *report both*.

**Reference:** Byrt, T., Bishop, J., & Carlin, J. B. (1993). "Bias, prevalence and kappa." *Journal of Clinical Epidemiology* 46 (5): 423–429.

## Kendall's τ-b — ordinal correlation

Robust ordinal correlation handling ties. Reported alongside quadratic κ for ordinal fields.

**Implementation:** `scipy.stats.kendalltau(variant="b")`. CI via Fisher-z transform.
**Reference:** Kendall, M. G. (1948). *Rank Correlation Methods*. Griffin.

## McNemar's test — paired binary

Tests whether two raters' disagreements are *symmetric*. Asymmetric disagreement (rater A says yes when B says no much more often than the reverse) suggests systematic bias.

**Implementation:** `digital_registrar_research.benchmarks.eval.ci.mcnemar_test` (in-house, with continuity correction and exact binomial fallback for small n). Library equivalent: `statsmodels.stats.contingency_tables.mcnemar`.
**Reference:** McNemar, Q. (1947). "Note on the sampling error of the difference between correlated proportions or percentages." *Psychometrika* 12 (2): 153–157.

## Coverage κ — agreement on attempted-vs-null

A separate κ on the binary indicator `is_attempted(annotator, field)`. Tells you whether the two annotators agree on *which* fields are documented in the report, separately from agreement on the *values*. High coverage κ + low value κ = "they're looking at the same fields but disagreeing on what they say"; low coverage κ = "they're not even looking at the same fields."

**Output:** Coverage κ rows in `iaa/pair_*.csv` (`stat_name = coverage_kappa`).

## Reading `iaa/pair_*.csv`

Long-form: one row per `(organ, field, stat_name)`. Columns: `n` (number of paired observations), `estimate`, `ci_lo`, `ci_hi`, `observed_agreement`, `n_categories`. Fields without enough variance (degenerate case) get NaN — these are filtered when consuming downstream.

## Whole-report headline

`iaa/whole_report.csv` reports:

- Case-level exact-match rate (fraction of cases where every field agreed across both annotators).
- Mean field accuracy (averaged across fields in the case).
- Per-section mean κ (top-level / scalar pathology / nested-list).
- Krippendorff α per measurement level.

One row per statistic per pair.

## Pair-focused headline (`iaa_pair`)

`iaa/whole_report.csv` gives Krippendorff α per type bucket but **no single overall Cohen's κ** for the pair you actually care about. The `iaa_pair` subcommand fills that gap. It scopes a comprehensive report to one or more requested annotator pairs and emits four headline κ flavours side-by-side, plus per-section / per-organ roll-ups, confusion matrices, and a markdown summary.

### When to use it

- "What's the overall κ between gold and `nhc_with_preann`?" → use `iaa_pair`.
- "I want a confusion matrix for `pt_category` between two annotators." → use `iaa_pair`.
- "I want everything `iaa` already produces, but for arbitrary pairs." → use `iaa`. `iaa_pair` is *additive* — it doesn't replace the per-field stats, it adds the missing summary layer.

### Why four flavours of κ?

There is no canonical pooled-κ for heterogeneous fields (binary + ordinal + nominal + continuous + nested-list). Different definitions answer different questions, so we report all four:

| Statistic | What it answers | Caveats |
|---|---|---|
| `mean_per_field_kappa` | "On average, how much do they agree per field?" | Equal weight to every field. CI omitted (would require field_count × n_boot recomputations). |
| `n_weighted_mean_per_field_kappa` | Same, but fields with more observations contribute more. | Same CI caveat. Closer to a "per-observation" interpretation. |
| `pooled_categorical_kappa` | One Cohen's κ over every (case, field) tuple, restricted to categorical fields (binary / ordinal / nominal). Missing-on-one-side encoded as `__missing__`. | Continuous and nested-list fields excluded — their value spaces don't share a κ category structure. Case-level bootstrap CI included. |
| `agree_disagree_pabak` | Single chance-corrected agreement rate across **all** fields, including continuous (within tolerance) and nested-list (bipartite F1 = 1.0). PABAK = 2·p_o − 1. | Loses information about *which* category the disagreements fall in. Use as a complement, not a substitute. |

`iaa_pair` also forwards Krippendorff α (nominal / ordinal / interval) and the case-exact-match rate from `iaa.whole_report_stats`, so the headline table is self-contained.

### Why PABAK and not Cohen's κ for agree/disagree?

If you build the (case, field) → agree(1)/disagree(0) indicator and try to compute Cohen's κ on it, one "rater" is constant (always 1 by construction). The chance baseline `p_e` then equals `p_o`, so κ collapses to 0 regardless of agreement rate. **PABAK** = 2·p_o − 1 is the principled chance-corrected statistic in this setting (range [−1, +1]; 0 = chance-50/50, +1 = perfect agreement). See Byrt et al. (1993) cited above under PABAK.

### Bootstrap unit

**Case** (with `strata = organ_per_case`). Multiple (case, field) observations from the same source case move together under a single bootstrap draw — this respects within-case correlation across fields and matches the standard for IAA bootstraps.

### Pre-annotation effect — same numbers, different reading

`iaa_pair` is **pair-agnostic**: it answers *"how different are these two streams?"* For inter-observer pairs (`gold:nhc_with_preann`, `kpc_with_preann:nhc_with_preann`) that's the right question.

For within-annotator preann pairs (`kpc_with_preann:kpc_without_preann`), the same κ measures the *magnitude of preann-induced change* rather than whether preann helps or hurts. Two readings:

- High κ: either preann had little effect, **or** the human rubber-stamped suggestions (anchoring risk).
- Low κ: either the human read carefully and overrode the LLM, **or** the LLM was so bad it had to be fixed everywhere.

The valence is **ambiguous** — the causal "does preann pull the annotator toward gold?" question requires Δκ vs gold + anchoring index, which live in the existing `iaa` subcommand's `preann/` outputs. The markdown summary auto-detects within-annotator pairs (matching `nhc_*` or `kpc_*` prefix on both sides) and shows a banner pointing readers to that output tree.

### Output tree

```
iaa_pair/
    manifest.json
    pair_<a>_vs_<b>/
        headline.csv          — 8 rows: 4 κ flavours + 3 α + case_exact_match
        per_field_kappa.csv   — long-form, one row per (organ × field × κ-stat)
        per_section.csv       — section roll-up
        per_organ.csv         — 4 stats × n_organs rows
        summary.md            — human-readable headline
        confusion/<field>.csv — top-N categorical fields by disagreement
```

### CLI

```
python -m scripts.eval.cli iaa_pair \
    --root workspace --dataset cmuh \
    --pair gold:nhc_with_preann \
    --pair kpc_with_preann:nhc_with_preann \
    --pair kpc_with_preann:kpc_without_preann \
    --out workspace/results/eval/iaa_pair \
    --n-boot 2000 --seed 0
```

`--pair` is repeatable. Both sides must be valid annotators (`gold`, `nhc_with_preann`, `nhc_without_preann`, `kpc_with_preann`, `kpc_without_preann`). Self-pairs (`a:a`) are rejected; duplicates are dropped with a warning; both `a:b` and `b:a` are kept when both are requested (confusion-matrix orientation differs).

Optional flags:
- `--n-confusion N` — top-N categorical fields by disagreement count to write as confusion matrices (default 20).
- `--top-fields K` — top-K most-disagreed fields shown in the markdown summary (default 10).

**Implementation:** `scripts.eval.iaa.run_iaa_pair`; library at `digital_registrar_research.benchmarks.eval.iaa_headline`.

## Disagreement resolution dynamics

When NHC and KPC disagree, gold acts as the tie-breaker. `iaa/disagreement_resolution.csv` reports, per (organ, field):

- Total disagreements between NHC and KPC.
- Of those, how many gold matched NHC vs KPC.
- Chi-square test on the asymmetry.

A significantly asymmetric distribution suggests one annotator is more closely aligned with the gold-setting consensus than the other — useful annotator-quality signal.

**Implementation:** `iaa.disagreement_resolution` plus `scipy.stats.chi2_contingency`.

## What to do if your numbers are bad

- **κ << observed agreement:** class imbalance. Report both. Consider PABAK for binary fields.
- **Wide CI on κ:** small n. The bootstrap CI accounts for this honestly — don't try to narrow it.
- **CCC much lower than Pearson r:** systematic offset between raters. Report Bland-Altman bias to make the offset visible.
- **Coverage κ low but value κ high:** annotators don't agree on what to fill in but agree when they fill. Likely a guideline ambiguity issue, not a quality issue.
- **High disagreement_resolution chi-square p-value (close to 0):** asymmetric resolution. Lift into Discussion.

For the headline pre-annotation analysis, see [preann_effect.md](preann_effect.md).
