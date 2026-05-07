# Multiple-comparisons correction

With ~30 fields × 10 organs × multiple methods, the eval suite produces hundreds of p-values. Without correction, ~5% will appear "significant" by chance alone — the standard statistical hygiene is multiple-comparisons control, applied consistently and separated by endpoint tier so the headline findings don't get diluted by exploratory ones.

## ELI5

Roll a die 100 times. Each individual roll has a 1/6 chance of landing on a 6 — but across 100 rolls, you'd be shocked *not* to see at least one 6. P-values work the same way: testing 100 hypotheses each at α = 0.05 will yield ~5 "significant" results purely by chance, even if all the null hypotheses are true.

Multiple-comparisons correction is the standard statistical hygiene for this. We apply it consistently across the eval pipeline, separated by **endpoint tier** so we don't dilute the primary findings.

## Endpoint pre-registration

`configs/eval_endpoints.yaml` lists fields as `primary` (the headline 5–10 fields the paper makes specific claims about) or `secondary` (everything else, exploratory). Multiple-comparisons correction applies **separately within each tier**.

Today's primary endpoints:

```
cancer_excision_report, cancer_category,
pt_category, pn_category, pm_category,
tumor_size, grade,
lymphovascular_invasion, perineural_invasion,
biomarker_er, biomarker_pr, biomarker_her2
```

Adding a field to `primary` is a **commitment** — the paper must defend why that field deserves headline scrutiny. Adding to `secondary` is cheap; secondary endpoints are reported with FDR control for transparency.

## Holm-Bonferroni — for primary endpoints

Strictest control: family-wise error rate (FWER). Probability of *any* false positive across the family is bounded by α.

For sorted p-values `p_(1) ≤ p_(2) ≤ ... ≤ p_(m)`:

```
adjusted_p_(i) = max{ (m − i + 1) · p_(i),  adjusted_p_(i−1) }
```

Less conservative than vanilla Bonferroni (which multiplies every p-value by m); strictly more powerful with the same FWER guarantee.

**Use when:** you want to make a specific claim about a small set of headline endpoints. FWER control is the standard for "the paper's main result" — it bounds the probability of *any* false positive across the family.

**Implementation:** `statsmodels.stats.multitest.multipletests(method="holm")` (Seabold & Perktold, 2010), wrapped by `_common/stats_extra.py:adjust_pvalues`.
**Reference:** Holm, S. (1979). "A simple sequentially rejective multiple test procedure." *Scandinavian Journal of Statistics* 6 (2): 65–70.

## Benjamini-Hochberg (FDR) — for secondary endpoints

Controls **expected** false discovery rate, not FWER. Among the rejections you make, no more than (in expectation) `q · 100%` will be false positives. For exploratory analysis with many tests, FDR is the standard.

For sorted p-values:

```
adjusted_p_(i) = (m / i) · p_(i)
```

(With monotone enforcement so adjusted p-values don't decrease.)

**Use when:** you have many tests and want to surface "anything interesting" without a strict guarantee on individual claims.

**Implementation:** `statsmodels.stats.multitest.multipletests(method="fdr_bh")`.
**Reference:** Benjamini, Y., & Hochberg, Y. (1995). "Controlling the false discovery rate: a practical and powerful approach to multiple testing." *JRSS B* 57 (1): 289–300.

## Bonferroni — kept available, rarely used

The vanilla `α / m` correction. Strictly conservative; powerless when m is large. Kept available for contexts that explicitly require Bonferroni; otherwise prefer Holm, which is uniformly more powerful with the same FWER guarantee.

**Implementation:** `multipletests(method="bonferroni")`.

## Effect sizes — report alongside p-values

A statistically significant effect can be substantively trivial. Effect sizes quantify practical importance. We report:

### Cohen's d — continuous comparisons

`d = (mean_a − mean_b) / pooled_SD`. Conventional thresholds: 0.2 small, 0.5 medium, 0.8 large.

**Implementation:** `_common/stats_extra.py:cohens_d` (in-house pooled-SD formula). Library equivalent: `pingouin.compute_effsize(eftype="cohen")`.
**Reference:** Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum.

### Odds ratio with Wald CI — binary comparisons

For paired McNemar-style tables. Haldane-Anscombe correction for zero cells.

**Implementation:** `_common/stats_extra.py:odds_ratio_with_ci` (in-house). Library equivalent: `scipy.stats.contingency.odds_ratio` (SciPy ≥ 1.10).

### Cliff's δ — non-parametric, ordinal

Distribution-free dominance statistic in [-1, 1]. Conventional thresholds (Romano et al. 2006): |δ| < 0.147 negligible, < 0.33 small, < 0.474 medium, ≥ 0.474 large.

**Implementation:** `_common/stats_extra.py:cliffs_delta` (in-house O(n_a · n_b) dominance count).
**Reference:** Cliff, N. (1993). "Dominance statistics: ordinal analyses to answer ordinal questions." *Psychological Bulletin* 114 (3): 494–509.

## Where corrections appear in the output

Every CSV that contains a `p_value` column also has `p_holm` and `p_bh` columns added by the orchestrator (when present in the implementation pass). Every comparison CSV gains `effect_size` and `effect_size_ci_lo/hi` columns.

**Cross-cutting columns** (added when relevant):

| Column | Source |
|---|---|
| `p_value` | Raw p-value |
| `p_holm` | Holm-Bonferroni adjusted (within tier) |
| `p_bh` | Benjamini-Hochberg adjusted (within tier) |
| `endpoint_tier` | `primary` or `secondary` per `configs/eval_endpoints.yaml` |
| `effect_size` | Cohen's d / odds ratio / Cliff's δ |
| `effect_size_ci_lo`, `effect_size_ci_hi` | CI on the effect size |

## Reporting

Every reported p-value carries its adjusted form (`p_holm` for primary, `p_bh` for secondary), and primary endpoints are pre-registered separately in `configs/eval_endpoints.yaml` so the published claims can be traced to a locked endpoint set rather than post-hoc selection.

**Sample paper-Methods sentence:**

> "Statistical significance was assessed at the field level using McNemar's test (McNemar, 1947) for binary comparisons and paired-bootstrap Δ (Efron, 1987; n = 2000 replicates) for continuous comparisons. P-values were adjusted within pre-registered endpoint tiers using the Holm-Bonferroni procedure (Holm, 1979) for primary endpoints (12 fields listed in `configs/eval_endpoints.yaml`) and the Benjamini-Hochberg false-discovery-rate procedure (Benjamini & Hochberg, 1995) for secondary endpoints. Both corrections were computed via `statsmodels.stats.multitest.multipletests` (Seabold & Perktold, 2010). Effect sizes are reported alongside p-values: Cohen's d (Cohen, 1988) for continuous outcomes, odds ratios with Wald CI for binary outcomes, and Cliff's δ (Cliff, 1993) for ordinal outcomes."

## What to do if your numbers are bad

- **Many primary p-values significant before correction, none after Holm:** the family-wise correction is biting. Either accept the (more conservative) Holm result or move borderline fields to `secondary`.
- **A specific field consistently fails Holm but passes BH:** the secondary tier is the appropriate home for it.
- **Effect size small but p-value tiny:** large n with trivial effect. Don't oversell — report both.
- **Effect size large but CI wide:** under-powered. Get more data or accept the uncertainty.
