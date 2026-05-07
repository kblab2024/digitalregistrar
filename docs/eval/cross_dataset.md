# Cross-dataset generalization

A model that scores well on one institution's reports does not necessarily transfer to another. The `cross_dataset` subcommand compares per-field accuracy and gold-class distributions between two datasets (typically CMUH vs TCGA) and surfaces the external-validity story — separating "the model is genuinely worse here" from "the data distribution is different here."

## ELI5

A model that scores 90% on hospital A's data might score 75% on hospital B's. Why? Two possible explanations:

1. **Model failure** — the model is genuinely worse at hospital B's reports. Bad.
2. **Distribution shift** — hospital B has a different case mix (older patients, rarer cancers, different reporting style). Not necessarily a model issue, but the model's "external validity" is bounded.

Cross-dataset analysis distinguishes these by reporting both **per-field Δ-accuracy** AND **distribution-shift indicators**, so the reader can attribute observed performance gaps to the right cause.

## Per-field Δ-accuracy

For each (organ, field), compute model accuracy on each dataset and the difference:

```
delta = accuracy_left − accuracy_right
```

CI on the difference uses **independent bootstrap** (cases are not paired across datasets — different patients):

- Resample left and right independently with replacement.
- Compute `accuracy_left_resampled − accuracy_right_resampled`.
- 95% percentile interval.

**Implementation:** `scripts/eval/cross_dataset/run_cross_dataset.py:_per_field_delta`. Bootstrap with `numpy.random.Generator.integers` for reproducible resampling.
**Output:** `cross_dataset/per_field_delta.csv` with columns `organ`, `field`, `left_label`, `right_label`, `n_left`, `n_right`, `left_accuracy`, `right_accuracy`, `delta`, `delta_ci_lo`, `delta_ci_hi`.

## Distribution-shift indicators

Per-field gold-value distribution comparison. Different metric depending on field type:

### Categorical fields

- **Jensen-Shannon distance** — symmetric, bounded [0, 1] with log₂. JS = 0 means identical distributions; JS = 1 means disjoint.
- **KL divergence (left || right)** — asymmetric. NaN if right has zero where left has non-zero.
- **Chi-square test of independence** — p-value testing whether the two distributions differ significantly given the sample sizes.

**Implementation:**
- `scipy.spatial.distance.jensenshannon(p, q, base=2)` (Virtanen et al., 2020).
- `scipy.special.rel_entr(p, q).sum()` for KL.
- `scipy.stats.chi2_contingency` for χ².

**References:**
- KL: Kullback, S., & Leibler, R. A. (1951). "On information and sufficiency." *Annals of Mathematical Statistics* 22 (1): 79–86.
- JS: Lin, J. (1991). "Divergence measures based on the Shannon entropy." *IEEE Transactions on Information Theory* 37 (1): 145–151.

### Continuous fields

- **Wasserstein-1 distance** (Earth Mover's distance) — minimum work to transform one empirical distribution into the other. Robust to outliers, in original units.

**Implementation:** `scipy.stats.wasserstein_distance`.
**Reference:** Vaserstein, L. N. (1969). "Markov processes over denumerable products of spaces describing large systems of automata." *Problems in Information Transmission* 5 (3): 64–72.

**Output:** `cross_dataset/distribution_shift.csv`.

## Transferability score

Per organ:

- `mean_delta` — average across fields.
- `median_abs_delta` — robust to outliers, summarises typical gap size.
- `frac_left_ahead` and `frac_right_ahead` — fraction of fields where one dataset wins.

**Output:** `cross_dataset/transferability.csv`.

## Running the subcommand

```
# Run non_nested twice — once per dataset.
python -m scripts.eval.cli non_nested --root dummy --dataset cmuh --model gpt_oss_20b \
    --annotator gold --out /tmp/eval/non_nested_cmuh
python -m scripts.eval.cli non_nested --root dummy --dataset tcga --model gpt_oss_20b \
    --annotator gold --out /tmp/eval/non_nested_tcga

# Then compare.
python -m scripts.eval.cli cross_dataset \
    --left /tmp/eval/non_nested_cmuh \
    --right /tmp/eval/non_nested_tcga \
    --out /tmp/eval/cross_dataset
```

## How to interpret the table

| Pattern | Interpretation |
|---|---|
| Small Δ, large JS | Different case mix but model handles both. Good external validity. |
| Large Δ, large JS | Could be either — distribution shift matters and the model is sensitive to it. Inspect which fields specifically. |
| Large Δ, small JS | Model failure. The data looks similar but the model performs differently — likely a prompt / training issue. |
| Small Δ, small JS | Datasets are similar AND model performs similarly. Best case. |

## What to do if your numbers are bad

- **Many fields with `delta_ci` excluding zero in the same direction:** the model has a systematic preference for one dataset. Check whether the prompt was tuned on one dataset (overfit signal).
- **Wasserstein distances large for continuous fields:** the underlying populations differ. Tumor sizes might be larger at one institution because the patient cohort differs in stage at presentation. Annotate this in the writeup.
- **JS distance ≈ 0 but Δ-accuracy large:** model failure. Investigate.
- **Per-organ transferability scores skewed:** some organs transfer well, others don't. Often the rarest organs in your training data transfer worst — note this for Discussion.

## Sample paper-Methods sentence

> "External validity was assessed by comparing per-field accuracy across the two institutions (CMUH and TCGA) and quantifying gold-class distribution shift. Bootstrap 95% CIs on the per-field accuracy difference (n = 2000 independent resamples per side) used `numpy.random.Generator`. Distribution shift was measured per field using Jensen-Shannon distance (Lin, 1991) for categorical fields and Wasserstein-1 distance (Vaserstein, 1969) for continuous fields, computed via `scipy.spatial.distance.jensenshannon` and `scipy.stats.wasserstein_distance` (Virtanen et al., 2020). Chi-square tests of distribution equality (`scipy.stats.chi2_contingency`) were Holm-Bonferroni corrected within fields (Holm, 1979)."
