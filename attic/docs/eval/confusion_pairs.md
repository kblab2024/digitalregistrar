# Confusion pairs and semantic neighbors

Some categorical errors are clinically equivalent. Minor variations in TNM staging — for example "anatomic" vs "pathologic" descriptors, or T2 vs T2a substaging — do not change downstream treatment decisions, but a strict string-equality check counts them as wrong. The confusion-pair / semantic-neighbor analysis surfaces these systematically so a reader can separate clinically meaningful errors from cosmetic ones.

## ELI5

If a doctor writes "T2a" and the model outputs "T2", that's not really a clinically meaningful disagreement — the substage is a refinement of the major stage, and many treatment guidelines act on the major stage alone. Calling that an "error" inflates the apparent error rate.

But if the model outputs "T4" when gold says "T1", that's a real, large error.

**Strict accuracy** treats both alike. **Semantic-neighbor-aware accuracy** treats the first as a "near miss" and the second as a real error. We report both.

## Curated semantic-neighbor list

Defined in [src/digital_registrar_research/benchmarks/eval/semantic_neighbors.py](../../src/digital_registrar_research/benchmarks/eval/semantic_neighbors.py). Each pair has:

- `field` — the field where the confusion is plausible.
- `a`, `b` — the two interchangeable values.
- `reason` — short clinical justification.

**Adding a pair here loosens the accuracy definition** — only add pairs where the literature supports the equivalence. Conservative defaults today:

- `tnm_descriptor`: `anatomic` ↔ `pathologic` — clinically equivalent in most treatment contexts.
- `pt_category`: t1 ↔ t1a/t1b/t1c, t2 ↔ t2a/t2b, t3 ↔ t3a/t3b, t4 ↔ t4a/t4b.
- `pn_category`: n1 ↔ n1a/n1b, n2 ↔ n2a/n2b.
- `pm_category`: m0 ↔ mx (clinically equivalent absent distant-metastasis workup).
- `lymph_node_category`: sentinel ↔ axillary_level_i (overlap when only one node sampled).
- `histology`: invasive_carcinoma_no_special_type ↔ invasive_ductal_carcinoma (legacy renaming, 2012 WHO update).

## Outputs

### `non_nested/confusion_pairs.csv`

Top-N most-frequent confusion pairs per (field, organ), with an `is_semantic_neighbor` flag for each. Sorted descending by count.

Columns: `field`, `organ`, `gold_value`, `pred_value`, `count`, `is_semantic_neighbor`.

This is the **descriptive** view: what pairs is the model confusing, and are they clinically reasonable?

### `non_nested/accuracy_collapsing_neighbors.csv`

Per (field, organ):

- `accuracy_strict` — exact match.
- `accuracy_collapsing_neighbors` — alternative accuracy where curated neighbor pairs are counted as correct.
- `n_neighbor_recovered` — how many cases moved from "wrong" to "correct" under the relaxed definition.

This is the **headline** number for the clinical-equivalence writeup paragraph.

## Mean rank distance for ordinals

For ordinal fields, an "off-by-one" error is materially less clinically bad than "off-by-three." We separately report:

- `mean_rank_distance` — average `|rank(pred) − rank(gold)|` across wrong predictions.
- `top1_off_count`, `top2_off_count` — cumulative counts of wrong predictions within k ranks.
- `histogram` — full distribution of rank distances.

**Output:** `non_nested/rank_distance.csv`.

**Implementation:** `scripts/eval/_common/stats_extra.py:rank_distance_distribution` and `top_k_ordinal_accuracy`. No direct sklearn equivalent (`top_k_accuracy_score` requires probability scores).

## How to lift this into the writeup

> "Of the model's pt_category errors in lung, 47% were within one substage of the gold (e.g. T2 instead of T2a) — an error class that does not change AJCC stage assignment. Treating curated semantic-neighbor pairs as correct (`anatomic` ↔ `pathologic`, T-substages within the same major stage; full list in `src/.../eval/semantic_neighbors.py`) raised pt_category accuracy from 0.78 (95% CI [0.74, 0.81]) to 0.91 (95% CI [0.88, 0.93])."

## What to do if your numbers are bad

- **High `n_neighbor_recovered`:** the model is making clinically benign near-misses. Quote the relaxed accuracy in addition to the strict one.
- **High `mean_rank_distance` for an ordinal:** errors are large (off-by-many), not just off-by-one. Investigate whether the prompt is exposing the ordering structure correctly.
- **A specific confusion pair appears repeatedly across organs but isn't in the curated list:** consider adding it to `semantic_neighbors.py` if literature supports the clinical equivalence.
- **Confusion involves null vs a value (`None` ↔ something):** that's missingness, not a confusion pair — see [completeness.md](completeness.md).

## References

The semantic-neighbor concept follows medical-NLP standard practice — see e.g. evaluations of clinical NER systems where "exact" and "relaxed" matching are typically reported side-by-side. The specific clinical justifications for individual pairs trace to:

- AJCC Cancer Staging Manual (8th ed., 2017) — for substage equivalence.
- WHO Classification of Tumours of the Breast (5th ed., 2019) — for histology renaming.
