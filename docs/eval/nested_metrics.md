# Nested-list field metrics — lymph nodes, margins, biomarkers

> **Cascade-redesign note (2026-05).** Two material changes:
>
> 1. **Lymph nodes are no longer scored by bipartite matching on `station_name`.** The free-text `station_name` is unstable across annotators (e.g. `"nonsentinel #1"`, `"nonsentinel #2"`), so `examined`/`involved` are now summed per `(lymph_node_side, lymph_node_category)` group on each side and groups are matched by the deterministic key. New columns: `ln_group_recall`, `ln_group_precision`, `ln_n_groups_gold`, `ln_n_groups_pred`. The legacy `ln_station_*` keys still exist and now refer to GROUPS (not per-list rows).
> 2. **Biomarker scoring is restricted to a per-organ whitelist.** Only `{er, pr, her2, ki67}` are scored for breast and `{msh2, msh6, pms2, mlh1}` for colorectal. Other biomarker entries (e.g. colon `her2`, `p53`) are filtered from gold and prediction before bipartite matching. The `description` inner key on margins is also stripped before matching.
>
> Output paths: `nested/...` → `chapter3_field_extraction/nested_*.csv`. See [CHANGELOG.md](CHANGELOG.md) and [../stat_methods.md](../stat_methods.md) §1.

For list-of-dict fields where each case has 0–N items (e.g. multiple lymph node stations or margin measurements). Scoring uses **bipartite matching** between gold and predicted items, then per-attribute accuracy on matched pairs.

## ELI5

A pathology report mentions, say, three lymph node stations: sentinel #1 (1/2 involved), level II (0/3), level III (0/2). A model predicts: sentinel #1 (1/2), level II (0/3), level III (1/2). To score this you need to:

1. **Match** the predicted items to the gold items (which "level III" is the same as which?).
2. **Score** each matched pair on its inner attributes (involved count off by 1, examined count correct, category correct).
3. **Count** unmatched items as false positives or false negatives.

Bipartite matching with a similarity score handles step 1; per-attribute conditional accuracy handles step 2; precision / recall / F1 handles step 3.

## Bipartite matching

Each (gold, pred) item pair gets a similarity score:

- **Lymph nodes:** `+3` if `station_name` matches, `+2` if `lymph_node_category` matches (excluding "others"), `+1` if `lymph_node_side` matches; tie-break `+0.5` each if `involved` and `examined` are within ±1 node tolerance.
- **Margins:** `+3` if `margin_category` matches (excluding "others"), `+0–3` Jaccard similarity on description text, tie-break `+1` if `margin_involved` agrees.
- **Biomarkers:** `+3` if `biomarker_category` matches; `+1` if `expression` matches; `+0.5` if `percentage` is within ±5pp.

**Greedy bipartite assignment:** iteratively pick the (g, p) pair with the highest positive similarity, remove both from the pool, repeat. Quadratic but adequate for the ~1–10 items per case observed in this corpus.

**Implementation:** `digital_registrar_research.benchmarks.eval.nested_metrics._greedy_match`. Library alternative for *optimal* assignment: `scipy.optimize.linear_sum_assignment` (Hungarian algorithm; Kuhn 1955) — not used here because the greedy approximation has matched the optimal in spot-checks and is simpler.

## Item-level precision, recall, F1 (micro)

After matching:

- **TP** = matched pairs where the primary key (station_name / margin_category / biomarker_category) agrees.
- **FP** = predicted items unmatched to any gold (hallucination).
- **FN** = gold items unmatched to any predicted (miss).

Then:

- `precision = TP / (TP + FP)`
- `recall = TP / (TP + FN)`
- `F1 = 2 · P · R / (P + R)`

Reported in two flavors:

- **Attempted F1** — F1 on cases where the model produced the field. Quality on what it tried.
- **Effective F1** — F1 across all eligible cases, treating missing-field cases as F1 = 0. Quality at deployment.

The gap is the **completeness penalty for nested fields** — see [completeness.md](completeness.md).

**BCa bootstrap CI** on macro F1 (mean per-case F1 across attempted cases) via case-stratified resampling. Reference: Efron, B. (1987). "Better bootstrap confidence intervals." *JASA* 82 (397): 171–185.

**Output:** `nested/per_field_per_organ.csv`.

## Hallucination rate vs miss rate

- **`hallucination_rate = FP / (FP + TP)`** — of the items the model predicted, what fraction had no gold counterpart? With Wilson 95% CI.
- **`miss_rate = FN / (FN + TP)`** — of the items in gold, what fraction did the model miss?

These are *item-level* rates, distinct from *field-level* missingness (where the entire field key is absent from the prediction). Both are informative — high hallucination means the model is making things up; high miss means the model is undercounting.

**Implementation:** in-house formulas; Wilson CI via `digital_registrar_research.benchmarks.eval.ci.wilson_ci`. Reference for Wilson: Wilson (1927).
**Output:** `nested/per_field_per_organ.csv` (`hallucination_rate`, `miss_rate` and their CIs).

## Count MAE and count correlation

How well does the model predict *how many* items the case has?

- `count_mae = mean(|len(gold) − len(pred)|)` per case.
- `count_correlation = pearson(gold_counts, pred_counts)` — does the model rank-order correctly even if it's off in absolute count?

**Implementation:** `numpy.corrcoef`.
**Output:** `nested/per_field_per_organ.csv` (`count_mae`, `count_correlation`); per-case distributions in `nested/support_distribution.csv`.

## Per-attribute conditional accuracy

For matched pairs, what fraction got each inner attribute right?

| Field | Attributes |
|---|---|
| `regional_lymph_node` | station_name, lymph_node_category, lymph_node_side, involved (±1), examined (±1) |
| `margins` | margin_category, margin_involved, distance (±2 mm), description (Jaccard) |
| `biomarkers` | biomarker_category (always 1.0 if matched), expression, percentage (±5pp), score |

Each with Wilson 95% CI. **Output:** `nested/per_attribute_per_organ.csv`.

## Case-level summary metrics

The clinically actionable headlines that drive treatment decisions:

- **Lymph nodes:** any-positive accuracy (does the model agree with gold on whether *any* node was involved?), examined-total MAE, involved-total MAE, ±1 tolerance accuracy on each.
- **Margins:** any-involved accuracy, closest-uninvolved-distance MAE, ±2 mm tolerance.

**Output:** `nested/case_level_per_organ.csv`.

## Four-level field missingness

Nested fields decompose missingness more finely than scalars:

| Level | Meaning |
|---|---|
| `parse_error` | Whole-case load failed. |
| `field_key_absent` | Case loaded, but the nested field key (`margins` etc.) is entirely missing. |
| `empty_list` | Field key present but `[]` — model "tried but found nothing" — when gold has items. |
| `partial_list` | Items present, some matched some missing — captured by F1 / miss_rate. |

Each level reported with Wilson CI. Field-level missingness is *separate* from item-level miss — the latter is "model produced a list but missed an item"; the former is "model didn't produce the list at all."

**Output:** `nested/nested_missingness.csv`.

## Multi-run consistency

When ≥ 2 runs exist:

- Mean per-case F1 across runs.
- Per-case F1 SD across runs (mean and max — high values flag brittle cases).
- Missing-flip rate: fraction of cases where ≥ 1 run had the field absent AND ≥ 1 run produced it.

**Implementation:** `nested/run_nested.py:_multirun_consistency`.
**Output:** `nested/multirun_consistency.csv`.

## Per-organ stratification

Mandatory: schemas differ across organs (margins for prostate has different sub-structure than for breast; biomarkers exist only for breast and colorectal in the canonical schemas). When the field doesn't exist in an organ's schema, output an explicit row with NaN values rather than silently dropping — keeps the per-organ picture exhaustive.

## What to do if your numbers are bad

- **High `hallucination_rate` but reasonable F1:** model is producing extra items that don't exist in gold. Could be over-reading the report. Inspect a few cases manually.
- **High `miss_rate` but reasonable F1:** model is undercounting. Check whether the missed items are systematically the *low-prevalence* sub-categories.
- **High `count_mae` with low `count_correlation`:** the model is *both* miscounting absolutely *and* not even ordering correctly. Likely a prompt issue (model isn't told what to count).
- **High `field_key_absent_rate`:** model is dropping the field entirely. This is missingness — see [completeness.md](completeness.md).
- **Per-attribute accuracy on `station_name` ≪ on `lymph_node_category`:** the model is getting the right *type* of node but the wrong *name* — often means the report's terminology doesn't match the schema enum. Could be a mapping issue.

## References

Foundational citations are in [methods_citations.md](methods_citations.md). Most directly relevant:

- Bipartite matching (greedy approximation): Kuhn, H. W. (1955). "The Hungarian method for the assignment problem." *Naval Research Logistics Quarterly* 2 (1–2): 83–97.
- Jaccard similarity: Jaccard, P. (1912). "The distribution of the flora in the alpine zone." *New Phytologist* 11 (2): 37–50.
- F1 score: van Rijsbergen, C. J. (1979). *Information Retrieval* (2nd ed.). Butterworths. Ch. 7.
- Wilson CI: Wilson (1927).
- BCa bootstrap: Efron (1987).
