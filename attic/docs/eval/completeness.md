# Completeness — why missing ≠ wrong

The single most consequential design decision in this evaluation suite is the **three-way outcome model**: every model-vs-gold prediction is classified as `correct`, `wrong`, or `missing`. Today's binary correct/wrong scoring conflates two completely different failure modes; separating them is the core of the modularity ablation argument.

## ELI5

Imagine grading a multiple-choice test. There are three ways to get a question wrong:

1. **Wrong answer** — student picked B; correct answer was C. The student tried, but missed.
2. **Skipped** — student left the question blank. They didn't try.
3. **Test paper torn** — the printer messed up and the question wasn't even on the paper. The student couldn't try.

Combining all three into "wrong" hides what's actually happening. A student who answers everything but is occasionally wrong (case 1) needs better preparation. A student who skips the hard ones (case 2) might be triaging time. A student whose paper was torn (case 3) — that's a process bug, not a knowledge bug.

For language models doing structured extraction, the three buckets are:

1. **Wrong** — model produced a value, value disagrees with gold. (Knowledge / reasoning failure.)
2. **Field missing** — model produced output for the case, but this specific field is absent or null. (The model "skipped".)
3. **Parse error** — the entire case-level output failed to load (malformed JSON, schema violation, timeout, refusal). (Process failure.)

The distinction matters because the **modularity ablation** hypothesis is exactly this: a monolithic raw-JSON pipeline drops fields under context-window pressure or schema strain — but a modular DSPy pipeline doesn't, because each field gets its own focused prompt. If we score "missing" as "wrong," we hide the difference. Calling it out is what makes the ablation evidence sharp.

## Outcome flags (every atomic row)

| Flag | Definition |
|---|---|
| `gold_present` | Gold has this field non-null. (Eligibility for accuracy.) |
| `parse_error` | Whole-case load failed. Entire case is parse_error for every field. |
| `field_missing` | Case loaded but this specific field is absent / null on the prediction side. |
| `attempted` | `not (parse_error or field_missing)` — model produced an actual value. |
| `correct` | `attempted AND value matches gold` (with field-specific tolerances). |
| `wrong` | `attempted AND not correct`. |
| `error_mode` | (when `parse_error`) one of `json_parse`, `schema_invalid`, `timeout`, `refusal`, `file_missing`, `other`. |

**Sum invariant:** across an eligible cohort (gold_present rows),
`n_correct + n_wrong + n_field_missing + n_parse_error == n_total`.

## Two accuracy flavors — always reported side by side

For every (field, organ) summary:

- **`attempted_accuracy = n_correct / n_attempted`** — accuracy among the cases the model didn't skip. Useful when comparing model *quality* on the questions it answered.
- **`effective_accuracy = n_correct / n_total`** — accuracy across the full eligible cohort, treating skipped fields as wrong. Useful when comparing model *deployment readiness*.

The gap, **`completeness_penalty = attempted_accuracy − effective_accuracy`**, is the headline of the modularity ablation. If two models report `attempted_accuracy = 0.92` but `effective_accuracy = 0.91 vs 0.65`, the second model is "skipping" a third of its questions. Reporting only the first number hides this.

**Implementation:** `scripts/eval/_common/outcome.py:classify_outcome`. Per-field aggregation: `digital_registrar_research.benchmarks.eval.completeness.aggregate_missingness`.
**Reference:** This three-way decomposition is foundational to the project's modularity argument; not a published metric per se but standard practice in structured-extraction evaluation (see e.g. Jurafsky & Martin, *Speech and Language Processing*, Ch. on Information Extraction).

## Missingness rates with Wilson CI

For each (field, organ):

- `parse_error_rate = n_parse_error / n_total` — whole-case load failure. The most severe missingness.
- `field_missing_rate = n_field_missing / n_total` — case loaded, this specific field skipped.
- `total_missing_rate = parse_error_rate + field_missing_rate` — combined.
- `attempted_rate = 1 − total_missing_rate` — fraction of cases the model produced any value for.

Each rate is reported with a Wilson 95% confidence interval.

**Implementation:** `statsmodels.stats.proportion.proportion_confint(method="wilson")` (Seabold & Perktold, 2010), wrapped by `digital_registrar_research.benchmarks.eval.ci.wilson_ci`.
**Reference:** Wilson, E. B. (1927). "Probable inference, the law of succession, and statistical inference." *JASA* 22 (158): 209–212.

## Error-mode decomposition

When the prediction file fails to load, we record *why*. The decomposition uses the prediction runner's `_log.jsonl` if available:

| `error_mode` | Meaning |
|---|---|
| `json_parse` | `json.JSONDecodeError` — output wasn't valid JSON. |
| `schema_invalid` | JSON parsed but failed Pydantic validation (wrong type / missing required field). |
| `timeout` | The runner reported a timeout. |
| `refusal` | The model refused (policy / safety). |
| `file_missing` | The prediction file isn't on disk. |
| `other` | Something else — read the message in the log. |

**Implementation:** `scripts/eval/_common/loaders.py:classify_log_error`. Output: `completeness/error_mode_decomposition.csv`.

## Schema-conformance / out-of-vocabulary rate

For categorical fields, even when the model produces a value, that value can be **outside the allowed enum**. This is *not* missingness (the model tried) and *not* a wrong answer in the usual sense (it's not even a valid answer). It's a separate failure mode that **only affects unconstrained generation** (raw JSON), not schema-constrained pipelines (DSPy / Pydantic).

For each (field, organ) we report:

- `n_oov` — count of attempted predictions whose value isn't in the allowed enum.
- `oov_rate = n_oov / n_attempted` with Wilson CI.

The headline pairing: a modular schema-constrained pipeline should have `oov_rate ≈ 0`; a raw-JSON pipeline typically does not. The gap is the *measurable benefit of schema constraints*.

**Implementation:** `digital_registrar_research.benchmarks.eval.completeness.out_of_vocab_rate`. Output: `non_nested/schema_conformance.csv`, `completeness/schema_conformance_per_method.csv`.

## Refusal calibration — justified vs lazy missingness

When the model returns null, is the model *correctly* admitting "this isn't in the report" (justified missingness) or *wrongly* giving up despite the answer being there (lazy missingness)?

| State | Interpretation |
|---|---|
| pred=null AND gold=null | `correct_refusal` — model knows when it doesn't know. |
| pred=null AND gold non-null | `lazy_missingness` — model gave up despite the answer being present in the report. |

For each (field, organ):

- `correct_refusal_rate = n_correct_refusal / n_pred_null` with Wilson CI.
- `lazy_missing_rate = n_lazy / n_pred_null` with Wilson CI.
- `justified_missingness_share = correct_refusal_rate / (correct_refusal_rate + lazy_missing_rate)`.

Distinct from `parse_error_rate` (runtime failure) and `field_missing_rate` (just counting nulls without checking gold). This calibration ratio tells you whether the model's silence is principled or evasive.

**Implementation:** `digital_registrar_research.benchmarks.eval.completeness.refusal_calibration`. Output: `non_nested/refusal_calibration.csv`, `completeness/refusal_calibration.csv`.

## Position-in-schema correlation (context-window-pressure test)

If the model drops fields preferentially **late in the schema**, that's a smoking gun for context-window pressure: by the time the model gets to the bottom of a long structured output, it's running out of attention budget.

We compute the Spearman correlation ρ between (a) the field's index in the canonical schema and (b) its `field_missing_rate`. Positive ρ → late-schema fields are missing more often.

**Implementation:** `scipy.stats.spearmanr` (Virtanen et al., 2020). Wrapped by `digital_registrar_research.benchmarks.eval.completeness.position_in_schema_correlation`. Output: `completeness/position_in_schema_correlation.csv`.

## Method-pair Δ — the modularity-advantage table

Cross-method comparison: for every pair of methods × every (field, organ), report:

- `attempted_rate_a`, `attempted_rate_b` — paired on the same case set.
- `delta_attempted_rate = attempted_rate_a − attempted_rate_b`.
- McNemar's test on the paired binary outcome (attempted vs not), with continuity correction or exact binomial fallback for small n.

Sorted descending by `|delta_attempted_rate|`, this is **the** ablation headline: the fields where method A's modular schema produces *measurably higher* completeness than method B's monolithic raw JSON.

**Implementation:** McNemar via `digital_registrar_research.benchmarks.eval.ci.mcnemar_test`. Aggregation: `completeness.method_pair_deltas`. Output: `completeness/method_pairwise_deltas.csv`, `completeness/modularity_advantage.csv`.
**Reference:** McNemar, Q. (1947). "Note on the sampling error of the difference between correlated proportions or percentages." *Psychometrika* 12 (2): 153–157.

## What to do if your numbers are bad

- **High `parse_error_rate`** (> 5%): the model is choking on the task structure. Check `error_mode_decomposition.csv` — if mostly `json_parse`, decoding settings (temperature, JSON-mode) need tuning. If `schema_invalid`, the schema is too strict or the model doesn't understand it.
- **High `field_missing_rate` for late-schema fields specifically**: context-window pressure. Check `position_in_schema_correlation.csv` — if Spearman ρ > 0.5, this is real. Mitigations: shorter schema, modular per-field prompts.
- **Low `attempted_accuracy` but high `effective_accuracy`**: the model is *very confident* on the cases it answers but selective. Check `correct_refusal_rate` vs `lazy_missing_rate` — a high `lazy_missing_rate` means the model is dropping easy answers, which is a quality issue.
- **High `oov_rate`** for raw-JSON method but ≈ 0 for DSPy: this is exactly the modularity advantage. Lift it into the writeup.

## Reading the CSVs

Every completeness output ships its own provenance: `manifest.json` in the output dir captures `git_sha`, `utc_timestamp`, full CLI args, and case counts per organ. Cite the manifest's `git_sha` next to any number quoted in the paper.
