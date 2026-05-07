# Pre-annotation effect — measuring the anchoring bias

When LLM pre-annotations are presented as a starting point for human annotators, the human's final answer can be *anchored* on the model's suggestion — they may skim, see something plausible, and click through rather than reading the report from scratch. If the model is mostly right this is a useful workflow accelerant; if the model is consistently wrong in subtle ways it actively contaminates the gold standard.

The pre-annotation (preann) effect analysis quantifies this: by re-annotating a subset of cases *from scratch* (without seeing the model's pre-fill), we can compare the with-preann gold against the without-preann gold and measure how much the model's suggestions shifted the human's answer.

## ELI5

Imagine showing a doctor a pathology report and asking them to fill out 30 fields. Two scenarios:

1. **With pre-annotation:** the form is already filled in by an LLM. The doctor's job is to verify and correct.
2. **Without pre-annotation:** the form is blank. The doctor reads the report and fills it in from scratch.

If the LLM is mostly right, scenario 1 saves time AND should produce more accurate annotations (the doctor catches mistakes). But if the LLM is *consistently wrong in subtle ways*, scenario 1 will *anchor* the doctor on those wrong answers — they'll skim, see something plausible, and click through.

The **preann effect** is the difference between scenarios 1 and 2 measured on the *same* doctors annotating the *same* cases.

## Study design — paired by case

The without-preann set is a **strict subset** of the with-preann set: the same NHC annotator wrote both for those cases. Same for KPC.

```
nhc_with_preann/    ⊃ nhc_without_preann/    (same patients, both modes)
kpc_with_preann/    ⊃ kpc_without_preann/    (same patients, both modes)
```

The paired cohort is the intersection. Every preann-effect metric below is computed on this intersection — so observed differences cannot be attributed to different case mix.

**Implementation:** `scripts/eval/_common/pairing.py:discover_paired_cases`.

## Metric 1 — Δκ (with vs without preann)

For each (annotator, organ, field):

- `κ_with` = Cohen's κ between `<annotator>_with_preann` and `gold`.
- `κ_without` = Cohen's κ between `<annotator>_without_preann` and `gold`.
- `Δκ = κ_with − κ_without`.

A **positive** Δκ means preann *helped* the annotator agree with gold (good — the model's suggestions were correct and the human caught the mistakes). A **negative** Δκ means preann *hurt* — the annotator converged on wrong answers because they didn't think hard enough.

The CI on Δκ uses **paired-bootstrap resampling** of case ids. Each bootstrap draw picks indices `i ∈ [0, n)` with replacement, then recomputes both κ's on the resampled records and takes the difference. This gives a CI on the *paired* difference, not on each κ independently.

**Implementation:** `digital_registrar_research.benchmarks.eval.preann.paired_delta_kappa` using `sklearn.metrics.cohen_kappa_score` (Pedregosa et al., 2011).
**References:** Cohen, J. (1960, 1968) for κ; Efron, B. (1979) for the bootstrap; the paired-by-case design is standard in psychometric within-subjects analysis.
**Output:** `iaa/preann/delta_kappa_per_field__<annotator>.csv`.

## Metric 2 — Convergence-to-preann rate

Of cases in with-preann mode where the preann produced a non-null prediction, how often does the human's final answer match the preann's?

- `P(human=preann | with_preann)` overall, with Wilson 95% CI.
- Stratified by `preann_correct` ∈ {True, False}:
  - `p_when_preann_correct` — when preann was right, did the human keep the right answer?
  - `p_when_preann_incorrect` — when preann was wrong, did the human change it or rubber-stamp it?

A high `p_when_preann_correct` is good (the workflow is efficient). A high `p_when_preann_incorrect` is **bad** — it means the human is anchoring on wrong answers. The two together tell the full story.

**Implementation:** `preann.convergence_to_preann`, `wilson_ci`.
**Output:** `iaa/preann/convergence_to_preann__<annotator>.csv`.

## Metric 3 — Anchoring index (AI)

The anchoring index isolates the *causal* effect of seeing the preann:

```
AI = P(human = preann | with_preann) − P(human = preann | without_preann)
```

The without-preann human still happens to coincide with what preann would have produced sometimes, by chance or shared knowledge. AI subtracts that baseline.

- **AI ≈ 0**: the annotator is unaffected by seeing preann. They produce the same answer either way.
- **AI > 0**: seeing preann shifts the human toward preann.
- **AI < 0**: seeing preann shifts the human *away* from preann (rare).

Reported in three flavors:

- `ai_overall` — across all cases.
- `ai_correct` — restricted to cases where preann was right. Positive AI here is innocuous or even good (the human accepts a correct suggestion).
- `ai_incorrect` — restricted to cases where preann was wrong. **Positive AI here is the dangerous direction** — the model is teaching humans wrong answers.

**Implementation:** `preann.anchoring_index`.
**Output:** `iaa/preann/anchoring_index__<annotator>.csv`.

## Metric 4 — Disagreement reduction

When both NHC and KPC see the same preann, do they converge on the same answer (because both copy preann), or do they genuinely converge to truth?

```
disag_with    = 1 − Cohen κ(nhc_with, kpc_with)
disag_without = 1 − Cohen κ(nhc_without, kpc_without)
Δdisag        = disag_with − disag_without
```

- **Negative Δdisag** means preann *reduced* inter-annotator disagreement.
- This must be read **alongside** Δκ-vs-gold:
  - Δdisag < 0 AND Δκ-gold > 0 → preann reduced disagreement *and* improved truth-tracking. Good.
  - Δdisag < 0 AND Δκ-gold ≈ 0 → preann reduced disagreement by making both humans copy preann. Suspicious.
  - Δdisag ≈ 0 AND Δκ-gold > 0 → preann improved truth-tracking without homogenising. Best case.

Paired-bootstrap CI on Δdisag (resample case-ids together, recompute both κ's per resample).

**Implementation:** `preann.disagreement_reduction`.
**Output:** `iaa/preann/disagreement_reduction.csv`.

## Metric 5 — Edit distance from preann

For each case in with-preann mode, count how many fields the human edited away from the preann. Distributional summaries:

- `mean_changes` — average number of fields edited per case.
- `median_changes`, `max_changes`.
- `mean_share_edited` — fraction of fields touched.

Stratified by whether the human's final answer matched gold. **Editing rate when wrong → edited correctly** is a positive workflow signal: the human's edits are doing real work.

**Implementation:** `preann.edit_distance_from_preann`.
**Output:** `iaa/preann/edit_distance__<annotator>.csv`.

## What the metrics together tell you

- **Quantifying anchoring bias.** Δκ, AI, and convergence-rate stratified by `preann_correct` together quantify how much the LLM pre-fill influenced the gold standard. If `ai_incorrect` is small (close to zero) the bias is bounded; if it is large and positive the bias is real and the Discussion section needs to acknowledge it.
- **Validating gold representativeness.** The without-preann subset is the *unbiased reference* — gold for these cases was assembled without any LLM pre-fill, so agreement between the with-preann and without-preann golds validates that the with-preann gold isn't systematically distorted. Cross-reading the IAA and preann tables together gives a richer picture than either alone.

## What to do if your numbers are bad

- **`ai_incorrect` strongly positive (> 0.2):** real anchoring on wrong answers. Mitigations: train annotators to override more aggressively, or remove preann from the workflow entirely on the most affected fields.
- **`Δdisag` strongly negative AND `Δκ-gold` ≈ 0:** preann is silencing disagreement without improving truth. Bad outcome — the gold is artificially homogenous. Discussion section needs a caveat.
- **Per-field heterogeneity:** some fields will show strong anchoring, others none. Report a few headline cases (the worst-anchored field per organ) rather than a single global number.

## Implementation summary

All preann-effect metrics live in `src/digital_registrar_research/benchmarks/eval/preann.py` (unit-testable independently of the CLI). The orchestrator is `scripts/eval/iaa/preann_effect.py`. All paired metrics use `scripts/eval/_common/pairing.py:discover_paired_cases` to materialise the paired cohort.

Library citations: `sklearn.metrics.cohen_kappa_score` (Pedregosa et al., 2011), Wilson CI via `digital_registrar_research.benchmarks.eval.ci.wilson_ci`. Paired bootstrap is in-house — see `ci.paired_bootstrap_diff` for the canonical reference.

For the full bibliography of κ, the bootstrap, and Wilson CI, see [methods_citations.md](methods_citations.md).
