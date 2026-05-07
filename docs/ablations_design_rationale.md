# Design rationale

> **Cascade-redesign note (2026-05).** The "how we measure success" question is now answered by the cascade chapter outputs (Stage A eligibility, Stage B organ classification, Stage C field extraction) rather than by a single flat correctness vector. Per-cell modular-vs-monolithic deltas in `cell_deltas.csv` are computed from the same restricted Stage-C cohort across all cells, so comparisons remain apples-to-apples. Cross-cell paired tests live in `model_pair_tests/` (McNemar + Cochran-Q + paired-bootstrap deltas). See [eval/CHANGELOG.md](eval/CHANGELOG.md) and [stat_methods.md](stat_methods.md) for details.

This document records the reasoning behind each ablation cell and how
the results should be read. Decisions here should carry forward if the
ablation grid is extended.

## Entanglement caveat — why this isn't a clean factorial

A natural framing of the ablation is "what does each component
contribute?". The honest answer is that the components are
**not orthogonal**:

- **DSPy is itself a prompting framework.** "Removing DSPy"
  simultaneously removes the framework's automated prompt construction,
  parse-retry logic, structured-output handling, *and* its `Literal[…]`-
  typed output channel.
- **The schema constraint is realised through DSPy's `Literal` type
  hints.** "Removing schema constraints" without removing DSPy means
  swapping enum-typed outputs for `str` outputs and parsing post-hoc —
  a hybrid lesion, not an isolated one.
- **The router (`is_cancer`) and intermediate JSON step (`ReportJsonize`)
  themselves use DSPy.** Removing DSPy from one stage and not another
  introduces a discontinuity worth reporting separately.

We therefore frame the ablations as **lesion studies on engineering
choices** (modular-vs-monolithic decomposition, with-vs-without
ReportJsonize, DSPy-vs-raw-JSON output channel) rather than as a clean
factorial of independent components. Where one knob necessarily co-varies
with another we say so explicitly and report both endpoints separately.

## Why these three cells?

The published Digital Registrar pipeline makes two joint design
choices that are candidate explanations for its accuracy:

1. It splits each organ's extraction into 5–7 small DSPy signatures
   (e.g. `BreastCancerNonnested`, `BreastCancerStaging`,
   `BreastCancerMargins`, `BreastCancerLN`, `BreastCancerBiomarkers`,
   `BreastCancerGrading`, `DCIS`).
2. It uses DSPy as the LM-calling framework, which carries the
   schema constraint via `Literal[…]` type hints.

A 2×2 separates them:

|          | Modular                 | Monolithic                |
|----------|-------------------------|---------------------------|
| DSPy     | **A** (baseline)        | **B** (test modularity)   |
| Raw LLM  | *(not run — see below)* | **C** (test framework)    |

The modular-raw cell is skipped because (a) without DSPy's automatic
output-schema management, running N raw calls per organ with
inter-call dependency on partial outputs becomes significantly harder
to implement correctly, and (b) a clean A→B→C ladder already answers
the questions the ablation is meant to answer:

- **A vs B**: modularity effect, with DSPy held constant.
- **B vs C**: framework effect, with modularity held constant at
  monolithic.

The chain lets us attribute any A→C gap to modularity (A→B) and
framework (B→C) separately.

A **fourth condition** in the lesion sequence — Cell B with
`--skip-jsonize` — isolates the contribution of the intermediate
`ReportJsonize` structuring step. See
[ablations.md](ablations.md) "The Grid 1 lesion study" for the full
sequence.

## Parts held constant

To isolate the modularity and framework variables, three upstream
components stay the same across all cells:

1. **`is_cancer` classifier** — the initial "is this an eligible
   cancer excision report, and which of the 10 organs?" routing step.
   Runs as the existing DSPy signature in Cells A and B, and as an
   equivalent raw JSON-mode call in Cell C. In all cases it returns
   the same flag + organ label.
2. **`ReportJsonize` step** — the intermediate "rough JSON structuring"
   signature. Kept on by default in Cells A, B (matches the baseline);
   omitted in Cell C (the monolithic raw call already has the full
   report as context). The `--skip-jsonize` flag on
   [`scripts/ablations/run_cell_b.py`](../scripts/ablations/run_cell_b.py)
   adds a supplementary "no jsonize" variant of Cell B for the lesion
   study.
3. **Case set** — every cell predicts on the same fixture set under
   `<folder>/data/<dataset>/`. No train/test split is applied: the
   dataset is the test set, and the gold annotations under
   `annotations/gold/` are the reference.

## Schema source of truth

The raw-JSON runner in Cell C loads its per-organ JSON schemas via
`digital_registrar_research.schemas.load_json_schema(organ)` — the
**same** JSON the annotation UI consumes, generated from the **same**
Pydantic case-models the DSPy pipeline targets. Agreement between
Cells A/B/C therefore reflects only model and framework behaviour, not
schema drift.

## Model-framework interaction

Running each cell against multiple models (locally:
`gpt-oss:20b`, `gemma3:27b`, `qwen3:30b`; cloud: `gpt-4-turbo`) lets us
see whether DSPy's scaffolding is more valuable for smaller local
models than for frontier models. A priori we expect:

- On `gpt-oss:20b` and similar local LMs: **A > B ≫ C**. Modularity
  saves context; DSPy saves JSON reliability on smaller models.
- On `gpt-4-turbo`: **A ≈ B ≈ C**. Frontier models handle both a full
  organ schema in one shot and raw JSON output reliably.

If this pattern is observed, it directly supports the paper's core
narrative that the Digital Registrar's schema-first modular design is
what makes a local LLM competitive in the first place — i.e. the
contribution is **the engineering**, not the model.

## Future axes (not yet wired into the runners)

The full menu of axes for future expansion includes:

- **Output structuring discipline** — beyond the DSPy-Literal vs raw-
  JSON contrast, integrate constrained decoding (`outlines`,
  `lm-format-enforcer`) or GBNF grammars for a stronger structuring
  baseline (axis B4 / B5).
- **Prompting strategy** — few-shot demos, `dspy.ChainOfThought`,
  compiled DSPy programs (`BootstrapFewShotWithRandomSearch`) — the
  axes that target prompting strategy independent of the framework
  question (axes C2–C5).
- **Schema specificity** — narrow per-organ Literal enums (current) vs
  union-across-organs vs flat schema (axes F1–F3).

Each of these requires a new cell-runner under
[`src/digital_registrar_research/ablations/runners/`](../src/digital_registrar_research/ablations/runners/);
[`scripts/ablations/run_grid.py`](../scripts/ablations/run_grid.py)
already has TODO markers in `CELL_DISPATCH` showing where to register
them.

## Metrics

All cells use the shared evaluator in
[`benchmarks.eval`](eval/index.md) — same field definitions, same
bipartite match for nested lists, same coverage accounting. Headline
per-cell numbers:

- **Per-field macro accuracy on `FAIR_SCOPE`** — primary endpoint
  (Wilson 95 % CI; paired-bootstrap CI vs. the full pipeline). Uses the
  fair-scope whitelist in
  [`scope.py`](../src/digital_registrar_research/benchmarks/eval/scope.py)
  to ensure every cell competes on fields it can actually populate.
- **Completeness breakdown** — `parse_error / field_missing / attempted /
  correct` from
  [`completeness.py`](../src/digital_registrar_research/benchmarks/eval/completeness.py).
  We expect Cell C's `parse_error` rate to spike on `gpt-oss:20b` vs.
  Cells A/B, demonstrating the DSPy framework's structural-validity
  contribution.
- **Nested-list F1** — bipartite-matched F1 for margins, biomarkers,
  regional lymph nodes ([`nested_metrics.md`](eval/nested_metrics.md)).
  Modularity's main payoff lives here — monolithic single-call cells
  tend to exhaust context on multi-list organs first.
- **Per-cell latency** — mean / median seconds from each
  `_ledger.json`. Useful for the "efficiency" axis of the paper:
  modular pays N× the DSPy-bootstrap cost; monolithic pays once;
  raw-JSON pays once with no DSPy overhead.
- **Retry / validation rate** — DSPy's automatic parse retry (Cells A,
  B); manual Pydantic-validation retry count (Cell C, populated by
  `RawJSONRunner.validation_retries`).

Statistical analysis (Wilson + BCa + paired bootstrap CIs, McNemar at
the case level, Fleiss κ across seeds, Holm-Bonferroni within axis,
Cohen's d / Cliff's δ effect sizes) goes through the shared toolkit at
[`benchmarks.eval.ci`](eval/ci_methods.md),
[`benchmarks.eval.multirun`](eval/multirun.md), and
[`scripts/eval/_common/stats_extra.py`](../scripts/eval/_common/stats_extra.py).

## Known risks / threats to validity

- **Context-window saturation on `gpt-oss:20b` × monolithic**. The
  monolithic Cell B may *also* be context-window limited for breast
  (≥ 7 nested field groups) — specifically the biomarkers + LN +
  margins combination can overflow a 16k-token context when combined
  with a long report. We detect and flag this in
  [`runners/dspy_monolithic.py`](../src/digital_registrar_research/ablations/runners/dspy_monolithic.py);
  if it fires, the finding itself is a result (modularity is not
  merely *helpful* but *necessary* for that organ at that model size).
- **Schema drift**. The JSON schemas shipped under
  [`schemas/data/`](../src/digital_registrar_research/schemas/data/)
  may drift from the DSPy signatures over time. The
  `registrar-schemas --check` console script asserts top-level
  Literal-vocab parity in CI; run it before every ablation kickoff.
- **DSPy version pinning**. DSPy's prompt-construction behaviour has
  changed materially between minor releases. Pin the version in
  [`pyproject.toml`](../pyproject.toml) and capture it in the run
  manifest via `_run_meta.json`'s git SHA + DSPy version field.
- **Seed scope**. Currently the seed is set in
  [`configs/dspy_ollama_<model>.yaml`](../configs/) and applies to all
  cells using that model. To run multi-seed ablations, copy the grid
  YAML, change `slug` per seed, and bump `decoding.seed` between
  invocations. Per-run seed override is on the TODO list in
  [run_grid.py](../scripts/ablations/run_grid.py).

## Smoke-first discipline

Before any multi-day grid, run [grid-wide smoke](ablations.md#grid-wide-smoke)
end-to-end and confirm exit code 0 + non-empty `ablation_summary.csv`.
The smoke contract is documented in [ablations.md §
"Smoke runners"](ablations.md#smoke-runners--pre-flight-before-a-multi-day-sweep).
Cost: ≤ 10 minutes. Insurance value: catching a typo or a schema-binding
regression before it consumes 8 hours of a real sweep.

## Pre-registration

Before kicking off a real grid:
1. Pin the gold-annotation set — record the SHA of `<folder>/data/<dataset>/annotations/gold/` before the grid runs.
2. Pre-register primary + secondary endpoints in
   [`configs/eval_endpoints.yaml`](../configs/eval_endpoints.yaml).
3. Lock the decoding seed in the relevant model config; capture it in
   `_run_meta.json` per cell.
4. Reference the locked endpoint config (with git SHA) in the paper
   Methods.

Pre-registration matters because the eval suite produces hundreds of
p-values across fields × organs × methods × cells; without locking the
primary-endpoint set in advance, post-hoc selection of "significant"
results is statistically unsound.

## 2026-04 redesign — what changed and why

A live audit on the M2 mac surfaced a class of silent bugs that made
the prior numbers uninterpretable. The relevant changes:

* **`monolithic` signature was empty.** `signatures.monolithic._iter_output_fields`
  inspected `cls.__dict__`, but DSPy stores fields in
  `cls.output_fields` — so every merged signature came back with zero
  output fields. `dspy_monolithic`, `str_outputs`, `chain_of_thought`,
  and `fewshot_demos` all built degenerate predictors and returned
  empty `cancer_data`. Fixed by reading `output_fields` directly.
* **`str_outputs` destroyed nested structure.** Coercing every type
  to `str | None` flattened `list[BreastBiomarker]` etc. into prose.
  Now only scalar Literal/int/float/bool leaves are coerced; nested
  Pydantic-list fields keep their structure so the ablation tests
  enum-discipline loss without conflating it with structure loss.
* **Folder-number → organ name was alphabetical.** `IMPLEMENTED_ORGANS[idx-1]`
  was wrong for both TCGA and CMUH (their folder numbering does not
  match the alphabetical order). Replaced with two-stage resolution:
  rule-based keyword classifier on report text
  (`ablations.utils.organ_classifier`), with the dataset-aware
  `benchmarks.organs.organ_n_to_name` as a last-resort fallback. Used
  in `no_router`, `per_section`, `build_fewshot_demos`, and
  `utils.demos`. The `is_cancer` LLM router is left unchanged in the
  four DSPy-routed cells.
* **`stats._split_method` mis-parsed multi-underscore methods.**
  `rsplit("_", 1)` on `"free_text_regex_gpt_oss_20b"` returned
  `("free_text_regex_gpt_oss", "20b")`. Switched to reading the
  explicit `cell` and `model` columns the aggregator already emits;
  the string-split path remains as a defensive fallback that walks
  known cell-ids longest-first.
* **Efficiency stats double-counted overlapping errors.** A case with
  both `_schema_errors` and `_error` flags went into both buckets, so
  the rates could exceed 1.0. Now the aggregator tracks
  `schema_only` / `parse_only` / `both` and exposes `failed_total` for
  the non-overlapping union.
* **Median-latency CI ledger path was wrong.** The old path
  `{cell}_{model}/_ledger.json` did not match the canonical layout, so
  the CI was always `None`. Fixed by aggregating per-case latencies
  from `results_root/{cell}/{model}/{run_id}/_log.jsonl`.
* **DSPy execution tracing.** `--trace-dspy` (or `-v`) on any runner
  now dumps every DSPy LM call (rendered prompt + raw response) into
  `_dspy_trace.jsonl` in the run dir. Combined with the
  `is_cancer -> excision=… category=…` and
  `invoking <organ> predictor (n_fields=…)` decision-point logs and
  the `NOT_CANCER=… UNKNOWN_ORGAN=… DOWNSTREAM=…` summary counters,
  the silent-skip class of bugs is now immediately visible.
* **Aggregator validates organ-folder alignment.**
  `build_grid_dataframe` compares each prediction's `cancer_category`
  against `organ_n_to_name(dataset, organ_n)` and surfaces mismatches
  as a per-case warning + an `organ_folder_mismatch` column in the
  grid CSV. The smoke contract spot-checks this too.
* **Grid driver pre-flight + per-cell try/catch.** `run_grid.py` now
  validates cell-ids, model aliases, required artifacts (`compiled_dspy`'s
  `compiled:` path, `fewshot_demos`'s `fewshot_demos.yaml`), and the
  data-layout existence BEFORE the first cell starts. Per-cell
  try/catch (with `--continue-on-cell-error`) writes a
  `grid_failures.json` so re-runs only target the failed subset.
* **`reference` folder shorthand.** `--folder reference` builds a
  one-time symlink staging tree under `reference/_staged/` from the
  canonical TCGA reports at `reference/reports/<organ_n>/*.txt` so
  M2-mac smoke runs use real TCGA data without restructuring the
  on-disk source. The reference checkout is reports-only; gold
  annotations aren't staged.
