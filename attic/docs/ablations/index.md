# Ablation studies

> Last updated: 2026-05-11 · Reflects: `d11d072`

> **Cascade-redesign note (2026-05).** The ablation harness grades predictions through the cascade-aware `score_case`. Cases that fail the upstream gates (Stage A `cancer_excision_report` or Stage B `cancer_category`) contribute fewer field-level rows to `ablation_grid.csv`; per-cell accuracy denominators reflect the cohort that reached each stage. Per-cell accuracy may rise vs the pre-cascade tables because the cohort is honestly restricted. Field exclusions (`ajcc_version`, `treatment_effect`, margin `description`, lymph-node `station_name`) and the per-organ biomarker whitelist (`{er, pr, her2, ki67}` for breast; `{msh2, msh6, pms2, mlh1}` for colorectal) apply uniformly across all cells, so modular-vs-monolithic comparisons remain apples-to-apples. See [../eval/CHANGELOG.md](../eval/CHANGELOG.md) and [../reference/stat_methods.md](../reference/stat_methods.md).

The Digital Registrar pipeline makes joint design choices — DSPy as the
LM-calling framework, schema constraints realised through DSPy `Literal`
type hints, and a per-organ modular decomposition into 5–7
sub-signatures chained through an intermediate `ReportJsonize` step.
The ablation suite attributes the headline accuracy back to those
choices, one knob at a time.

## 1. Why this isn't a clean factorial

A natural framing is "what does each component contribute?". The honest
answer is that the components are **not orthogonal**:

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

The ablations are framed as **lesion studies on engineering choices**
(modular-vs-monolithic decomposition, with-vs-without ReportJsonize,
DSPy-vs-raw-JSON output channel) rather than as a clean factorial of
independent components. Where one knob necessarily co-varies with
another we say so and report both endpoints separately.

## 2. Why these three cells

The pipeline makes two joint design choices that are candidate
explanations for its accuracy:

1. Splits each organ's extraction into 5–7 small DSPy signatures
   (e.g. `BreastCancerNonnested`, `BreastCancerStaging`,
   `BreastCancerMargins`, `BreastCancerLN`, `BreastCancerBiomarkers`,
   `BreastCancerGrading`, `DCIS`).
2. Uses DSPy as the LM-calling framework, which carries the
   schema constraint via `Literal[…]` type hints.

A 2×2 separates them:

|          | Modular                 | Monolithic                |
|----------|-------------------------|---------------------------|
| DSPy     | **A** (baseline)        | **B** (test modularity)   |
| Raw LLM  | *(not run — see below)* | **C** (test framework)    |

The modular-raw cell is skipped because (a) without DSPy's automatic
output-schema management, running N raw calls per organ with inter-call
dependency on partial outputs becomes hard to implement correctly, and
(b) a clean A→B→C ladder already answers the relevant questions:

- **A vs B**: modularity effect, with DSPy held constant.
- **B vs C**: framework effect, with modularity held constant at
  monolithic.

A **fourth condition** in the lesion sequence — Cell B with
`--skip-jsonize` — isolates the contribution of the intermediate
`ReportJsonize` structuring step.

### Parts held constant

To isolate the modularity and framework variables, three upstream
components stay the same across all cells:

1. **`is_cancer` classifier** — initial routing step. Runs as the
   existing DSPy signature in Cells A and B, and as an equivalent raw
   JSON-mode call in Cell C. All paths return the same flag + organ
   label.
2. **`ReportJsonize` step** — intermediate "rough JSON structuring"
   signature. Kept on by default in A, B (matches baseline); omitted
   in C (the monolithic raw call already has the full report as
   context). The `--skip-jsonize` flag on
   [`scripts/ablations/run_cell_b.py`](../../scripts/ablations/run_cell_b.py)
   adds a supplementary "no jsonize" variant of Cell B.
3. **Case set** — every cell predicts on the same fixture set under
   `<folder>/data/<dataset>/`. No train/test split: the dataset is the
   test set; `annotations/gold/` is the reference.

### Schema source of truth

The raw-JSON runner in Cell C loads its per-organ JSON schemas via
`digital_registrar.schemas.load_json_schema(organ)` — the
**same** JSON the annotation UI consumes, generated from the **same**
Pydantic case-models the DSPy pipeline targets. Agreement between
Cells A/B/C therefore reflects only model and framework behaviour,
not schema drift.

## 3. Canonical layout

Every ablation runner uses the same `--folder/--dataset/--model`
contract as
[`scripts/pipeline/run_factory_ollama_single.py`](../../scripts/pipeline/run_factory_ollama_single.py)
so input data, output predictions, and model aliases share one
directory convention with the rest of the toolkit.

```
{folder}/                                      # 'examples/dummy' | 'workspace' | 'workspace_obfustrated' | 'reference' | abs path
├── data/{dataset}/                            # 'cmuh' | 'tcga'
│   ├── reports/{organ_n}/{case_id}.txt        # input
│   └── annotations/gold/{organ_n}/{case_id}.json   # gold for grading
└── results/ablations/{dataset}/
    └── {cell_id}/{model_slug}/                # e.g. dspy_monolithic/gpt_oss_20b/
        ├── _manifest.yaml                     # all runs for this cell × model
        └── {run_id}/                          # e.g. run01
            ├── _summary.json                  # n_cases, n_ok, parse_error_rate, ...
            ├── _log.jsonl                     # per-case row
            ├── _run.log                       # full-verbosity log
            ├── _run_meta.json                 # git SHA, UTC, decoding kwargs
            ├── _dspy_trace.jsonl              # OPTIONAL — set --trace-dspy / -v
            └── {organ_n}/{case_id}.json       # one prediction per case
```

All cell-runner wrappers also accept `--obfustrated` and
`--folder obfustrated` to point at the synthetic workspace at
`workspace_obfustrated/`. See
[../workflows/obfuscation.md](../workflows/obfuscation.md).

### `{organ_n}` mapping

`{organ_n}` is a dataset-specific numeric folder name, **not** an
alphabetical ordering. The mapping lives in
[`configs/organ_code.yaml`](../../configs/organ_code.yaml) and is
loaded by
[`benchmarks.organs`](../../src/digital_registrar/benchmarks/organs.py):

| folder | TCGA          | CMUH       |
|--------|---------------|------------|
| 1      | breast        | pancreas   |
| 2      | colorectal    | breast     |
| 3      | esophagus     | cervix     |
| 4      | stomach       | colorectal |
| 5      | liver         | esophagus  |
| 6      | —             | liver      |
| 7      | —             | lung       |
| 8      | —             | prostate   |
| 9      | —             | stomach    |
| 10     | —             | thyroid    |

Always convert `organ_n` ↔ `organ_name` via
`benchmarks.organs.organ_n_to_name(dataset, organ_n)` /
`organ_name_to_n(dataset, name)` — never `IMPLEMENTED_ORGANS[idx-1]`
(alphabetical, therefore wrong for both datasets).

`--folder examples/dummy`, `--folder workspace`, and
`--folder reference` are standard shortcuts. The `reference` shortcut
builds a one-time symlink staging tree under
`tests/fixtures/reference/_staged/` from canonical TCGA reports at
`tests/fixtures/reference/reports/<organ_n>/*.txt` so M2-mac smoke runs
can use real TCGA data without restructuring it. (Reports-only — gold
annotations aren't staged, so eval-with-gold needs a `workspace`
checkout.) Absolute paths and other relatives resolve via
[`_config_loader.resolve_folder`](../../scripts/_helpers/_config_loader.py).

Models pass by alias from `models.common.UNIFIED_MODELS`
(`gptoss | gemma3 | gemma4 | gemma4e2b | qwen3_5 | medgemmalarge |
medgemmasmall`). The model slug in the output path mirrors the pipeline
runner: `ollama_chat/gpt-oss:20b` → `gpt_oss_20b`.

### Pre-run validation checklist

Run before any multi-day grid:

1. `python -c "from drr_attic.benchmarks.organs import dataset_organs; print(dataset_organs('tcga'))"` — confirm the loader sees the YAML.
2. `ls {folder}/data/{dataset}/reports/` — folder names must be numeric and match the table for your dataset.
3. Run a single cell with `--limit 1 --trace-dspy --verbose` and read the printed summary line — it ends with `NOT_CANCER=… UNKNOWN_ORGAN=… DOWNSTREAM=…`. If `DOWNSTREAM=0` the runner never invoked the organ-specific predictor — check `_dspy_trace.jsonl` for prompts and responses.
4. Run `scripts/ablations/run_grid.py --config <smoke-yaml>` — pre-flight validation rejects typos in `cell:` / `model:` and missing artifacts before the first cell starts.

### Skip taxonomy

Every DSPy-routed cell (`dspy_monolithic`, `str_outputs`,
`chain_of_thought`, `fewshot_demos`) emits a `_skip_reason` flag on each
per-case JSON when the downstream organ predictor is NOT invoked:

- `not_cancer` — upstream `is_cancer` router said the report is not a primary-excision report.
- `unknown_organ` — `is_cancer.cancer_category` returned `"others"` (or a value not in `models.modellist.organmodels`).
- (no `_skip_reason`) — downstream predictor ran; `_downstream_called: true` is set on the payload.

These tally into `n_skipped_not_cancer` / `n_skipped_unknown_organ` /
`n_downstream_called` in `_summary.json` and are surfaced in the
printed summary line.

### Source-tree map

```
src/digital_registrar/ablations/
├── runners/
│   ├── _base.py              # canonical args, path resolution, run loop
│   ├── reuse_baseline.py     # Cell A — copy pipeline outputs into ablations tree
│   ├── dspy_monolithic.py    # Cell B — one DSPy signature per organ
│   ├── raw_json.py           # Cell C — raw OpenAI-compatible chat API + JSON mode
│   ├── no_router.py          # A4 — drop the is_cancer router
│   ├── per_section.py        # A5 — per-section decomposition
│   ├── str_outputs.py        # B2 — DSPy with str outputs + post-hoc parser
│   ├── constrained_decoding.py  # B4 — outlines (vLLM/HF backend)
│   ├── free_text_regex.py    # B6 — degenerate baseline
│   ├── fewshot_demos.py      # C2/C3 — N curated demos per organ
│   ├── chain_of_thought.py   # C4 — dspy.ChainOfThought wrap
│   ├── compiled_dspy.py      # C5 — BootstrapFewShotWithRandomSearch
│   ├── minimal_prompt.py     # C6 — single-sentence raw prompt
│   ├── union_schema.py       # F2 — single union schema across organs
│   └── flat_schema.py        # F3 — denested per-organ schema
├── signatures/
│   ├── monolithic.py         # merges per-subsection signatures (B baseline)
│   ├── str_outputs.py        # strips Literals → str (B2)
│   └── per_section.py        # per-organ × per-section variant (A5)
├── extractors/               # post-hoc projection helpers (B2 / B6 / F3)
├── utils/                    # section_splitter (A5), demos loader (C2/C3)
└── eval/
    ├── run_ablations.py      # canonical-tree aggregator → grid/summary/table CSVs
    └── stats.py              # paired-bootstrap CI, McNemar, GLMM, Fleiss κ, …

scripts/ablations/
├── _common.py                # CELL_MAP, smoke-root, aggregator round-trip
├── run_cell_<short>.py       # thin per-cell wrappers (a, b, c, a4, …, f3)
├── run_cell_smoke.py         # per-cell smoke (≤ 2 cases, fail-loud)
├── run_grid_smoke.py         # full grid-wide smoke (≤ 10 min)
├── run_grid.py               # YAML-driven full grid driver
├── run_stats.py              # regenerate the stats pack from ablation_grid.csv
├── compile_dspy.py           # build the compiled DSPy artifact (C5)
└── build_fewshot_demos.py    # build configs/ablations/fewshot_demos.yaml (C2/C3)

configs/ablations/
├── smoke.yaml                # smoke defaults
├── grid_1.yaml               # Grid 1 minimum-viable lesion study
├── grid_2.yaml               # Grid 2 27-cell factorial
├── axes.yaml                 # cell → axis mapping for family-wise correction
└── fewshot_demos.yaml        # generated by build_fewshot_demos.py
```

Runners expose both `run(args: argparse.Namespace) -> int` and
`main(argv=None) -> int`. Wrappers under `scripts/ablations/` go through
`main()`; the YAML grid driver constructs a Namespace and calls `run()`
directly.

## 4. The six axes

### Axis 1 — Pipeline decomposition

| Level | Description | Status |
|---|---|---|
| A1 | Full modular (5–7 signatures per organ) | Cell A — `[shipped]` |
| A2 | Monolithic single signature per organ | Cell B — `[shipped]` |
| A3 | Monolithic, no `ReportJsonize` | Cell B `--skip-jsonize` — `[shipped]` |
| A4 | Monolithic, no `is_cancer` router | `runners/no_router.py` — `[shipped]` (uses gold organ; upper-bound router estimate) |
| A5 | Per-section decomposition (header / gross / micro / dx / comments) | `runners/per_section.py` — `[shipped]` |

### Axis 2 — Output structuring discipline

| Level | Description | Status |
|---|---|---|
| B1 | DSPy + Literal enums + Pydantic | Cells A, B — current default |
| B2 | DSPy with `str` outputs + post-hoc parser | `runners/str_outputs.py` — `[shipped]` |
| B3 | Raw JSON-mode (`response_format=json_object`) | Cell C — `[shipped]` |
| B4 | Constrained decoding (outlines / lm-format-enforcer) | `runners/constrained_decoding.py` — `[shipped]` (requires `outlines` + vLLM/HF backend) |
| B5 | GBNF grammar (llama.cpp) | `[skipped]` — not on Ollama path |
| B6 | Free-text + regex post-extractor | `runners/free_text_regex.py` — `[shipped]` |

### Axis 3 — Prompting strategy

| Level | Description | Status |
|---|---|---|
| C1 | Zero-shot signature docstring | current default |
| C2 | + 3 in-context examples (curated from train) | `runners/fewshot_demos.py --n-shots 3` — `[shipped]` |
| C3 | + 5 in-context examples (curated from train) | `runners/fewshot_demos.py --n-shots 5` — `[shipped]` |
| C4 | `dspy.ChainOfThought` wrapper | `runners/chain_of_thought.py` — `[shipped]` |
| C5 | Compiled DSPy program (`BootstrapFewShotWithRandomSearch`) | `runners/compiled_dspy.py` + `scripts/ablations/compile_dspy.py` — `[shipped]` |
| C6 | Minimal raw prompt (degenerate baseline) | `runners/minimal_prompt.py` — `[shipped]` |

### Axes 4–6 — Decoding, model identity, schema specificity

Decoding (temperature / num_ctx / self-consistency) and model identity
(`gemma3:4b/27b`, `gpt-oss:20b`, `qwen3:30b`, `medgemma`, `llama3-med42`)
are config-driven via [`configs/dspy_ollama_<model>.yaml`](../../configs/);
no new runners needed for sweeps along those axes — pass a different
`--model` key.

Schema specificity (Axis 6) gets two new runners:

| Level | Description | Status |
|---|---|---|
| F1 | Per-organ schema | Cell C — current default |
| F2 | Union schema across all organs | `runners/union_schema.py` — `[shipped]` |
| F3 | Flat (denested) per-organ schema | `runners/flat_schema.py` — `[shipped]` |

Per-axis multi-seed sweeps are `[future, no timeline]` — per-run seed
override and a config-driven seed scheduler are wired only as TODO
markers in
[`scripts/ablations/run_grid.py`](../../scripts/ablations/run_grid.py).

### Model-framework interaction

Running each cell against multiple models (local: `gpt-oss:20b`,
`gemma3:27b`, `qwen3:30b`; cloud: `gpt-4-turbo`) lets us see whether
DSPy's scaffolding is more valuable for smaller local models than for
frontier models. A priori expectation:

- On `gpt-oss:20b` and similar local LMs: **A > B ≫ C**. Modularity
  saves context; DSPy saves JSON reliability on smaller models.
- On `gpt-4-turbo`: **A ≈ B ≈ C**. Frontier models handle a full
  organ schema in one shot and raw JSON output reliably.

If observed, this pattern supports the paper's core narrative that the
Digital Registrar's schema-first modular design is what makes a local
LLM competitive — the contribution is **the engineering**, not the
model.

## 5. Grid 1 — the minimum-viable lesion study

The minimum-viable lesion study lives at
[`configs/ablations/grid_1.yaml`](../../configs/ablations/grid_1.yaml).

Conditions:

1. **Full pipeline** — modular DSPy + ReportJsonize + Literal enums
2. **Monolithic DSPy** — drops the modular per-section chain
3. **Monolithic DSPy without ReportJsonize** — also drops the intermediate JSON structuring step
4. **No DSPy** — raw OpenAI-compatible JSON-mode against local Ollama
5. **No schema** — free-text generation + regex post-extractor

Single backbone (`gptoss` → `ollama_chat/gpt-oss:20b`), single seed for
the first pass; for multi-seed reproducibility, invoke the script
multiple times — each invocation auto-picks the next free `runNN` slot
under each cell's directory, and decoding seeds come from
[`configs/dspy_ollama_<alias>.yaml`](../../configs/) (the legacy
`run_dspy_ollama_multirun.py` driver — now at
[`scripts/_legacy/`](../../scripts/_legacy/) — wraps multiple
invocations with a master seed for reproducibility).

Wall-clock estimate: ~2–3 days on a single 48 GB GPU.

## 6. Smoke runners — pre-flight before a multi-day sweep

Catching a typo or schema-binding regression eight hours into a sweep
is catastrophic. Every ablation kickoff goes through smoke first.

### Smoke contract

- **1 model**, **1 seed**, **2–3 cases** (default `--n 2`)
- Output dir prefixed with `_smoke_<YYYYMMDD-HHMM>/` so the regular
  aggregator's directory glob ignores it.
- **Fail loud** — any cell exception propagates; exit code ≠ 0.
- **Round-trip the aggregator** — smoke calls
  `drr_attic.ablations.eval.run_ablations.main()`
  with `--results-root <smoke dir>` and asserts
  `ablation_summary.csv` is non-empty before returning success.
- **Wall-time target**: ≤ 10 minutes for grid-wide smoke.

### Per-cell smoke

```bash
# Cell C × local gptoss — fastest smoke (no DSPy bootstrap)
python scripts/ablations/run_cell_smoke.py --cell c \
    --folder examples/dummy --dataset tcga --model gptoss

# Cell B × local gptoss
python scripts/ablations/run_cell_smoke.py --cell b \
    --folder examples/dummy --dataset tcga --model gptoss --n 2

# Cell A — copies the most recent completed pipeline run
python scripts/ablations/run_cell_smoke.py --cell a \
    --folder examples/dummy --dataset tcga --model gptoss
```

### Grid-wide smoke

```bash
python scripts/ablations/run_grid_smoke.py \
    --folder examples/dummy --dataset tcga --model gptoss

# Subset cells (skip B if Ollama bootstrap is slow)
python scripts/ablations/run_grid_smoke.py \
    --folder examples/dummy --dataset tcga --model gptoss --cells c b6 c6
```

Extend [`scripts/repo/install_git_hooks.sh`](../../scripts/repo/install_git_hooks.sh)
to run the grid-wide smoke when files under
`src/digital_registrar/ablations/` change. Ten minutes of
pre-push insurance against pushing a broken cell that wastes a
multi-day sweep.

## 7. Running a real grid

After a green smoke:

```bash
# grid_1.yaml ships pointing at folder=examples/dummy / dataset=tcga / model=gptoss.
python scripts/ablations/run_grid.py --config configs/ablations/grid_1.yaml
python scripts/ablations/run_grid.py --config configs/ablations/grid_1.yaml \
    --folder workspace --dataset tcga
```

Each runner writes (see canonical layout above):

- `{cell_id}/{model_slug}/{run_id}/{organ_n}/{case_id}.json` — per-case prediction
- `{cell_id}/{model_slug}/{run_id}/_summary.json` — run-level totals
- `{cell_id}/{model_slug}/{run_id}/_log.jsonl` — one row per case
- `{cell_id}/{model_slug}/{run_id}/_run.log` — full-verbosity log
- `{cell_id}/{model_slug}/{run_id}/_run_meta.json` — git SHA, UTC, decoding kwargs
- `{cell_id}/{model_slug}/_manifest.yaml` — accumulated across all runs

Top-level: `{folder}/results/ablations/{dataset}/_grid_meta.json` — full grid manifest.

For ad-hoc per-cell runs, wrappers all take the same args as the
underlying runners:

```bash
python scripts/ablations/run_cell_b.py --folder examples/dummy --dataset tcga --model gptoss
python scripts/ablations/run_cell_c.py --folder examples/dummy --dataset tcga --model gptoss
python scripts/ablations/run_cell_b6.py --folder examples/dummy --dataset tcga --model gptoss
python scripts/ablations/run_cell_c2.py --folder examples/dummy --dataset tcga --model gptoss
python scripts/ablations/run_cell_c5.py --folder examples/dummy --dataset tcga --model gptoss \
    --compiled workspace/compiled/dspy_compiled_gptoss.json
```

## 8. Aggregator outputs

`run_ablations.main()` writes the following under `--results-root`
(default: `workspace/results/ablations/`):

| File | Contents |
|---|---|
| `ablation_grid.csv` | Long-form: one row per (cell, model, case, field) |
| `ablation_summary.csv` | Per-(cell, model, field): accuracy + coverage + nested F1 |
| `ablation_table.csv` | Pivot: rows=field, cols=`<cell>_<model>`, cells=accuracy |
| `cell_deltas.csv` | A→B and B→C per-field deltas per model |
| `efficiency.csv` | Mean / median latency, schema-error rate, parse-error rate |

When `--with-stats` is on (default for any non-smoke results-root) the
aggregator calls
[`ablations.eval.stats.run_all`](../../src/digital_registrar/ablations/eval/stats.py)
to emit the full statistics pack:

| File | Contents |
|---|---|
| `ablation_paired_deltas.csv` | Per (target cell × model × field) Δ vs baseline with paired-bootstrap 95% CI; McNemar discordant counts + p for binary fields. |
| `ablation_paired_deltas_corrected.csv` | Same with `p_holm` (primary endpoints, FWER) and `p_bh` (secondary, FDR), grouped by `(axis, endpoint_tier)` per [`configs/ablations/axes.yaml`](../../configs/ablations/axes.yaml) and [`configs/eval_endpoints.yaml`](../../configs/eval_endpoints.yaml). |
| `ablation_glmm.csv` | Per-(cell × field) marginal accuracy from a mixed-effects logistic GLMM with random intercepts for case and seed; falls back to two-source bootstrap when convergence fails. Multi-seed only. |
| `ablation_seed_consistency.csv` | Fleiss κ across seeds, flip rate, min pairwise Spearman ρ — diagnoses cell determinism. |
| `ablation_factorial.csv` + `ablation_marginal_means.csv` | Grid 2 only: term-level effects from `correct ~ A * B * C + (1|case) + (1|model)` plus per-axis-level marginal accuracy with Wilson CI. |
| `ablation_efficiency_stats.csv` | Schema/parse error rate with Wilson CI, median latency with bootstrap CI per cell. |
| `ablation_effect_sizes.csv` | Cohen's d, Cliff's δ, and odds ratio (binary fields, Haldane–Anscombe corrected) for each cell vs baseline. |

Scoring is reused verbatim from
[`benchmarks.eval.{scope, metrics, completeness}`](../eval/index.md) so
ablation cell numbers slot directly into the benchmark comparison
tables. The stats module is a thin wrapper over the shared toolkit
documented in [`eval/ci_methods.md`](../eval/ci_methods.md),
[`eval/multirun.md`](../eval/multirun.md), and
[`eval/multiple_comparisons.md`](../eval/multiple_comparisons.md) — it
does **not** reimplement bootstrap, McNemar, GLMM, Fleiss κ, or
Holm/BH correction, only wires them to the ablation grid.

To regenerate the stats pack from an existing grid:

```bash
python scripts/ablations/run_stats.py --results-root workspace/results/ablations
```

## 9. Reading the result for the paper

The headline figure is the **lesion table**: full → −Decomposition →
−ReportJsonize → −DSPy → −Schema. Each column reports per-field macro
accuracy (FAIR_SCOPE) with paired-bootstrap 95% CI vs. the full
pipeline. The completeness columns show *where* each lesion bleeds
quality:

- A → B (modularity off) typically loses on **breast-biomarker** and
  **regional_lymph_node** because monolithic context fills up first on
  multi-list organs.
- B → C (DSPy off) typically loses on **schema conformance** —
  Cell C's parse-error rate climbs on local LMs.
- B `--skip-jsonize` typically loses on **fields buried in narrative
  prose** (anything in "Comments" or "Final Diagnosis" sections), which
  the intermediate JSON structuring step normally surfaces.

The OFAT factorial in Grid 2 is reported as supplementary depth.

## 10. Known risks / threats to validity

- **Context-window saturation on `gpt-oss:20b` × monolithic.** The
  monolithic Cell B may also be context-window limited for breast
  (≥ 7 nested field groups) — specifically the biomarkers + LN +
  margins combination can overflow a 16k-token context with a long
  report. Detected and flagged in
  [`runners/dspy_monolithic.py`](../../src/digital_registrar/ablations/runners/dspy_monolithic.py);
  if it fires, the finding itself is a result (modularity is necessary,
  not merely helpful, for that organ at that model size).
- **Schema drift.** JSON schemas under
  [`schemas/data/`](../../src/digital_registrar/schemas/data/)
  may drift from the Pydantic source. `registrar-schemas --check`
  asserts top-level Literal-vocab parity in CI; run before every
  ablation kickoff.
- **DSPy version pinning.** DSPy's prompt-construction behaviour has
  changed materially between minor releases. Pin the version in
  [`pyproject.toml`](../../pyproject.toml) and capture it in the run
  manifest via `_run_meta.json`'s git SHA + DSPy version field.
- **Seed scope.** Currently the seed is set in
  [`configs/dspy_ollama_<model>.yaml`](../../configs/) and applies to
  all cells using that model. To run multi-seed ablations, copy the
  grid YAML, change `slug` per seed, and bump `decoding.seed` between
  invocations. Per-run seed override is `[future]` — TODO markers in
  [`run_grid.py`](../../scripts/ablations/run_grid.py).

## 11. Pre-registration discipline

Before kicking off a real grid:

1. Pin the gold-annotation set — record the SHA-256 of the relevant
   `<folder>/data/<dataset>/annotations/gold/` tree before kicking off
   a grid, so a later run on the same fixture is comparable.
2. Pre-register endpoints in
   [`configs/eval_endpoints.yaml`](../../configs/eval_endpoints.yaml)
   — primary endpoint = per-field macro accuracy on FAIR_SCOPE;
   secondary = nested F1, completeness, latency.
3. Multiple-comparisons correction within each axis: Holm-Bonferroni
   ([`scripts/eval/_common/stats_extra.py`](../../scripts/eval/_common/stats_extra.py)).
4. Reference the locked endpoint config (with git SHA) in the paper
   Methods section.

Pre-registration matters because the eval suite produces hundreds of
p-values across fields × organs × methods × cells; without locking the
primary-endpoint set in advance, post-hoc selection of "significant"
results is statistically unsound.

## 12. 2026-04 redesign — what changed and why

> Date-stamped audit log (2026-04). Below is the list of silent-bug
> fixes that made earlier ablation numbers unreliable. Anyone reading
> a pre-2026-04 result should consult this section before drawing
> comparisons.

- **`monolithic` signature was empty.** `signatures.monolithic._iter_output_fields`
  inspected `cls.__dict__`, but DSPy stores fields in
  `cls.output_fields` — so every merged signature came back with zero
  output fields. `dspy_monolithic`, `str_outputs`, `chain_of_thought`,
  and `fewshot_demos` all built degenerate predictors and returned
  empty `cancer_data`. Fixed by reading `output_fields` directly.
- **`str_outputs` destroyed nested structure.** Coercing every type
  to `str | None` flattened `list[BreastBiomarker]` etc. into prose.
  Now only scalar Literal/int/float/bool leaves are coerced; nested
  Pydantic-list fields keep their structure so the ablation tests
  enum-discipline loss without conflating it with structure loss.
- **Folder-number → organ name was alphabetical.** `IMPLEMENTED_ORGANS[idx-1]`
  was wrong for both TCGA and CMUH. Replaced with two-stage
  resolution: rule-based keyword classifier on report text
  (`ablations.utils.organ_classifier`), with the dataset-aware
  `benchmarks.organs.organ_n_to_name` as a last-resort fallback. Used
  in `no_router`, `per_section`, `build_fewshot_demos`, and
  `utils.demos`. The `is_cancer` LLM router is unchanged in the four
  DSPy-routed cells.
- **`stats._split_method` mis-parsed multi-underscore methods.**
  `rsplit("_", 1)` on `"free_text_regex_gpt_oss_20b"` returned
  `("free_text_regex_gpt_oss", "20b")`. Switched to reading the
  explicit `cell` and `model` columns; the string-split path remains
  as a defensive fallback that walks known cell-ids longest-first.
- **Efficiency stats double-counted overlapping errors.** A case with
  both `_schema_errors` and `_error` flags went into both buckets, so
  rates could exceed 1.0. Now the aggregator tracks `schema_only` /
  `parse_only` / `both` and exposes `failed_total` for the
  non-overlapping union.
- **Median-latency CI ledger path was wrong.** The old path
  `{cell}_{model}/_ledger.json` did not match the canonical layout,
  so the CI was always `None`. Fixed by aggregating per-case latencies
  from `results_root/{cell}/{model}/{run_id}/_log.jsonl`.
- **DSPy execution tracing.** `--trace-dspy` (or `-v`) now dumps every
  DSPy LM call (rendered prompt + raw response) into
  `_dspy_trace.jsonl`. Combined with `is_cancer -> excision=… category=…`
  and `invoking <organ> predictor (n_fields=…)` decision-point logs
  and the `NOT_CANCER=… UNKNOWN_ORGAN=… DOWNSTREAM=…` summary
  counters, the silent-skip class of bugs is now immediately visible.
- **Aggregator validates organ-folder alignment.**
  `build_grid_dataframe` compares each prediction's `cancer_category`
  against `organ_n_to_name(dataset, organ_n)` and surfaces mismatches
  as a per-case warning + an `organ_folder_mismatch` column in the
  grid CSV.
- **Grid driver pre-flight + per-cell try/catch.** `run_grid.py`
  validates cell-ids, model aliases, required artifacts
  (`compiled_dspy`'s `compiled:` path, `fewshot_demos`'s
  `fewshot_demos.yaml`), and the data-layout existence BEFORE the
  first cell starts. Per-cell try/catch (with `--continue-on-cell-error`)
  writes a `grid_failures.json` so re-runs only target the failed
  subset.
- **`reference` folder shorthand.** `--folder reference` builds a
  one-time symlink staging tree under `tests/fixtures/reference/_staged/`
  from the canonical TCGA reports at
  `tests/fixtures/reference/reports/<organ_n>/*.txt` so M2-mac smoke
  runs use real TCGA data without restructuring the on-disk source.
