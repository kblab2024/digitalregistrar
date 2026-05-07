# Ablation studies

> **Cascade-redesign note (2026-05).** The ablation harness now grades predictions through the cascade-aware `score_case`. Cases that fail the upstream gates (Stage A `cancer_excision_report` or Stage B `cancer_category`) contribute fewer field-level rows to `ablation_grid.csv`; per-cell accuracy denominators reflect the cohort that reached each stage. Per-cell accuracy may rise vs the pre-cascade tables because the cohort is honestly restricted. Field exclusions (`ajcc_version`, `treatment_effect`, margin `description`, lymph-node `station_name`) and the per-organ biomarker whitelist (`{er, pr, her2, ki67}` for breast; `{msh2, msh6, pms2, mlh1}` for colorectal) apply uniformly across all cells, so modular-vs-monolithic comparisons remain apples-to-apples. See [eval/CHANGELOG.md](eval/CHANGELOG.md) and [stat_methods.md](stat_methods.md).

The Digital Registrar pipeline makes joint design choices — DSPy as the
LM-calling framework, schema constraints realised through DSPy `Literal`
type hints, and a per-organ modular decomposition into 5–7 sub-signatures
chained through an intermediate `ReportJsonize` step. The ablation suite
attributes the headline accuracy back to those choices, one knob at a time.

## Canonical layout

Every ablation runner uses the same `--folder/--dataset/--model` contract
as [`scripts/pipeline/run_dspy_ollama_single.py`](../scripts/pipeline/run_dspy_ollama_single.py)
so input data, output predictions, and model aliases share one directory
convention with the rest of the toolkit.

```
{folder}/                                      # 'dummy' | 'workspace' | 'workspace_obfustrated' | 'reference' | abs path
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

All cell-runner wrappers also accept `--obfustrated` (additive shortcut)
and `--folder obfustrated` to point at the schema-conformant synthetic
workspace at `workspace_obfustrated/`. Useful for end-to-end debugging
without exposing PHI; see [obfuscation.md](obfuscation.md).

**`{organ_n}` is a dataset-specific numeric folder name**, NOT an
alphabetical ordering. The mapping lives in
[`configs/organ_code.yaml`](../configs/organ_code.yaml) and is loaded by
[`benchmarks.organs`](../src/digital_registrar_research/benchmarks/organs.py):

| folder | TCGA          | CMUH      |
|--------|---------------|-----------|
| 1      | breast        | pancreas  |
| 2      | colorectal    | breast    |
| 3      | esophagus     | cervix    |
| 4      | stomach       | colorectal|
| 5      | liver         | esophagus |
| 6      | —             | liver     |
| 7      | —             | lung      |
| 8      | —             | prostate  |
| 9      | —             | stomach   |
| 10     | —             | thyroid   |

Always convert `organ_n` ↔ `organ_name` via
`benchmarks.organs.organ_n_to_name(dataset, organ_n)` /
`organ_name_to_n(dataset, name)` — never `IMPLEMENTED_ORGANS[idx-1]`,
which is alphabetical and therefore wrong for both datasets.

`--folder dummy`, `--folder workspace`, and `--folder reference` are the
standard shortcuts. The `reference` shortcut builds a one-time symlink
staging tree under `reference/_staged/` from the canonical TCGA reports
at `reference/reports/<organ_n>/*.txt` so the M2-mac smoke runs can use
real TCGA data without restructuring it. (The reference checkout is
reports-only — gold annotations aren't staged, so eval-with-gold needs a
`workspace` checkout.) Absolute paths and other relatives are accepted
and resolved against the repo root via
[`_config_loader.resolve_folder`](../scripts/_config_loader.py).
Models are passed by alias from `models.common.UNIFIED_MODELS`
(`gptoss | gemma3 | gemma4 | gemma4e2b | qwen3_5 | medgemmalarge |
medgemmasmall`). The model slug used in the output path is the same as
the pipeline runner's — `ollama_chat/gpt-oss:20b` → `gpt_oss_20b`.

### Pre-run validation checklist

Run before any multi-day grid:

1. `python -c "from digital_registrar_research.benchmarks.organs import dataset_organs; print(dataset_organs('tcga'))"` — confirm the loader sees the YAML.
2. `ls {folder}/data/{dataset}/reports/` — folder names must be numeric and match the table above for your dataset.
3. Run a single cell with `--limit 1 --trace-dspy --verbose` and read the printed summary line — it ends with
   `NOT_CANCER=… UNKNOWN_ORGAN=… DOWNSTREAM=…`. If `DOWNSTREAM=0` the
   runner never invoked the organ-specific predictor — check
   `_dspy_trace.jsonl` for the rendered prompts and raw responses.
4. Run `scripts/ablations/run_grid.py --config <smoke-yaml>` — pre-flight
   validation will reject typos in `cell:` / `model:` and missing
   artifacts before the first cell starts.

### Skip taxonomy

Every DSPy-routed cell (`dspy_monolithic`, `str_outputs`,
`chain_of_thought`, `fewshot_demos`) emits a `_skip_reason` flag on
each per-case JSON when the downstream organ predictor is NOT
invoked:

* `not_cancer` — the upstream `is_cancer` router said the report is
  not a primary-excision report.
* `unknown_organ` — `is_cancer.cancer_category` returned `"others"`
  (or some value not in `models.modellist.organmodels`).
* (no `_skip_reason`) — the downstream predictor ran;
  `_downstream_called: true` is set on the payload.

These tally into `n_skipped_not_cancer` / `n_skipped_unknown_organ` /
`n_downstream_called` in `_summary.json` and are surfaced in the printed
summary line so you can immediately tell whether the cell is doing the
work the design says it should.

```
src/digital_registrar_research/ablations/
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
    └── stats.py              # paired-bootstrap CI, McNemar, GLMM, Fleiss κ, ...

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
├── smoke.yaml                # smoke defaults (n, cells, folder, dataset, model)
├── grid_1.yaml               # Grid 1 minimum-viable lesion study
├── grid_2.yaml               # Grid 2 27-cell factorial
├── axes.yaml                 # cell → axis mapping for the stats family correction
└── fewshot_demos.yaml        # generated by build_fewshot_demos.py
```

The runners expose both a `run(args: argparse.Namespace) -> int` entry
point and a `main(argv=None) -> int`. Wrappers under
`scripts/ablations/` go through `main()`; the YAML grid driver
constructs a Namespace and calls `run()` directly.

## Ablation axes

The pipeline's design choices fall into six axes — pipeline decomposition,
output structuring, prompting, decoding, model identity, and schema
specificity. This document tracks **what's currently implemented** vs
what's wired up as a TODO in [run_grid.py](../scripts/ablations/run_grid.py).

### Axis 1 — Pipeline decomposition

| Level | Description | Status |
|---|---|---|
| A1 | Full modular (5–7 signatures per organ) | Cell A — implemented |
| A2 | Monolithic single signature per organ | Cell B — implemented |
| A3 | Monolithic, no `ReportJsonize` | Cell B `--skip-jsonize` |
| A4 | Monolithic, no `is_cancer` router | `runners/no_router.py` — implemented (uses gold organ; upper-bound estimate of router contribution) |
| A5 | Per-section decomposition (header / gross / micro / dx / comments) | `runners/per_section.py` — implemented |

### Axis 2 — Output structuring discipline

| Level | Description | Status |
|---|---|---|
| B1 | DSPy + Literal enums + Pydantic | Cells A, B — current default |
| B2 | DSPy with `str` outputs + post-hoc parser | `runners/str_outputs.py` — implemented |
| B3 | Raw JSON-mode (`response_format=json_object`) | Cell C — implemented |
| B4 | Constrained decoding (outlines / lm-format-enforcer) | `runners/constrained_decoding.py` — implemented (requires `outlines` + vLLM/HF backend) |
| B5 | GBNF grammar (llama.cpp) | skipped (not on Ollama path) |
| B6 | Free-text + regex post-extractor | `runners/free_text_regex.py` — implemented |

### Axis 3 — Prompting strategy

| Level | Description | Status |
|---|---|---|
| C1 | Zero-shot signature docstring | current default |
| C2 | + 3 in-context examples (curated from train) | `runners/fewshot_demos.py --n-shots 3` — implemented |
| C3 | + 5 in-context examples (curated from train) | `runners/fewshot_demos.py --n-shots 5` — implemented |
| C4 | `dspy.ChainOfThought` wrapper | `runners/chain_of_thought.py` — implemented |
| C5 | Compiled DSPy program (`BootstrapFewShotWithRandomSearch`) | `runners/compiled_dspy.py` (+ `scripts/ablations/compile_dspy.py`) — implemented |
| C6 | Minimal raw prompt (degenerate baseline) | `runners/minimal_prompt.py` — implemented |

### Axes 4–6 — Decoding, model identity, schema specificity

Decoding (temperature / num_ctx / self-consistency) and model identity
(`gemma3:4b/27b`, `gpt-oss:20b`, `qwen3:30b`, `medgemma`, `llama3-med42`)
are config-driven via [`configs/dspy_ollama_<model>.yaml`](../configs/);
no new runners needed for sweeps along those axes — pass a different
`--model` key.

Schema specificity (Axis 6) gets two new runners:

| Level | Description | Status |
|---|---|---|
| F1 | Per-organ schema | Cell C — current default |
| F2 | Union schema across all organs | `runners/union_schema.py` — implemented |
| F3 | Flat (denested) per-organ schema | `runners/flat_schema.py` — implemented |

## The Grid 1 lesion study

The minimum-viable lesion study lives at
[`configs/ablations/grid_1.yaml`](../configs/ablations/grid_1.yaml).

Conditions:

1. **Full pipeline** — modular DSPy + ReportJsonize + Literal enums
2. **Monolithic DSPy** — drops the modular per-section chain
3. **Monolithic DSPy without ReportJsonize** — also drops the
   intermediate JSON structuring step
4. **No DSPy** — raw OpenAI-compatible JSON-mode against local Ollama
5. **No schema** — free-text generation + regex post-extractor

Single backbone (`gptoss` → `ollama_chat/gpt-oss:20b`), single seed for
the first pass; for multi-seed reproducibility, invoke the script
multiple times — each invocation auto-picks the next free `runNN` slot
under each cell's directory, and decoding seeds come from
[`configs/dspy_ollama_<alias>.yaml`](../configs/) (the
`run_dspy_ollama_multirun.py` driver wraps multiple invocations with a
master seed for reproducibility).

Wall-clock estimate: ~2–3 days on a single 48 GB GPU.

## Smoke runners — pre-flight before a multi-day sweep

Catching a typo or schema-binding regression eight hours into a sweep is
catastrophic. Every ablation kickoff goes through smoke first.

### Smoke contract

- **1 model**, **1 seed**, **2–3 cases** (default `--n 2`)
- Output dir prefixed with `_smoke_<YYYYMMDD-HHMM>/` so the regular
  aggregator's directory glob ignores it. Smoke never contaminates the
  real grid.
- **Fail loud** — any cell exception propagates; exit code ≠ 0 on
  partial failure.
- **Round-trip the aggregator** — smoke calls
  `digital_registrar_research.ablations.eval.run_ablations.main()` with
  `--results-root <smoke dir>` and asserts `ablation_summary.csv` is
  non-empty before returning success. A green smoke means **both** the
  cell and the eval reader work end-to-end.
- **Wall-time target**: ≤ 10 minutes for grid-wide smoke.

### Per-cell smoke

```bash
# Cell C × local gptoss — fastest smoke (no DSPy bootstrap)
python scripts/ablations/run_cell_smoke.py --cell c \
    --folder dummy --dataset tcga --model gptoss

# Cell B × local gptoss
python scripts/ablations/run_cell_smoke.py --cell b \
    --folder dummy --dataset tcga --model gptoss --n 2

# Cell A — copies the most recent completed pipeline run
python scripts/ablations/run_cell_smoke.py --cell a \
    --folder dummy --dataset tcga --model gptoss
```

### Grid-wide smoke

```bash
# Smoke the default cells on local gptoss
python scripts/ablations/run_grid_smoke.py \
    --folder dummy --dataset tcga --model gptoss

# Subset cells (skip B if Ollama bootstrap is slow on this machine)
python scripts/ablations/run_grid_smoke.py \
    --folder dummy --dataset tcga --model gptoss --cells c b6 c6
```

### Recommended pre-push hook

Extend [`scripts/repo/install_git_hooks.sh`](../scripts/repo/install_git_hooks.sh)
to run the grid-wide smoke when files under
`src/digital_registrar_research/ablations/` are touched. Ten minutes of
pre-push insurance against pushing a broken cell that wastes a multi-day
sweep.

## Running a real grid

After a green smoke:

```bash
# grid_1.yaml ships pointing at folder=dummy / dataset=tcga / model=gptoss.
# Override --folder / --dataset on the CLI or edit the YAML for workspace runs.

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

For ad-hoc per-cell runs without the grid driver, the wrappers all take
the same args as the underlying runners:

```bash
python scripts/ablations/run_cell_b.py --folder dummy --dataset tcga --model gptoss
python scripts/ablations/run_cell_c.py --folder dummy --dataset tcga --model gptoss
python scripts/ablations/run_cell_b6.py --folder dummy --dataset tcga --model gptoss
python scripts/ablations/run_cell_c2.py --folder dummy --dataset tcga --model gptoss   # 3-shot
python scripts/ablations/run_cell_c5.py --folder dummy --dataset tcga --model gptoss \
    --compiled workspace/compiled/dspy_compiled_gptoss.json
```

## Aggregator outputs

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
aggregator also calls
[`ablations.eval.stats.run_all`](../src/digital_registrar_research/ablations/eval/stats.py)
to emit the full statistics pack:

| File | Contents |
|---|---|
| `ablation_paired_deltas.csv` | Per (target cell × model × field) Δ vs baseline with paired-bootstrap 95 % CI; McNemar discordant counts + p for binary fields. |
| `ablation_paired_deltas_corrected.csv` | The same table with `p_holm` (primary endpoints, FWER) and `p_bh` (secondary, FDR), grouped by `(axis, endpoint_tier)` per [`configs/ablations/axes.yaml`](../configs/ablations/axes.yaml) and [`configs/eval_endpoints.yaml`](../configs/eval_endpoints.yaml). |
| `ablation_glmm.csv` | Per-(cell × field) marginal accuracy from a mixed-effects logistic GLMM with random intercepts for case and seed; falls back to two-source bootstrap when convergence fails. Multi-seed grids only. |
| `ablation_seed_consistency.csv` | Fleiss κ across seeds, flip rate, min pairwise Spearman ρ — diagnoses cell determinism. |
| `ablation_factorial.csv` + `ablation_marginal_means.csv` | Grid 2 only: term-level effects from `correct ~ A * B * C + (1|case) + (1|model)` plus per-axis-level marginal accuracy with Wilson CI. |
| `ablation_efficiency_stats.csv` | Schema/parse error rate with Wilson CI, median latency with bootstrap CI per cell. |
| `ablation_effect_sizes.csv` | Cohen's d, Cliff's δ, and odds ratio (binary fields, Haldane–Anscombe corrected) for each cell vs baseline. |

Scoring is reused verbatim from
[`benchmarks.eval.{scope, metrics, completeness}`](eval/index.md) so
ablation cell numbers slot directly into the benchmark comparison
tables. The stats module is a thin wrapper over the shared toolkit
documented in [`eval/ci_methods.md`](eval/ci_methods.md),
[`eval/multirun.md`](eval/multirun.md), and
[`eval/multiple_comparisons.md`](eval/multiple_comparisons.md) — it
does **not** reimplement bootstrap, McNemar, GLMM, Fleiss κ, or
Holm/BH correction, only wires them to the ablation grid.

To regenerate the stats pack from an existing grid run:

```bash
python scripts/ablations/run_stats.py --results-root workspace/results/ablations
```

## Reading the ablation result for the paper

The headline figure is the **lesion table**: full → −Decomposition →
−ReportJsonize → −DSPy → −Schema. Each column reports per-field macro
accuracy (FAIR_SCOPE) with paired-bootstrap 95 % CI vs. the full
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

## Pre-registration discipline

Before kicking off a real grid:

1. Pin the gold-annotation set: record the SHA-256 of the relevant
   `<folder>/data/<dataset>/annotations/gold/` tree before kicking off
   a grid, so a later run on the same fixture is comparable.
2. Pre-register endpoints in
   [`configs/eval_endpoints.yaml`](../configs/eval_endpoints.yaml) —
   primary endpoint = per-field macro accuracy on FAIR_SCOPE; secondary
   = nested F1, completeness, latency.
3. Multiple-comparisons correction within each axis: Holm-Bonferroni
   ([`scripts/eval/_common/stats_extra.py`](../scripts/eval/_common/stats_extra.py)).
4. Reference the locked endpoint config (with git SHA) in the paper
   Methods section.

## Related documentation

- [ablations_design_rationale.md](ablations_design_rationale.md) — the
  why-these-cells reasoning
- [eval/index.md](eval/index.md) — the metric explanations every
  ablation number traces back to
- [eval/ci_methods.md](eval/ci_methods.md) — Wilson / BCa / paired
  bootstrap / GLMM CIs
- [eval/multirun.md](eval/multirun.md) — Fleiss κ and multi-run
  consistency
- [eval/multiple_comparisons.md](eval/multiple_comparisons.md) —
  Holm-Bonferroni vs BH-FDR
