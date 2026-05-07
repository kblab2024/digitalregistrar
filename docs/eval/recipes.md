# Canonical CLI invocations (recipes)

Copy-paste examples for every common evaluation task. All examples assume you're at the repo root.

> **Cascade-redesign note (2026-05).** Recipes that reference `cli non_nested` or `cli nested` should be replaced with `cli cascade` — the cascade subcommand is the unified replacement. See [CHANGELOG.md](CHANGELOG.md) for the migration map. New canonical recipe:
>
> ```bash
> # Cascade run on cmuh, single model, two runs.
> python -m scripts.eval.cli cascade \
>     --root workspace --dataset cmuh --annotator gold \
>     --method llm --model gpt_oss_20b --run-ids run01 run02 \
>     --out workspace/results/eval/cascade/cmuh_gpt_oss_20b
> ```
>
> Output: `chapter1_eligibility/`, `chapter2_organ_classification/` (with `others/` ledger), `chapter3_field_extraction/` (with `cascade_funnel.csv`), and `cascade_atomic.parquet` at the root.

## Score one model on /dummy

```
python -m scripts.eval.cli non_nested \
    --root dummy --dataset cmuh \
    --model gpt_oss_20b --annotator gold \
    --out workspace/results/eval/non_nested/cmuh/gpt_oss_20b
```

Auto-discovers all `run*` subdirectories under the model. To restrict to specific runs:

```
python -m scripts.eval.cli non_nested \
    --root dummy --dataset cmuh \
    --model gpt_oss_20b --annotator gold \
    --run-ids run01 run02 \
    --out workspace/results/eval/non_nested/cmuh/gpt_oss_20b__r01_r02
```

## Score one model on /workspace_obfustrated (PHI-free debugging)

The obfuscator produces a schema-conformant synthetic copy of `workspace/`
so eval logic can be exercised end-to-end without exposing patient data.

```
# 1. Generate the synthetic workspace (one-time per workspace state).
python scripts/obfuscate_workspace.py --seed 42

# 2. Run any eval subcommand against it via the --obfustrated shortcut.
python -m scripts.eval.cli non_nested \
    --obfustrated --dataset tcga \
    --model gpt_oss_20b --annotator gold
# Reads from workspace_obfustrated/, writes under workspace_obfustrated/results/eval/non_nested/.

# Equivalent explicit form:
python -m scripts.eval.cli non_nested \
    --root workspace_obfustrated --dataset tcga \
    --model gpt_oss_20b --annotator gold \
    --out workspace_obfustrated/results/eval/non_nested/tcga/gpt_oss_20b
```

Eval signal is meaningful (annotations are deliberately consistent with
report text up to per-field noise — see [obfuscation.md](../obfuscation.md)
for the layered noise pipeline). Existing `--root dummy` / `--root workspace`
recipes are unchanged.

## Single-run mode

Multi-run is the default; for a single-run analysis just pass one ID:

```
python -m scripts.eval.cli non_nested \
    --root workspace --dataset cmuh \
    --model gpt_oss_20b --run-ids run01 \
    --annotator gold --out workspace/results/eval/non_nested/single_run
```

## Score nested fields

```
python -m scripts.eval.cli nested \
    --root dummy --dataset cmuh \
    --model gpt_oss_20b --annotator gold \
    --field regional_lymph_node \
    --out workspace/results/eval/nested/lymph_nodes

python -m scripts.eval.cli nested \
    --root dummy --dataset cmuh \
    --model gpt_oss_20b --annotator gold \
    --field margins \
    --out workspace/results/eval/nested/margins

python -m scripts.eval.cli nested \
    --root dummy --dataset cmuh \
    --model gpt_oss_20b --annotator gold \
    --field biomarkers \
    --out workspace/results/eval/nested/biomarkers
```

## IAA across all annotators

```
python -m scripts.eval.cli iaa \
    --root workspace --dataset cmuh \
    --annotators gold nhc_with_preann nhc_without_preann \
                 kpc_with_preann kpc_without_preann \
    --preann-model gpt_oss_20b \
    --out workspace/results/eval/iaa
```

To run only a specific subset of pairs:

```
python -m scripts.eval.cli iaa \
    --root workspace --dataset cmuh \
    --annotators gold nhc_with_preann kpc_with_preann \
    --pairs gold:nhc_with_preann gold:kpc_with_preann \
            nhc_with_preann:kpc_with_preann \
    --out workspace/results/eval/iaa
```

## Pair-focused IAA (overall κ + roll-ups for one pair)

Use `iaa_pair` when you want a single overall Cohen's κ (and per-section / per-organ roll-ups, confusion matrices, and a markdown summary) scoped to one or more specific annotator pairs. The general `iaa` subcommand stops at per-field κ + Krippendorff α; `iaa_pair` fills the headline gap. See [iaa_basics.md](iaa_basics.md#pair-focused-headline-iaa_pair) for what each headline statistic means.

### Inter-observer variance — gold vs human

```
python -m scripts.eval.cli iaa_pair \
    --root workspace --dataset cmuh \
    --pair gold:nhc_with_preann \
    --pair gold:kpc_with_preann \
    --out workspace/results/eval/iaa_pair/inter_observer
```

### Inter-observer variance — human vs human

```
python -m scripts.eval.cli iaa_pair \
    --root workspace --dataset cmuh \
    --pair kpc_with_preann:nhc_with_preann \
    --out workspace/results/eval/iaa_pair/human_vs_human
```

### Pre-annotation effect (descriptive — same human, with vs without preann)

```
python -m scripts.eval.cli iaa_pair \
    --root workspace --dataset cmuh \
    --pair kpc_with_preann:kpc_without_preann \
    --pair nhc_with_preann:nhc_without_preann \
    --out workspace/results/eval/iaa_pair/preann_effect_descriptive
```

The summary.md auto-flags within-annotator preann pairs and points to the `iaa preann/` outputs for the **causal** Δκ-vs-gold and anchoring analyses. `iaa_pair` only describes magnitude of change here; it does not answer "does preann help?".

### All pairs the user typically wants in one shot

```
python -m scripts.eval.cli iaa_pair \
    --root workspace --dataset cmuh \
    --pair gold:nhc_with_preann \
    --pair gold:kpc_with_preann \
    --pair kpc_with_preann:nhc_with_preann \
    --pair kpc_with_preann:kpc_without_preann \
    --pair nhc_with_preann:nhc_without_preann \
    --out workspace/results/eval/iaa_pair --n-boot 2000 --seed 0
ls workspace/results/eval/iaa_pair/
# pair_gold_vs_nhc_with_preann/  pair_gold_vs_kpc_with_preann/  ...
cat workspace/results/eval/iaa_pair/pair_gold_vs_nhc_with_preann/headline.csv
```

### Tweak knobs

```
# Smoke run with fast bootstrap (CI noisier but point estimate exact).
python -m scripts.eval.cli iaa_pair --root dummy --dataset cmuh \
    --pair gold:nhc_with_preann --n-boot 100 --seed 0 \
    --out /tmp/iaa_pair_smoke

# More confusion matrices in the output (default 20).
python -m scripts.eval.cli iaa_pair --root workspace --dataset cmuh \
    --pair gold:nhc_with_preann --n-confusion 50 \
    --out workspace/results/eval/iaa_pair/full_confusion
```

## Compare humans with vs without preann (preann-effect headline)

This is built into the `iaa` subcommand — the `preann/` subdirectory of the output captures Δκ, anchoring index, convergence-to-preann, disagreement reduction, and edit distance:

```
python -m scripts.eval.cli iaa \
    --root workspace --dataset cmuh \
    --annotators nhc_with_preann nhc_without_preann \
                 kpc_with_preann kpc_without_preann gold \
    --out workspace/results/eval/iaa
ls workspace/results/eval/iaa/preann/
```

## Modularity-advantage table (ablation headline)

Compare multiple methods on the same case set. The `completeness` subcommand emits the modularity-advantage CSV.

```
python -m scripts.eval.cli completeness \
    --root workspace --dataset cmuh \
    --methods llm:gpt_oss_20b llm:qwen3_30b \
              clinicalbert:v2_finetuned rule_based: \
    --annotator gold \
    --out workspace/results/eval/completeness
cat workspace/results/eval/completeness/modularity_advantage.csv | head
```

## Ablation grid — smoke before sweep

Pre-flight one cell against a 2-case sample. Cell C (raw JSON) is the fastest because there's no DSPy bootstrapping; use it as the canonical smoke target.

```
python scripts/ablations/run_cell_smoke.py --cell c --model gpt-oss:20b --n 2
```

The smoke writes to `workspace/results/_smoke_<YYYYMMDD-HHMM>/` (the leading underscore keeps it out of the regular aggregator's directory glob), round-trips through the aggregator, and asserts `ablation_summary.csv` is non-empty before exiting 0.

Grid-wide smoke (Cells B + C, ≤ 10 minutes):

```
python scripts/ablations/run_grid_smoke.py --model gpt-oss:20b --n 2
```

To include Cell A (which copies pre-computed modular-DSPy predictions), point at the source directory:

```
python scripts/ablations/run_grid_smoke.py --model gpt-oss:20b --n 2 \
    --modular-source-dir E:/experiment/20260422/gpt-oss
```

Both smoke runners are **fail-loud** — any cell exception or empty aggregator CSV propagates as a non-zero exit. A green smoke is the go/no-go for a real sweep.

## Ablation grid — full sweep (Grid 1)

After a green smoke, edit [`configs/ablations/grid_1.yaml`](../../configs/ablations/grid_1.yaml) to point `modular_gpt_oss_dir` at the parent project's full-pipeline output, then:

```
python scripts/ablations/run_grid.py --config configs/ablations/grid_1.yaml
```

Per-cell outputs land at `workspace/results/ablations/<cell_id>_<slug>/{<case_id>.json, _ledger.json, _run_meta.json}`. The driver runs the aggregator at the end; outputs match `registrar-ablate` (`ablation_grid.csv`, `ablation_summary.csv`, `ablation_table.csv`, `cell_deltas.csv`, `efficiency.csv`).

For ad-hoc per-cell runs that bypass the grid driver:

```
python scripts/ablations/run_cell_b.py --model gpt \
    --out workspace/results/ablations/dspy_monolithic_gpt-oss

python scripts/ablations/run_cell_c.py --model gpt-oss:20b \
    --out workspace/results/ablations/raw_json_gpt-oss
```

See [ablations.md](../ablations.md) for the per-cell argument reference and the lesion-study reading guide.

## Ablation aggregator across smoke + real outputs

The aggregator's `--results-root` flag scopes discovery to a specific tree, which is what the smoke runners use under the hood:

```
python -m digital_registrar_research.ablations.eval.run_ablations \
    --results-root workspace/results/_smoke_20260427-1015 \
    --cells dspy_monolithic raw_json \
    --models gpt-oss-20b
```

Without `--results-root` it defaults to `workspace/results/ablations/` — the canonical location for real-sweep outputs.

## Error-source diagnostics

After running `non_nested` and `iaa`:

```
python -m scripts.eval.cli diagnostics \
    --non-nested-out workspace/results/eval/non_nested/cmuh/gpt_oss_20b \
    --iaa-out workspace/results/eval/iaa \
    --out workspace/results/eval/diagnostics
```

Outputs `error_source_decomposition.csv` (model_error vs report_ambiguity vs report_silent buckets) and `accuracy_by_difficulty_tier.csv` (accuracy stratified by IAA-derived field difficulty).

## Cross-dataset generalisation

Score the same model on both datasets, then compare:

```
python -m scripts.eval.cli non_nested \
    --root workspace --dataset cmuh \
    --model gpt_oss_20b --annotator gold \
    --out workspace/results/eval/non_nested/cmuh

python -m scripts.eval.cli non_nested \
    --root workspace --dataset tcga \
    --model gpt_oss_20b --annotator gold \
    --out workspace/results/eval/non_nested/tcga

python -m scripts.eval.cli cross_dataset \
    --left workspace/results/eval/non_nested/cmuh \
    --right workspace/results/eval/non_nested/tcga \
    --out workspace/results/eval/cross_dataset
```

## Joint headline forest plot

Combines non_nested + IAA outputs into a single long-form CSV for plotting:

```
python -m scripts.eval.cli headline \
    --non-nested-out workspace/results/eval/non_nested/cmuh/gpt_oss_20b \
    --iaa-out workspace/results/eval/iaa \
    --out workspace/results/eval/headline
```

## Subset to specific organs

```
python -m scripts.eval.cli non_nested \
    --root dummy --dataset cmuh \
    --model gpt_oss_20b --annotator gold \
    --organs breast lung colorectal \
    --out workspace/results/eval/non_nested/breast_lung_colon
```

Accepts either organ names or numeric indices: `--organs 1 6 3` (breast=1, lung=6, colorectal=3).

## Subset to specific cases

```
python -m scripts.eval.cli non_nested \
    --root dummy --dataset cmuh \
    --model gpt_oss_20b --annotator gold \
    --cases cmuh1_42 cmuh1_43 cmuh1_44 \
    --out workspace/results/eval/non_nested/spot_check

# Or from a file:
python -m scripts.eval.cli non_nested \
    --root dummy --dataset cmuh \
    --model gpt_oss_20b --annotator gold \
    --cases @/tmp/case_list.txt \
    --out workspace/results/eval/non_nested/spot_check
```

## Faster smoke-tests with fewer bootstrap reps

Default `n_boot = 2000` — for smoke tests use `--n-boot 100` for ~20× speedup:

```
python -m scripts.eval.cli non_nested \
    --root dummy --dataset cmuh \
    --model gpt_oss_20b --annotator gold \
    --n-boot 100 \
    --out /tmp/smoke
```

## Score against a non-gold annotator

The `--annotator` arg accepts any annotator subdirectory:

```
python -m scripts.eval.cli non_nested \
    --root workspace --dataset cmuh \
    --model gpt_oss_20b --annotator nhc_with_preann \
    --out workspace/results/eval/non_nested/vs_nhc_with_preann
```

Useful for measuring "how close to NHC's annotations" rather than to gold.

## Score ClinicalBERT or rule_based

These methods don't have run IDs:

```
python -m scripts.eval.cli non_nested \
    --root workspace --dataset cmuh \
    --method clinicalbert --model v2_finetuned \
    --annotator gold \
    --out workspace/results/eval/non_nested/clinicalbert

python -m scripts.eval.cli non_nested \
    --root workspace --dataset cmuh \
    --method rule_based \
    --annotator gold \
    --out workspace/results/eval/non_nested/rule_based
```

## Inspecting outputs

Every output dir contains a `manifest.json` with full provenance:

```
cat workspace/results/eval/non_nested/cmuh/gpt_oss_20b/manifest.json | jq .
```

Use `git_sha` from the manifest when citing numbers in the paper — it's the canonical reproducibility key.
