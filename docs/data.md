# Datasets — layout and conventions

The repo holds two datasets and the outputs of every method / annotator /
run that touches them. Layout covers the full cross-product so you never
have to guess which file belongs to which condition.

## Top-level layout

```
with_preann/data/
├── cmuh/
│   ├── reports/{organ_n}/{case_id}.txt              # raw pathology reports
│   ├── preannotation/gpt_oss_20b/{organ_n}/{case_id}.json
│   ├── annotations/
│   │   ├── nhc/{organ_n}/{case_id}.json
│   │   ├── kpc/{organ_n}/{case_id}.json
│   │   └── gold/{organ_n}/{case_id}.json            # consensus
│   └── dataset_manifest.yaml
└── tcga/                                             # mirrors cmuh/

without_preann/data/
├── cmuh/                                             # independent subset (no preannotation/)
│   ├── reports/{organ_n}/{case_id}.txt
│   ├── annotations/{nhc,kpc,gold}/{organ_n}/{case_id}.json
│   └── dataset_manifest.yaml
└── tcga/

results/
├── predictions/{dataset}/
│   ├── llm/{model}/run{NN}/{organ_n}/{case_id}.json
│   │                       + _summary.json, _log.jsonl
│   │   {model}/_manifest.yaml
│   ├── clinicalbert/{variant}/[run{NN}/]{organ_n}/{case_id}.json  + _summary.json
│   │                          # run{NN}/ slot present iff trained via train_bert_multirun.py
│   └── rule_based/{organ_n}/{case_id}.json              + _summary.json
└── evaluation/{dataset}/
    ├── iaa/*.csv                                    # inter-annotator agreement
    ├── accuracy/*.csv                               # method vs gold
    ├── ensembles/{model}/{organ_n}/{case_id}.json   # majority-vote across runs
    └── comparisons/*.csv

configs/
├── datasets/{cmuh,tcga}.yaml
├── models/{gpt_oss_20b,gemma4_30b,qwen3_30b,gemma4_e2b,clinicalbert_v1,clinicalbert_v2,rule_based}.yaml
└── annotators/annotators.yaml

models/clinicalbert/{v1_baseline,v2_finetuned}/{checkpoint.pt,config.yaml}
```

## Naming conventions

| Thing | Pattern | Example |
|---|---|---|
| Dataset | lowercase, no date | `cmuh`, `tcga` |
| Case ID | `{dataset}{N}_{idx}` | `tcga1_37`, `cmuh1_1` |
| Organ partition | numeric dir, **dataset-specific** per [`configs/organ_code.yaml`](../configs/organ_code.yaml) | `tcga/.../1/` = breast, `cmuh/.../1/` = pancreas |
| LLM model | snake_case with size | `gpt_oss_20b`, `gemma4_e2b` |
| Run directory | zero-padded, optional machine suffix | `run01`..`run10`, or `run01-alpha`..`run10-alpha` |
| Mode subtree | `{with,without}_preann/` | `with_preann/` |
| Annotator dir | `{annotator}/` (mode implied by parent path) | `nhc/` |
| Sidecar files | leading underscore | `_summary.json`, `_manifest.yaml`, `_log.jsonl` |
| Case files | `{case_id}.json` — annotator/run/model is encoded by the folder | `tcga1_1.json` |

The **leading-underscore sidecar rule** lets any glob of `**/*.json` filter
out metadata files: `[p for p in paths if not p.name.startswith("_")]`.

## The four annotation modes

Two annotators × two conditions = four files per case, plus gold. The
mode is encoded in the top-level subtree (`with_preann/` vs
`without_preann/`); under each subtree, annotator folders drop the mode
suffix:

| Path | Who | What they saw |
|---|---|---|
| `with_preann/data/{dataset}/annotations/nhc/` | Annotator NHC | gpt-oss:20b pre-annotation pre-filled, annotator edits in place |
| `without_preann/data/{dataset}/annotations/nhc/` | Annotator NHC | blank template, annotator fills from scratch |
| `with_preann/data/{dataset}/annotations/kpc/` | Annotator KPC | gpt-oss:20b pre-annotation pre-filled |
| `without_preann/data/{dataset}/annotations/kpc/` | Annotator KPC | blank template |
| `{mode}/data/{dataset}/annotations/gold/` | Consensus | produced by adjudication of the four above |

`with_preann` and `without_preann` are independent datasets — in
production, `without_preann` may hold a different (smaller) subset of
cases than `with_preann`.

This layout enables two comparisons:
1. **Inter-annotator agreement** (`pairwise_nhc_vs_kpc_with_preann.csv`, `…_without_preann.csv`).
2. **Pre-annotation effect** (`preann_effect_nhc.csv`, `preann_effect_kpc.csv`) — same annotator, with vs without the LLM draft.

## Predictions layout — why per-run directories

LLMs are run ≥10 times per model with fixed seeds (42..51) for
stochastic confidence intervals. Each run gets its own directory so the
per-run seed, token count, parse-error rate, and log stream are kept
adjacent to the predictions they describe:

```
results/predictions/cmuh/llm/gpt_oss_20b/
├── run01/
│   ├── 1/cmuh1_1.json
│   ├── ...
│   ├── _summary.json                 # seed, total_tokens, parse_errors, wall_time
│   └── _log.jsonl                    # one JSON per case: case_id, run, seed, tokens, latency
├── run02/ ... run10/
└── _manifest.yaml                    # lists all runs + config hash + validity flags
```

### Multi-machine sweeps

When the same `(dataset, model)` is processed on more than one host, set a
short stable slug per machine so each one writes to a disjoint slot space:

  * Env var (one-shot): `DRR_MACHINE_ID=alpha python scripts/pipeline/run_…`
  * Persistent: `machine_id: alpha` in `configs/local/runtime.yaml`
    (the `configs/local/` tree is gitignored, so each checkout sets its own).

Run dirs then become `run01-alpha .. run10-alpha` on machine *alpha* and
`run01-beta .. run10-beta` on machine *beta*. Both forms still match the
`startswith("run")` discovery glob in
[`benchmarks/eval/multirun.py`](../src/digital_registrar_research/benchmarks/eval/multirun.py),
so the eval aggregator naturally treats them as additional samples for the
confidence interval. Slug format: `^[a-z0-9][a-z0-9-]{0,11}$`.

Majority-vote ensembles live under
`results/evaluation/{dataset}/ensembles/{model}/` and are produced from
the individual runs — they're a derived artifact, not a separate model.

## Paths in code

All hardcoded paths go through [`paths.py`](../src/digital_registrar_research/paths.py).
Downstream code uses the resolver rather than string literals:

```python
from digital_registrar_research.paths import dataset, predictions_dir, evaluation_dir

ds = dataset("cmuh", mode="with_preann")        # -> DatasetPaths
ds.reports, ds.annotations, ds.preannotation    # Path objects
ds.gold_dir, ds.annotator_dir("nhc")            # annotator helpers (mode is on ds)
predictions_dir("cmuh", "llm/gpt_oss_20b", run="run03")
evaluation_dir("cmuh", "iaa")
```

## Dummy skeleton

`python scripts/data/gen_dummy_skeleton.py --out dummy --clean` writes the
entire layout with schema-valid but trivial content — 2 datasets × 2 organs × 3 cases
— so eval scripts can be smoke-tested before real data lands. Regenerate
anytime; it is deterministic (seed = `20251117`).

## Obfuscated workspace (`workspace_obfustrated/`)

A third workspace root, sibling of `workspace/` and `dummy/`, holds a
schema-conformant **synthetic-from-real** copy of the live workspace —
real-shape, real-scale, but byte-for-byte different from `workspace/` and
PHI-free. Generated by `python scripts/obfuscate_workspace.py` (or
`pip install -e obfuscator/ && obfuscate-workspace`). Use it to give a
Claude session (or any external collaborator) a workspace they can read
freely without exposing patient data.

Pointing scripts at it (additive — existing `--folder dummy` / `--folder workspace`
invocations are unchanged):

- `--folder obfustrated` shorthand in any script that calls `resolve_folder`
  (`scripts/_config_loader.py`)
- `--obfustrated` boolean flag on eval, ablation, and pipeline scripts
- `DIGITAL_REGISTRAR_WORKSPACE=workspace_obfustrated` env var (lowest
  precedence — only consulted when `--folder` and `--obfustrated` are absent)

See [obfuscation.md](obfuscation.md) for the threat model, generation pipeline,
and full integration recipe.

## Migration from the legacy TCGA layout

The old convention — `data/tcga_{dataset,result,annotation}_20251117/`
with `_{suffix}` annotator filename tags — is being retired. See
[`testing_migration`](branching_strategy.md) branch and
`scripts/migrate_tcga_layout.py` (to be added) for the byte-equal copy
procedure.

## Provenance

- **TCGA**: public pathology reports sampled to cover ten organs. Initial 151 cases (100 breast + 51 colorectal) produced during the 2025-11 round.
- **CMUH**: local hospital cohort; data collection for the 2026-04 experiment round begins on [`experiment_cmuh_pilot`](branching_strategy.md).

Pre-annotations across both datasets are produced by a single gpt-oss:20b
pass (Ollama / vLLM) so the "with_preann" condition is held constant across annotators.
