# Canonical data + output layout

Every method (rule, BERT, LLM) reads inputs from the same place and writes outputs to the same place. The single source of truth is `scripts/eval/_common/paths.py:Paths` — eval subcommands and run scripts must never construct path strings directly, they go through that class.

## Input layout (`{folder}/data/{dataset}/`)

`{folder}` is one of `dummy`, `workspace`, `workspace_obfustrated`, or an
absolute path. The first three are sibling directories at the repo root that
serve different purposes (real PHI / small unit-test fixture / real-shape
synthetic-from-real for debugging — see [../obfuscation.md](../obfuscation.md)).

```
{folder}/data/{dataset}/
├── reports/{organ_n}/{case_id}.txt           Pathology report text, one file per case
├── annotations/
│   ├── gold/{organ_n}/{case_id}.json         Curated ground truth
│   ├── nhc_with_preann/{organ_n}/{case_id}.json    Annotator NHC, with pre-annotation
│   ├── nhc_without_preann/{organ_n}/{case_id}.json Annotator NHC, no pre-annotation
│   ├── kpc_with_preann/{organ_n}/{case_id}.json    Annotator KPC, with pre-annotation
│   └── kpc_without_preann/{organ_n}/{case_id}.json Annotator KPC, no pre-annotation
└── preannotation/{model}/{organ_n}/{case_id}.json  LLM-generated pre-annotation seeds
```

`{organ_n}` is the 1-based integer organ index. The mapping is **dataset-specific** — single source of truth is [`configs/organ_code.yaml`](../../configs/organ_code.yaml), loaded by `src/digital_registrar_research/benchmarks/organs.py` (and re-exposed via `scripts/eval/_common/stratify.py` for eval scripts).

**TCGA** (5 organs):

| `organ_n` | Organ |
|---|---|
| 1 | breast |
| 2 | colorectal |
| 3 | esophagus |
| 4 | stomach |
| 5 | liver |

**CMUH** (10 organs):

| `organ_n` | Organ |
|---|---|
| 1 | pancreas |
| 2 | breast |
| 3 | cervix |
| 4 | colorectal |
| 5 | esophagus |
| 6 | liver |
| 7 | lung |
| 8 | prostate |
| 9 | stomach |
| 10 | thyroid |

The cross-corpus baseline (train on CMUH, predict on TCGA) operates on the **5 organs both datasets share**: `breast, colorectal, esophagus, stomach, liver`.

`{case_id}` follows the pattern `{dataset}{organ_n}_{idx}` — and `organ_n` is interpreted via the dataset-specific mapping. So `cmuh1_17` is *pancreas* case 17 in CMUH; `tcga1_17` is *breast* case 17 in TCGA.

### Generating dummy data

```bash
python scripts/data/gen_dummy_skeleton.py --out dummy --clean \
    --cases-per-organ cmuh:100,tcga:50 --llm-runs 3
```

The `dummy/` tree is fully synthetic but schema-valid, so the entire benchmark + eval pipeline can be exercised end-to-end without touching real data.

### Real data (`workspace/`)

The contract: drop your reports + gold annotations into `workspace/data/{dataset}/...` matching the same tree. The runners and eval scripts default to `--folder workspace` and treat it exactly the same as `dummy` — only the contents differ.

### No train/test split inside a corpus

The cross-corpus baseline does not partition any single corpus into train and test folds. CMUH is the training corpus; TCGA is the prediction corpus. Disjointness is guaranteed by the dataset boundary, and `run_bert.py` enforces it at predict time by reading the checkpoint's `datasets` metadata and refusing to predict on any dataset that was in training.

Cases are discovered by walking `{folder}/data/{dataset}/annotations/gold/<organ_n>/*.json`. Adding new cases is a matter of dropping new annotation+report files in those folders; no manifest or split file needs to be regenerated.

## Output layout (`{folder}/results/predictions/{dataset}/`)

```
{folder}/results/predictions/{dataset}/
├── rule_based/                               No model, no run_id — deterministic
│   ├── {organ_n}/{case_id}.json
│   ├── _summary.json     run-level totals + per-organ counts
│   ├── _log.jsonl        one JSONL row per case
│   ├── _run.log          full-verbosity log
│   └── _run_meta.json    provenance (git sha, host, argv, ...)
├── clinicalbert/{model}/                     model ∈ {cls, qa, merged}
│   ├── {organ_n}/{case_id}.json
│   └── _{summary,run_meta}.json (at parent clinicalbert/ level)
└── llm/{model}/                              model ∈ {gpt_oss_20b, gemma3_27b, ...}
    ├── _manifest.yaml                        aggregated across all runs
    └── {run_id}/                             e.g. run01, run02-alpha
        ├── {organ_n}/{case_id}.json
        ├── _summary.json
        ├── _log.jsonl
        ├── _run.log
        └── _run_meta.json
```

The depth of the model / run_id path depends on the method:

| Method | Path template |
|---|---|
| rule_based | `rule_based/{organ_n}/{case_id}.json` (flat) |
| clinicalbert | `clinicalbert/{model}/{organ_n}/{case_id}.json` |
| llm | `llm/{model}/{run_id}/{organ_n}/{case_id}.json` |

`Paths.prediction(method, model, run_id, organ_idx, case_id)` resolves the right shape per method — no hand-built path strings anywhere in eval / run scripts.

## Per-case prediction JSON shape

Every method's per-case JSON file uses the same shape (matching the gold annotation):

```json
{
  "cancer_excision_report": true,
  "cancer_category": "breast",
  "cancer_category_others_description": null,
  "cancer_data": {
    "pt_category": "t2",
    "pn_category": "n1mi",
    "pm_category": "mx",
    "grade": "2",
    "histology": "invasive_carcinoma_no_special_type",
    "lymphovascular_invasion": true,
    "tumor_size": 25
  }
}
```

**Coverage convention:** A field is "attempted" if the method emits an explicit entry (a value or explicit `null`). Missing keys count as "not attempted" and drop out of the accuracy denominator. ClinicalBERT always emits every field its scope covers (sometimes as `null`); rule_based emits only when its regex/lexicon matches. This is intentional and shows up as honest coverage asymmetry in the eval outputs.

## Eval output layout (`{folder}/results/eval/{subcommand}/`)

Each eval subcommand writes to its own subdir. The structure is uniform:

```
{folder}/results/eval/{subcommand}/
├── manifest.json                    CLI args + git sha + UTC timestamp
├── correctness_table.parquet        atomic per-(case, organ, field, run) outcome (non_nested only)
├── per_field_overall.csv            per-field accuracy + Wilson CI
├── per_field_by_organ.csv           per-(field, organ) accuracy
├── ... and other subcommand-specific outputs
```

`scripts.eval.compare.run_compare` consumes the `correctness_table.parquet` files from multiple per-method runs and emits a side-by-side comparison under `{folder}/results/eval/compare/{label}/`.
