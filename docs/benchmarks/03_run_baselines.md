# Running predictions (rule, BERT, LLM)

Each baseline has a dedicated runner under `scripts/baselines/` (and `scripts/pipeline/` for LLM). All three write to the canonical predictions tree, share the same `--folder` / `--dataset` arg shape, and emit comparable side files (`_summary.json`, `_log.jsonl`, `_run.log`, `_run_meta.json`).

## Rule-based

```bash
python scripts/baselines/run_rule.py \
    [--folder workspace] [--datasets tcga] \
    [--organs 1 2 3] [--limit 50] [--overwrite] [-v]
```

| Flag | Default | Effect |
|---|---|---|
| `--folder` | `workspace` | Experiment root (`dummy` / `workspace` / `workspace_obfustrated` / abs path). Also accepts the `obfustrated` shorthand for `workspace_obfustrated/` — see [../obfuscation.md](../obfuscation.md). |
| `--datasets` | `tcga` | The LLM-comparable evaluation corpus. Pass `cmuh tcga` for both, or `cmuh` for intra-corpus ablations. |
| `--organs` | Numeric organ subdirs to keep, e.g. `1 2 3`. Default: every organ subdir under `reports/` that has at least one `.txt`. |
| `--limit N` | Cap cases per organ (debugging). |
| `--overwrite` | Reprocess cases even if a valid output already exists. |
| `--tolerate-errors` | Always exit 0 if the script completes (even with per-case failures). |
| `-v` | DEBUG console logging. |

**Notes:**

- The rule baseline classifies the organ from the report itself (lexicon vote, `rules.classify_organ`). It does **not** read the gold annotation. This keeps the floor honest — when the lexicon classifier is wrong, the per-organ extraction emits an empty `cancer_data`.
- Runtime: <1 second per case on a laptop CPU. Pure Python, no GPU, no model.
- **Deterministic — no multirun needed.** Given the same input, the rule baseline produces the same output, so running it K times has no value. The CI you see for rule in the cross-method comparison comes from case-level Wilson intervals (within-method) and case-level paired bootstrap (vs other methods), not from run-level variance.

Output: `{folder}/results/predictions/{dataset}/rule_based/{organ_n}/{case_id}.json` plus side files.

## ClinicalBERT (CLS + QA + merged)

```bash
python scripts/baselines/run_bert.py \
    [--folder workspace] [--datasets tcga] \
    [--heads cls qa merged] \
    [--ckpt-cls ckpts/clinicalbert_cls.pt] \
    [--ckpt-qa  ckpts/clinicalbert_qa] \
    [--organs breast colorectal esophagus liver stomach] [--overwrite] [-v]
```

| Flag | Default | Effect |
|---|---|---|
| `--folder` | `workspace` | Experiment root. |
| `--datasets` | `tcga` | TCGA is held out from CMUH-only training. The leakage guard refuses to predict on a dataset that was in the checkpoint's training set. |
| `--heads` | `cls qa merged` | Heads to run. `merged` requires both `cls` and `qa` outputs (run them in the same call or beforehand). |
| `--ckpt-cls` | `ckpts/clinicalbert_cls.pt` | Path to CLS checkpoint. |
| `--ckpt-qa` | `ckpts/clinicalbert_qa` | Path to QA checkpoint dir. |
| `--organs` | `breast colorectal esophagus liver stomach` | Cancer-category names to keep. |
| `--overwrite` | off | Reprocess cases even if valid outputs exist. |

**Notes:**

- Default predicts on the **full TCGA corpus** since TCGA was held out from CMUH-only training. The leakage guard in `clinicalbert_*.predict` reads the checkpoint's `datasets` metadata and refuses to predict on any dataset that was in training (so `--datasets cmuh` against a CMUH-trained checkpoint will fail loudly).
- Device auto-detected: MPS / CUDA / CPU.
- `merged` does a per-case key-merge: CLS provides the base (carries `cancer_category` + `cancer_excision_report`); QA's `cancer_data` scalars overlay onto CLS's, with CLS winning on collisions.

Output (single-seed): `{folder}/results/predictions/{dataset}/clinicalbert/{cls|qa|merged}/{organ_n}/{case_id}.json`.

Output (multirun): `{folder}/results/predictions/{dataset}/clinicalbert/{cls|qa|merged}/run{NN}/{organ_n}/{case_id}.json`. The `run{NN}/` slot is present iff the multirun trainer was used — see the next subsection.

### Multi-run training (K-seed sweep)

LLMs are stochastic at inference time; BERT is stochastic at *training* time (random head init, dropout, data-shuffle order). To put BERT on the same statistical footing as the LLM K-run sweep, train K seeds and feed each seed's predictions into the cascade as a separate `run_id`:

```bash
# Smoke (2 seeds × 1 epoch each, dummy data — ~3 min total)
python scripts/baselines/train_bert_multirun.py \
    --folder dummy --num-runs 2 --master-seed 1234 \
    --epochs-cls 1 --epochs-qa 1 --datasets tcga

# Full K=10 sweep on workspace (~6.5–7 hr on A6000 Ada)
python scripts/baselines/train_bert_multirun.py \
    --folder workspace --num-runs 10 --master-seed 42 --datasets tcga
```

| Flag | Default | Effect |
|---|---|---|
| `--num-runs N` | 10 | Number of seeds to train. Mirrors LLM `--n`. |
| `--master-seed M` | random (`secrets.randbelow(2**31)`) | Makes the *sequence* of per-iteration seeds reproducible: same master seed → same K seeds → same K checkpoints (modulo CUDA non-determinism, which is intentional). |
| `--epochs-cls`, `--epochs-qa` | inherited from `train_bert.py` | Per-head epoch counts. Same defaults as the single-seed trainer. |
| `--folder`, `--datasets`, `--heads` | as `train_bert.py` | Forwarded verbatim to the per-seed train + predict subprocesses. |

Each iteration draws a fresh 31-bit seed, trains both heads, runs inference into `clinicalbert/{head}/run{NN}/...`, then **deletes the checkpoint** to keep transient disk to ~1× the per-seed footprint. The model-level `_manifest.yaml` at `clinicalbert/merged/_manifest.yaml` lists every seed and its validity flag.

Cascade auto-discovers these run slots — there is no extra flag to pass at evaluation time. See [04_evaluate.md](04_evaluate.md) for the consumption side and [05_compare.md](05_compare.md) for the multirun-vs-multirun-vs-rule comparison wrapper.

## LLM (DSPy + Ollama)

The LLM pipeline is unchanged from the existing canonical workflow — see `scripts/pipeline/run_dspy_ollama_single.py` for the single-run runner and `run_dspy_ollama_multirun.py` / `run_gpt_oss_multirun.py` for multi-run sweeps.

```bash
python scripts/pipeline/run_dspy_ollama_single.py \
    --model gptoss --folder workspace --dataset tcga \
    [--run run01] [--organs 1 2] [--limit N] [--overwrite] [-v]
```

`--model` accepts: `gptoss`, `gemma3`, `gemma4`, `qwen3_5`, `medgemmalarge`, `medgemmasmall` (each auto-loads `configs/dspy_ollama_{alias}.yaml` for decoding overrides).

Output: `{folder}/results/predictions/{dataset}/llm/{model_slug}/{run_id}/{organ_n}/{case_id}.json`. The model slug is derived from the model id (`ollama_chat/gpt-oss:20b` → `gpt_oss_20b`).

**Multi-run pattern:** LLMs are stochastic, so the canon is to run 3-10 seeded runs per model. The aggregated `_manifest.yaml` at `{folder}/results/predictions/{dataset}/llm/{model_slug}/_manifest.yaml` lists every run and its parse-error rate.

## LLM (DSPy + OpenAI hosted)

OpenAI counterpart of the Ollama runner — drives the same loose-JSON `pipeline.py` through `dspy.LM(model="openai/...")`. Used to add a "yet another top-class model" comparator (e.g. `gpt-5.4-mini`) on the public TCGA corpus alongside the local Ollama LLM baselines (`gpt_oss_20b`, `qwen3`, ...) in the **full cascade** comparison. Note that the rule-based and ClinicalBERT baselines do not produce the nested margins / lymph-node / biomarker fields, so they belong to a separate scope-restricted comparison (`scripts/baselines/eval_rule_bert_llm.py`), not to the full cascade.

**Set up the API key (out of repo).** Create `~/.config/digital-registrar/.env` (Unix / `%USERPROFILE%\.config\digital-registrar\.env` on Windows) with one line:

```
OPENAI_API_KEY=sk-...
```

The loader at [`util.secrets.load_openai_key`](../../src/digital_registrar_research/util/secrets.py) reads from this path (or from `$OPENAI_API_KEY` if already exported). Both locations are outside the repository tree, so the key cannot be committed or shipped in a tarball.

```bash
# Single run (one seed, full TCGA)
python scripts/pipeline/run_pipeline_openai_single.py \
    --model gpt5_4_mini --folder workspace --dataset tcga \
    [--run run01] [--organs 1 2] [--limit N] [--overwrite] [-v]

# K-seed multirun for statistical comparison (paired bootstrap, ICC, flip rate)
python scripts/pipeline/run_pipeline_openai_multirun.py \
    --model gpt5_4_mini --folder workspace --dataset tcga \
    --n 10 --master-seed 42
```

`--model` aliases live in `models.common.model_list` and must resolve to `openai/...`. Add a new alias + `MODEL_PROFILES[...]` entry there to onboard another OpenAI model.

Output: same canonical layout as the Ollama runner — `{folder}/results/predictions/{dataset}/llm/{model_slug}/{run_id}/{organ_n}/{case_id}.json`. `openai/gpt-5.4-mini` → slug `gpt_5_4_mini` (no `_dspy_strict` suffix; the loose-pipeline runner is the only OpenAI driver).

**Provider-agnostic discovery.** OpenAI-hosted runs land in the **same canonical namespace as Ollama runs** — `llm/{model_slug}/run{NN}/...`. Cascade auto-discovery (when `--llm-runs` is omitted) walks that directory and is provider-blind; the only thing that distinguishes a hosted run from a local run downstream is the model slug, e.g. `gpt_5_4_mini` vs `gpt_oss_20b`. So the multi-method comparison wrapper `eval_rule_bert_llm.py` finds OpenAI runs automatically — no extra flag, no separate code path.

The runner writes one extra side file per run: `_cost_ledger.json` (per-case wall-time as a proxy for spend; pricing is not queried).

## Caching and resume

All three runners check for valid existing outputs before re-running a case. To force a fresh run, pass `--overwrite`. Partial / errored outputs (those with `_pipeline_error: true` in the JSON) are NOT considered valid and will be retried automatically.

## Side files (uniform across all three)

| File | Content |
|---|---|
| `_summary.json` | Run-level totals: `n_cases`, `n_ok`, `n_pipeline_error`, `n_cached`, per-organ counts, wall time, parse error rate. |
| `_log.jsonl` | One JSONL row per case: `{case_id, organ, status, latency_s, parse_success, is_cancer, cancer_category, error, started_at}`. |
| `_run.log` | Full-verbosity log. |
| `_run_meta.json` | Provenance: git sha, host, argv, started_at / finished_at, full path resolution. |

For LLM runs only, the model-level `_manifest.yaml` is appended/updated idempotently with one entry per run (so re-running a particular run slot updates its row in place).
