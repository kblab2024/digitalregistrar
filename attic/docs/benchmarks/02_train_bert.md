# Training the ClinicalBERT heads

ClinicalBERT is the only baseline that needs training. Rule-based is deterministic; LLMs are zero / few-shot.

## Default: CMUH-only training (cross-corpus baseline)

The default training set is **CMUH only**. This is driven by the privacy constraint that the LLM comparator (especially the OpenAI API) cannot see CMUH — TCGA is the only corpus all three methods can be evaluated on. To make BERT a fair baseline against the LLM on TCGA, BERT must NOT have seen TCGA at training time.

```bash
# Default: trains on cmuh, both heads.
python scripts/baselines/train_bert.py
```

The acknowledged trade-off: a 110M-parameter encoder trained on ~100 CMUH cases is undertrained. That's accepted as the methodologically honest baseline — pooling would let BERT see TCGA at training time and the cross-method comparison would no longer be cross-domain. For an upper-bound ablation, pass `--datasets cmuh tcga`:

```bash
# Pooled-training ablation (NOT the canonical baseline).
python scripts/baselines/train_bert.py --datasets cmuh tcga
```

## Disjoint corpora — no train/test split within a corpus

The cross-corpus baseline does **not** split a corpus into train and test folds. CMUH is the training corpus in full; TCGA is the prediction corpus in full. Disjointness is guaranteed by the dataset boundary — CMUH cases never appear in TCGA's case ids and vice versa.

Cases are discovered by walking the gold-annotations tree:

- Training: `{folder}/data/cmuh/annotations/gold/<organ_n>/*.json`
- Prediction: `{folder}/data/tcga/annotations/gold/<organ_n>/*.json`

`train_bert.py` runs a corpus pre-flight at the top of every training run:

```
============================================================
Training corpus summary (cross-corpus: train CMUH, predict TCGA)
============================================================
[cmuh] cases=500
  organ 2: cases=120
  organ 4: cases=110
  organ 5: cases=98
  organ 6: cases=92
  organ 9: cases=80
============================================================
pooled training cases: 500
============================================================
```

It fails fast if any requested dataset has no gold annotations under `{folder}/data/{dataset}/annotations/gold/`.

The training datasets are saved in the checkpoint metadata:

- CLS: `torch.save(...)` includes a `"datasets"` key.
- QA: a sidecar `{ckpt_dir}/_train_meta.json` carries `"datasets"` (HuggingFace `save_pretrained` doesn't pass through custom keys).

`run_bert.py:predict()` reads this back at inference time and **refuses to predict** on any dataset that was in the checkpoint's training set:

```
refusing to predict: dataset(s) ['cmuh'] were in the checkpoint's training set
(['cmuh']). Cross-corpus baseline predicts on datasets disjoint from training.
```

This dataset-disjointness guard is the only safety check needed; it's much simpler than per-case overlap tracking and is the actual invariant the cross-corpus baseline depends on.

## What the two heads do

| Head | What it predicts | Loss | Output shape |
|---|---|---|---|
| **CLS** | Categorical / boolean fields per organ (TNM, histology, procedure, LVI, perineural, ...) and the top-level `cancer_category` | Multi-head softmax cross-entropy over `[CLS]` pooled embedding | One classification per field |
| **QA** | Numeric span fields (tumor_size, AJCC version, maximal LN size, Gleason percentages, ...) | Span-prediction (start/end logits) on per-organ question bank | Integer per field |

The two heads are trained **independently** on the **pooled** train split across all configured datasets, then merged at predict time. See [`docs/benchmarks/06_methods.md`](06_methods.md) for the architectural-scope rationale.

## Training command

```bash
# Default canonical training: CMUH-only.
python scripts/baselines/train_bert.py \
    --heads cls qa \
    --ckpt-cls ckpts/clinicalbert_cls.pt \
    --ckpt-qa  ckpts/clinicalbert_qa \
    --epochs-cls 5 --epochs-qa 3
```

Arguments:

| Flag | Default | What it controls |
|---|---|---|
| `--folder` | `workspace` | Experiment root (`dummy` / `workspace` / abs path). |
| `--datasets` | `cmuh` | Training datasets. Default is CMUH only (TCGA held out for cross-corpus eval). Pass `cmuh tcga` for the pooled-training ablation. |
| `--heads` | `cls qa` | Which heads to train. Both run sequentially in a single invocation. |
| `--organs` | breast colorectal esophagus liver stomach | Cancer-category names to keep. Other organs' rows are dropped from the train set. |
| `--ckpt-cls` | `ckpts/clinicalbert_cls.pt` | Output checkpoint path for CLS. |
| `--ckpt-qa` | `ckpts/clinicalbert_qa` | Output checkpoint dir for QA (HuggingFace `save_pretrained` format). |
| `--epochs-cls` | 5 | CLS epochs. |
| `--epochs-qa` | 3 | QA epochs. |
| `--included-only` | off | CLS-only flag: drop train cases where `cancer_excision_report=False` (no organ-specific labels to learn from). |

## Device selection

The trainer auto-detects:

1. Apple-silicon **MPS** (Mac M1/M2/M3) — preferred when available.
2. **CUDA** GPUs.
3. **CPU** fallback.

No flag is needed; the device is picked once per head. `PYTORCH_ENABLE_MPS_FALLBACK=1` and `TOKENIZERS_PARALLELISM=false` are set automatically so the trainer is friendly to laptop runs.

## Why CMUH-only by default (and why pooling is now an ablation)

Earlier sessions defaulted to pooled CMUH+TCGA training, on the reasoning that ~100 cases per dataset would undertrain a 110M-parameter encoder. That reasoning still applies — BERT trained on CMUH alone IS undertrained. But the privacy constraint dominates:

- The LLM comparator (OpenAI API) cannot see CMUH. So the only corpus all three methods can be benchmarked on is TCGA.
- Pooled training would have BERT see TCGA at training time. The cross-method comparison on TCGA would no longer be a held-out test — BERT would have an unfair informational advantage.
- An undertrained BERT trained on CMUH only is the **methodologically honest baseline**. It shows what BERT-as-a-baseline actually delivers when the test set is genuinely held out.

The structural choices that remain unchanged:
- One shared multi-organ encoder per head (CLS, QA). The CLS multi-head architecture *requires* a shared encoder so the top-level `cancer_category` head can exist; per-organ encoders would also waste capacity re-learning shared pathology vocabulary.
- Predict-time eval stays per-organ (cross-organ rollups are added at the eval-table level).

Pooled training is still available as `--datasets cmuh tcga` for upper-bound ablations. Reconsider making it the default only if (a) CMUH alone exceeds ~500 annotated cases and (b) there's a privacy-sandboxed LLM that can see CMUH.

## Training output

```
ckpts/
├── clinicalbert_cls.pt      torch.save dict with state + field_to_idx + card metadata
└── clinicalbert_qa/
    ├── config.json
    ├── pytorch_model.bin     or model.safetensors (HF defaults)
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── special_tokens_map.json
```

The CLS checkpoint stores both the model weights and the per-field index→value vocabularies, so prediction time doesn't need to re-derive them from the schema.

## Smoke training on dummy data

The dummy fixtures are too small for meaningful learning, but the smoke run validates the training loop end-to-end:

```bash
python scripts/data/gen_dummy_skeleton.py --out dummy --clean --cases-per-organ 5 --llm-runs 1
python scripts/baselines/train_bert.py --folder dummy --datasets cmuh --heads cls qa \
    --epochs-cls 1 --epochs-qa 1
python scripts/baselines/run_bert.py --folder dummy --datasets tcga
```

Expect both heads to write checkpoints in <1 minute on CPU; accuracy will be near-random because the dummy data is synthetic noise.
