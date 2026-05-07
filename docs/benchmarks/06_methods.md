# Methods reference

Detailed description, scope, and known limitations of each baseline.

## Rule-based

**Module:** `digital_registrar_research.benchmarks.baselines.rules`
**Runner:** `scripts/baselines/run_rule.py`
**Code size:** ~900 LOC, pure Python (no torch / transformers / DSPy).

A **deterministic, per-organ regex + lexicon extractor**. For each report it:

1. **Classifies the organ** via a lexicon vote across 10 organs (`classify_organ`).
2. **Detects excision vs biopsy** via an excision/resection lexicon vs a biopsy/cytology lexicon (`detect_excision_report`). Fixes the legacy bug where biopsy reports were wrongly flagged as excision because they mentioned a known cancer category.
3. **Dispatches to per-organ extractors** (`extract_for_organ`) that attempt every field in `bert_scope_for_organ(organ)`:
   - **TNM** — driven by `ORGAN_CATEGORICAL[organ]` enum lists (longest-match), so cervix `t1a1` and lung `t1mi` are caught despite their long suffixes.
   - **Stage groups** — emits `pathologic_stage_group` + `anatomic_stage_group` for breast (both slots), `stage_group` / `overall_stage` for other organs.
   - **Grade** — integer (1–4) for most organs; **Gleason → ISUP grade group** for prostate (`group_<isup>_<primary>_<secondary>`, with the standard 2014 ISUP mapping: 4+3 → group_3, 3+4 → group_2, 5+5 → group_5).
   - **Boolean fields** — phrase + outcome-word proximity match (LVI, perineural, distant_metastasis, extranodal_extension, dcis_present, signet_ring, STAS, ...). Abstain phrases ("not assessed", "unable to assess") emit no key (coverage = 0 rather than wrong).
   - **Span fields** — integer mm with multi-dimension max-take (`Tumor size: 2.5 x 1.8 x 1.0 cm` → 25 mm), AJCC version, maximal LN size, Gleason percentages, prostate size/weight, tumor budding, tumor percentage.
   - **Histology / procedure / surgical_technique** — per-organ lexicon → enum mapping (e.g. "infiltrating ductal carcinoma" → `invasive_carcinoma_no_special_type`).
   - **Cancer laterality / quadrant / sideness / clock** — generic enum-name match against the schema.

**Coverage policy:** Omit-on-no-signal. A field is emitted only when the regex/lexicon matches with confidence. Missing keys count as "not attempted" — they drop out of the accuracy denominator. This is intentional and produces honest per-field coverage asymmetry vs ClinicalBERT (which always emits a class, possibly `null`).

**Schema-conformance guard:** `extract_for_organ` filters its output to fields in `bert_scope_for_organ(organ)`, so a regex that fires on the wrong organ (e.g. breast `nuclear_grade` matching a colorectal report) cannot leak into the prediction. Tested explicitly in `tests/baselines/test_rules_schema.py`.

**Out of scope (intentional):**
- Nested-list fields (`margins`, `regional_lymph_node`, `biomarkers`, `histological_patterns`, `vascular_invasion`, `liver.tumor_extent`, `prostate.involved_margin_list`) — BERT doesn't emit them either; rules-vs-BERT must compare on the same scope. Legacy breast-biomarker emission still happens in the back-compat `extract()` wrapper for the older `FAIR_SCOPE` use cases.
- LLM rephrasing or semantic similarity — defeats the purpose of a non-ML floor.

**Known weaknesses:**
- Organ classifier is lexicon-based — fails on reports with rare phrasings or multiple organ candidates.
- Tumor-size regex covers common phrasings but not all ("2.5 x 1.8 cm pink-tan mass located..." with no "tumor size" cue).
- Histology lexicons are seed-quality (~10–20 phrases per enum); will under-cover novel phrasings.

## ClinicalBERT

**Modules:** `digital_registrar_research.benchmarks.baselines.clinicalbert_cls` + `clinicalbert_qa`
**Runners:** `scripts/baselines/train_bert.py`, `scripts/baselines/run_bert.py`
**Foundation model:** `emilyalsentzer/Bio_ClinicalBERT` (110M params).

Two heads on top of one shared encoder per head — see [`02_train_bert.md`](02_train_bert.md) for training details.

### CLS head

Multi-head softmax classifier over the `[CLS]` pooled embedding. One head per categorical / boolean field. Nullable fields get an extra `null` class so the model can say "absent / not mentioned". The `cancer_category` head is shared across all organs; per-organ field heads only fire when the predicted category implies them.

**Predicts:**
- `cancer_category`, `cancer_excision_report`
- All `Literal[...]`-typed categorical fields per organ (TNM, stage groups, grade, histology, procedure, surgical_technique, primary site, ...)
- All boolean fields (LVI, perineural, distant_metastasis, ...)

**Always emits:** every field in its scope, possibly as `null`. So coverage ≈ 100% on its scope.

### QA head

`AutoModelForQuestionAnswering` (Bio_ClinicalBERT span-prediction head). For each numeric field, asks a per-organ question (e.g. "What is the tumor size in mm?") against the report and extracts the highest-confidence span. Trained with **silver supervision**: at training time the gold value is auto-located in the report via substring search (with cm/mm variants); cases where the value isn't findable are dropped.

**Predicts:** `tumor_size`, `dcis_size`, `maximal_ln_size`, `ajcc_version`, integer `grade` for organs that use it, Gleason percentages, prostate size/weight, tumor_percentage, tumor_budding.

**Coverage:** Emits only when the span-prediction head returns a confidently-extracted span; can omit for low-confidence spans.

### Merged head

`run_bert.py --heads merged` does a per-case key-merge: CLS prediction is the base (carries `cancer_category` + `cancer_excision_report`), QA's `cancer_data` scalars overlay onto CLS's, with CLS winning on collisions. The merged output is what gets compared against rule and LLM in the canonical eval contract.

**Out of scope (architectural):** Nested-list fields (lymph nodes, margins as multiple sites, biomarkers as list-of-dicts) — encoder-only architectures cannot emit variable-length nested objects without a generative decoder. These are reported as N/A in the metrics.

### Why pooled training

See [`02_train_bert.md#why-pool-train-splits`](02_train_bert.md#why-pool-train-splits). The annotated pool is too small (~100 train cases per dataset) to fine-tune a 110M-parameter encoder per dataset or per organ. One shared multi-organ checkpoint per head is the only setup that converges meaningfully at this scale.

## LLM

**Module / runner:** `scripts/pipeline/run_dspy_ollama_single.py` (and multirun variants).

Existing canonical pipeline — unchanged by this baseline integration. Uses DSPy signatures + Ollama / OpenAI-compatible endpoints to run a per-organ subsection extraction pipeline. Reads the report, routes to per-organ DSPy modules, returns a structured prediction.

**Multi-run by design:** LLMs are stochastic, so the canon is to run 3-10 seeded runs per (model, dataset). Each run's seed is baked into the LM kwargs at load time, so runs remain individually reproducible. The aggregated `_manifest.yaml` at the model level lists every run and its parse-error rate.

**Models supported via the unified runner:**
| Alias | Backend |
|---|---|
| `gptoss` | `ollama_chat/gpt-oss:20b` |
| `gemma3` | `ollama_chat/gemma3` |
| `gemma4` | `ollama_chat/gemma4` |
| `qwen3_5` | `ollama_chat/qwen3:30b` |
| `medgemmalarge` | `ollama_chat/medgemma:large` |
| `medgemmasmall` | `ollama_chat/medgemma:small` |

Each alias auto-loads `configs/dspy_ollama_{alias}.yaml` for decoding overrides (temperature, top_p, num_ctx, max_tokens, ...).

**Always emits** every field its DSPy signatures cover. Parse errors (malformed JSON, refusal, timeout) get a sentinel JSON `{"_pipeline_error": true, "reason": ..., "message": ...}` rather than being dropped — the eval pipeline classifies these as `parse_error` rather than `field_missing`.

## Side-by-side comparability

The three methods produce per-case JSONs in **identical shape** (top-level `cancer_excision_report`, `cancer_category`, `cancer_category_others_description`, `cancer_data`). The only differences are coverage and accuracy — which is exactly what the eval surface is designed to measure.

The `non_nested` subcommand pulls fields **per organ** from the schema, so all three methods are scored against the same per-organ field set automatically. A method that doesn't attempt a field doesn't get penalized on accuracy (drops out of the denominator); it only gets penalized on coverage.
