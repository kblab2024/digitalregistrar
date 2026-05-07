# Experiment protocol (2026-04 round)

This document is the reference for the in-flight experiment. Any change
that invalidates a headline number (new seed, new prompt, new scoring
rule) requires bumping the relevant `experiment_id` in
`configs/experiments/*.yaml` — that's the pre-registration contract.

## Cross-product under study

| Axis | Values | Count |
|---|---|--:|
| Dataset | `cmuh`, `tcga` | 2 |
| Human annotator | `nhc`, `kpc` | 2 |
| Annotation mode | `with_preann`, `without_preann` | 2 |
| LLM model | `gpt_oss_20b`, `gemma4_30b`, `qwen3_30b`, `gemma4_e2b`* | 3 (+1 optional) |
| LLM run | `run01`..`run10` (fixed seeds 42..51) | 10 |
| BERT variant | `v1_baseline`, `v2_finetuned` | 2 |
| Rule-based | single deterministic run | 1 |

\* `gemma4_e2b` is tentative; decide before seeds 42..51 are burned on the headline sweep.

## Per-case artifact inventory

For one case (e.g. `cmuh1_3`) the experiment produces:

- 1 raw report (`data/cmuh/reports/1/cmuh1_3.txt`)
- 1 gpt-oss:20b pre-annotation (`data/cmuh/preannotation/gpt_oss_20b/1/cmuh1_3.json`)
- 4 human annotations (2 annotators × 2 modes)
- 1 gold annotation (after consensus)
- ≥30 LLM predictions (3 models × ≥10 runs)
- 2 BERT predictions
- 1 rule-based prediction

Total: ~40+ artifacts per case, fully tracked by folder convention.

## Evaluation questions and the outputs that answer them

### Q1 — How accurate is each method against gold?
- **Output**: `results/evaluation/{dataset}/accuracy/by_method.csv`, `per_field.csv`, `per_organ.csv`, `per_fieldtype.csv`
- **Unit of analysis**: case × field, scored by `score_case` in [`metrics.py`](../src/digital_registrar_research/benchmarks/eval/metrics.py)
- **CI**: case-level bootstrap (n=2000), run-level bootstrap over the 10 LLM runs, and a total-CI combining both

### Q2 — How consistent are LLM runs?
- **Output**: `results/evaluation/{dataset}/accuracy/run_consistency.csv`
- **Metric**: Fleiss κ across the 10 runs per field
- **Input**: `results/predictions/{dataset}/llm/{model}/run{01..10}/`

### Q3 — Do majority-vote ensembles beat single runs?
- **Output**: `results/evaluation/{dataset}/ensembles/{model}/{organ_n}/{case_id}.json` + `ensemble_vs_single.csv`
- **Procedure**: majority vote per field across the 10 runs (ties broken by first-encountered value)
- **CI**: paired bootstrap Δ(accuracy) between ensemble and single-run mean

### Q4 — How much do annotators agree?
- **Output**: `results/evaluation/{dataset}/iaa/pairwise_nhc_vs_kpc_with_preann.csv`, `…_without_preann.csv`
- **Metrics**: Cohen's κ (binary/nominal), quadratic-weighted κ (ordinal), Lin's CCC (continuous), matched F1 (nested lists — margins / biomarkers / LNs)

### Q5 — Does seeing the LLM pre-annotation change the annotator's output?
- **Output**: `preann_effect_nhc.csv`, `preann_effect_kpc.csv`
- **Procedure**: same annotator, same case, with-preann vs without-preann
- **Interpretation**: large drift from their own "without" version means the LLM draft is anchoring them; small drift means they're reviewing thoroughly

### Q6 — How do methods rank against each other and against annotators?
- **Output**: `results/evaluation/{dataset}/comparisons/model_vs_annotator.csv`
- **Procedure**: score each method against gold, score each annotator against gold, compare distributions

## Protocol invariants

These cannot change without bumping `experiment_id`:

1. Seeds `[42, 43, ..., 51]` for LLM runs.
2. Decoding `temperature=0.7, top_p=1.0, max_tokens=2048` for the headline sweep (ablations use their own configs).
3. Prompt template hash (`prompts/gpt_oss_pathology_v1.jinja` — stored in
   `configs/experiments/multirun_{model}.yaml`).
4. Schema scope — `FAIR_SCOPE ∪ {margins, regional_lymph_node, breast biomarkers}` from
   [`scope.py`](../src/digital_registrar_research/benchmarks/eval/scope.py).
5. Acceptance thresholds: `parse_error_rate_max = 0.05`, `missing_case_rate_max = 0.01`.
6. Annotator identity + mode assignment (fixed in `configs/annotators/annotators.yaml`).

## Timeline

| Phase | Milestone | Branch |
|---|---|---|
| Data collection | CMUH reports curated, gold schema frozen | `experiment_cmuh_pilot` |
| Pre-annotation | gpt-oss:20b sweep over CMUH + TCGA | `testing_llm` → `experiment_cmuh_pilot` |
| Human annotation round 1 | Both annotators, both modes, all cases | `experiment_cmuh_pilot` (UI via `testing_ui`) |
| Gold consensus | Adjudication produces `annotations/gold/` | `experiment_cmuh_pilot` |
| Method sweeps | 10 runs × 3–4 LLMs, 2 BERT variants, rule-based | `testing_llm`, `testing_bert`, `testing_rule` |
| Evaluation | All six questions above | `testing_iaa`, `testing_ensemble` |
| Writeup | Headline CIs, tables, figures | `experiment_cmuh_pilot` → `main` |

## Reproducing a result

Every CSV under `results/evaluation/` is reproducible from:

1. The report `.txt` files under `data/{dataset}/reports/`
2. The gold annotations under `data/{dataset}/annotations/gold/`
3. The per-run predictions under `results/predictions/{dataset}/.../run{NN}/`
4. The pinned config at `configs/experiments/multirun_{model}.yaml`

No hidden state — re-running the eval scripts against the above produces
byte-equivalent CSVs modulo bootstrap RNG.
