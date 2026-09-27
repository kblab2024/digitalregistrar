# `registrar-eval` recipes

Copy-paste commands for the common evaluation tasks. File pairing, scoring and output files are explained in [index.md](index.md), and every column is described in [reading_outputs.md](reading_outputs.md).

## Score one run

```bash
registrar-pipeline --input reports/ --output runs/gpt_oss_20b --model gpt
registrar-eval metrics --pred runs/gpt_oss_20b --gold gold/ --out eval_out/gpt_oss_20b
```

`--pred` is the `registrar-pipeline --output` folder (`<stem>_output.json`). `--gold` holds `<stem>_annotation.json` or `<stem>.json`. The command prints the pairing counts and then the per-field summary:

```
runs/gpt_oss_20b: 41/42 gold cases matched (1 missing, 0 unreadable, 0 predictions without gold)
Wrote 1018 rows to eval_out/gpt_oss_20b/atomic.csv
Wrote 25 rows to eval_out/gpt_oss_20b/summary.csv
     method stage                              field   metric  attempted  total  coverage  accuracy_attempted    ci_lo  ci_hi
gpt_oss_20b     A             cancer_excision_report accuracy         41     42   0.97619                 1.0 0.914332    1.0
gpt_oss_20b     B                    cancer_category accuracy         41     41   1.00000                 1.0 0.914332    1.0
...
```

The case with no prediction file counts against Stage A coverage (41 / 42) and never reaches Stage B or C.

The `method` column defaults to the prediction folder's name. Set it explicitly with `--label`.

## Score a run whose output is split into dataset subfolders

`registrar-pipeline` without `--input` writes `<run>/<dataset>/<stem>_output.json`. Both folders are searched recursively, so nothing changes:

```bash
registrar-eval metrics --pred runs/20260501_120000 --gold gold/ --out eval_out/run
```

`atomic.csv` records the gold file's subfolder in the `dataset` column. To get per-dataset numbers:

```python
import pandas as pd
from digital_registrar.eval import summary_table

atomic = pd.read_csv("eval_out/run/atomic.csv")
print(summary_table(atomic, by=["dataset", "field"]))
```

## Head-to-head field list only (`--scope fair`)

This scores just the `FAIR_SCOPE` fields (the gate fields, pT/pN/pM, grade, LVI, PNI, tumor size) on every case, without cascade gating. It's useful for comparing against baselines that only emit those fields.

```bash
registrar-eval metrics --pred runs/rule_based --gold gold/ --scope fair --out eval_out/rule_based_fair
```

## Compare two runs / models

```bash
registrar-eval compare \
    --pred-a runs/gpt_oss_20b --pred-b runs/qwen3_30b \
    --gold gold/ --out eval_out/gpt_vs_qwen
```

This writes `compare.csv`, with one row per field: Δ = A − B, a paired-bootstrap 95% CI, and McNemar's test. It also writes both runs' `atomic.csv` / `summary.csv`. See [comparing_runs.md](comparing_runs.md) for how to read it.

## Missingness, refusals and out-of-vocab values

```bash
registrar-eval completeness --pred runs/gpt_oss_20b --gold gold/ --out eval_out/gpt_oss_20b_completeness
```

- `completeness.csv`: per (field, organ). Parse errors (missing or unreadable prediction files), missing or null fields, attempted rate, and accuracy on attempted values, with Wilson CIs.
- `refusal_calibration.csv`: when the model returned null, was gold null too?
- `out_of_vocab.csv`: categorical predictions outside the schema's allowed values.

## Find the worst fields / cases

```python
import pandas as pd

summary = pd.read_csv("eval_out/gpt_oss_20b/summary.csv")
print(summary.sort_values("accuracy_attempted").head(10))

atomic = pd.read_csv("eval_out/gpt_oss_20b/atomic.csv")
wrong = atomic[(atomic["metric"] == "accuracy") & (atomic["correct"] == False)]  # noqa: E712
print(wrong.groupby("case_id").size().sort_values(ascending=False).head(10))
```

## Score in Python

Score whole folders:

```python
from digital_registrar.eval import load_pairs, score_pairs, summarize_scores

pairs, stats = load_pairs("runs/gpt_oss_20b", "gold/")
atomic = score_pairs(pairs, method="gpt_oss_20b")   # scope=None → cascade
summary = summarize_scores(atomic)
```

Score a single case (both arguments are full output records):

```python
import json
from digital_registrar.eval import score_case, score_lymph_nodes

gold = json.load(open("gold/tcga4_1_annotation.json"))
pred = json.load(open("runs/gpt_oss_20b/tcga4_1_output.json"))

result = score_case(gold=gold, pred=pred)
result["stage_c_eligible"]      # passed eligibility + organ gates?
result["pt_category"]           # True / False / None (not attempted)
result["_nested"]["margins"]    # {"tp", "fp", "fn", "f1"}

ln = score_lymph_nodes(gold, pred)
ln["ln_examined_total_correct_tol"], ln["ln_involved_total_correct_tol"]
```

To compare two in-memory runs, concatenate their `score_pairs` tables and call `compare_runs(atomic, "run_a", "run_b")`.
