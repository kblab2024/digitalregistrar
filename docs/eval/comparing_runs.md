# Comparing runs

How to answer "which of these two runs is better, and on which fields?" with `registrar-eval compare`.

## Run it

```bash
registrar-eval compare \
    --pred-a runs/gpt_oss_20b --pred-b runs/qwen3_30b \
    --gold gold/ --out eval_out/gpt_vs_qwen
```

| Flag | Meaning |
|---|---|
| `--pred-a`, `--pred-b` | the two prediction folders (`registrar-pipeline --output` folders) |
| `--gold` | gold folder shared by both runs |
| `--label-a`, `--label-b` | names used in the `method` columns. Default: the folder names, or `a` / `b` if the two names are the same |
| `--scope` | `cascade` (default) or `fair`; see [index.md](index.md#how-scoring-works---scope-cascade-the-default) |
| `--n-boot` | bootstrap resamples for the Δ CI (default 2000) |

Outputs in `--out`:

- `compare.csv`: the comparison, one row per (stage, field, metric) plus an `ALL` row.
- `summary.csv`: each run's own per-field summary (same columns as `registrar-eval metrics`).
- `atomic.csv`: both runs' per-(case, field) rows, told apart by the `method` column.

## Reading `compare.csv`

| Column | Meaning |
|---|---|
| `stage`, `field`, `metric` | as in `summary.csv`. `metric` is `accuracy` or `f1` |
| `method_a`, `method_b` | run labels |
| `n_attempted_a`, `n_attempted_b` | cases each run was scored on for this field |
| `n_paired` | cases **both** runs were scored on. All the statistics below use only these cases |
| `score_a`, `score_b` | accuracy (or mean per-case F1) on the paired cases |
| `delta` | `score_a − score_b`. **Positive means A is better** |
| `delta_ci_lo`, `delta_ci_hi` | 95% percentile paired bootstrap over cases ([ci_methods.md](ci_methods.md#paired-bootstrap-δ--for-paired-comparisons)) |
| `mcnemar_b` | cases where A is right and B is wrong (accuracy rows only) |
| `mcnemar_c` | cases where A is wrong and B is right |
| `mcnemar_p_value` | McNemar's test on (b, c): exact binomial when b + c < 25, otherwise χ² with continuity correction. NaN for `f1` rows |

The last row has `field == "ALL"`. It compares **per-case mean accuracy** over all paired accuracy rows and bootstraps over cases. Use it as the single headline number. Its `n_attempted_*` / `n_paired` columns count cases, not rows.

### Interpreting a row

- **CI excludes 0 and McNemar p < 0.05**: the difference on this field is unlikely to be noise.
- **CI includes 0**: no evidence of a difference. Don't rank the runs on this field.
- **`mcnemar_b` and `mcnemar_c` are both small**: the runs rarely disagree, and the p-value will be large however the disagreements split.
- **Many fields**: expect about 1 in 20 fields to reach p < 0.05 by chance. Adjust for multiplicity (e.g. Holm) before claiming per-field wins, or lead with the `ALL` row.

## Why `n_paired` can be smaller than the case count

With cascade scoring, a run only gets Stage-C rows for a case if it passed both gates on that case: eligibility (`cancer_excision_report`) and organ (`cancer_category`). A missing or unreadable prediction file fails the first gate. The comparison uses only the cases where **both** runs were scored.

So compare `n_attempted_a` / `n_attempted_b` with `n_paired`:

- **A big gap** means one run lost cases at the gates. Look at the `cancer_excision_report` (stage A) and `cancer_category` (stage B) rows first. Those are scored for every gold case and gold-eligible case respectively, so they capture the gating difference directly.
- **Similar counts** mean the per-field deltas compare like with like.

## Three or more runs

Run `compare` once per pair you care about, usually each candidate against a reference run. Then read the `ALL` rows side by side. For per-field accuracy of every run in one table, concatenate their `summary.csv` files:

```python
import pandas as pd

runs = ["gpt_oss_20b", "qwen3_30b", "gemma3_27b"]
summary = pd.concat(pd.read_csv(f"eval_out/{r}/summary.csv") for r in runs)
wide = summary.pivot_table(index="field", columns="method", values="accuracy_attempted")
print(wide.round(3))
```

## Pitfalls

- **Different gold folders.** Both runs are scored against the single `--gold` you pass, so this can't happen by accident within one `compare` call. Comparing `summary.csv` files from separate `metrics` calls can go wrong, though, if the gold folders differed.
- **Different case sets.** Cases missing from one run drop out of the paired statistics but still count against that run's stage-A coverage. Check the stats line the CLI prints (`41/42 gold cases matched …`).
- **Comparing a run with itself.** Every `delta` is 0 and every McNemar p is 1.0. That makes a quick sanity check.
- **Cross-dataset comparisons.** Different datasets have different case ids, so nothing pairs. Score each with `metrics` and compare the `summary.csv` files instead.
