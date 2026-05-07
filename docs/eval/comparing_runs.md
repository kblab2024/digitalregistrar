# Comparing cascade runs — cookbook

How to answer the three questions you'll ask once you have more than one cascade output:

1. **"Which run is better?"** — different methodology, different model, or just two stochastic reruns of the same config.
2. **"How reproducible is this model?"** — across N runs of the same config.
3. **"Where do two configurations disagree?"** — per-field deltas with significance tests.

Two tools cover all three:

| Tool | When | Output |
|---|---|---|
| `cli cascade` with multiple `--run-ids` | Reliability of one config across N runs | `multirun_consistency.csv` (ICC, Cronbach α, accuracy-flip-rate) |
| `cascade.compare_runs` | Side-by-side comparison of multiple cascade outputs | `verdict.csv`, `summary.md`, paired-bootstrap deltas |

---

## Recipe 1 — "Which of these two runs is better?"

The most common question. Answers it with a one-line verdict to stdout plus a full report on disk.

### Step 1 — score each run separately

```bash
python -m scripts.eval.cli cascade \
    --root workspace --dataset cmuh --annotator gold \
    --method llm --model gpt_oss_20b --run-ids run01-alpha \
    --out workspace/results/eval/cascade/gpt_oss_20b__run01-alpha

python -m scripts.eval.cli cascade \
    --root workspace --dataset cmuh --annotator gold \
    --method llm --model gpt_oss_20b --run-ids run02-beta \
    --out workspace/results/eval/cascade/gpt_oss_20b__run02-beta
```

Each command produces a directory containing `cascade_atomic.parquet` plus the three chapter folders. The two runs are scored independently against the same gold annotations.

### Step 2 — compare and get a verdict

```bash
python -m scripts.eval.cascade.compare_runs \
    --runs run01-alpha=workspace/results/eval/cascade/gpt_oss_20b__run01-alpha \
           run02-beta=workspace/results/eval/cascade/gpt_oss_20b__run02-beta \
    --out workspace/results/eval/cascade_compare/gpt_oss_20b__run01_vs_run02
```

Stdout prints the verdict immediately:

```
VERDICT — Stage C (field extraction): **RUN02-BETA IS BETTER**
(run02-beta=0.922 vs run01-alpha=0.818, Δ = +0.105,
 95% CI [+0.065, +0.152], McNemar p = 0.0000, n = 400).
```

### Step 3 — interpret

The verdict reports a paired-McNemar test across every (case_id, field) Stage-C pair, pooled across all fields and organs.

| Verdict | Meaning | What to do |
|---|---|---|
| **"`<run>` better"** | McNemar p < 0.05 *and* the higher-accuracy run wins on the pooled disagreement asymmetry. | Use the winning run as the headline. Inspect `pairwise_deltas.csv` to see *which fields* drove the difference. |
| **"tie"** | McNemar p ≥ 0.05 — disagreement is noise, not signal. | Don't cherry-pick. Switch to Recipe 2 (multi-run cascade) and report ICC instead. |
| **"undetermined"** | No overlapping cases between the two runs. | Check `cascade_atomic.parquet` — likely a path or filter mismatch. |

### Output file map

Under `--out`:

| File | What's inside |
|---|---|
| `summary.md` | Verdict + headline tables. **Read this first.** |
| `verdict.csv` | `stage, run_a, run_b, n_pairs, acc_a, acc_b, delta_acc_b_minus_a, delta_ci_lo, delta_ci_hi, mcnemar_p, verdict`. Stage A, B, C separately. |
| `headline.csv` | Per-run Stage A/B/C accuracy with Wilson CIs and Cohen's κ on Stage B. |
| `pairwise_deltas.csv` | Every `(field, run_a, run_b)` combo with paired-bootstrap CI, McNemar p, and Holm/BH-adjusted q. The drill-down for "which fields differ?". |
| `chapter3_per_field_wide.csv` | Wide pivot — rows = field, columns = `<run>_acc`, `<run>_n`, `<run>_correct`. Spreadsheet-friendly. |
| `chapter3_per_organ_wide.csv` | Same shape but rows = organ. |
| `cascade_funnel_compare.csv` | Cohort attrition per run (n passed Stage A, B, scored at C). |
| `others_compare.csv` | Others-ledger summary (dual-primary / out-of-scope counts) per run when present. |

---

## Recipe 2 — "How reproducible is this model?"

When you want a reliability claim for a single model configuration across N runs (no ranking — just "is this stable?"). Pass all run IDs to a single cascade invocation.

```bash
python -m scripts.eval.cli cascade \
    --root workspace --dataset cmuh --annotator gold \
    --method llm --model gpt_oss_20b \
    --run-ids run01-alpha run02-beta run03-gamma \
    --out workspace/results/eval/cascade/gpt_oss_20b__multirun
```

Read `chapter3_field_extraction/multirun_consistency.csv`:

| Column | Meaning |
|---|---|
| `icc_2_1` | Single-run reliability (Shrout-Fleiss two-way mixed effects). The "if I rerun, how much does per-case correctness vary?" answer. |
| `icc_3_k` | Reliability of the run-averaged accuracy. The "how stable is the headline number I'm reporting?" answer. |
| `cronbach_alpha` | Internal consistency across runs. > 0.9 excellent; > 0.7 acceptable. |
| `accuracy_flip_rate` | Fraction of cases that change correct↔wrong across runs. Direct read on instability. |
| `per_case_sd_mean`, `per_case_sd_p90` | Mean and 90th-percentile of per-case SD across runs. |

This is the right tool when the verdict from Recipe 1 was "tie" — instead of forcing a winner, report the reliability of the average.

---

## Recipe 3 — "Compare three or more runs"

`compare_runs` accepts arbitrary `LABEL=PATH` pairs. Useful for benchmarking N different models, methodologies, or hyperparameter settings.

```bash
python -m scripts.eval.cascade.compare_runs \
    --runs gpt_oss_20b=workspace/results/eval/cascade/gpt_oss_20b \
           qwen3_30b=workspace/results/eval/cascade/qwen3_30b \
           gemma3_27b=workspace/results/eval/cascade/gemma3_27b \
    --out workspace/results/eval/cascade_compare/three_way
```

Verdict output adapts: with ≥ 3 runs, stdout prints one line per pair at Stage C:

```
VERDICT (Stage C, pairwise):
  gpt_oss_20b vs qwen3_30b:   0.943 / 0.929, Δ = -0.014, p = 0.046 → gpt_oss_20b better
  gpt_oss_20b vs gemma3_27b:  0.943 / 0.898, Δ = -0.045, p < 0.001 → gpt_oss_20b better
  qwen3_30b vs gemma3_27b:    0.929 / 0.898, Δ = -0.031, p = 0.012 → qwen3_30b better
```

For an omnibus "are all 3 different?" test instead of all pairwise, use Cochran's Q via the cascade orchestrator's `--multi-model-roots` flag (see [stat_methods.md](../stat_methods.md) §2.2).

---

## Recipe 4 — "Which fields drove the difference?"

After Recipe 1 or 3, drill into `pairwise_deltas.csv`:

```python
import pandas as pd
df = pd.read_csv("workspace/results/eval/cascade_compare/.../pairwise_deltas.csv")
# Show fields where the difference is significant after Holm correction
significant = df[df["mcnemar_p_holm"] < 0.05].sort_values(
    "delta_acc", key=abs, ascending=False,
)
print(significant[["field", "run_a", "run_b", "acc_a", "acc_b",
                   "delta_acc", "delta_ci_lo", "delta_ci_hi",
                   "mcnemar_p_holm"]].head(20))
```

Or read `summary.md` directly — it lists the top-K deltas above the pre-registered `HEADLINE_ACCURACY_DELTA_THRESHOLD = 0.02` filter.

---

## Recipe 5 — "Compare across datasets (cmuh vs tcga)"

The same model on different corpora. Run cascade once per dataset, then compare:

```bash
python -m scripts.eval.cli cascade --root workspace --dataset cmuh \
    --method llm --model gpt_oss_20b --annotator gold \
    --out workspace/results/eval/cascade/gpt_oss_20b__cmuh

python -m scripts.eval.cli cascade --root workspace --dataset tcga \
    --method llm --model gpt_oss_20b --annotator gold \
    --out workspace/results/eval/cascade/gpt_oss_20b__tcga

python -m scripts.eval.cascade.compare_runs \
    --runs cmuh=workspace/results/eval/cascade/gpt_oss_20b__cmuh \
           tcga=workspace/results/eval/cascade/gpt_oss_20b__tcga \
    --out workspace/results/eval/cascade_compare/gpt_oss_20b__cmuh_vs_tcga
```

Caveat: cases don't pair across datasets (different `case_id` namespaces). The pairwise McNemar pivot will return zero pairs, so `verdict.csv` will be empty and `summary.md` will report only the unpaired headline accuracies. The right test for cross-dataset comparison is the [`cross_dataset` subcommand](cross_dataset.md) (per-field Δ accuracy with bootstrap CI), not `compare_runs`.

---

## Common pitfalls

- **Forgetting `--annotator gold`**. Default exists but always set it explicitly to avoid scoring against the wrong annotator subdir.
- **Comparing runs with different `--organs` filters**. `compare_runs` pairs by `(case_id, field)` so cases absent from one side are dropped. Use the same `--organs` arg on both cascade invocations.
- **Paths with spaces on Windows**. Quote the whole `LABEL=PATH` entry: `--runs "alpha=C:/My Path/cascade_alpha"`.
- **Comparing the same model with itself**. If you accidentally point both runs at the same cascade dir, `verdict` reports `acc_a == acc_b` and McNemar p = 1.0. Sanity check.
- **`UnicodeEncodeError` in stdout on Windows**. The verdict line uses Greek letters (Δ, κ). The Python file IO uses UTF-8 unconditionally; the terminal display is the only thing that mangles glyphs. The CSV outputs and `summary.md` are correct.

---

## Decision tree — which recipe?

```
                Do you have multiple cascade outputs to compare?
                              /                \
                            yes                 no
                             |                   |
                             |          Run cascade first;
                             |          then come back here.
                             ▼
                  How many runs / configurations?
                       /                  \
                  exactly 2              ≥ 3
                      |                    |
                      ▼                    ▼
        Are they the same                Recipe 3
        model+config (just              (compare_runs
        different runs)?                 with N labels)
                  /          \
                yes           no
                 |             |
                 ▼             ▼
            Recipe 2        Recipe 1
         (multirun        (compare_runs
          cascade)         with 2 labels)
```

For "where do they disagree?" follow with Recipe 4. For cross-dataset, switch to Recipe 5 + the `cross_dataset` subcommand.
