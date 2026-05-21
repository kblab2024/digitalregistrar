# GPU acceleration for eval statistics

The eval pipeline's CI / hypothesis-testing primitives now have an opt-in
GPU path (CUDA + MPS) for the loops that dominate wall-time on large evals.
The original CPU code in
[`ci.py`](../../src/digital_registrar_research/benchmarks/eval/ci.py) is
preserved as the **safety net** — its outputs are unchanged unless you
explicitly opt in.

## What this accelerates

- **`bootstrap_mean_ci`** — bootstrap CI on the (weighted) mean of a 1-D
  vector. Used by `non_nested` (per-field, per-organ, per-subgroup
  bootstrap CIs and `section_rollup`).
- **`paired_bootstrap_diff`** — paired bootstrap of `mean(a) - mean(b)`.
  Used by `canonical_stats` (`_headline`, `_per_field`,
  `_modularity_advantage`) and `ablations.eval.stats.paired_deltas_vs_baseline`.
- **`independent_bootstrap_diff`** — unpaired bootstrap of mean
  difference. Used by `cross_dataset._per_field_delta`.
- **`mcnemar_batch`** — vectorized McNemar test on a batch of 2×2 paired
  tables. Replaces the ~15,000-McNemar triple loop in
  `completeness.method_pair_deltas`.
- **`bootstrap_kappa_ci`** — BCa bootstrap CI on unweighted Cohen's κ.
  Used by IAA's binary/nominal and ordinal-unweighted κ paths.
- **`paired_kappa_delta_ci`** — paired bootstrap on κ(a₁,b₁) − κ(a₂,b₂).
  Used by `preann.paired_delta_kappa` and `preann.disagreement_reduction`.
- **`fleiss_kappa_batch`** — vectorized Fleiss κ across a batch of
  rating matrices. Used by `metrics_non_nested.run_consistency_extended`.

It does **not** accelerate:
- Model inference (DSPy / Ollama owns its own device dispatch).
- The PyTorch ClinicalBERT baselines (`scripts/baselines/`) — those have
  separate device handling already.
- The per-field sklearn loops in
  [`run_non_nested.py:162-196`](../../scripts/eval/non_nested/run_non_nested.py#L162-L196)
  (confusion matrix, P/R/F1, MCC). These are sklearn-bound; the right
  fix is vectorizing across (field, organ) in numpy/pandas, not GPU.
- Quadratic-weighted κ in IAA — the GPU primitive is unweighted-only;
  the quadratic branch stays on the safety-net path.
- p-value adjustment (Holm, Benjamini-Hochberg) — already vectorized in
  statsmodels.

## How to use

Every `python -m scripts.eval.cli <subcmd>` invocation accepts
`--device {auto,cpu,cuda,mps}` (default **`cpu`** — explicit opt-in).

### MacBook Air M3 (MPS)

```bash
python -m scripts.eval.cli non_nested \
    --root workspace --dataset cmuh \
    --method llm --model gpt_oss_20b \
    --device mps
```

### RTX 6000 Ada workstation (CUDA)

```bash
python -m scripts.eval.cli non_nested \
    --root workspace --dataset cmuh \
    --method llm --model gpt_oss_20b \
    --device cuda
```

### Cross-machine scripts: `--device auto`

```bash
python -m scripts.eval.cli completeness \
    --root workspace --dataset cmuh \
    --methods llm:gpt_oss_20b dspy:gpt_oss_20b \
    --device auto
```

`auto` picks **mps → cuda → cpu** in that order.

### Other entry points

| Entry point                                | Flag                       | Default  |
|--------------------------------------------|----------------------------|----------|
| `scripts/eval/cli.py` (all subcommands)    | `--device`                 | `cpu`    |
| `scripts/eval/cross_dataset/run_cross_dataset.py` | `--device`          | `cpu`    |
| `scripts/eval/canonical/make_paper_tables.py`     | `--device`          | `cpu`    |
| `src/.../ablations/eval/run_ablations.py`  | `--device`                 | `cpu`    |

Inside library code, every public stats function takes a `device:
str = "cpu"` kwarg that is threaded through from the entry point.

## Default behavior (`--device cpu`)

Without `--device`, the eval routes through `ci_gpu`'s
**vectorized-numpy CPU path**. This already eliminates the per-iteration
Python loop over `n_boot=2000` and the O(n) Python jackknife loop in
`ci.bootstrap_ci`, so even on the same machine you get a meaningful
speedup over the pre-change implementation. Numerical output is bit-equal
to the safety-net `ci.py` path on the simple-mean statistic at the same
seed (asserted by
[`tests/eval/test_ci_gpu.py:test_bootstrap_mean_cpu_matches_ci_py_simple_mean`](../../tests/eval/test_ci_gpu.py)).

If you need byte-identical safety-net behavior (e.g. to reproduce a
historical run), call the underlying functions directly without a
`device` kwarg — they default to `cpu` and the original
`ci.bootstrap_ci`/`ci.mcnemar_test`/etc. paths remain reachable through
`from digital_registrar_research.benchmarks.eval.ci import ...`.

## Per-primitive coverage

| Primitive                       | Call sites                                                                                           |
|---------------------------------|-------------------------------------------------------------------------------------------------------|
| `bootstrap_mean_ci`             | `metrics_non_nested._overall_row`, `metrics_non_nested._per_field_row`, `metrics_non_nested.section_rollup` |
| `paired_bootstrap_diff`         | `canonical_stats._headline`, `canonical_stats._per_field`, `canonical_stats._modularity_advantage`, `ablations.eval.stats.paired_deltas_vs_baseline` |
| `independent_bootstrap_diff`    | `scripts/eval/cross_dataset/run_cross_dataset.py:_per_field_delta`                                    |
| `mcnemar_batch`                 | `completeness.method_pair_deltas`                                                                     |
| `bootstrap_kappa_ci`            | `iaa.score_field_pair` (binary, nominal, ordinal-unweighted), `iaa.pairwise_iaa` (coverage κ)         |
| `paired_kappa_delta_ci`         | `preann.paired_delta_kappa`, `preann.disagreement_reduction`                                          |
| `fleiss_kappa_batch`            | `metrics_non_nested.run_consistency_extended`                                                          |

## Numerical equivalence

- **Point estimates** — the `point` field of `BootstrapResult`, McNemar
  χ², Fleiss κ — are deterministic functions of the input and match
  exactly across devices to within float64 roundoff (~1e-12).
- **CI endpoints** are stochastic. The
  [`test_bootstrap_mean_cpu_matches_ci_py_*`](../../tests/eval/test_ci_gpu.py)
  tests pin the CPU path's `lo`/`hi` to within `1e-12` of
  `ci.bootstrap_ci`'s output (because the vectorized RNG draws are the
  same flat stream as the Python-loop draws). GPU-vs-CPU agreement is
  within the natural Monte-Carlo noise floor of `n_boot` (~1% absolute
  on a proportion at `n_boot=2000`).
- **Hard error on unavailable backend** — explicit `--device cuda` on a
  Mac or `--device mps` on the workstation exits with a clear message
  rather than silently falling back to CPU. Silent fallback would
  mislead reported wall-times in this benchmarking codebase. `auto`
  remains the explicit "do whatever works" knob.

## Troubleshooting

- **MPS lacks an op** — set `PYTORCH_ENABLE_MPS_FALLBACK=1` so unsupported
  ops fall back to CPU instead of erroring. The bootstrap primitives
  here use only `gather`, `mean`, `scatter_add_`, and basic arithmetic.
- **GPU dtype is float32 (not float64)** — the per-row reductions on
  any GPU (`--device cuda` or `--device mps`) run in float32. Reasons:
  - MPS (Metal) is float32-only for floating-point compute.
  - Consumer / workstation CUDA cards — RTX 6000 Ada, RTX 30/40-series,
    A6000 — have heavily throttled float64 throughput (~1:32 to 1:64
    of float32). Using float64 there would defeat the GPU speedup.
  - Data-center cards (A100, H100) do full-rate float64, but they're
    not the target environment for this codebase.

  Float32 precision (~7 decimal digits) is well above the natural
  Monte-Carlo noise floor of bootstrap CIs at typical `n_boot=2000`
  (~2/√B ≈ 4%). The bootstrap distribution is cast back to float64
  on CPU before quantile / BCa, so endpoint arithmetic remains
  full-precision. The CPU path stays at float64 for bit-equivalence
  with the safety-net `ci.py`.
- **CUDA OOM** — unlikely at typical `n_boot=2000` and `n ≤ a few
  thousand`, but if you scale `n_boot` to 100k+ you may need to chunk
  the index matrix. File an issue if this comes up in practice.
- **MPS reproducibility** — MPS may produce slightly different
  floating-point results than CUDA/CPU on some torch versions due to
  Metal kernel fusion ordering. Pin a torch version if you need
  cross-machine bit-equality on GPU; CPU paths are bit-identical at
  the same seed.

## Where the original code lives

[`src/digital_registrar_research/benchmarks/eval/ci.py`](../../src/digital_registrar_research/benchmarks/eval/ci.py)
is the canonical reference implementation for the bootstrap, McNemar,
and Wilson primitives. It is **not** modified by the GPU work — it is
the safety net that all `device="cpu"` callers fall back to (when they
don't already route through `ci_gpu`'s vectorized-numpy path).

For the GPU implementations, see
[`src/digital_registrar_research/benchmarks/eval/ci_gpu.py`](../../src/digital_registrar_research/benchmarks/eval/ci_gpu.py).
The `pick_device(...)` helper there is the single source of truth for
auto/cpu/cuda/mps resolution.
