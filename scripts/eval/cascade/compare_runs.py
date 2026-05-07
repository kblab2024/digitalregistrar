"""Side-by-side comparison of cascade runs (the canonical ``compare`` subcommand).

Registered in :mod:`scripts.eval.cli` as ``compare``. Invoked via::

    python -m scripts.eval.cli compare \\
        --runs gpt_oss_20b=workspace/results/eval/cascade/cmuh_gpt_oss \\
               qwen3_30b=workspace/results/eval/cascade/cmuh_qwen3 \\
               gemma3_27b=workspace/results/eval/cascade/cmuh_gemma3 \\
        --out workspace/results/eval/cascade_compare/cmuh_3way

Inputs: each ``--runs LABEL=PATH`` points at a directory produced by
``scripts.eval.cli cascade``. The directory must contain
``cascade_atomic.parquet`` and the chapter folders.

Output tree under ``--out``:

    manifest.json
    summary.md                            brief human-readable report
    verdict.csv                           pooled paired tests per (stage, run-pair)
    headline.csv                          one row per run: Stage A/B/C accuracy + κ
    pairwise_deltas.csv                   per (field, run-pair) Stage-C scalar deltas
    cascade_funnel_compare.csv            funnel attrition per run
    others_compare.csv                    others-ledger summary per run
    chapter1_eligibility/
        comparison.csv                    Stage A acc + Wilson CI per run
        pairwise.csv                      Stage A McNemar + paired bootstrap per run-pair
    chapter2_organ_classification/
        comparison.csv                    Stage B acc + κ per run
        pairwise.csv                      Stage B McNemar + paired bootstrap per run-pair
        confusion_per_class_compare.csv   per-gold-class recall across runs
    chapter3_field_extraction/
        per_field_wide.csv                wide pivot rows=field, cols=runs (scalar only)
        per_organ_wide.csv                wide pivot rows=organ, cols=runs (scalar only)
        pairwise.csv                      Stage-C scalar McNemar + Δ per (field, run-pair)
    chapter4_margins/
        comparison.csv                    margins F1 + missingness per run
        pairwise.csv                      paired bootstrap on per-case F1 per run-pair
    chapter5_lymph_nodes/
        comparison.csv                    regional_lymph_node F1 + missingness per run
        pairwise.csv                      paired bootstrap on per-case F1 per run-pair

The summary.md intentionally stays brief — one screen of headline tables.
Read the supporting CSVs for full per-field detail.
"""
from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from digital_registrar_research.benchmarks.eval.stats import (
    HEADLINE_ACCURACY_DELTA_THRESHOLD,
    bh_fdr,
    cohens_kappa,
    holm,
    mcnemar,
    paired_bootstrap_delta,
    wilson_ci,
)

from .._common.loaders import coerce_cascade_bool, coerce_cascade_correct
from .._common.reporting import setup_logging, write_csv, write_manifest

logger = logging.getLogger("scripts.eval.cascade.compare_runs")


# --- Argparse -------------------------------------------------------------

def _parse_runs(raw: Sequence[str]) -> dict[str, Path]:
    """Parse a list of ``LABEL=PATH`` strings into an ordered dict."""
    runs: dict[str, Path] = {}
    for entry in raw:
        if "=" not in entry:
            raise SystemExit(
                f"--runs entry must be LABEL=PATH; got {entry!r}"
            )
        label, path = entry.split("=", 1)
        label = label.strip()
        if not label:
            raise SystemExit(f"empty label in --runs entry {entry!r}")
        if label in runs:
            raise SystemExit(f"duplicate label {label!r} in --runs")
        runs[label] = Path(path)
    if len(runs) < 2:
        raise SystemExit("--runs requires at least 2 entries to compare")
    return runs


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``compare`` subcommand on the shared CLI dispatcher."""
    parser = subparsers.add_parser(
        "compare",
        help="Side-by-side comparison of cascade runs (chapter1-5 layout).",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--runs", nargs="+", required=True, metavar="LABEL=PATH",
        help="Cascade output directories to compare, labelled. "
             "At least two required.",
    )
    parser.add_argument(
        "--out", type=Path, required=True,
        help="Output directory for the comparison report.",
    )
    parser.add_argument(
        "--n-boot", type=int, default=2000,
        help="Bootstrap replicates for paired delta CIs (default: %(default)s).",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="CI coverage level (default: %(default)s).",
    )
    parser.add_argument(
        "--top-k", type=int, default=20,
        help="How many top per-field deltas to render in summary.md (default: %(default)s).",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed for paired bootstrap (default: %(default)s).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="DEBUG-level logging.",
    )
    parser.set_defaults(_handler=_main)


def _main(args: argparse.Namespace) -> int:
    setup_logging(args.verbose)

    runs = _parse_runs(args.runs)
    args.out.mkdir(parents=True, exist_ok=True)

    # Load each cascade run's atomic table. Coerce the JSON-stringified
    # correct/attempted columns back to numeric — see
    # :func:`scripts.eval._common.loaders.coerce_cascade_correct` for why
    # this is necessary on parquets that mix scalar bool with nested float.
    atomics: dict[str, pd.DataFrame] = {}
    for label, path in runs.items():
        atomic_path = path / "cascade_atomic.parquet"
        if not atomic_path.is_file():
            raise SystemExit(f"missing cascade_atomic.parquet under {path}")
        df = pd.read_parquet(atomic_path)
        df["correct"] = coerce_cascade_correct(df["correct"])
        if "attempted" in df.columns:
            df["attempted"] = coerce_cascade_bool(df["attempted"])
        df["__run_label__"] = label
        atomics[label] = df
        logger.info("loaded %s: %d rows from %s", label, len(df), atomic_path)

    # --- Top-level cross-cutting CSVs --------------------------------
    headline = _build_headline(atomics, alpha=args.alpha)
    write_csv(headline, args.out / "headline.csv")

    pairwise = _build_pairwise(
        atomics, n_boot=args.n_boot, alpha=args.alpha,
        random_state=args.seed,
    )
    write_csv(pairwise, args.out / "pairwise_deltas.csv")

    funnel = _build_funnel_compare(atomics)
    write_csv(funnel, args.out / "cascade_funnel_compare.csv")

    verdict = _build_verdict(
        atomics, n_boot=args.n_boot, alpha=args.alpha,
        random_state=args.seed,
    )
    write_csv(verdict, args.out / "verdict.csv")

    others = _build_others_compare(runs)
    if not others.empty:
        write_csv(others, args.out / "others_compare.csv")

    # --- Chapter 1: eligibility (Stage A) ----------------------------
    ch1_dir = args.out / "chapter1_eligibility"
    write_csv(_build_stage_comparison(atomics, stage="A", alpha=args.alpha),
              ch1_dir / "comparison.csv")
    write_csv(_build_stage_pairwise(
        atomics, stage="A",
        n_boot=args.n_boot, alpha=args.alpha, random_state=args.seed,
    ), ch1_dir / "pairwise.csv")

    # --- Chapter 2: organ classification (Stage B) -------------------
    ch2_dir = args.out / "chapter2_organ_classification"
    write_csv(_build_stage_comparison(atomics, stage="B", alpha=args.alpha),
              ch2_dir / "comparison.csv")
    write_csv(_build_stage_pairwise(
        atomics, stage="B",
        n_boot=args.n_boot, alpha=args.alpha, random_state=args.seed,
    ), ch2_dir / "pairwise.csv")
    write_csv(_build_stage_b_confusion_compare(atomics),
              ch2_dir / "confusion_per_class_compare.csv")

    # --- Chapter 3: field extraction (Stage C scalar) ----------------
    ch3_dir = args.out / "chapter3_field_extraction"
    per_field_wide = _build_per_field_wide(atomics)
    write_csv(per_field_wide, ch3_dir / "per_field_wide.csv")
    per_organ_wide = _build_per_organ_wide(atomics)
    write_csv(per_organ_wide, ch3_dir / "per_organ_wide.csv")
    # Stage C scalar pairwise is the same set of rows already in the
    # top-level pairwise_deltas.csv (which filters to Stage C scalar by
    # construction). Mirror it here for chapter discoverability.
    write_csv(pairwise, ch3_dir / "pairwise.csv")

    # --- Chapter 4: margins (nested) ---------------------------------
    ch4_dir = args.out / "chapter4_margins"
    write_csv(_build_nested_comparison(
        atomics, field="margins", alpha=args.alpha,
        n_boot=args.n_boot, random_state=args.seed,
    ), ch4_dir / "comparison.csv")
    write_csv(_build_nested_pairwise(
        atomics, field="margins",
        n_boot=args.n_boot, alpha=args.alpha, random_state=args.seed,
    ), ch4_dir / "pairwise.csv")

    # --- Chapter 5: regional lymph nodes (nested) --------------------
    ch5_dir = args.out / "chapter5_lymph_nodes"
    write_csv(_build_nested_comparison(
        atomics, field="regional_lymph_node", alpha=args.alpha,
        n_boot=args.n_boot, random_state=args.seed,
    ), ch5_dir / "comparison.csv")
    write_csv(_build_nested_pairwise(
        atomics, field="regional_lymph_node",
        n_boot=args.n_boot, alpha=args.alpha, random_state=args.seed,
    ), ch5_dir / "pairwise.csv")

    # --- Brief markdown report ---------------------------------------
    report = _render_report(
        headline=headline, per_field_wide=per_field_wide,
        per_organ_wide=per_organ_wide, pairwise=pairwise,
        funnel=funnel, others=others, verdict=verdict,
        runs=runs, top_k=args.top_k, alpha=args.alpha,
    )
    report_path = args.out / "summary.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info("wrote %s", report_path)

    # --- Manifest ----------------------------------------------------
    write_manifest(
        args.out, args, subcommand="compare",
        extra={
            "n_runs": len(runs),
            "run_labels": list(runs.keys()),
            "input_paths": {k: str(v) for k, v in runs.items()},
            "n_atomic_rows_per_run": {k: int(len(v)) for k, v in atomics.items()},
        },
    )

    # Also print the verdict directly to stdout so the user gets the
    # answer without opening a file.
    print()
    print(_render_verdict_one_liner(verdict, runs, alpha=args.alpha))
    print(f"\nFull report: {report_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Backward-compat entry point for ``python -m scripts.eval.cascade.compare_runs``.

    Delegates to the canonical ``scripts.eval.cli compare`` dispatcher so
    existing scripts that invoke this module directly keep working.
    """
    import sys
    from scripts.eval.cli import main as cli_main
    if argv is None:
        argv = sys.argv[1:]
    return cli_main(["compare", *argv])


# --- Builders -------------------------------------------------------------

def _safe_acc(correct: pd.Series) -> tuple[float, int, int]:
    correct = pd.to_numeric(correct, errors="coerce").dropna().astype(int)
    n = len(correct)
    k = int(correct.sum())
    return (k / n if n else float("nan"), k, n)


def _build_headline(
    atomics: dict[str, pd.DataFrame],
    *, alpha: float,
) -> pd.DataFrame:
    """One row per run: cascade-stage accuracies + Cohen's κ."""
    rows: list[dict] = []
    for label, df in atomics.items():
        # Stage A
        stage_a = df[df["cascade_stage"] == "A"]
        a_acc, a_k, a_n = _safe_acc(stage_a["correct"])
        a_lo, a_hi = wilson_ci(a_k, a_n, alpha) if a_n else (float("nan"),) * 2
        # Stage B
        stage_b = df[df["cascade_stage"] == "B"]
        b_acc, b_k, b_n = _safe_acc(stage_b["correct"])
        b_lo, b_hi = wilson_ci(b_k, b_n, alpha) if b_n else (float("nan"),) * 2
        # Stage C aggregate accuracy across all fields
        stage_c = df[df["cascade_stage"] == "C"]
        attempted = stage_c[stage_c["attempted"] == True]  # noqa: E712
        c_acc, c_k, c_n = _safe_acc(attempted["correct"])
        c_lo, c_hi = wilson_ci(c_k, c_n, alpha) if c_n else (float("nan"),) * 2
        # Stage B Cohen's κ
        b_paired = [
            (g, p) for g, p in zip(stage_b["gold_value"], stage_b["pred_value"])
            if g is not None and p is not None
        ]
        b_kappa = float("nan")
        if b_paired:
            try:
                gv, pv = zip(*b_paired)
                b_kappa = cohens_kappa(list(gv), list(pv))
            except Exception:
                pass
        rows.append({
            "run": label,
            "n_cases_total": int(df["case_id"].nunique()),
            "stage_a_n": a_n, "stage_a_acc": a_acc,
            "stage_a_lo": a_lo, "stage_a_hi": a_hi,
            "stage_b_n": b_n, "stage_b_acc": b_acc,
            "stage_b_lo": b_lo, "stage_b_hi": b_hi,
            "stage_b_kappa": b_kappa,
            "stage_c_n_attempted": c_n, "stage_c_acc": c_acc,
            "stage_c_lo": c_lo, "stage_c_hi": c_hi,
        })
    return pd.DataFrame(rows)


def _stage_c_scalar(df: pd.DataFrame) -> pd.DataFrame:
    """Subset cascade_atomic to Stage-C scalar rows.

    Mirrors :func:`scripts.eval._common.loaders.cascade_scalar_only`. Inlined
    here (rather than imported) so this module remains usable on cascade
    outputs produced before that helper landed.
    """
    sub = df[df["cascade_stage"] == "C"]
    if "field_kind" in sub.columns:
        sub = sub[sub["field_kind"] != "nested_list"]
    return sub


def _build_per_field_wide(atomics: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Wide pivot: rows=field, columns=`<run>_acc`, `<run>_n`. Scalar Stage-C only."""
    pieces: list[pd.DataFrame] = []
    for label, df in atomics.items():
        stage_c = _stage_c_scalar(df)
        attempted = stage_c[stage_c["attempted"] == True]  # noqa: E712
        if attempted.empty:
            continue
        grp = attempted.groupby("field").agg(
            n_attempted=("correct", "size"),
            n_correct=("correct", lambda s: int(pd.to_numeric(s, errors="coerce").fillna(0).sum())),
        )
        grp["acc"] = grp["n_correct"] / grp["n_attempted"]
        grp = grp.rename(columns={
            "n_attempted": f"{label}_n",
            "acc": f"{label}_acc",
            "n_correct": f"{label}_correct",
        })
        pieces.append(grp[[f"{label}_acc", f"{label}_n", f"{label}_correct"]])
    if not pieces:
        return pd.DataFrame()
    out = pd.concat(pieces, axis=1).reset_index()
    return out


def _build_per_organ_wide(atomics: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Wide pivot: rows=organ, columns=`<run>_acc`. Scalar Stage-C only."""
    pieces: list[pd.DataFrame] = []
    for label, df in atomics.items():
        stage_c = _stage_c_scalar(df)
        attempted = stage_c[stage_c["attempted"] == True]  # noqa: E712
        if attempted.empty:
            continue
        grp = attempted.groupby("organ").agg(
            n_attempted=("correct", "size"),
            n_correct=("correct", lambda s: int(pd.to_numeric(s, errors="coerce").fillna(0).sum())),
        )
        grp["acc"] = grp["n_correct"] / grp["n_attempted"]
        grp = grp.rename(columns={
            "acc": f"{label}_acc", "n_attempted": f"{label}_n",
            "n_correct": f"{label}_correct",
        })
        pieces.append(grp[[f"{label}_acc", f"{label}_n", f"{label}_correct"]])
    if not pieces:
        return pd.DataFrame()
    out = pd.concat(pieces, axis=1).reset_index()
    return out


def _build_pairwise(
    atomics: dict[str, pd.DataFrame],
    *, n_boot: int, alpha: float, random_state: int,
) -> pd.DataFrame:
    """For every (run_A, run_B) pair × Stage-C scalar field, paired-bootstrap delta + McNemar.

    Pairing key is ``(case_id, field)``. Cases without a row in both
    runs are dropped from that field's pairing. Nested-list rows are
    excluded (their ``correct`` is float F1 — not a binary outcome).
    """
    rows: list[dict] = []
    labels = list(atomics.keys())
    # Pre-pivot each method to (case_id, field) -> mean correctness over
    # whatever runs the atomic table carries. For single-run methods this
    # is identity. For multi-run methods (K=10 BERT, K=5 LLM, ...) this
    # collapses to a per-case expected-accuracy in [0, 1] so the same
    # pairing logic works regardless of K_a vs K_b. McNemar still works
    # because we binarize via >= 0.5 below before calling it.
    pivots: dict[str, pd.DataFrame] = {}
    for label, df in atomics.items():
        stage_c = _stage_c_scalar(df)
        if stage_c.empty:
            continue
        sc = stage_c.copy()
        sc["correct_num"] = pd.to_numeric(sc["correct"], errors="coerce")
        pivots[label] = sc.pivot_table(
            index="case_id", columns="field",
            values="correct_num", aggfunc="mean",
        )

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a_label, b_label = labels[i], labels[j]
            if a_label not in pivots or b_label not in pivots:
                continue
            a_pivot, b_pivot = pivots[a_label], pivots[b_label]
            common_fields = set(a_pivot.columns) & set(b_pivot.columns)
            for field in sorted(common_fields):
                a_col = a_pivot[field]
                b_col = b_pivot[field]
                pair = pd.concat([a_col, b_col], axis=1, keys=["a", "b"]).dropna()
                if pair.empty:
                    continue
                a_vals = pair["a"].astype(float).tolist()
                b_vals = pair["b"].astype(float).tolist()
                # McNemar needs binary outcomes. Binarize by majority vote
                # across runs (a value already in {0, 1} is a no-op).
                a_bin = [1 if v >= 0.5 else 0 for v in a_vals]
                b_bin = [1 if v >= 0.5 else 0 for v in b_vals]
                mc = mcnemar(a_bin, b_bin)
                boot = paired_bootstrap_delta(
                    a_vals, b_vals,
                    n_boot=n_boot, alpha=alpha, random_state=random_state,
                )
                rows.append({
                    "field": field,
                    "run_a": a_label, "run_b": b_label,
                    "n_pairs": mc.n,
                    "acc_a": float(np.mean(a_vals)),
                    "acc_b": float(np.mean(b_vals)),
                    "delta_acc": boot.effect_size,
                    "delta_ci_lo": boot.effect_ci_lo,
                    "delta_ci_hi": boot.effect_ci_hi,
                    "mcnemar_p": mc.p_raw,
                    "mcnemar_method": mc.notes,
                    "above_threshold": (
                        abs(boot.effect_size) >= HEADLINE_ACCURACY_DELTA_THRESHOLD
                        if not np.isnan(boot.effect_size) else False
                    ),
                })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Holm + BH adjust within (run_a, run_b) family.
    df["mcnemar_p_holm"] = float("nan")
    df["mcnemar_p_bh"] = float("nan")
    for (a, b), sub in df.groupby(["run_a", "run_b"]):
        df.loc[sub.index, "mcnemar_p_holm"] = holm(sub["mcnemar_p"].tolist())
        df.loc[sub.index, "mcnemar_p_bh"] = bh_fdr(sub["mcnemar_p"].tolist())
    return df


def _build_funnel_compare(atomics: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-run cascade attrition counts."""
    rows: list[dict] = []
    for label, df in atomics.items():
        n_total = int(df["case_id"].nunique())
        a_pass = int(df[(df["cascade_stage"] == "A")
                        & (df["gate_pass"] == True)]["case_id"].nunique())  # noqa: E712
        b_pass = int(df[(df["cascade_stage"] == "B")
                        & (df["gate_pass"] == True)]["case_id"].nunique())  # noqa: E712
        c_scored = int(df[df["cascade_stage"] == "C"]["case_id"].nunique())
        rows.append({
            "run": label,
            "n_total": n_total,
            "n_passed_a": a_pass,
            "n_passed_b": b_pass,
            "n_scored_c": c_scored,
            "attrition_a": (n_total - a_pass) / n_total if n_total else float("nan"),
            "attrition_b_given_a": ((a_pass - b_pass) / a_pass) if a_pass else float("nan"),
            "stage_c_yield": c_scored / n_total if n_total else float("nan"),
        })
    return pd.DataFrame(rows)


def _build_verdict(
    atomics: dict[str, pd.DataFrame],
    *, n_boot: int, alpha: float, random_state: int,
) -> pd.DataFrame:
    """Pooled paired comparison per (run_A, run_B) at each cascade stage.

    For each stage, every paired (case_id[, field]) correctness datum
    contributes to a single accuracy delta with paired-bootstrap CI and
    McNemar p-value. This is the *headline* answer to "which run is
    better?" — Stage C pools across all fields × cases.

    Stages:
        A: paired by case_id, field=cancer_excision_report.
        B: paired by case_id, field=cancer_category, only on cases that
           passed A in BOTH runs.
        C: paired by (case_id, field) across all Stage-C scalar fields,
           only on case-field cells where both runs scored.
    """
    rows: list[dict] = []
    labels = list(atomics.keys())

    def _pivot_stage(df: pd.DataFrame, stage: str) -> pd.Series:
        sub = df[df["cascade_stage"] == stage].copy()
        if sub.empty:
            return pd.Series(dtype=float)
        sub["correct_num"] = pd.to_numeric(sub["correct"], errors="coerce")
        # Mean-aggregate over runs so multi-run methods (K=10 BERT,
        # K=5 LLM, ...) yield a unique per-(case_id, field) expected
        # accuracy in [0, 1]. Single-run methods are unchanged.
        return (sub.groupby(["case_id", "field"])["correct_num"]
                  .mean(numeric_only=False))

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a_label, b_label = labels[i], labels[j]
            a_df, b_df = atomics[a_label], atomics[b_label]

            for stage in ("A", "B", "C"):
                a_ser = _pivot_stage(a_df, stage)
                b_ser = _pivot_stage(b_df, stage)
                if a_ser.empty or b_ser.empty:
                    continue
                pair = pd.concat([a_ser, b_ser], axis=1, keys=["a", "b"]).dropna()
                if pair.empty:
                    continue
                a_vals = pair["a"].astype(float).tolist()
                b_vals = pair["b"].astype(float).tolist()
                # McNemar needs binary; binarize via >=0.5 majority vote.
                a_bin = [1 if v >= 0.5 else 0 for v in a_vals]
                b_bin = [1 if v >= 0.5 else 0 for v in b_vals]
                mc = mcnemar(a_bin, b_bin)
                # paired_bootstrap_delta(x, y) gives mean(x) - mean(y).
                # We want delta = b - a, so pass (b, a). Continuous values
                # are fine; bootstrap CI on per-case expected accuracies.
                boot = paired_bootstrap_delta(
                    b_vals, a_vals,
                    n_boot=n_boot, alpha=alpha, random_state=random_state,
                )
                acc_a = float(np.mean(a_vals))
                acc_b = float(np.mean(b_vals))
                # Conclusion logic:
                #   - If McNemar p < alpha, the run with the larger pooled
                #     accuracy is *significantly* better.
                #   - Else, no significant difference.
                if np.isnan(mc.p_raw):
                    verdict = "undetermined"
                elif mc.p_raw < alpha:
                    verdict = (f"{b_label} better"
                               if acc_b > acc_a else f"{a_label} better")
                else:
                    verdict = "tie"
                rows.append({
                    "stage": stage,
                    "run_a": a_label, "run_b": b_label,
                    "n_pairs": mc.n,
                    "acc_a": acc_a, "acc_b": acc_b,
                    "delta_acc_b_minus_a": acc_b - acc_a,
                    "delta_ci_lo": boot.effect_ci_lo,
                    "delta_ci_hi": boot.effect_ci_hi,
                    "mcnemar_p": mc.p_raw,
                    "alpha": alpha,
                    "verdict": verdict,
                    "notes": mc.notes,
                })
    return pd.DataFrame(rows)


def _render_verdict_one_liner(
    verdict: pd.DataFrame,
    runs: dict[str, Path],
    *, alpha: float,
) -> str:
    """One-paragraph plain-language verdict for stdout + summary.md header."""
    if verdict.empty:
        return "VERDICT: undetermined (no overlapping cases between runs)."

    labels = list(runs.keys())
    if len(labels) == 2:
        a, b = labels
        # Use Stage C as the headline; fall back to B then A if absent.
        for stage in ("C", "B", "A"):
            row = verdict[(verdict["stage"] == stage)
                          & (verdict["run_a"] == a)
                          & (verdict["run_b"] == b)]
            if row.empty:
                continue
            r = row.iloc[0]
            acc_a = r["acc_a"]
            acc_b = r["acc_b"]
            delta = r["delta_acc_b_minus_a"]
            p = r["mcnemar_p"]
            n = int(r["n_pairs"])
            lo = r["delta_ci_lo"]
            hi = r["delta_ci_hi"]
            v = r["verdict"]
            label = {"A": "Stage A (eligibility)",
                     "B": "Stage B (organ classification)",
                     "C": "Stage C (field extraction)"}[stage]
            if v == "tie":
                conclusion = (
                    f"VERDICT — {label}: NO SIGNIFICANT DIFFERENCE. "
                    f"{a} = {acc_a:.3f}, {b} = {acc_b:.3f} "
                    f"(Δ = {delta:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}], "
                    f"McNemar p = {p:.3f}, n = {n})."
                )
            elif "better" in v:
                winner, loser = (b, a) if v.startswith(b) else (a, b)
                w_acc, l_acc = (acc_b, acc_a) if winner == b else (acc_a, acc_b)
                conclusion = (
                    f"VERDICT — {label}: **{winner.upper()} IS BETTER** "
                    f"({winner}={w_acc:.3f} vs {loser}={l_acc:.3f}, "
                    f"Δ = {delta:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}], "
                    f"McNemar p = {p:.4f}, n = {n})."
                )
            else:
                conclusion = (
                    f"VERDICT — {label}: undetermined ({v})."
                )
            return conclusion
        return "VERDICT: no Stage A/B/C overlap between runs."

    # >2 runs: list all pairwise verdicts at Stage C.
    lines = ["VERDICT (Stage C, pairwise):"]
    stage_c = verdict[verdict["stage"] == "C"]
    for _, r in stage_c.iterrows():
        lines.append(
            f"  {r['run_a']} vs {r['run_b']}: "
            f"{r['acc_a']:.3f} / {r['acc_b']:.3f}, "
            f"Δ = {r['delta_acc_b_minus_a']:+.3f}, "
            f"p = {r['mcnemar_p']:.3f} → {r['verdict']}"
        )
    return "\n".join(lines)


def _build_stage_comparison(
    atomics: dict[str, pd.DataFrame],
    *, stage: str, alpha: float,
) -> pd.DataFrame:
    """Per-run accuracy + Wilson CI (+ Cohen's κ for Stage B) on a cascade stage.

    Stage A is binary eligibility; Stage B is multiclass organ. For Stage
    B, ``cohens_kappa`` is computed on (gold_value, pred_value) pairs.
    """
    rows: list[dict] = []
    for label, df in atomics.items():
        sub = df[df["cascade_stage"] == stage]
        acc, k, n = _safe_acc(sub["correct"])
        lo, hi = wilson_ci(k, n, alpha) if n else (float("nan"),) * 2
        row: dict = {
            "run": label, "n": n, "n_correct": k,
            "accuracy": acc, "ci_lo": lo, "ci_hi": hi,
        }
        if stage == "B":
            paired = [
                (g, p) for g, p in zip(sub["gold_value"], sub["pred_value"])
                if g is not None and p is not None
            ]
            kappa = float("nan")
            if paired:
                try:
                    gv, pv = zip(*paired)
                    kappa = cohens_kappa(list(gv), list(pv))
                except Exception:
                    pass
            row["cohens_kappa"] = kappa
        rows.append(row)
    return pd.DataFrame(rows)


def _build_stage_pairwise(
    atomics: dict[str, pd.DataFrame],
    *, stage: str, n_boot: int, alpha: float, random_state: int,
) -> pd.DataFrame:
    """Pairwise McNemar + paired-bootstrap delta for a single cascade stage.

    Pairing key is ``case_id`` (one row per case at the given stage).
    Cases not present in both runs at this stage are dropped.
    """
    rows: list[dict] = []
    labels = list(atomics.keys())
    pivots: dict[str, pd.Series] = {}
    for label, df in atomics.items():
        sub = df[df["cascade_stage"] == stage].copy()
        if sub.empty:
            continue
        sub["correct_num"] = pd.to_numeric(sub["correct"], errors="coerce")
        # Mean over runs so multi-run methods yield one expected-accuracy
        # per case; single-run is identity.
        pivots[label] = (sub.groupby("case_id")["correct_num"]
                           .mean(numeric_only=False))

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a_label, b_label = labels[i], labels[j]
            if a_label not in pivots or b_label not in pivots:
                continue
            pair = pd.concat(
                [pivots[a_label], pivots[b_label]],
                axis=1, keys=["a", "b"],
            ).dropna()
            if pair.empty:
                continue
            a_vals = pair["a"].astype(float).tolist()
            b_vals = pair["b"].astype(float).tolist()
            a_bin = [1 if v >= 0.5 else 0 for v in a_vals]
            b_bin = [1 if v >= 0.5 else 0 for v in b_vals]
            mc = mcnemar(a_bin, b_bin)
            boot = paired_bootstrap_delta(
                b_vals, a_vals,
                n_boot=n_boot, alpha=alpha, random_state=random_state,
            )
            acc_a = float(np.mean(a_vals))
            acc_b = float(np.mean(b_vals))
            rows.append({
                "stage": stage,
                "run_a": a_label, "run_b": b_label,
                "n_pairs": mc.n,
                "acc_a": acc_a, "acc_b": acc_b,
                "delta_acc_b_minus_a": acc_b - acc_a,
                "delta_ci_lo": boot.effect_ci_lo,
                "delta_ci_hi": boot.effect_ci_hi,
                "mcnemar_p": mc.p_raw,
                "notes": mc.notes,
            })
    return pd.DataFrame(rows)


def _build_stage_b_confusion_compare(
    atomics: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Per-gold-class recall across runs (Stage B).

    Wide pivot with columns ``<run>_n_gold``, ``<run>_n_correct``,
    ``<run>_recall`` so a reader can see where each run drops Stage-B
    cases. Pred-only ("hallucination") classes are not surfaced — that
    information lives in the others ledger.
    """
    pieces: list[pd.DataFrame] = []
    for label, df in atomics.items():
        sub = df[df["cascade_stage"] == "B"]
        if sub.empty:
            continue
        rows: list[dict] = []
        for organ_class, group in sub.groupby("gold_value"):
            if organ_class is None:
                continue
            n_gold = len(group)
            n_correct = int(
                pd.to_numeric(group["correct"], errors="coerce").fillna(0).sum()
            )
            rows.append({
                "gold_class": organ_class,
                f"{label}_n_gold": n_gold,
                f"{label}_n_correct": n_correct,
                f"{label}_recall": n_correct / n_gold if n_gold else float("nan"),
            })
        if rows:
            pieces.append(pd.DataFrame(rows).set_index("gold_class"))
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, axis=1).reset_index()


def _bootstrap_mean_ci(
    values: Sequence[float], *, alpha: float, n_boot: int, random_state: int,
) -> tuple[float, float]:
    """Percentile-bootstrap CI on the mean of ``values``."""
    if not values:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(random_state)
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    boots = np.empty(n_boot)
    for r in range(n_boot):
        boots[r] = rng.choice(arr, size=n, replace=True).mean()
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return (lo, hi)


def _build_nested_comparison(
    atomics: dict[str, pd.DataFrame],
    *, field: str, alpha: float, n_boot: int, random_state: int,
) -> pd.DataFrame:
    """Per-run mean F1 + four-level missingness counts for a nested field.

    Reads atomic rows where ``field == <field>`` and ``field_kind ==
    "nested_list"``. The headline F1 is the mean of ``correct`` over
    rows where ``attempted == True`` (cascade emits a float F1 in those
    rows). Missingness counts use ``nested_missingness_level`` if
    present.
    """
    rows: list[dict] = []
    for label, df in atomics.items():
        sub = df[
            (df["cascade_stage"] == "C")
            & (df["field"] == field)
        ]
        if "field_kind" in sub.columns:
            sub = sub[sub["field_kind"] == "nested_list"]
        n_total = int(len(sub))
        if n_total == 0:
            rows.append({
                "run": label, "field": field,
                "n_total": 0, "n_attempted": 0,
                "mean_f1": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan"),
                "n_parse_error": 0, "n_field_key_absent": 0,
                "n_empty_list": 0, "n_partial_list": 0,
            })
            continue
        attempted = sub[sub["attempted"] == True]  # noqa: E712
        f1_vals = (
            pd.to_numeric(attempted["correct"], errors="coerce")
            .dropna().tolist()
        )
        n_att = len(f1_vals)
        mean_f1 = float(np.mean(f1_vals)) if n_att else float("nan")
        ci_lo, ci_hi = _bootstrap_mean_ci(
            f1_vals, alpha=alpha, n_boot=n_boot, random_state=random_state,
        )
        miss = sub.get("nested_missingness_level")
        if miss is not None:
            n_pe = int((miss == "parse_error").sum())
            n_fka = int((miss == "field_key_absent").sum())
            n_el = int((miss == "empty_list").sum())
            n_pl = int((miss == "partial_list").sum())
        else:
            n_pe = n_fka = n_el = n_pl = 0
        rows.append({
            "run": label, "field": field,
            "n_total": n_total, "n_attempted": n_att,
            "mean_f1": mean_f1,
            "ci_lo": ci_lo, "ci_hi": ci_hi,
            "n_parse_error": n_pe,
            "n_field_key_absent": n_fka,
            "n_empty_list": n_el,
            "n_partial_list": n_pl,
        })
    return pd.DataFrame(rows)


def _build_nested_pairwise(
    atomics: dict[str, pd.DataFrame],
    *, field: str, n_boot: int, alpha: float, random_state: int,
) -> pd.DataFrame:
    """Paired-bootstrap delta on per-case F1 for a nested field.

    No McNemar — the per-case outcome is continuous (F1 ∈ [0, 1]), so a
    binary contingency test isn't appropriate. We report Δ mean F1 with
    a paired-bootstrap CI.
    """
    rows: list[dict] = []
    labels = list(atomics.keys())
    pivots: dict[str, pd.Series] = {}
    for label, df in atomics.items():
        sub = df[
            (df["cascade_stage"] == "C")
            & (df["field"] == field)
            & (df["attempted"] == True)  # noqa: E712
        ].copy()
        if "field_kind" in sub.columns:
            sub = sub[sub["field_kind"] == "nested_list"]
        if sub.empty:
            continue
        sub["f1_num"] = pd.to_numeric(sub["correct"], errors="coerce")
        pivots[label] = (sub.groupby("case_id")["f1_num"]
                           .mean(numeric_only=False))

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a_label, b_label = labels[i], labels[j]
            if a_label not in pivots or b_label not in pivots:
                continue
            pair = pd.concat(
                [pivots[a_label], pivots[b_label]],
                axis=1, keys=["a", "b"],
            ).dropna()
            if pair.empty:
                continue
            a_vals = pair["a"].astype(float).tolist()
            b_vals = pair["b"].astype(float).tolist()
            boot = paired_bootstrap_delta(
                b_vals, a_vals,
                n_boot=n_boot, alpha=alpha, random_state=random_state,
            )
            mean_a = float(np.mean(a_vals))
            mean_b = float(np.mean(b_vals))
            rows.append({
                "field": field,
                "run_a": a_label, "run_b": b_label,
                "n_pairs": len(a_vals),
                "mean_f1_a": mean_a, "mean_f1_b": mean_b,
                "delta_f1_b_minus_a": mean_b - mean_a,
                "delta_ci_lo": boot.effect_ci_lo,
                "delta_ci_hi": boot.effect_ci_hi,
            })
    return pd.DataFrame(rows)


def _build_others_compare(runs: dict[str, Path]) -> pd.DataFrame:
    """Sum the others ledger from each run, if present."""
    rows: list[dict] = []
    for label, path in runs.items():
        ledger_path = path / "chapter2_organ_classification" / "others" / "others_ledger.csv"
        if not ledger_path.is_file():
            continue
        try:
            ledger = pd.read_csv(ledger_path)
        except Exception as e:
            logger.warning("skipping others ledger for %s: %s", label, e)
            continue
        rows.append({
            "run": label,
            "n_others_rows": len(ledger),
            "n_gold_only": int((ledger["stage_b_disposition"] == "gold_only").sum())
                if "stage_b_disposition" in ledger else 0,
            "n_pred_only": int((ledger["stage_b_disposition"] == "pred_only").sum())
                if "stage_b_disposition" in ledger else 0,
            "n_both_others": int((ledger["stage_b_disposition"] == "both_others").sum())
                if "stage_b_disposition" in ledger else 0,
            "n_dual_primary_subtype": int((ledger["others_subtype"] == "dual_primary").sum())
                if "others_subtype" in ledger else 0,
        })
    return pd.DataFrame(rows)


# --- Markdown rendering ---------------------------------------------------

def _fmt_pct(v: float, prec: int = 3) -> str:
    if v is None or (isinstance(v, float) and (np.isnan(v) or v != v)):
        return "—"
    return f"{v:.{prec}f}"


def _md_table(df: pd.DataFrame, columns: list[str] | None = None,
              col_labels: dict[str, str] | None = None) -> str:
    if df is None or df.empty:
        return "_(no rows)_\n"
    if columns is None:
        columns = list(df.columns)
    labels = [col_labels.get(c, c) if col_labels else c for c in columns]
    out = ["| " + " | ".join(labels) + " |",
           "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in columns:
            v = row.get(c)
            if isinstance(v, float):
                cells.append(_fmt_pct(v))
            else:
                cells.append(str(v) if v is not None else "—")
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def _render_report(
    *,
    headline: pd.DataFrame,
    per_field_wide: pd.DataFrame,
    per_organ_wide: pd.DataFrame,
    pairwise: pd.DataFrame,
    funnel: pd.DataFrame,
    others: pd.DataFrame,
    verdict: pd.DataFrame,
    runs: dict[str, Path],
    top_k: int,
    alpha: float,
) -> str:
    parts: list[str] = []
    parts.append("# Cascade run comparison\n")
    parts.append("Brief accuracy comparison across "
                 f"{len(runs)} runs:\n")
    for label, path in runs.items():
        parts.append(f"- **{label}** — `{path}`")
    parts.append("\nFull tables in supporting CSVs in this directory.\n")

    # --- Verdict (lead with the answer) -------------------------------
    parts.append("\n## Verdict\n")
    parts.append(_render_verdict_one_liner(verdict, runs, alpha=alpha))
    parts.append("")
    if not verdict.empty:
        parts.append("\nFull pairwise pooled-paired tests at every stage:\n")
        parts.append(_md_table(
            verdict,
            columns=["stage", "run_a", "run_b", "n_pairs",
                     "acc_a", "acc_b", "delta_acc_b_minus_a",
                     "delta_ci_lo", "delta_ci_hi",
                     "mcnemar_p", "verdict"],
            col_labels={
                "n_pairs": "n",
                "delta_acc_b_minus_a": "Δ (b−a)",
                "delta_ci_lo": "Δ lo", "delta_ci_hi": "Δ hi",
                "mcnemar_p": "McNemar p",
            },
        ))

    # --- Headline -----------------------------------------------------
    parts.append("\n## Headline (Stage A / B / C)\n")
    parts.append(_md_table(
        headline,
        columns=["run", "n_cases_total",
                 "stage_a_acc", "stage_a_lo", "stage_a_hi",
                 "stage_b_acc", "stage_b_kappa",
                 "stage_c_n_attempted", "stage_c_acc",
                 "stage_c_lo", "stage_c_hi"],
        col_labels={
            "n_cases_total": "n cases",
            "stage_a_acc": "A acc",
            "stage_a_lo": "A 95% lo", "stage_a_hi": "A 95% hi",
            "stage_b_acc": "B acc", "stage_b_kappa": "B κ",
            "stage_c_n_attempted": "C n",
            "stage_c_acc": "C acc",
            "stage_c_lo": "C 95% lo", "stage_c_hi": "C 95% hi",
        },
    ))

    # --- Funnel attrition --------------------------------------------
    parts.append("\n## Cascade funnel — cohort attrition\n")
    parts.append(_md_table(
        funnel,
        columns=["run", "n_total", "n_passed_a", "n_passed_b",
                 "n_scored_c", "attrition_a", "attrition_b_given_a",
                 "stage_c_yield"],
    ))

    # --- Per-organ accuracy ------------------------------------------
    parts.append("\n## Stage C per-organ accuracy\n")
    if per_organ_wide.empty:
        parts.append("_(no Stage C rows)_\n")
    else:
        labels = list(runs.keys())
        cols = ["organ"] + [f"{lbl}_acc" for lbl in labels]
        parts.append(_md_table(per_organ_wide, columns=cols))

    # --- Top per-field deltas ----------------------------------------
    parts.append(f"\n## Top {top_k} per-field deltas (sorted by |Δ|)\n")
    if pairwise.empty:
        parts.append("_(no pairwise rows)_\n")
    else:
        # Sort by absolute delta descending; show only |Δ| ≥ threshold rows by default.
        flagged = pairwise[pairwise["above_threshold"] == True].copy()  # noqa: E712
        if flagged.empty:
            parts.append(
                "_(no pair × field exceeded the pre-registered threshold "
                f"|Δacc| ≥ {HEADLINE_ACCURACY_DELTA_THRESHOLD}; "
                "see `pairwise_deltas.csv` for the full table)_\n"
            )
        else:
            flagged["abs_delta"] = flagged["delta_acc"].abs()
            flagged = flagged.sort_values("abs_delta", ascending=False).head(top_k)
            parts.append(_md_table(
                flagged,
                columns=["field", "run_a", "run_b",
                         "acc_a", "acc_b",
                         "delta_acc", "delta_ci_lo", "delta_ci_hi",
                         "mcnemar_p", "mcnemar_p_holm",
                         "n_pairs"],
                col_labels={
                    "delta_acc": "Δ", "delta_ci_lo": "Δ lo",
                    "delta_ci_hi": "Δ hi",
                    "mcnemar_p_holm": "p (Holm)",
                    "n_pairs": "n",
                },
            ))

    # --- Others ledger ------------------------------------------------
    if others is not None and not others.empty:
        parts.append("\n## Others-ledger summary\n")
        parts.append(_md_table(others))

    parts.append("\n---\n")
    parts.append(f"_Pre-registered Δacc threshold: {HEADLINE_ACCURACY_DELTA_THRESHOLD}._\n")
    return "\n".join(parts)


if __name__ == "__main__":
    raise SystemExit(main())
