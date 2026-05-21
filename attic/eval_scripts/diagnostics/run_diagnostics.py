"""Diagnostics subcommand orchestrator.

Joins a cascade ``cascade_atomic.parquet`` with IAA outputs to classify
each model error into ``model_error`` / ``report_ambiguity`` /
``report_silent``, stratifies model accuracy by IAA-derived field
difficulty, and writes a worst-cases catalog.

Operates on **Stage-C scalar** rows only (Stage A/B are summarised in
their own chapter outputs and the nested fields use a continuous F1
that doesn't fit the binary error-decomposition taxonomy).

Output tree:
    manifest.json
    error_source_decomposition.csv
    accuracy_by_difficulty_tier.csv
    worst_cases.csv
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .._common.loaders import (
    cascade_scalar_only, coerce_cascade_bool, coerce_cascade_correct,
)
from .._common.reporting import setup_logging, write_csv, write_manifest

logger = logging.getLogger("scripts.eval.diagnostics")


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "diagnostics",
        help="Source-of-error decomposition + difficulty tiers + worst cases.",
        description=__doc__,
    )
    parser.add_argument(
        "--cascade-out", type=Path, required=True,
        help="Path to cascade output directory containing cascade_atomic.parquet.",
    )
    parser.add_argument(
        "--iaa-out", type=Path, required=True,
        help="Path to iaa output directory.",
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("workspace") / "results" / "eval" / "diagnostics",
        help="Output directory (default: %(default)s).",
    )
    parser.add_argument(
        "--top-n-worst", type=int, default=20,
        help="Per-field, how many worst cases to include (default: %(default)s).",
    )
    parser.add_argument(
        "--difficulty-thresholds", default="0.5,0.8",
        help="Comma-separated κ thresholds for hard/medium/easy "
             "(default: %(default)s).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
    )
    parser.set_defaults(_handler=_main)


def _main(args: argparse.Namespace) -> int:
    setup_logging(args.verbose)
    args.out.mkdir(parents=True, exist_ok=True)

    # 1. Load cascade atomic and restrict to Stage-C scalar rows.
    cascade_path = args.cascade_out / "cascade_atomic.parquet"
    if not cascade_path.exists():
        raise SystemExit(
            f"missing {cascade_path}; run `python -m scripts.eval.cli cascade` first.",
        )
    full = pd.read_parquet(cascade_path)
    atomic = cascade_scalar_only(full).copy()
    # cascade_atomic stores correct as object (bool|float|None) and may
    # be JSON-stringified by write_parquet when bool/float coexist
    # (Stage C scalar + Stage C nested). Coerce explicitly so downstream
    # boolean aggregations don't silently turn "false" strings into True.
    if "correct" in atomic.columns:
        atomic["correct"] = (
            coerce_cascade_correct(atomic["correct"]).fillna(0.0).astype(bool)
        )
    for col in ("wrong", "field_missing", "parse_error",
                "attempted", "gold_present"):
        if col in atomic.columns:
            atomic[col] = coerce_cascade_bool(atomic[col])
    logger.info(
        "loaded cascade atomic: %d total rows, %d Stage-C scalar rows",
        len(full), len(atomic),
    )

    # 2. Load IAA per-field κ — try the cross-human pair first, fall back
    # to gold-vs-human pair if cross-human is unavailable.
    kappa_per_field = _load_iaa_kappa_per_field(args.iaa_out)
    logger.info("loaded IAA κ for %d (organ, field) keys", len(kappa_per_field))

    # 3. Source-of-error decomposition.
    eds = _error_source_decomposition(atomic, kappa_per_field)
    write_csv(eds, args.out / "error_source_decomposition.csv")

    # 4. Difficulty-tier stratification.
    thresholds = [float(t) for t in args.difficulty_thresholds.split(",")]
    if len(thresholds) != 2:
        raise SystemExit("--difficulty-thresholds must be 'hard,easy'")
    dt = _difficulty_tier(atomic, kappa_per_field, thresholds=thresholds)
    write_csv(dt, args.out / "accuracy_by_difficulty_tier.csv")

    # 5. Worst-cases catalog.
    wc = _worst_cases(atomic, top_n=args.top_n_worst)
    write_csv(wc, args.out / "worst_cases.csv")

    write_manifest(
        args.out, args, subcommand="diagnostics",
        extra={
            "n_atomic_rows": int(len(atomic)),
            "n_iaa_kappa_keys": len(kappa_per_field),
            "thresholds_hard_easy": thresholds,
        },
    )
    logger.info("done. outputs in %s", args.out)
    return 0


# --- IAA κ loading ----------------------------------------------------------


def _load_iaa_kappa_per_field(iaa_out: Path) -> dict[tuple[str, str], float]:
    """Build ``{(organ, field): kappa_humans}`` from IAA pair CSVs.

    Prefers ``pair_nhc_with_preann_vs_kpc_with_preann.csv`` (cross-human
    pair) since that's the canonical "humans disagree" signal. Falls
    back to ``pair_gold_vs_*.csv`` if the cross-human pair is absent.
    """
    candidates = [
        "pair_nhc_with_preann_vs_kpc_with_preann.csv",
        "pair_nhc_without_preann_vs_kpc_without_preann.csv",
        "pair_gold_vs_nhc_with_preann.csv",
        "pair_gold_vs_kpc_with_preann.csv",
    ]
    for name in candidates:
        path = iaa_out / name
        if path.exists():
            df = pd.read_csv(path)
            return _extract_kappa(df)
    return {}


def _extract_kappa(df: pd.DataFrame) -> dict[tuple[str, str], float]:
    """Pull ``cohen_kappa_unweighted`` (or fallback) per (organ, field)
    from a pair_*.csv long-form table.
    """
    if df.empty:
        return {}
    needed = {"organ", "field", "stat_name", "estimate"}
    if not needed.issubset(df.columns):
        return {}
    sub = df[df["stat_name"].isin(["cohen_kappa_unweighted",
                                    "cohen_kappa_quadratic"])]
    if sub.empty:
        return {}
    out: dict[tuple[str, str], float] = {}
    for _, row in sub.iterrows():
        key = (row["organ"], row["field"])
        # Prefer unweighted; only overwrite if no value yet.
        if key not in out or row["stat_name"] == "cohen_kappa_unweighted":
            try:
                out[key] = float(row["estimate"])
            except (TypeError, ValueError):
                continue
    return out


# --- Source-of-error decomposition ------------------------------------------


def _error_source_decomposition(
    atomic: pd.DataFrame,
    kappa_per_field: dict[tuple[str, str], float],
) -> pd.DataFrame:
    """Bucket each model error into model_error / report_ambiguity /
    report_silent / parse_error.

    Heuristic:
        - parse_error  → bucket "parse_error"
        - field_missing AND gold null → "report_silent" (ambient null)
        - field_missing AND gold present → "model_error"
        - wrong AND κ_humans ≥ 0.8 → "model_error"
        - wrong AND κ_humans < 0.5 → "report_ambiguity"
        - wrong AND 0.5 ≤ κ_humans < 0.8 → "borderline"
        - wrong AND κ unknown → "model_error" (default to model fault)
    """
    rows: list[dict] = []
    for (organ, field), sub in atomic.groupby(["organ", "field"]):
        kappa = kappa_per_field.get((organ, field))
        n = len(sub)
        n_correct = int(sub["correct"].sum())
        n_parse = int(sub["parse_error"].sum())
        # field_missing further split by gold presence
        miss_with_gold = int((sub["field_missing"] & sub["gold_present"]).sum())
        miss_without_gold = int((sub["field_missing"] & ~sub["gold_present"]).sum())
        wrong = sub[sub["wrong"]]
        n_wrong = int(len(wrong))
        if kappa is not None and not pd.isna(kappa):
            if kappa >= 0.8:
                model_err = n_wrong + miss_with_gold
                ambig = 0
                borderline = 0
            elif kappa < 0.5:
                model_err = miss_with_gold
                ambig = n_wrong
                borderline = 0
            else:
                model_err = miss_with_gold
                ambig = 0
                borderline = n_wrong
        else:
            model_err = n_wrong + miss_with_gold
            ambig = 0
            borderline = 0
        rows.append({
            "organ": organ, "field": field,
            "n_total": n,
            "n_correct": n_correct,
            "n_parse_error": n_parse,
            "n_report_silent": miss_without_gold,
            "n_model_error": model_err,
            "n_report_ambiguity": ambig,
            "n_borderline": borderline,
            "kappa_humans": kappa,
            "model_bound_accuracy_ceiling": (
                1.0 - (model_err / n) if n else float("nan")
            ),
        })
    return pd.DataFrame(rows)


def _difficulty_tier(
    atomic: pd.DataFrame,
    kappa_per_field: dict[tuple[str, str], float],
    *, thresholds: list[float],
) -> pd.DataFrame:
    """Stratify model accuracy by human-IAA tier."""
    hard, easy = thresholds[0], thresholds[1]

    def _tier(k: float | None) -> str:
        if k is None or pd.isna(k):
            return "unknown"
        if k >= easy:
            return "easy"
        if k < hard:
            return "hard"
        return "medium"

    atomic = atomic.copy()
    atomic["kappa_humans"] = [
        kappa_per_field.get((o, f)) for o, f in
        zip(atomic["organ"], atomic["field"], strict=True)
    ]
    atomic["difficulty_tier"] = [_tier(k) for k in atomic["kappa_humans"]]
    out = (
        atomic[atomic["attempted"]]
        .groupby("difficulty_tier")
        .agg(
            n_attempted=("attempted", "sum"),
            n_correct=("correct", "sum"),
            attempted_accuracy=("correct", "mean"),
            n_unique_fields=("field", "nunique"),
        )
        .reset_index()
    )
    return out


def _worst_cases(atomic: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    """Top-N worst cases per field by run-disagreement + wrong-rate.

    Uses pivoted correctness across runs: cases where ≥half the runs got
    it wrong AND there's run-to-run variance are flagged as worst.
    """
    out_rows: list[dict] = []
    for field, sub in atomic.groupby("field"):
        if sub["run_id"].nunique() < 1:
            continue
        # Score per case = (1 − mean correctness) + run-variance signal.
        per_case = (
            sub.groupby("case_id")
            .agg(
                organ=("organ", "first"),
                mean_correct=("correct", "mean"),
                n_runs=("run_id", "nunique"),
                std_correct=("correct", "std"),
                attempted_rate=("attempted", "mean"),
            )
            .reset_index()
        )
        per_case["score"] = (1 - per_case["mean_correct"].fillna(0)) + \
                             per_case["std_correct"].fillna(0)
        per_case = per_case.sort_values("score", ascending=False).head(top_n)
        per_case["field"] = field
        out_rows.append(per_case)
    if not out_rows:
        return pd.DataFrame()
    return pd.concat(out_rows, ignore_index=True)


__all__ = ["register"]
