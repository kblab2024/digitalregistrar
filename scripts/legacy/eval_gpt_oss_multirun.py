#!/usr/bin/env python3
"""Accuracy + CI analysis for a gpt-oss multi-run experiment (Part C).

Consumes the artifacts produced by `run_gpt_oss_multirun.py`:

    <runs_root>/run1/<case>.json ... runK/<case>.json
    <runs_root>/_manifest.yaml          (optional; used to skip invalid runs)

Against a gold annotation root, produces:

    <out>/by_run.csv                    — atomic (run × case × field) table
    <out>/per_field_accuracy.csv        — point + case-CI + run-CI + total-CI
    <out>/per_organ.csv                 — organ-stratified point + CI
    <out>/per_fieldtype.csv             — field-type-stratified summary
    <out>/run_consistency.csv           — Fleiss κ across runs + flip rate
    <out>/ensemble_by_case.csv          — per-case ensemble correctness
    <out>/ensemble_vs_single.csv        — paired-bootstrap Δ per field

Usage:
    python scripts/eval_gpt_oss_multirun.py \\
        --runs-root results/benchmarks/gpt_oss \\
        --annotations data/cmuh_annotation_<date> \\
        --out results/gpt_oss_multirun
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import yaml

from digital_registrar_research.benchmarks.eval.iaa import classify_field
from digital_registrar_research.benchmarks.eval.multirun import (
    build_correctness_table,
    discover_runs,
    ensemble_vs_single,
    majority_vote_ensemble,
    per_field_ci,
    per_fieldtype_ci,
    per_organ_ci,
    run_consistency,
)

logger = logging.getLogger("eval_gpt_oss_multirun")


def _filter_valid_runs(runs, manifest_path: Path):
    if not manifest_path.exists():
        return runs
    try:
        with manifest_path.open(encoding="utf-8") as f:
            manifest = yaml.safe_load(f)
    except Exception:
        return runs
    valid_ids = {
        rid for rid, info in (manifest.get("runs") or {}).items()
        if info.get("valid", True)
    }
    if not valid_ids:
        return runs
    filtered = [(rid, p) for rid, p in runs if rid in valid_ids]
    if len(filtered) != len(runs):
        dropped = [rid for rid, _ in runs if rid not in valid_ids]
        logger.warning("excluding invalid runs per manifest: %s", dropped)
    return filtered


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", type=Path, required=True,
                    help="Folder containing run1/, run2/, ... subdirs.")
    ap.add_argument("--annotations", type=Path,
                    help="Gold annotation root (required unless --splits is used).")
    ap.add_argument("--splits", type=Path,
                    help="Legacy splits.json for TCGA-style layout.")
    ap.add_argument("--gold-suffix", default="_gold",
                    help="Suffix for gold annotation files (default: %(default)s).")
    ap.add_argument("--out", type=Path, default=Path("results/gpt_oss_multirun"),
                    help="Output directory (default: %(default)s).")
    ap.add_argument("--n-boot", type=int, default=2000,
                    help="Bootstrap replicates (default: %(default)s).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if args.annotations is None and args.splits is None:
        ap.error("either --annotations or --splits must be provided")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.out.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(args.runs_root)
    logger.info("discovered %d runs under %s", len(runs), args.runs_root)
    runs = _filter_valid_runs(runs, args.runs_root / "_manifest.yaml")
    if not runs:
        logger.error("no valid runs; aborting")
        return

    gold_root = args.annotations or args.splits.parent
    logger.info("building correctness table (this may take a while)")
    df = build_correctness_table(
        runs, gold_root=gold_root,
        gold_suffix=args.gold_suffix,
    )
    by_run_path = args.out / "by_run.csv"
    df.to_csv(by_run_path, index=False)
    logger.info("wrote %s (%d rows)", by_run_path, len(df))

    # Per-field CI (case + run + total)
    logger.info("computing per-field CIs")
    per_field = per_field_ci(df, n_boot=args.n_boot, random_state=args.seed)
    per_field.to_csv(args.out / "per_field_accuracy.csv", index=False)

    # Per organ
    per_org = per_organ_ci(df, n_boot=args.n_boot, random_state=args.seed)
    per_org.to_csv(args.out / "per_organ.csv", index=False)

    # Per field-type: derive the mapping from the first-seen organ per field.
    # Use the iaa classifier for the field types.
    first_organ_by_field = (
        df.dropna(subset=["organ"]).groupby("field")["organ"].first().to_dict()
    )
    field_to_type = {
        f: classify_field(f, first_organ_by_field.get(f))
        for f in df["field"].unique()
    }
    per_ft = per_fieldtype_ci(df, field_to_type,
                              n_boot=args.n_boot, random_state=args.seed)
    per_ft.to_csv(args.out / "per_fieldtype.csv", index=False)

    # Run consistency diagnostics
    cons = run_consistency(df)
    cons.to_csv(args.out / "run_consistency.csv", index=False)

    # Majority-vote ensemble — write predictions, then score them.
    logger.info("building majority-vote ensemble")
    ensemble_dir = args.out / "ensemble_predictions"
    majority_vote_ensemble(runs, ensemble_dir)
    df_ens = build_correctness_table(
        [("ensemble", ensemble_dir)], gold_root=gold_root,
        gold_suffix=args.gold_suffix,
    )
    df_ens.to_csv(args.out / "ensemble_by_case.csv", index=False)

    delta = ensemble_vs_single(df, df_ens,
                               n_boot=args.n_boot, random_state=args.seed)
    delta.to_csv(args.out / "ensemble_vs_single.csv", index=False)

    logger.info("done. outputs in %s", args.out)


if __name__ == "__main__":
    main()
