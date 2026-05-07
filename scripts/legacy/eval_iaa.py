#!/usr/bin/env python3
"""Compute interobserver-agreement (IAA) statistics for the CMUH dataset.

Scans an annotations directory for `_nhc` / `_kpc` / `_gold` file trios,
then writes the Part A output CSVs described in the statistical plan:

    <out>/pairwise_nhc_vs_kpc.csv       — primary IAA matrix (long form)
    <out>/pairwise_nhc_vs_gold.csv      — descriptive: annotator A vs consensus
    <out>/pairwise_kpc_vs_gold.csv      — descriptive: annotator B vs consensus
    <out>/whole_report.csv              — case-exact-match + Krippendorff α
    <out>/disagreement_resolution.csv   — how _gold resolves _nhc ≠ _kpc

Usage:
    python scripts/eval_iaa.py \\
        --annotations data/cmuh_annotation_<date> \\
        --out results/iaa

Works on the TCGA sample too, provided the files are renamed with the
`_nhc` / `_kpc` / `_gold` suffixes expected by discover_cases.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from digital_registrar_research.benchmarks.eval.iaa import (
    disagreement_resolution,
    discover_cases,
    pairwise_iaa,
    whole_report_stats,
)

logger = logging.getLogger("eval_iaa")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotations", type=Path, required=True,
                    help="Root folder containing <case>_<annotator>.json files.")
    ap.add_argument("--out", type=Path, default=Path("results/iaa"),
                    help="Output directory (default: %(default)s).")
    ap.add_argument("--pairs", nargs="*",
                    default=["_nhc:_kpc", "_nhc:_gold", "_kpc:_gold"],
                    help="Colon-separated pairs to score (default: all three).")
    ap.add_argument("--n-boot", type=int, default=2000,
                    help="Bootstrap replicates (default: %(default)s).")
    ap.add_argument("--seed", type=int, default=0,
                    help="Bootstrap RNG seed (default: %(default)s).")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.out.mkdir(parents=True, exist_ok=True)

    annotators = {"_nhc", "_kpc", "_gold"}
    cases = discover_cases(args.annotations, tuple(annotators))
    logger.info("discovered %d cases in %s", len(cases), args.annotations)
    if not cases:
        logger.error("no cases found; check --annotations path and file suffixes")
        return

    # Per-pair IAA tables
    for spec in args.pairs:
        ann_a, ann_b = spec.split(":")
        if ann_a not in annotators or ann_b not in annotators:
            logger.warning("skipping unknown pair %s", spec)
            continue
        logger.info("scoring IAA: %s vs %s", ann_a, ann_b)
        df = pairwise_iaa(cases, ann_a=ann_a, ann_b=ann_b,
                          n_boot=args.n_boot, random_state=args.seed)
        out_path = args.out / f"pairwise_{ann_a.lstrip('_')}_vs_{ann_b.lstrip('_')}.csv"
        df.to_csv(out_path, index=False)
        logger.info("wrote %s (%d rows)", out_path, len(df))

    # Whole-report headline for each pair
    wr_rows = []
    for spec in args.pairs:
        ann_a, ann_b = spec.split(":")
        if ann_a not in annotators or ann_b not in annotators:
            continue
        wr = whole_report_stats(cases, ann_a=ann_a, ann_b=ann_b)
        wr_rows.append(wr)
    if wr_rows:
        import pandas as pd
        combined = pd.concat(wr_rows, ignore_index=True)
        combined_path = args.out / "whole_report.csv"
        combined.to_csv(combined_path, index=False)
        logger.info("wrote %s (%d rows)", combined_path, len(combined))

    # Disagreement resolution: requires all three annotators
    if "_gold" in {a for pair in args.pairs for a in pair.split(":")}:
        logger.info("computing disagreement resolution")
        res = disagreement_resolution(cases)
        res_path = args.out / "disagreement_resolution.csv"
        res.to_csv(res_path, index=False)
        logger.info("wrote %s (%d rows)", res_path, len(res))


if __name__ == "__main__":
    main()
