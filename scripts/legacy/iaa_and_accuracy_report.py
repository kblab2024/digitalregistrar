#!/usr/bin/env python3
"""Join Part A (IAA) and Part C (multi-run accuracy) outputs into a single
forest-plot CSV that puts human interobserver agreement and model accuracy
on the same axes.

Output (long form), one row per (organ × field × source):
    organ, field, field_type, section, source, point, ci_lo, ci_hi, n, note

Where `source` is one of:
    nhc_vs_kpc         — human IAA (primary estimand)
    nhc_vs_gold        — human vs adjudicated consensus
    kpc_vs_gold        — human vs adjudicated consensus
    gpt_oss_mean_run   — per-run accuracy mean with total (two-source) CI
    gpt_oss_ensemble   — majority-vote ensemble accuracy
    <baseline>         — optional additional methods loaded via --baseline

For binary and nominal fields the IAA entry is Cohen's κ; for ordinal it
is quadratic-weighted κ; for continuous it is Lin's CCC; for nested lists
it is matched F1. Each is compared to the model's accuracy on the gold
set (a different metric, but the point of the overlay is visual: does
the model's accuracy CI overlap or exceed the human IAA CI?).

Usage:
    python scripts/iaa_and_accuracy_report.py \\
        --iaa results/iaa \\
        --multirun results/gpt_oss_multirun \\
        --out results/headline_forest.csv \\
        [--baseline clinicalbert_cls results/benchmarks/clinicalbert_cls_by_method.csv] ...
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from digital_registrar_research.benchmarks.eval.iaa import (
    ORDINAL_FIELDS,
    classify_field,
    classify_section,
)

logger = logging.getLogger("iaa_and_accuracy_report")

# Which IAA stat we treat as the "headline" for each field type.
PRIMARY_STAT_BY_TYPE = {
    "binary": "cohen_kappa",
    "nominal": "cohen_kappa",
    "ordinal": "cohen_kappa_quadratic",
    "continuous": "lins_ccc",
    "nested_list": "matched_f1",
}


def _load_iaa_pair(iaa_dir: Path, pair_label: str) -> pd.DataFrame:
    p = iaa_dir / f"pairwise_{pair_label}.csv"
    if not p.exists():
        logger.warning("missing IAA file %s — skipping source %s", p, pair_label)
        return pd.DataFrame()
    df = pd.read_csv(p)
    keep_stats = set(PRIMARY_STAT_BY_TYPE.values())
    df = df[df["stat_name"].isin(keep_stats)].copy()
    df = df.rename(columns={"estimate": "point"})
    df["source"] = pair_label
    return df[["organ", "section", "field", "field_type",
               "stat_name", "point", "ci_lo", "ci_hi", "source"]]


def _load_multirun(multirun_dir: Path) -> pd.DataFrame:
    p = multirun_dir / "per_field_accuracy.csv"
    if not p.exists():
        logger.warning("missing per_field_accuracy.csv under %s", multirun_dir)
        return pd.DataFrame()
    df = pd.read_csv(p)
    # gpt-oss mean per run with total (two-source) CI
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "organ": "ALL", "field": r["field"],
            "field_type": classify_field(r["field"]),
            "section": classify_section(r["field"]),
            "stat_name": "accuracy",
            "point": r["point_estimate"],
            "ci_lo": r["total_ci_lo"], "ci_hi": r["total_ci_hi"],
            "source": "gpt_oss_mean_run",
        })
    # Per-organ
    per_org_path = multirun_dir / "per_organ.csv"
    if per_org_path.exists():
        po = pd.read_csv(per_org_path)
        for _, r in po.iterrows():
            rows.append({
                "organ": r["organ"], "field": r["field"],
                "field_type": classify_field(r["field"]),
                "section": classify_section(r["field"]),
                "stat_name": "accuracy",
                "point": r["point_estimate"],
                "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                "source": "gpt_oss_mean_run",
            })
    return pd.DataFrame(rows)


def _load_ensemble(multirun_dir: Path) -> pd.DataFrame:
    p = multirun_dir / "ensemble_by_case.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    # Aggregate per-field + per-organ
    df = df.dropna(subset=["correct"])
    rows = []
    for field, sub in df.groupby("field"):
        point = float(sub["correct"].mean())
        # Simple Wilson-like CI via normal approx (keep this lightweight —
        # the main CI story already lives in multirun per_field_accuracy).
        n = len(sub)
        se = (point * (1 - point) / n) ** 0.5 if n else 0.0
        rows.append({
            "organ": "ALL", "field": field,
            "field_type": classify_field(field),
            "section": classify_section(field),
            "stat_name": "accuracy",
            "point": point,
            "ci_lo": max(0.0, point - 1.96 * se),
            "ci_hi": min(1.0, point + 1.96 * se),
            "source": "gpt_oss_ensemble",
        })
        for organ, osub in sub.groupby("organ"):
            p_ = float(osub["correct"].mean())
            n_ = len(osub)
            se_ = (p_ * (1 - p_) / n_) ** 0.5 if n_ else 0.0
            rows.append({
                "organ": organ, "field": field,
                "field_type": classify_field(field, organ),
                "section": classify_section(field),
                "stat_name": "accuracy",
                "point": p_,
                "ci_lo": max(0.0, p_ - 1.96 * se_),
                "ci_hi": min(1.0, p_ + 1.96 * se_),
                "source": "gpt_oss_ensemble",
            })
    return pd.DataFrame(rows)


def _load_baseline(label: str, csv_path: Path) -> pd.DataFrame:
    """Load a baseline's by_method.csv (output of aggregate_to_csv) and
    reduce to per-field accuracy with Wilson CI. Treats the baseline as
    a single run."""
    if not csv_path.exists():
        logger.warning("baseline %s: %s does not exist", label, csv_path)
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    attempted = df[df["attempted"]]
    rows = []
    for field, sub in attempted.groupby("field"):
        vals = pd.to_numeric(sub["correct"], errors="coerce").dropna()
        if vals.empty:
            continue
        p = float(vals.mean())
        n = len(vals)
        se = (p * (1 - p) / n) ** 0.5 if n else 0.0
        rows.append({
            "organ": "ALL", "field": field,
            "field_type": classify_field(field),
            "section": classify_section(field),
            "stat_name": "accuracy",
            "point": p,
            "ci_lo": max(0.0, p - 1.96 * se),
            "ci_hi": min(1.0, p + 1.96 * se),
            "source": label,
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iaa", type=Path, required=True,
                    help="Directory produced by eval_iaa.py.")
    ap.add_argument("--multirun", type=Path, required=True,
                    help="Directory produced by eval_gpt_oss_multirun.py.")
    ap.add_argument("--out", type=Path, default=Path("results/headline_forest.csv"),
                    help="Output CSV path (default: %(default)s).")
    ap.add_argument("--baseline", nargs=2, metavar=("LABEL", "CSV"),
                    action="append", default=[],
                    help="Additional baselines to overlay (repeatable). "
                         "CSV is the by_method.csv from aggregate_to_csv.")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    frames: list[pd.DataFrame] = []
    for pair in ("nhc_vs_kpc", "nhc_vs_gold", "kpc_vs_gold"):
        frames.append(_load_iaa_pair(args.iaa, pair))
    frames.append(_load_multirun(args.multirun))
    frames.append(_load_ensemble(args.multirun))
    for label, csv_path in args.baseline:
        frames.append(_load_baseline(label, Path(csv_path)))

    frames = [f for f in frames if not f.empty]
    if not frames:
        logger.error("no inputs loaded — nothing to write")
        return
    combined = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.out, index=False)
    logger.info("wrote %s (%d rows)", args.out, len(combined))


if __name__ == "__main__":
    main()
