"""Joint headline forest report.

Combines:
    - per-field model accuracy from cascade
      ``chapter3_field_extraction/per_field_by_organ.csv`` (and
      ``per_field_overall.csv`` for the cross-organ row).
    - human-vs-human and human-vs-gold κ from ``iaa/pair_*.csv``
    - paired Δκ from ``iaa/preann/delta_kappa_per_field__*.csv``

into a single long-form CSV ``headline_forest.csv`` with columns:
    organ, field, source, point, ci_lo, ci_hi, n
where ``source`` ∈ {model_attempted_acc, model_effective_acc,
nhc_vs_kpc_kappa, gold_vs_nhc_kappa, gold_vs_kpc_kappa,
delta_kappa_with_preann_nhc, delta_kappa_with_preann_kpc, ...}.

Downstream plotting code overlays IAA bands on model-accuracy bars on
the same forest plot to visualise where the model lands relative to
human inter-rater agreement.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from .._common.reporting import setup_logging, write_csv, write_manifest

logger = logging.getLogger("scripts.eval.joint")


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "headline",
        help="Joint forest-plot CSV combining cascade chapter3 + IAA outputs.",
        description=__doc__,
    )
    parser.add_argument(
        "--cascade-out", type=Path, required=True,
        help="Path to cascade output directory (the parent of "
             "chapter3_field_extraction/).",
    )
    parser.add_argument(
        "--iaa-out", type=Path, required=True,
        help="Path to iaa output directory.",
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("workspace") / "results" / "eval" / "headline",
        help="Output directory for headline_forest.csv + manifest "
             "(default: %(default)s).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
    )
    parser.set_defaults(_handler=_main)


def _main(args: argparse.Namespace) -> int:
    setup_logging(args.verbose)
    args.out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    rows.extend(_load_model_accuracy(args.cascade_out))
    rows.extend(_load_iaa_pairs(args.iaa_out))
    rows.extend(_load_preann_delta(args.iaa_out))

    df = pd.DataFrame(rows)
    if df.empty:
        logger.error("no rows assembled — check input directories.")
        return 1
    write_csv(df, args.out / "headline_forest.csv")
    write_manifest(args.out, args, subcommand="headline",
                   extra={"n_rows": len(df),
                          "n_unique_sources": int(df["source"].nunique())})
    logger.info("done. outputs in %s", args.out)
    return 0


def _load_model_accuracy(cascade_out: Path) -> list[dict]:
    """Pull attempted_accuracy and effective_accuracy per (organ, field)
    from cascade chapter3 outputs with Wilson CI.

    Reads ``chapter3_field_extraction/per_field_by_organ.csv`` for the
    per-organ rows and ``per_field_overall.csv`` for the cross-organ
    rows (stamped as ``organ="ALL"``). Both CSVs carry the
    ``effective_accuracy`` + ``effective_acc_wilson_lo/hi`` columns
    added by the cascade redesign.
    """
    rows: list[dict] = []
    ch3_dir = cascade_out / "chapter3_field_extraction"
    by_organ_path = ch3_dir / "per_field_by_organ.csv"
    overall_path = ch3_dir / "per_field_overall.csv"
    if by_organ_path.is_file():
        df = pd.read_csv(by_organ_path)
        for _, r in df.iterrows():
            rows.append({
                "organ": r["organ"], "field": r["field"],
                "source": "model_attempted_acc",
                "point": r.get("attempted_accuracy"),
                "ci_lo": r.get("attempted_acc_boot_lo")
                          or r.get("attempted_acc_wilson_lo"),
                "ci_hi": r.get("attempted_acc_boot_hi")
                          or r.get("attempted_acc_wilson_hi"),
                "n": r.get("n_attempted"),
            })
            rows.append({
                "organ": r["organ"], "field": r["field"],
                "source": "model_effective_acc",
                "point": r.get("effective_accuracy"),
                "ci_lo": r.get("effective_acc_wilson_lo"),
                "ci_hi": r.get("effective_acc_wilson_hi"),
                "n": r.get("n_total"),
            })
    if overall_path.is_file():
        df = pd.read_csv(overall_path)
        for _, r in df.iterrows():
            rows.append({
                "organ": "ALL", "field": r["field"],
                "source": "model_attempted_acc",
                "point": r.get("attempted_accuracy"),
                "ci_lo": r.get("attempted_acc_boot_lo")
                          or r.get("attempted_acc_wilson_lo"),
                "ci_hi": r.get("attempted_acc_boot_hi")
                          or r.get("attempted_acc_wilson_hi"),
                "n": r.get("n_attempted"),
            })
            rows.append({
                "organ": "ALL", "field": r["field"],
                "source": "model_effective_acc",
                "point": r.get("effective_accuracy"),
                "ci_lo": r.get("effective_acc_wilson_lo"),
                "ci_hi": r.get("effective_acc_wilson_hi"),
                "n": r.get("n_total"),
            })
    return rows


def _load_iaa_pairs(iaa_out: Path) -> list[dict]:
    """For every ``pair_*.csv`` file, pull cohen_kappa stats per
    (organ, field) into the forest schema."""
    rows: list[dict] = []
    for path in sorted(iaa_out.glob("pair_*.csv")):
        # Extract source label from filename: pair_<a>_vs_<b>.csv
        label = path.stem.removeprefix("pair_")
        try:
            df = pd.read_csv(path)
        except Exception as e:
            logger.warning("failed to read %s: %s", path, e)
            continue
        if "stat_name" not in df.columns:
            continue
        sub = df[df["stat_name"] == "cohen_kappa_unweighted"]
        for _, r in sub.iterrows():
            rows.append({
                "organ": r.get("organ"), "field": r.get("field"),
                "source": f"iaa_kappa_{label}",
                "point": r.get("estimate"),
                "ci_lo": r.get("ci_lo"), "ci_hi": r.get("ci_hi"),
                "n": r.get("n"),
            })
    return rows


def _load_preann_delta(iaa_out: Path) -> list[dict]:
    """Pull Δκ + paired CI from iaa/preann/delta_kappa_per_field__*.csv."""
    rows: list[dict] = []
    preann_dir = iaa_out / "preann"
    if not preann_dir.is_dir():
        return rows
    for path in sorted(preann_dir.glob("delta_kappa_per_field__*.csv")):
        ann = path.stem.split("__")[-1]
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        for _, r in df.iterrows():
            rows.append({
                "organ": r.get("organ"), "field": r.get("field"),
                "source": f"delta_kappa_with_preann_{ann}",
                "point": r.get("delta"),
                "ci_lo": r.get("delta_ci_lo"),
                "ci_hi": r.get("delta_ci_hi"),
                "n": r.get("n_paired_cases"),
            })
    return rows


__all__ = ["register"]
