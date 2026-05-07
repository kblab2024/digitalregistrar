"""IAA subcommand orchestrator.

Pairwise IAA across human annotators and gold, plus the headline
preann-effect analyses (Δκ, anchoring index, convergence, disagreement
reduction, edit distance).

Output tree under ``--out``:
    manifest.json
    pair_<a>_vs_<b>.csv          — pairwise IAA per ordered pair
    whole_report.csv             — case-exact-match + α per pair
    disagreement_resolution.csv  — gold-resolution dynamics (when 3 annotators present)
    preann/
        delta_kappa_per_field.csv
        delta_kappa_per_organ.csv
        convergence_to_preann.csv
        anchoring_index.csv
        disagreement_reduction.csv
        edit_distance.csv
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

import pandas as pd

from digital_registrar_research.benchmarks.eval.iaa import (
    disagreement_resolution, pairwise_iaa, whole_report_stats,
)

from .._common.args import (
    add_common_args, add_iaa_args, parse_cases, parse_organs,
)
from .._common.paths import from_args
from .._common.reporting import setup_logging, write_csv, write_manifest
from . import preann_effect
from ._discovery import discover_cases_dir_layout

logger = logging.getLogger("scripts.eval.iaa")


# --- Argparse registration --------------------------------------------------


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "iaa",
        help="Inter-annotator agreement + preann effect.",
        description=__doc__,
    )
    add_common_args(parser, subcommand="iaa")
    add_iaa_args(parser)
    parser.add_argument(
        "--pairs", nargs="*", default=None,
        help="Optional explicit pair list (a:b). Defaults to all pairs of "
             "the --annotators set that include 'gold' or are within the "
             "same human (with vs without preann).",
    )
    parser.set_defaults(_handler=_main)


# --- Main entry --------------------------------------------------------------


def _main(args: argparse.Namespace) -> int:
    setup_logging(args.verbose)
    from digital_registrar_research.benchmarks.eval.ci_gpu import pick_device
    requested_device = getattr(args, "device", "cpu")
    resolved_device = pick_device(requested_device)
    logger.info("device: requested=%s resolved=%s",
                requested_device, resolved_device)
    args.resolved_device = resolved_device

    paths = from_args(args.root, args.dataset)
    paths.assert_exists()
    organs = parse_organs(args)
    case_filter = parse_cases(args)

    args.out.mkdir(parents=True, exist_ok=True)

    # --- Build CaseEntry dict from the new dir layout ---------------------
    cases, n_per_organ = discover_cases_dir_layout(
        paths=paths, annotators=tuple(args.annotators),
        organs=tuple(organs), case_filter=case_filter,
    )
    logger.info("discovered %d cases across %d annotators",
                len(cases), len(args.annotators))
    if not cases:
        logger.error("no cases discovered. Check --root, --dataset, --annotators.")
        return 1

    # --- Pairwise IAA per requested pair ----------------------------------
    pair_specs = _expand_pairs(args.pairs, args.annotators)
    for ann_a, ann_b in pair_specs:
        logger.info("scoring IAA: %s vs %s", ann_a, ann_b)
        df = pairwise_iaa(cases, ann_a=ann_a, ann_b=ann_b,
                          n_boot=args.n_boot, random_state=args.seed,
                          device=resolved_device)
        write_csv(df, args.out / f"pair_{ann_a}_vs_{ann_b}.csv")

    # --- Whole-report headline -------------------------------------------
    wr_rows = []
    for ann_a, ann_b in pair_specs:
        wr = whole_report_stats(cases, ann_a=ann_a, ann_b=ann_b)
        wr_rows.append(wr)
    if wr_rows:
        write_csv(pd.concat(wr_rows, ignore_index=True),
                  args.out / "whole_report.csv")

    # --- Disagreement resolution (needs 3 annotators including gold) -----
    annotator_set = set(args.annotators)
    if "gold" in annotator_set and len(annotator_set) >= 3:
        # Pick the first two non-gold annotators that look like a paired
        # human comparison; default to nhc_with_preann vs kpc_with_preann.
        ann_a, ann_b = _default_human_pair(args.annotators)
        if ann_a and ann_b:
            res = disagreement_resolution(
                cases, ann_a=ann_a, ann_b=ann_b, gold="gold",
            )
            write_csv(res, args.out / "disagreement_resolution.csv")

    # --- Preann effect ----------------------------------------------------
    preann_dir = args.out / "preann"
    preann_dir.mkdir(parents=True, exist_ok=True)

    # Combine paired records across both annotators into a single DataFrame.
    all_records = []
    for annotator in ("nhc", "kpc"):
        recs = preann_effect.collect_paired_records(
            paths, annotator,
            preann_model=args.preann_model, organs=organs,
        )
        for r in recs:
            d = r.__dict__.copy()
            d["annotator"] = annotator
            all_records.append(d)
    paired_df = pd.DataFrame(all_records)

    if not paired_df.empty:
        # Δκ + convergence + anchoring per annotator
        for annotator in ("nhc", "kpc"):
            sub = paired_df[paired_df["annotator"] == annotator]
            if sub.empty:
                continue
            from digital_registrar_research.benchmarks.eval.preann import (
                PairedRecord,
            )
            recs = [PairedRecord(**{k: v for k, v in r.items() if k != "annotator"})
                    for r in sub.to_dict(orient="records")]
            dk = preann_effect.delta_kappa_table(
                recs, n_boot=args.n_boot, seed=args.seed,
                device=resolved_device,
            )
            if not dk.empty:
                dk.insert(0, "annotator", annotator)
                write_csv(
                    dk,
                    preann_dir / f"delta_kappa_per_field__{annotator}.csv",
                )
            conv = preann_effect.convergence_table(recs)
            if not conv.empty:
                conv.insert(0, "annotator", annotator)
                write_csv(
                    conv,
                    preann_dir / f"convergence_to_preann__{annotator}.csv",
                )
            ai = preann_effect.anchoring_index_table(recs)
            if not ai.empty:
                ai.insert(0, "annotator", annotator)
                write_csv(ai, preann_dir / f"anchoring_index__{annotator}.csv")
            ed = preann_effect.edit_distance_summary(recs)
            if not ed.empty:
                ed.insert(0, "annotator", annotator)
                write_csv(ed, preann_dir / f"edit_distance__{annotator}.csv")

        # Disagreement reduction (needs both annotators)
        dual = preann_effect.collect_dual_paired_records(
            paths, preann_model=args.preann_model, organs=organs,
        )
        dr = preann_effect.disagreement_reduction_table(
            dual, n_boot=args.n_boot, seed=args.seed,
            device=resolved_device,
        )
        if not dr.empty:
            write_csv(dr, preann_dir / "disagreement_reduction.csv")

    write_manifest(
        args.out, args, subcommand="iaa",
        n_cases_per_organ=n_per_organ,
        extra={
            "n_cases_total": len(cases),
            "annotators": list(args.annotators),
            "preann_model": args.preann_model,
            "pairs_scored": [f"{a}:{b}" for a, b in pair_specs],
        },
    )
    logger.info("done. outputs in %s", args.out)
    return 0


# --- Pair expansion ----------------------------------------------------------


def _expand_pairs(
    raw: list[str] | None, annotators: list[str],
) -> list[tuple[str, str]]:
    """Produce the ordered pairs to score.

    If ``--pairs`` is given, parse colon-separated pairs verbatim.
    Otherwise, default to: every (gold, X) pair for X in annotators
    other than gold, plus (with_preann, without_preann) within each
    annotator.
    """
    if raw:
        out = []
        for s in raw:
            if ":" not in s:
                raise SystemExit(f"--pairs entry malformed: {s!r}")
            a, b = s.split(":", 1)
            out.append((a, b))
        return out
    out: list[tuple[str, str]] = []
    annotators_set = set(annotators)
    if "gold" in annotators_set:
        for a in sorted(annotators_set - {"gold"}):
            out.append(("gold", a))
    # Within-annotator with vs without preann
    for human in ("nhc", "kpc"):
        w = f"{human}_with_preann"
        wo = f"{human}_without_preann"
        if w in annotators_set and wo in annotators_set:
            out.append((w, wo))
    # Cross-annotator at same mode
    for mode in ("with_preann", "without_preann"):
        a = f"nhc_{mode}"
        b = f"kpc_{mode}"
        if a in annotators_set and b in annotators_set:
            out.append((a, b))
    return out


def _default_human_pair(annotators: list[str]) -> tuple[str | None, str | None]:
    """Pick the canonical NHC vs KPC pair for disagreement resolution."""
    s = set(annotators)
    for mode in ("with_preann", "without_preann"):
        a, b = f"nhc_{mode}", f"kpc_{mode}"
        if a in s and b in s:
            return a, b
    return None, None


__all__ = ["register"]
