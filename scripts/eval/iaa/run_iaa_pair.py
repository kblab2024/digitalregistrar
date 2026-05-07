"""IAA pair-focused subcommand.

Comprehensive IAA report scoped to one or more requested annotator
pairs (e.g. ``gold:nhc_with_preann``,
``kpc_with_preann:nhc_with_preann``,
``kpc_with_preann:kpc_without_preann``). Unlike the broader ``iaa``
subcommand, this one always emits a single overall Cohen's κ headline
per pair plus per-section / per-organ roll-ups, confusion matrices,
and a markdown summary.

Output tree under ``--out``:
    manifest.json
    pair_<a>_vs_<b>/
        headline.csv
        per_field_kappa.csv
        per_section.csv
        per_organ.csv
        summary.md
        confusion/<field>.csv ...
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from digital_registrar_research.benchmarks.eval.iaa_headline import (
    CONFUSION_COLUMNS,
    HEADLINE_COLUMNS,
    PER_FIELD_COLUMNS,
    PER_ORGAN_COLUMNS,
    PER_SECTION_COLUMNS,
    compute_headline,
    confusion_matrix_for_field,
    disagreement_count_per_field,
    headline_rows_to_df,
    per_field_kappas,
    per_organ_rollup,
    per_section_rollup,
)

from .._common.args import (
    KNOWN_ANNOTATORS, add_common_args, parse_cases, parse_organs,
)
from .._common.paths import from_args
from .._common.reporting import setup_logging, write_csv, write_manifest
from ._discovery import discover_cases_dir_layout
from ._summary_md import render_summary

logger = logging.getLogger("scripts.eval.iaa_pair")


# --- Argparse registration --------------------------------------------------


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "iaa_pair",
        help="Pair-focused IAA report (overall κ + roll-ups + confusion).",
        description=__doc__,
    )
    add_common_args(parser, subcommand="iaa_pair")
    parser.add_argument(
        "--pair", action="append", dest="pairs", default=None, metavar="A:B",
        help="Pair to score, formatted A:B. Repeat for multiple pairs. "
             "Both sides must be valid annotators "
             f"({', '.join(KNOWN_ANNOTATORS)}).",
    )
    parser.add_argument(
        "--n-confusion", type=int, default=20,
        help="Top-N categorical fields by disagreement count to write "
             "as confusion matrices (default: %(default)s).",
    )
    parser.add_argument(
        "--top-fields", type=int, default=10,
        help="Top-K most-disagreed fields shown in the markdown summary "
             "(default: %(default)s).",
    )
    parser.set_defaults(_handler=_main)


# --- Pair parsing -----------------------------------------------------------


def _parse_pairs(raw: list[str] | None) -> list[tuple[str, str]]:
    if not raw:
        raise SystemExit(
            "iaa_pair: at least one --pair A:B argument is required."
        )
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for s in raw:
        if ":" not in s:
            raise SystemExit(f"--pair entry malformed (need 'A:B'): {s!r}")
        a, b = s.split(":", 1)
        a, b = a.strip(), b.strip()
        if a not in KNOWN_ANNOTATORS or b not in KNOWN_ANNOTATORS:
            raise SystemExit(
                f"--pair {s!r} contains unknown annotator. "
                f"Valid options: {', '.join(KNOWN_ANNOTATORS)}"
            )
        if a == b:
            raise SystemExit(
                f"--pair {s!r} is degenerate (A == B); refusing."
            )
        if (a, b) in seen:
            logger.warning("dropping duplicate --pair %s", s)
            continue
        if (b, a) in seen:
            logger.info(
                "both orientations of %s requested; keeping both "
                "(confusion matrices are asymmetric).", s,
            )
        seen.add((a, b))
        out.append((a, b))
    return out


def _annotators_for(pairs: list[tuple[str, str]]) -> list[str]:
    return sorted({s for ab in pairs for s in ab})


# --- Main entry --------------------------------------------------------------


def _main(args: argparse.Namespace) -> int:
    setup_logging(args.verbose)
    from digital_registrar_research.benchmarks.eval.ci_gpu import pick_device
    requested_device = getattr(args, "device", "cpu")
    resolved_device = pick_device(requested_device)
    logger.info("device: requested=%s resolved=%s",
                requested_device, resolved_device)

    pair_specs = _parse_pairs(args.pairs)
    annotators = _annotators_for(pair_specs)

    paths = from_args(args.root, args.dataset)
    paths.assert_exists()
    organs = parse_organs(args)
    case_filter = parse_cases(args)

    args.out.mkdir(parents=True, exist_ok=True)

    cases, n_per_organ = discover_cases_dir_layout(
        paths=paths, annotators=tuple(annotators),
        organs=tuple(organs), case_filter=case_filter,
    )
    logger.info(
        "discovered %d cases across %d annotators (%s)",
        len(cases), len(annotators), ", ".join(annotators),
    )
    if not cases:
        logger.error("no cases discovered. Check --root, --dataset, --organs.")
        return 1

    pairs_with_outputs: list[dict] = []
    for ann_a, ann_b in pair_specs:
        pair_dir = args.out / f"pair_{ann_a}_vs_{ann_b}"
        pair_dir.mkdir(parents=True, exist_ok=True)
        n_overlap = sum(
            1 for e in cases.values()
            if ann_a in e.annotations and ann_b in e.annotations
        )
        logger.info("scoring iaa_pair: %s vs %s (%d overlapping cases)",
                    ann_a, ann_b, n_overlap)

        if n_overlap == 0:
            _write_empty_pair(pair_dir, ann_a, ann_b)
            pairs_with_outputs.append({
                "pair": f"{ann_a}_vs_{ann_b}", "n_cases": 0,
                "outputs": [], "reason": "no overlapping cases",
            })
            continue

        # Per-field κ table — drives mean / n-weighted-mean / top-K /
        # per-organ helpers.
        pf_df = per_field_kappas(
            cases, ann_a=ann_a, ann_b=ann_b,
            n_boot=args.n_boot, random_state=args.seed,
            device=resolved_device,
        )
        write_csv(pf_df, pair_dir / "per_field_kappa.csv")

        # Headline κ stats.
        headline_rows = compute_headline(
            cases, ann_a=ann_a, ann_b=ann_b,
            n_boot=args.n_boot, random_state=args.seed,
            device=resolved_device,
            per_field_df=pf_df,
        )
        headline_df = headline_rows_to_df(
            headline_rows, pair_label=f"{ann_a}_vs_{ann_b}",
        )
        write_csv(headline_df, pair_dir / "headline.csv")

        # Roll-ups.
        section_df = per_section_rollup(pf_df, cases,
                                        ann_a=ann_a, ann_b=ann_b)
        write_csv(section_df, pair_dir / "per_section.csv")

        organ_df = per_organ_rollup(
            cases, ann_a=ann_a, ann_b=ann_b,
            n_boot=args.n_boot, random_state=args.seed,
            device=resolved_device,
            per_field_df=pf_df,
        )
        write_csv(organ_df, pair_dir / "per_organ.csv")

        # Confusion matrices: top-N categorical fields by disagreement count.
        confusion_dir = pair_dir / "confusion"
        confusion_files = _write_confusion_matrices(
            cases, ann_a=ann_a, ann_b=ann_b,
            out_dir=confusion_dir, top_n=args.n_confusion,
        )

        # Markdown summary.
        all_rows = pf_df[pf_df["organ"] == "ALL"] if not pf_df.empty else pf_df
        summary_md = render_summary(
            ann_a=ann_a, ann_b=ann_b,
            headline_rows=headline_rows,
            per_field_df=all_rows,
            per_section_df=section_df,
            per_organ_df=organ_df,
            confusion_files=[
                (field, f"confusion/{_safe_name(field)}.csv")
                for field in confusion_files
            ],
            n_cases=n_overlap,
            n_boot=args.n_boot,
            top_k=args.top_fields,
        )
        (pair_dir / "summary.md").write_text(summary_md, encoding="utf-8")
        logger.info("wrote %s", pair_dir / "summary.md")

        pairs_with_outputs.append({
            "pair": f"{ann_a}_vs_{ann_b}", "n_cases": n_overlap,
            "n_confusion": len(confusion_files),
        })

    write_manifest(
        args.out, args, subcommand="iaa_pair",
        n_cases_per_organ=n_per_organ,
        extra={
            "n_cases_total": len(cases),
            "annotators": annotators,
            "pairs_scored": [f"{a}:{b}" for a, b in pair_specs],
            "pairs_summary": pairs_with_outputs,
        },
    )
    logger.info("done. outputs in %s", args.out)
    return 0


# --- Helpers ----------------------------------------------------------------


def _write_empty_pair(pair_dir: Path, ann_a: str, ann_b: str) -> None:
    """Create header-only CSVs + a stub markdown for a pair with no cases."""
    pair_label = f"{ann_a}_vs_{ann_b}"
    pd.DataFrame(columns=list(HEADLINE_COLUMNS)).to_csv(
        pair_dir / "headline.csv", index=False)
    pd.DataFrame(columns=list(PER_FIELD_COLUMNS)).to_csv(
        pair_dir / "per_field_kappa.csv", index=False)
    pd.DataFrame(columns=list(PER_SECTION_COLUMNS)).to_csv(
        pair_dir / "per_section.csv", index=False)
    pd.DataFrame(columns=list(PER_ORGAN_COLUMNS)).to_csv(
        pair_dir / "per_organ.csv", index=False)
    (pair_dir / "summary.md").write_text(
        f"# IAA: {ann_a} vs {ann_b}\n\n"
        f"_No overlapping cases between {ann_a} and {ann_b}._\n",
        encoding="utf-8",
    )
    logger.info("pair %s has no overlapping cases — wrote empty outputs",
                pair_label)


def _safe_name(field: str) -> str:
    """Filesystem-safe filename for a field name (alnum + underscore)."""
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in field)


def _write_confusion_matrices(
    cases, *, ann_a: str, ann_b: str, out_dir: Path, top_n: int,
) -> list[str]:
    """Write top-N categorical-field confusion matrices.

    Returns the list of fields (in disagreement-rank order) that were
    actually written. Fields with zero disagreements are skipped.
    """
    counts = disagreement_count_per_field(cases, ann_a=ann_a, ann_b=ann_b)
    if not counts:
        return []
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    ranked = [f for f, n in ranked if n > 0][:top_n]
    if ranked:
        out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for field in ranked:
        df = confusion_matrix_for_field(
            cases, field, ann_a=ann_a, ann_b=ann_b,
        )
        if df.empty:
            continue
        df = df.reindex(columns=list(CONFUSION_COLUMNS))
        write_csv(df, out_dir / f"{_safe_name(field)}.csv")
        written.append(field)
    return written


__all__ = ["register"]
