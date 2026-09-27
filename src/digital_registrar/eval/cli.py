"""``registrar-eval`` — prediction-vs-annotation evaluation CLI.

Scores a folder of ``registrar-pipeline`` outputs against a folder of
gold annotations. Files are paired by case id: one trailing ``_output``
or ``_annotation`` is stripped from the file stem, so
``pred/tcga4_1_output.json`` pairs with ``gold/tcga4_1_annotation.json``
(or ``gold/tcga4_1.json``). Both folders are searched recursively.

Subcommands:

    metrics        — per-field accuracy (and F1 for nested lists) with 95% CIs.
                     Reads --pred / --gold; writes atomic.csv and summary.csv.

    compare        — paired comparison of two runs against the same gold.
                     Reads --pred-a / --pred-b / --gold; writes atomic.csv,
                     summary.csv and compare.csv (paired-bootstrap Δ + McNemar).

    completeness   — missingness per field: parse errors, missing fields,
                     refusal calibration and out-of-vocab values.
                     Reads --pred / --gold; writes completeness.csv,
                     refusal_calibration.csv and out_of_vocab.csv.

Scoring scope (--scope):

    cascade (default) — gated scoring: eligibility (cancer_excision_report),
                        then organ (cancer_category), then every field of the
                        gold organ's schema, plus margins / biomarkers /
                        lymph nodes. Stage-C fields are scored only for cases
                        that pass both gates.
    fair              — flat scoring of the fixed FAIR_SCOPE field list.

Paper-specific analyses (IAA, preann effect, multirun statistics,
cross-dataset) are not part of this CLI; see attic/ in the repository.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger("digital_registrar.eval.cli")


def _label(explicit: str | None, folder: Path) -> str:
    return explicit or folder.resolve().name or "pred"


def _resolve_scope(name: str):
    if name == "fair":
        from digital_registrar.eval.scope import FAIR_SCOPE
        return list(FAIR_SCOPE)
    return None


def _load(pred: Path, gold: Path):
    """Pair ``pred`` with ``gold``; print stats. Returns None on a fatal error."""
    from digital_registrar.eval.folders import load_pairs

    try:
        pairs, stats = load_pairs(pred, gold)
    except (FileNotFoundError, ValueError) as e:
        print(f"registrar-eval: error: {e}", file=sys.stderr)
        return None
    if not pairs:
        print(f"registrar-eval: error: no gold annotations (*.json) found under {gold}",
              file=sys.stderr)
        return None
    print(f"{pred}: {stats['n_matched']}/{stats['n_gold']} gold cases matched "
          f"({stats['n_missing_pred']} missing, {stats['n_unreadable_pred']} unreadable, "
          f"{stats['n_extra_pred']} predictions without gold)")
    return pairs


def _score(pred: Path, gold: Path, label: str, scope_name: str) -> pd.DataFrame | None:
    from digital_registrar.eval.metrics import score_pairs

    pairs = _load(pred, gold)
    if pairs is None:
        return None
    return score_pairs(pairs, method=label, scope=_resolve_scope(scope_name))


def _write(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    print(f"Wrote {len(df)} rows to {path}")


def _cmd_metrics(args: argparse.Namespace) -> int:
    from digital_registrar.eval.metrics import summarize_scores

    atomic = _score(args.pred, args.gold, _label(args.label, args.pred), args.scope)
    if atomic is None:
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    summary = summarize_scores(atomic)
    _write(atomic, args.out / "atomic.csv")
    _write(summary, args.out / "summary.csv")
    print(summary.to_string(index=False))
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    from digital_registrar.eval.metrics import summarize_scores
    from digital_registrar.eval.pairwise_compare import compare_runs

    label_a = _label(args.label_a, args.pred_a)
    label_b = _label(args.label_b, args.pred_b)
    if label_a == label_b:
        label_a, label_b = "a", "b"
    atomic_a = _score(args.pred_a, args.gold, label_a, args.scope)
    atomic_b = _score(args.pred_b, args.gold, label_b, args.scope)
    if atomic_a is None or atomic_b is None:
        return 1
    atomic = pd.concat([atomic_a, atomic_b], ignore_index=True)
    args.out.mkdir(parents=True, exist_ok=True)
    comparison = compare_runs(atomic, label_a, label_b, n_boot=args.n_boot)
    _write(atomic, args.out / "atomic.csv")
    _write(summarize_scores(atomic), args.out / "summary.csv")
    _write(comparison, args.out / "compare.csv")
    print(comparison[["stage", "field", "metric", "n_paired", "score_a", "score_b",
                      "delta", "delta_ci_lo", "delta_ci_hi",
                      "mcnemar_p_value"]].to_string(index=False))
    return 0


def _cmd_completeness(args: argparse.Namespace) -> int:
    from digital_registrar.eval.completeness import (
        aggregate_missingness,
        completeness_atomic,
        out_of_vocab_table,
        refusal_calibration,
    )

    pairs = _load(args.pred, args.gold)
    if pairs is None:
        return 1
    label = _label(args.label, args.pred)
    atomic = completeness_atomic(pairs, method=label, scope=_resolve_scope(args.scope))
    by = ("method", "field", "organ")
    report = aggregate_missingness(atomic, by=by)
    args.out.mkdir(parents=True, exist_ok=True)
    _write(report, args.out / "completeness.csv")
    _write(refusal_calibration(atomic, by=by), args.out / "refusal_calibration.csv")
    _write(out_of_vocab_table(pairs, method=label), args.out / "out_of_vocab.csv")
    print(report[["field", "organ", "n_total", "n_parse_error", "n_field_missing",
                  "attempted_rate", "attempted_accuracy"]].to_string(index=False))
    return 0


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gold", type=Path, required=True,
                        help="Folder of gold annotations: <case>_annotation.json "
                             "or <case>.json (searched recursively).")
    parser.add_argument("--out", type=Path, default=Path("eval_out"),
                        help="Output directory (created if missing; default: eval_out).")
    parser.add_argument("--scope", choices=["cascade", "fair"], default="cascade",
                        help="cascade (default): gated scoring of every field in the "
                             "gold organ's schema. fair: flat scoring of the fixed "
                             "FAIR_SCOPE field list.")


_PRED_HELP = ("Folder of predictions, e.g. a registrar-pipeline --output folder of "
              "<case>_output.json files (searched recursively).")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="registrar-eval",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    p_metrics = sub.add_parser(
        "metrics", help="per-field accuracy / F1 with 95%% CIs",
        description="Score one run. Writes atomic.csv (one row per case x field) "
                    "and summary.csv (per-field accuracy, coverage, 95% CI).")
    p_metrics.add_argument("--pred", type=Path, required=True, help=_PRED_HELP)
    p_metrics.add_argument("--label", help="Run name for the method column "
                                           "(default: the --pred folder name).")
    _add_common_args(p_metrics)
    p_metrics.set_defaults(_handler=_cmd_metrics)

    p_compare = sub.add_parser(
        "compare", help="paired comparison between two runs",
        description="Score two runs against the same gold and compare them on the "
                    "cases both runs scored. Writes atomic.csv, summary.csv and "
                    "compare.csv (delta = A - B, paired-bootstrap 95% CI, McNemar).")
    p_compare.add_argument("--pred-a", type=Path, required=True, help=_PRED_HELP)
    p_compare.add_argument("--pred-b", type=Path, required=True, help=_PRED_HELP)
    p_compare.add_argument("--label-a", help="Name for run A (default: folder name).")
    p_compare.add_argument("--label-b", help="Name for run B (default: folder name).")
    p_compare.add_argument("--n-boot", type=int, default=2000,
                           help="Bootstrap resamples for the delta CI (default: 2000).")
    _add_common_args(p_compare)
    p_compare.set_defaults(_handler=_cmd_compare)

    p_complete = sub.add_parser(
        "completeness", help="missingness / refusal / out-of-vocab analysis",
        description="Per-field missingness for one run. Writes completeness.csv, "
                    "refusal_calibration.csv and out_of_vocab.csv.")
    p_complete.add_argument("--pred", type=Path, required=True, help=_PRED_HELP)
    p_complete.add_argument("--label", help="Run name for the method column "
                                            "(default: the --pred folder name).")
    _add_common_args(p_complete)
    p_complete.set_defaults(_handler=_cmd_completeness)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "_handler", None)
    if handler is None:
        parser.error(f"subcommand {args.subcommand!r} did not register a handler")
    return int(handler(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
