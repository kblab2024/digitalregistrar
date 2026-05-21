"""``registrar-eval`` — prediction-vs-annotation evaluation CLI.

Three subcommands cover the regular workflow:

    metrics        — per-field exact-match accuracy + bootstrap CIs.
                     Reads {pred_dir, gold_dir}; emits a parquet
                     atomic table and a summary table.

    compare        — paired comparison between two runs on the same gold.
                     Reads {pred_dir_a, pred_dir_b, gold_dir}; emits a
                     side-by-side delta table and paired-bootstrap CIs.

    completeness   — missingness analysis per field + method-pair deltas.
                     Reads {pred_dir, gold_dir} (or multiple pred_dirs);
                     emits an out-of-vocab / refusal calibration report.

Paper-specific subcommands (IAA, preann, multirun statistical analysis,
cross-dataset, cascade-style joint reports) live in the ``drr-attic``
package — install it separately and use ``registrar-benchmark`` /
``scripts/eval/`` (under ``attic/``) to access them.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger("digital_registrar.eval.cli")


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gold", type=Path, required=True,
                        help="Directory of {sample_id}_annotation.json files.")
    parser.add_argument("--out", type=Path, default=Path("eval_out"),
                        help="Output directory (created if missing).")
    parser.add_argument("--scope", choices=["all", "attempted", "applicable"],
                        default="all",
                        help="Field-scope policy for denominators.")


def _cmd_metrics(args: argparse.Namespace) -> int:
    from digital_registrar.eval.metrics import aggregate_cases_to_df, summary_table

    args.out.mkdir(parents=True, exist_ok=True)
    df = aggregate_cases_to_df(
        pred_root=args.pred, gold_root=args.gold, scope=args.scope,
    )
    df.to_parquet(args.out / "atomic.parquet")
    summary = summary_table(df)
    summary.to_csv(args.out / "summary.csv", index=False)
    print(f"Wrote {len(df)} rows to {args.out/'atomic.parquet'}")
    print(summary.to_string(index=False))
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    from digital_registrar.eval.pairwise_compare import main as pairwise_main
    return pairwise_main([
        "--pred-a", str(args.pred_a),
        "--pred-b", str(args.pred_b),
        "--gold", str(args.gold),
        "--out", str(args.out),
    ])


def _cmd_completeness(args: argparse.Namespace) -> int:
    from digital_registrar.eval.completeness import aggregate_missingness

    args.out.mkdir(parents=True, exist_ok=True)
    report = aggregate_missingness(pred_root=args.pred, gold_root=args.gold)
    report.to_csv(args.out / "completeness.csv", index=False)
    print(report.to_string(index=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="registrar-eval",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    p_metrics = sub.add_parser("metrics", help="per-field accuracy + bootstrap CIs")
    p_metrics.add_argument("--pred", type=Path, required=True,
                           help="Directory of {sample_id}_output.json predictions.")
    _add_common_args(p_metrics)
    p_metrics.set_defaults(_handler=_cmd_metrics)

    p_compare = sub.add_parser("compare", help="paired comparison between two runs")
    p_compare.add_argument("--pred-a", type=Path, required=True)
    p_compare.add_argument("--pred-b", type=Path, required=True)
    _add_common_args(p_compare)
    p_compare.set_defaults(_handler=_cmd_compare)

    p_complete = sub.add_parser("completeness", help="missingness / out-of-vocab analysis")
    p_complete.add_argument("--pred", type=Path, required=True)
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
