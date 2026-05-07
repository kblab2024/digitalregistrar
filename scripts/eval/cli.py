"""Subcommand dispatcher for evaluation scripts.

Usage:
    python -m scripts.eval.cli <subcommand> [options]

Subcommands:
    cascade       — cascade-gated five-chapter evaluation (Stage A:
                    eligibility triage, Stage B: organ classification,
                    Stage C: field extraction; Chapter 4 margins, Chapter 5
                    lymph nodes). The primary subcommand.
    compare       — side-by-side comparison of cascade runs (chapter1-5
                    layout); paired bootstrap + McNemar.
    iaa           — inter-annotator agreement + preann effect.
    iaa_pair      — pair-focused IAA: overall κ headline, roll-ups,
                    confusion matrices, markdown summary, for one or
                    more requested pairs.
    completeness  — detailed missingness analysis across methods.
    diagnostics   — source-of-error decomposition, difficulty tiers,
                    worst cases (consumes cascade_atomic.parquet).
    cross_dataset — per-field Δ between datasets + distribution shift
                    (consumes cascade_atomic.parquet).
    headline      — joint forest-plot CSV combining IAA + cascade
                    chapter3 accuracy.

The legacy ``non_nested`` and ``nested`` subcommands were removed in
favor of ``cascade``. The cascade emits per-field accuracy (formerly
``non_nested``) and bipartite-F1 nested fields (formerly ``nested``)
under the same gating, with the gate cohorts honestly applied to
denominators. Downstream consumers (diagnostics / cross_dataset /
headline / compare) now read ``cascade_atomic.parquet`` directly.

All subcommands share a common argument schema (see _common/args.py).
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Callable

logger = logging.getLogger("scripts.eval.cli")

# Subcommand registry. Each entry maps name → (help_text, builder).
# Builders return (parser, main_callable). Lazy-imported so the CLI can
# load even if a subcommand has missing optional dependencies.

SubcommandBuilder = Callable[[argparse._SubParsersAction], None]


def _register_cascade(sub: argparse._SubParsersAction) -> None:
    from scripts.eval.cascade.run_cascade import register
    register(sub)


def _register_compare(sub: argparse._SubParsersAction) -> None:
    from scripts.eval.cascade.compare_runs import register
    register(sub)


def _register_iaa(sub: argparse._SubParsersAction) -> None:
    from scripts.eval.iaa.run_iaa import register
    register(sub)


def _register_iaa_pair(sub: argparse._SubParsersAction) -> None:
    from scripts.eval.iaa.run_iaa_pair import register
    register(sub)


def _register_completeness(sub: argparse._SubParsersAction) -> None:
    from scripts.eval.completeness.run_completeness import register
    register(sub)


def _register_diagnostics(sub: argparse._SubParsersAction) -> None:
    from scripts.eval.diagnostics.run_diagnostics import register
    register(sub)


def _register_cross_dataset(sub: argparse._SubParsersAction) -> None:
    from scripts.eval.cross_dataset.run_cross_dataset import register
    register(sub)


def _register_headline(sub: argparse._SubParsersAction) -> None:
    from scripts.eval.joint.headline_forest import register
    register(sub)


REGISTRARS: dict[str, SubcommandBuilder] = {
    "cascade": _register_cascade,
    "compare": _register_compare,
    "iaa": _register_iaa,
    "iaa_pair": _register_iaa_pair,
    "completeness": _register_completeness,
    "diagnostics": _register_diagnostics,
    "cross_dataset": _register_cross_dataset,
    "headline": _register_headline,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts.eval.cli",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    for name, register in REGISTRARS.items():
        try:
            register(sub)
        except ImportError as e:
            # Subcommand has missing dependency — register a stub so the
            # CLI lists it but errors usefully at invocation.
            stub = sub.add_parser(name, help=f"(unavailable: {e})")
            stub.set_defaults(_unavailable=str(e))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "_unavailable", None):
        logger.error(
            "subcommand %r is unavailable: %s",
            args.subcommand, args._unavailable,
        )
        return 2
    # Resolve --root/--out from --obfustrated when not explicitly set.
    # Only applies to subcommands that called add_common_args (which sets _subcommand).
    if hasattr(args, "_subcommand") and hasattr(args, "obfustrated"):
        from scripts.eval._common.args import apply_obfustrated_defaults
        apply_obfustrated_defaults(args)
    handler = getattr(args, "_handler", None)
    if handler is None:
        parser.error(f"subcommand {args.subcommand!r} did not register a handler")
    return int(handler(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
