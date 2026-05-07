#!/usr/bin/env python3
"""Evaluate `regional_lymph_node` predictions for generative methods.

Reports two axes per method:
  - Case-level: total examined / involved (MAE, accuracy within +/-1),
    any-positive (accuracy + F1).
  - Per-station: precision / recall / F1 on optimal bipartite matching,
    plus conditional accuracy on involved / examined counts, category,
    and side for the matched pairs.

Only generative methods (`digital_registrar`, `gpt4_dspy`) emit
list-of-dicts predictions for this field; rules / clinicalbert are
rejected with a clear error.

Usage (from the repo root):

    python scripts/eval_lymph_nodes.py
    python scripts/eval_lymph_nodes.py --methods digital_registrar
    python scripts/eval_lymph_nodes.py \
        --results-root results/benchmarks \
        --out results/benchmarks/nested/ln_by_method.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

from digital_registrar_research import paths
from digital_registrar_research.benchmarks.eval.nested_metrics import (
    GENERATIVE_METHODS,
    aggregate_ln_to_csv,
    summarize_ln,
)

DEFAULT_METHODS = ["digital_registrar", "gpt4_dspy"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--methods", nargs="*", default=DEFAULT_METHODS,
                    help="Generative method names. Default: %(default)s")
    ap.add_argument("--results-root", type=Path, default=paths.BENCHMARKS_RESULTS,
                    help="Parent folder of <method>/ prediction dirs "
                         "(default: %(default)s)")
    ap.add_argument("--splits", type=Path, required=True,
                    help="Path to a legacy splits.json (no longer packaged "
                         "in-repo; supply your own).")
    ap.add_argument("--out", type=Path,
                    default=paths.BENCHMARKS_RESULTS / "nested" / "ln_by_method.csv",
                    help="Output CSV of per-case metrics (default: %(default)s)")
    args = ap.parse_args()

    bad = [m for m in args.methods if m not in GENERATIVE_METHODS]
    if bad:
        raise SystemExit(
            f"[eval-ln] refuse to score non-generative methods {bad}: "
            f"regional_lymph_node is a list-of-dicts field that only "
            f"{sorted(GENERATIVE_METHODS)} can produce. "
            "Pass --methods with only generative method names."
        )

    method_to_preds = {m: args.results_root / m for m in args.methods
                       if (args.results_root / m).exists()}
    missing = set(args.methods) - set(method_to_preds)
    if missing:
        print(f"[eval-ln] skipping methods with no predictions: {sorted(missing)}")
    if not method_to_preds:
        raise SystemExit(
            f"[eval-ln] no prediction folders found under {args.results_root}. "
            f"Expected {args.results_root}/<method>/<case_id>.json."
        )

    df = aggregate_ln_to_csv(method_to_preds, args.splits, args.out)
    print(f"[eval-ln] wrote {args.out} ({len(df)} rows)")

    summary = summarize_ln(df)
    summary_csv = args.out.with_name("ln_summary.csv")
    summary.to_csv(summary_csv, index=False)
    print(f"[eval-ln] wrote {summary_csv}")
    if summary.empty:
        print("[eval-ln] no cases attempted by any method — nothing to summarise.")
    else:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
