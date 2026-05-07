#!/usr/bin/env python3
"""Re-emit the reviewer-grade statistics CSVs from an existing grid run.

Independent from the cell-runner pipeline: reads ``ablation_grid.csv``
and ``efficiency.csv`` already written by ``run_ablations.main()`` and
regenerates the statistical outputs (paired Δ + CI, McNemar,
multiple-comparison correction, multi-seed GLMM, Fleiss κ, factorial
GLMM, efficiency CIs, effect sizes).

Usage::

    python scripts/ablations/run_stats.py --folder dummy --dataset tcga
    python scripts/ablations/run_stats.py --results-root <path> --baseline dspy_modular_gpt_oss_20b
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO_ROOT  # noqa: E402,F401
from _config_loader import resolve_folder  # noqa: E402

from digital_registrar_research.ablations.eval import stats as ablation_stats  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--folder", dest="experiment_root", default=None,
                    type=resolve_folder,
                    help="Experiment root containing data/ and results/.")
    ap.add_argument("--dataset", default=None, choices=("cmuh", "tcga"))
    ap.add_argument("--results-root", type=Path, default=None,
                    help="Override the {folder}/results/ablations/{dataset}/ path")
    ap.add_argument("--baseline", default="dspy_modular_gpt_oss_20b",
                    help="<cell>_<model_slug> key for paired-Δ comparisons")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    if args.results_root is not None:
        results_root = args.results_root
    elif args.experiment_root is not None and args.dataset:
        results_root = (args.experiment_root / "results" / "ablations"
                        / args.dataset)
    else:
        sys.exit("Provide --folder + --dataset OR --results-root.")

    if not results_root.exists():
        sys.exit(f"Results root not found: {results_root}")

    outputs = ablation_stats.run_all(
        results_root,
        baseline_method=args.baseline,
        n_boot=args.n_boot,
    )
    if not outputs:
        sys.exit("No statistical outputs written — is ablation_grid.csv "
                 "present and non-empty?")
    for stage, path in outputs.items():
        print(f"  → {stage:25s} {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
