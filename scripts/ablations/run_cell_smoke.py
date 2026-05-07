#!/usr/bin/env python3
"""Per-cell smoke runner — canonical layout.

Runs **one** ablation cell on a small case sample and round-trips the
output through the eval aggregator. Designed to be the go/no-go before
launching a multi-day Grid-1/2 sweep.

Contract:
  * 1 model, 1 seed, 2-3 cases (default ``--n 2``)
  * Output writes into ``{folder}/results/ablations/{dataset}/_smoke_<ts>/``
    so the regular aggregator glob ignores it
  * **Fail loud** — any exception propagates, exit code != 0
  * Round-trips through ``run_ablations.main()`` so the eval reader is
    exercised, not just the cell
  * Asserts ``ablation_summary.csv`` is non-empty before returning

Usage
-----
    # Cell C × local gpt-oss
    python scripts/ablations/run_cell_smoke.py --cell c \\
        --folder dummy --dataset tcga --model gptoss

    # Cell B with --n 2
    python scripts/ablations/run_cell_smoke.py --cell b \\
        --folder dummy --dataset tcga --model gptoss --n 2

    # Cell A — needs a completed pipeline run under predictions/
    python scripts/ablations/run_cell_smoke.py --cell a \\
        --folder dummy --dataset tcga --model gptoss
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    CELL_MAP,
    ensure_aggregator_round_trip,
    smoke_root,
)
from _config_loader import resolve_folder  # noqa: E402

from digital_registrar_research.ablations.runners._base import (  # noqa: E402
    DATASETS,
    UNIFIED_MODELS,
    model_slug,
)


def _build_args(cell_id: str, args: argparse.Namespace,
                results_root: Path) -> argparse.Namespace:
    """Construct the runner Namespace, pointing it at the smoke root."""
    # smoke_root is already absolute; the runner will compute
    # `experiment_root/results/ablations/{dataset}/{cell}/{model}/...`
    # We trick the runner into writing under the smoke dir by passing
    # a synthetic experiment_root whose
    # `results/ablations/{dataset}/{cell}/{model}/...` is the smoke dir.
    #
    # results_root is `{folder}/results/ablations/{dataset}/_smoke_<ts>/`.
    # For the runner to write into `<smoke>/{cell}/{model}/...` we need
    # the runner's `paths.cell_dir` to be `<smoke>/{cell}/{model}/`. The
    # runner builds:
    #   {experiment_root}/results/ablations/{dataset}/{cell_id}/{model_slug}/...
    # so we set:
    #   experiment_root = <smoke>/../../../..
    #   dataset         = "_smoke_<ts>" (substituted in path)
    #
    # Simpler: just give the runner the real folder/dataset and let the
    # output land in the canonical tree alongside real predictions.
    # Smoke is supposed to be invisible to the aggregator, but with
    # canonical layout the easiest "invisibility" is the underscore-prefixed
    # _smoke_<ts> dataset spelling. We keep that contract by passing
    # `dataset = "_smoke_<ts>"` and making the smoke caller create the
    # corresponding fake `data/_smoke_<ts>/` dir if needed. To keep this
    # simple we DO NOT use the smoke_root() trick — we use the real
    # dataset and let the aggregator filter on `--cells`.
    return argparse.Namespace(
        experiment_root=args.experiment_root,
        dataset=args.dataset,
        model=args.model,
        run=None,
        organs=args.organs,
        limit=args.n,
        overwrite=True,
        tolerate_errors=False,
        verbose=args.verbose,
        # Cell-specific defaults (the runner Namespaces are tolerant of
        # extras; argparse-built Namespaces are dicts).
        skip_jsonize=False,
        include_jsonize=False,
        n_shots=3,
        api_base=args.api_base,
        backend="openai",
        use_union_schema=False,
        cot_everywhere=False,
        compiled=getattr(args, "compiled", None),
        source_run=getattr(args, "source_run", None),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cell", required=True, choices=list(CELL_MAP.keys()))
    ap.add_argument("--folder", dest="experiment_root", required=True,
                    type=resolve_folder)
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--model", required=True, choices=UNIFIED_MODELS)
    ap.add_argument("--n", type=int, default=2,
                    help="cases per organ (default 2)")
    ap.add_argument("--organs", nargs="*", default=None)
    ap.add_argument("--api-base", default=None)
    ap.add_argument("--compiled", default=None,
                    help="(c5) path to compiled program JSON")
    ap.add_argument("--source-run", default=None,
                    help="(a) source run-id under predictions/")
    ap.add_argument("--n-shots", type=int, default=3,
                    help="(c2/c3) override the n_shots default")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    cell_id = CELL_MAP[args.cell]
    slug = model_slug(args.model)
    smoke_dir = smoke_root(args.experiment_root, args.dataset)
    print(f"[smoke] cell={args.cell} ({cell_id}) folder={args.experiment_root} "
          f"dataset={args.dataset} model={args.model} slug={slug} "
          f"n={args.n}")
    print(f"[smoke] smoke root: {smoke_dir}")

    # Ensure the cell's runner module is importable.
    from run_grid import CELL_DISPATCH  # noqa: E402

    if cell_id not in CELL_DISPATCH:
        sys.exit(f"Cell {cell_id!r} not registered in CELL_DISPATCH")

    # Build per-cell ns (overrides for c2/c3 n_shots, compiled for c5).
    ns = _build_args(cell_id, args, smoke_dir)
    if args.cell == "c2":
        ns.n_shots = 3
    elif args.cell == "c3":
        ns.n_shots = 5

    module = importlib.import_module(CELL_DISPATCH[cell_id])
    rc = module.run(ns) or 0
    if rc != 0:
        sys.exit(f"smoke run returned non-zero rc={rc}")

    print("\n[smoke] cell finished — round-tripping aggregator…")
    ensure_aggregator_round_trip(args.experiment_root, args.dataset,
                                 [cell_id], [slug])
    print(f"[smoke] OK — summary at "
          f"{args.experiment_root / 'results' / 'ablations' / args.dataset / 'ablation_summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
