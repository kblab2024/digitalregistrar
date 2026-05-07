#!/usr/bin/env python3
"""Grid-wide smoke runner — canonical layout.

Runs every requested ablation cell on a 2-3 case sample, then
aggregates across all cells. Designed to catch inter-cell layout
regressions and aggregator-side bugs that a per-cell smoke would miss.

Usage::

    python scripts/ablations/run_grid_smoke.py --folder dummy --dataset tcga --model gptoss
    python scripts/ablations/run_grid_smoke.py --folder dummy --dataset tcga \\
        --model gptoss --cells b c b6 c4
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    CELL_MAP,
    ensure_aggregator_round_trip,
)
from _config_loader import resolve_folder  # noqa: E402

from digital_registrar_research.ablations.runners._base import (  # noqa: E402
    DATASETS,
    UNIFIED_MODELS,
    model_slug,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--folder", dest="experiment_root", required=True,
                    type=resolve_folder)
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--model", required=True, choices=UNIFIED_MODELS)
    ap.add_argument("--cells", nargs="+",
                    default=["b", "c", "b6", "c4", "c6", "f2", "f3"],
                    choices=list(CELL_MAP.keys()),
                    help="cells to smoke (default: b c b6 c4 c6 f2 f3)")
    ap.add_argument("--n", type=int, default=2, help="cases per organ")
    ap.add_argument("--organs", nargs="*", default=None)
    ap.add_argument("--api-base", default=None)
    ap.add_argument("--compiled", default=None,
                    help="(c5) compiled program JSON")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    slug = model_slug(args.model)
    cells_run: list[str] = []
    t0 = time.perf_counter()

    print(f"[grid-smoke] folder={args.experiment_root} dataset={args.dataset} "
          f"model={args.model} slug={slug} cells={args.cells} n={args.n}")

    from run_grid import CELL_DISPATCH  # noqa: E402

    for short in args.cells:
        cell_id = CELL_MAP[short]
        if cell_id not in CELL_DISPATCH:
            sys.exit(f"Cell {short!r} ({cell_id}) not registered.")
        print(f"\n=== smoking cell {short} ({cell_id}) ===")
        t_cell = time.perf_counter()

        ns = argparse.Namespace(
            experiment_root=args.experiment_root,
            dataset=args.dataset,
            model=args.model,
            run=None,
            organs=args.organs,
            limit=args.n,
            overwrite=True,
            tolerate_errors=False,
            verbose=args.verbose,
            skip_jsonize=False,
            include_jsonize=False,
            n_shots=3 if short == "c2" else (5 if short == "c3" else 3),
            api_base=args.api_base,
            backend="openai",
            use_union_schema=False,
            cot_everywhere=False,
            compiled=args.compiled,
            source_run=None,
        )
        module = importlib.import_module(CELL_DISPATCH[cell_id])
        rc = module.run(ns) or 0
        if rc != 0:
            sys.exit(f"smoke cell {short} returned rc={rc}")
        cells_run.append(cell_id)
        print(f"=== cell {short} done in {time.perf_counter() - t_cell:.1f}s")

    uniq_cells = list(dict.fromkeys(cells_run))

    print(f"\n[grid-smoke] all cells finished — round-tripping aggregator…")
    ensure_aggregator_round_trip(args.experiment_root, args.dataset,
                                 uniq_cells, [slug])

    elapsed = time.perf_counter() - t0
    print(f"\n[grid-smoke] OK — total wall time {elapsed:.1f}s")
    if elapsed > 600:
        print(f"[grid-smoke] WARNING: wall time exceeded 10-minute target.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
