#!/usr/bin/env python3
"""Chain N consecutive ``run_grid`` passes with seeded iterations.

Companion to ``scripts/pipeline/run_dspy_ollama_multirun.py``, lifted one
level up: instead of looping the *single full pipeline* N times, this
loops a *whole ablation grid* (e.g. ``configs/ablations/grid_1.yaml``)
N times. Each iteration draws one seed from the master sequence and
applies it as a ``decoding.seed`` override to every cell in that pass,
so all cells in iteration k share ``seed_k`` and land in the next free
``runNN/`` slot.

Why this exists
---------------
``run_grid`` does one pass with the seed baked into
``configs/dspy_ollama_<alias>.yaml``. The grid_1 docstring already
promises an ``--master-seed`` knob that doesn't exist anywhere — this
script is that knob.

Usage
-----
    python scripts/ablations/run_grid_multirun.py \\
        --config configs/ablations/grid_1.yaml \\
        --n 5 [--master-seed 1234] \\
        [--folder dummy] [--dataset tcga] \\
        [--continue-on-cell-error] \\
        [--skip-aggregate] [--tolerate-errors] [-v]

The seed sequence is reproducible: ``--master-seed 1234`` always
yields the same N seeds, so re-running last week's K-sweep is one flag.
Without it, each iteration draws a fresh ``secrets.randbelow(2**31)``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "ablations"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "pipeline"))

import yaml  # noqa: E402

from _config_loader import resolve_folder  # noqa: E402

import run_grid  # noqa: E402  (scripts/ablations/run_grid.py)
from run_dspy_ollama_multirun import draw_seeds  # noqa: E402

from digital_registrar_research.ablations.runners import _base as _ab_base  # noqa: E402
from digital_registrar_research.ablations.runners._base import MAX_RUN_SLOTS  # noqa: E402


@contextmanager
def _seed_override(seed: int):
    """Patch ``_base.load_decoding_overrides`` to merge in ``seed``.

    Every ablation runner calls ``_base.load_decoding_overrides(args.model)``
    inside its ``run(args)`` entry point, so this single patch propagates
    to all 14 cell runners with no edits to the runners themselves. The
    seed flows on into ``setup_dspy_lm`` and is persisted into
    ``_run_meta.json`` + ``_manifest.yaml`` via the existing
    ``finalize_run`` path.
    """
    original = _ab_base.load_decoding_overrides

    def patched(model_alias: str) -> dict:
        return {**original(model_alias), "seed": int(seed)}

    _ab_base.load_decoding_overrides = patched
    try:
        yield
    finally:
        _ab_base.load_decoding_overrides = original


def _resolve_grid_target(spec: dict, folder_override: Path | None,
                         dataset_override: str | None,
                         ) -> tuple[Path, str]:
    """Mirror ``run_grid.main``'s folder/dataset resolution so we can
    locate the ablations-root output dir without re-running the YAML."""
    experiment_root = (folder_override
                       if folder_override is not None
                       else resolve_folder(spec["folder"]))
    dataset = dataset_override or spec["dataset"]
    return experiment_root, dataset


def _build_inner_argv(args: argparse.Namespace) -> list[str]:
    """Forward CLI flags to ``run_grid.main``. Always pass
    ``--skip-aggregate`` — we run the aggregator once at the end."""
    argv = ["--config", str(args.config), "--skip-aggregate"]
    if args.experiment_root_override is not None:
        argv += ["--folder", str(args.experiment_root_override)]
    if args.dataset_override is not None:
        argv += ["--dataset", args.dataset_override]
    if args.continue_on_cell_error:
        argv.append("--continue-on-cell-error")
    return argv


def _rename_grid_meta(ablations_root: Path, k: int) -> Path | None:
    """Rename ``_grid_meta.json`` → ``_grid_meta_iter{k:02d}.json`` so the
    next iteration doesn't clobber it. Returns the new path, or None if
    the iteration crashed before run_grid wrote the file."""
    src = ablations_root / "_grid_meta.json"
    if not src.exists():
        return None
    dst = ablations_root / f"_grid_meta_iter{k:02d}.json"
    if dst.exists():
        dst.unlink()
    src.rename(dst)
    return dst


def _aggregate(experiment_root: Path, dataset: str, spec: dict) -> int:
    """Run the eval aggregator once across all runs. Mirrors the call
    shape ``run_grid.main`` uses internally."""
    cells_seen = sorted({run["cell"] for run in spec["runs"]})
    from digital_registrar_research.ablations.eval.run_ablations import (
        main as eval_main,
    )
    eval_argv = [
        "--folder", str(experiment_root),
        "--dataset", dataset,
        "--cells", *cells_seen,
    ]
    return eval_main(eval_argv) or 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", type=Path, required=True,
                    help="YAML grid spec (e.g. configs/ablations/grid_1.yaml).")
    ap.add_argument("--n", type=int, default=5,
                    help="Number of consecutive grid passes "
                         "(default: 5; capped to MAX_RUN_SLOTS=%d slots since "
                         "each cell runner only allocates run01..run%02d.)"
                         % (MAX_RUN_SLOTS, MAX_RUN_SLOTS))
    ap.add_argument("--master-seed", type=int, default=None,
                    help="If set, the per-iteration seed sequence is drawn "
                         "from random.Random(master_seed) so the whole sweep "
                         "is reproducible. Default: each iteration draws a "
                         "fresh seed via secrets.randbelow.")
    ap.add_argument("--folder", dest="experiment_root_override",
                    type=resolve_folder, default=None,
                    help="Override the YAML's `folder:` field (forwarded "
                         "to run_grid).")
    ap.add_argument("--dataset", dest="dataset_override", default=None,
                    help="Override the YAML's `dataset:` field (forwarded "
                         "to run_grid).")
    ap.add_argument("--continue-on-cell-error", action="store_true",
                    help="Forward to run_grid: don't halt the grid on a "
                         "single cell failure.")
    ap.add_argument("--skip-aggregate", action="store_true",
                    help="Skip the final aggregator after all iterations.")
    ap.add_argument("--tolerate-errors", action="store_true",
                    help="Continue the sweep even if an iteration returns "
                         "non-zero. Default: stop on the first failed "
                         "iteration.")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.n <= 0:
        print(f"error: --n must be >= 1 (got {args.n})", file=sys.stderr)
        return 2
    if args.n > MAX_RUN_SLOTS:
        print(f"error: --n={args.n} exceeds MAX_RUN_SLOTS={MAX_RUN_SLOTS}; "
              f"the cell runners only allocate run01..run{MAX_RUN_SLOTS:02d}.",
              file=sys.stderr)
        return 2

    if not args.config.is_file():
        print(f"error: grid config not found: {args.config}", file=sys.stderr)
        return 2

    spec = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not spec.get("runs"):
        print(f"error: no runs listed in {args.config}", file=sys.stderr)
        return 2

    experiment_root, dataset = _resolve_grid_target(
        spec, args.experiment_root_override, args.dataset_override)
    ablations_root = experiment_root / "results" / "ablations" / dataset

    seeds = draw_seeds(args.n, args.master_seed)

    print(f"grid_multirun: config={args.config}")
    print(f"grid_multirun: folder={experiment_root}  dataset={dataset}")
    print(f"grid_multirun: n={args.n}  master_seed={args.master_seed}  "
          f"seeds={seeds}")

    inner_argv = _build_inner_argv(args)

    iterations: list[dict] = []
    overall_t0 = time.perf_counter()
    last_rc = 0
    for i, seed in enumerate(seeds, start=1):
        print(f"\n=== grid_multirun iteration {i}/{args.n} (seed={seed}) ===")
        t0 = time.perf_counter()
        err: str | None = None
        try:
            with _seed_override(seed):
                rc = run_grid.main(inner_argv) or 0
        except SystemExit as exc:
            rc = int(exc.code) if exc.code is not None else 1
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            print(f"grid_multirun: iteration {i} raised {err}",
                  file=sys.stderr)
            rc = 1
        wall_s = round(time.perf_counter() - t0, 1)
        meta_path = _rename_grid_meta(ablations_root, i)
        iterations.append({
            "i": i,
            "seed": int(seed),
            "rc": rc,
            "wall_s": wall_s,
            "error": err,
            "grid_meta": meta_path.name if meta_path else None,
        })
        last_rc = rc
        print(f"=== grid_multirun iteration {i}/{args.n} done "
              f"rc={rc} wall={wall_s}s ===")
        if rc != 0 and not args.tolerate_errors:
            print(f"grid_multirun: stopping early (iteration {i} rc={rc}; "
                  f"pass --tolerate-errors to keep going).", file=sys.stderr)
            break

    overall_wall = round(time.perf_counter() - overall_t0, 1)

    # Persist the multirun-level summary regardless of partial completion.
    ablations_root.mkdir(parents=True, exist_ok=True)
    multirun_meta = {
        "config_path": str(args.config),
        "n_requested": args.n,
        "n_completed": len(iterations),
        "master_seed": args.master_seed,
        "seeds": [int(s) for s in seeds],
        "iterations": iterations,
        "overall_wall_s": overall_wall,
        "completed_utc": dt.datetime.utcnow().isoformat(timespec="seconds"),
    }
    (ablations_root / "_multirun_meta.json").write_text(
        json.dumps(multirun_meta, ensure_ascii=False, indent=2),
        encoding="utf-8")

    ok_runs = sum(1 for it in iterations if it["rc"] == 0)
    failed_runs = len(iterations) - ok_runs
    print(f"\nGRID_MULTIRUN K={len(iterations)}/{args.n} "
          f"OK_ITERS={ok_runs} FAILED_ITERS={failed_runs} "
          f"WALL={overall_wall}s")
    for it in iterations:
        suffix = f" err={it['error']}" if it["error"] else ""
        print(f"  iter {it['i']:>2}: seed={it['seed']:<11} "
              f"rc={it['rc']} wall={it['wall_s']}s{suffix}")

    if failed_runs and not args.tolerate_errors:
        return last_rc or 1

    if args.skip_aggregate:
        print("\ngrid_multirun: --skip-aggregate set; not running aggregator")
        return 0

    if ok_runs == 0:
        print("\ngrid_multirun: no successful iterations; skipping aggregator",
              file=sys.stderr)
        return last_rc or 1

    print("\ngrid_multirun: running aggregator across all runs…")
    rc = _aggregate(experiment_root, dataset, spec)
    if rc == 0:
        print(f"grid_multirun: OK — see {ablations_root}/ablation_summary.csv")
    return rc


if __name__ == "__main__":
    sys.exit(main())
