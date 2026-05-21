"""
C5 — Inference using a `BootstrapFewShotWithRandomSearch`-compiled DSPy program.

Loads a compiled monolithic pipeline JSON written by
``scripts/ablations/compile_dspy.py`` and runs it against the canonical
report tree. Otherwise structurally identical to Cell B.

Canonical layout:
    --folder dummy --dataset tcga --model gptoss --compiled <path> [--run runNN]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import _base
from .dspy_monolithic import MonolithicPipeline

CELL_ID = "compiled_dspy"


def _load_compiled(path: Path) -> MonolithicPipeline:
    if not path.exists():
        sys.exit(
            f"No compiled program at {path}. Run "
            f"`python scripts/ablations/compile_dspy.py --folder ... "
            f"--dataset ... --model <alias> --out {path}` first.")
    pipe = MonolithicPipeline()
    pipe.load(str(path))
    return pipe


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    _base.add_canonical_args(ap)
    ap.add_argument("--compiled", required=True, type=Path,
                    help="path to compiled program JSON written by compile_dspy.py")
    return ap.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    paths, organs, run_name = _base.resolve_run_paths(args, CELL_ID)
    overrides = _base.load_decoding_overrides(args.model)
    lm_kwargs = _base.setup_dspy_lm(args.model, overrides=overrides)

    logger = _base.make_logger("compiled_dspy", paths.run_dir(run_name),
                               args.verbose)
    logger.info("cell=%s model=%s slug=%s run=%s compiled=%s",
                CELL_ID, args.model, paths.model_slug, run_name, args.compiled)

    pipe = _load_compiled(args.compiled)

    def _predict(report_text: str, organ_n: str, case_id: str) -> dict:
        return pipe(report=report_text, logger=logger, fname=case_id)

    summary = _base.run_loop(
        paths, organs, run_name, model_alias=args.model,
        predict=_predict, args=args, logger=logger,
        decoding=lm_kwargs,
        manifest_extra={"compiled_artifact": str(args.compiled)},
        extra_meta={"compiled_artifact": str(args.compiled),
                    "dspy_lm_kwargs": lm_kwargs},
    )

    print(f"OK={summary.n_ok} ERR={summary.n_pipeline_error} "
          f"CACHED={summary.n_cached} N={summary.n_cases} "
          f"NOT_CANCER={summary.n_skipped_not_cancer} "
          f"UNKNOWN_ORGAN={summary.n_skipped_unknown_organ} "
          f"DOWNSTREAM={summary.n_downstream_called} "
          f"WALL={summary.wall_time_s:.1f}s")
    print(f"run dir: {paths.run_dir(run_name)}")

    if summary.n_pipeline_error > 0 and not args.tolerate_errors:
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
