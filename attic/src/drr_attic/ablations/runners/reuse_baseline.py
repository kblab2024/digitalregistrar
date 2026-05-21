"""
Cell A — reuse the existing modular-DSPy outputs.

The modular DSPy pipeline IS the parent project; predictions for the
test split already live in the canonical pipeline tree:

    {root}/results/predictions/{dataset}/llm/{model_slug}/{run}/{organ}/{case_id}.json

Cell A copies those per-case JSONs into the canonical ablation tree:

    {root}/results/ablations/{dataset}/dspy_modular/{model_slug}/{run}/{organ}/{case_id}.json

so every cell has a consistent on-disk layout for the aggregator.

By default we copy the **most recent** completed pipeline run for the
chosen model. Pass ``--source-run runNN`` to pin a specific source run.

Canonical layout:
    --folder dummy --dataset tcga --model gptoss [--source-run runNN] [--run runNN]
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from . import _base

CELL_ID = "dspy_modular"


def _pipeline_predictions_dir(experiment_root: Path, dataset: str,
                              model_slug_str: str) -> Path:
    return (experiment_root / "results" / "predictions" / dataset / "llm"
            / model_slug_str)


def _resolve_source_run(pipeline_dir: Path, source_run: str | None) -> str:
    if source_run:
        if not (pipeline_dir / source_run / "_summary.json").exists():
            raise SystemExit(
                f"--source-run {source_run!r} not present (or not finalised) "
                f"under {pipeline_dir}")
        return source_run
    if not pipeline_dir.is_dir():
        raise SystemExit(
            f"no pipeline predictions found at {pipeline_dir}. "
            f"Run scripts/pipeline/run_dspy_ollama_single.py first.")
    candidates = sorted(
        p for p in pipeline_dir.iterdir()
        if p.is_dir() and (p / "_summary.json").exists())
    if not candidates:
        raise SystemExit(
            f"no completed runs (with _summary.json) under {pipeline_dir}")
    return candidates[-1].name


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    _base.add_canonical_args(ap)
    ap.add_argument("--source-run", default=None,
                    help="pipeline run-id to copy from (default: most recent "
                         "completed run under predictions/{dataset}/llm/{slug}/)")
    return ap.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    paths, organs, run_name = _base.resolve_run_paths(args, CELL_ID)
    pipeline_dir = _pipeline_predictions_dir(
        args.experiment_root, args.dataset, paths.model_slug)
    source_run = _resolve_source_run(pipeline_dir, args.source_run)
    source_run_dir = pipeline_dir / source_run

    logger = _base.make_logger("reuse_baseline", paths.run_dir(run_name),
                               args.verbose)
    logger.info("cell=%s model=%s slug=%s run=%s source_run=%s",
                CELL_ID, args.model, paths.model_slug, run_name, source_run)
    logger.info("source dir: %s", source_run_dir)

    summary = _base.RunSummary(
        run=run_name, cell=CELL_ID, model_slug=paths.model_slug,
        model_alias=args.model, dataset=args.dataset,
        seed=None,
    )

    log_path = paths.run_dir(run_name) / "_log.jsonl"
    t_run = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as log_fh:
        for organ_n, _organ_dir in organs:
            src_organ = source_run_dir / organ_n
            if not src_organ.is_dir():
                logger.warning("source has no organ %s — skipping", organ_n)
                continue
            sources = sorted(src_organ.glob("*.json"))
            if args.limit:
                sources = sources[:args.limit]
            for src_path in sources:
                case_id = src_path.stem
                dst_path = paths.case_path(run_name, organ_n, case_id)
                if not args.overwrite and dst_path.exists():
                    summary.record(organ_n, "cached")
                    log_fh.write(json.dumps({
                        "case_id": case_id, "organ": organ_n,
                        "status": "cached",
                    }) + "\n")
                    continue
                try:
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_path, dst_path)
                    with src_path.open(encoding="utf-8") as f:
                        payload = json.load(f)
                    is_cancer = bool(payload.get("cancer_excision_report"))
                    summary.record(organ_n, "ok", is_cancer=is_cancer)
                    log_fh.write(json.dumps({
                        "case_id": case_id, "organ": organ_n,
                        "status": "ok",
                        "is_cancer": is_cancer,
                        "cancer_category": payload.get("cancer_category"),
                    }) + "\n")
                except Exception as exc:
                    summary.record(organ_n, "pipeline_error")
                    logger.error("[%s/%s] copy error: %s",
                                 organ_n, case_id, exc)
                    log_fh.write(json.dumps({
                        "case_id": case_id, "organ": organ_n,
                        "status": "pipeline_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }) + "\n")

    summary.wall_time_s = time.perf_counter() - t_run
    _base.finalize_run(
        paths, run_name, summary, model_alias=args.model,
        decoding={"copied_from": source_run},
        manifest_extra={"source_run": source_run,
                        "source_dir": str(source_run_dir)},
        extra_meta={"source_run": source_run,
                    "source_dir": str(source_run_dir)},
    )

    print(f"OK={summary.n_ok} ERR={summary.n_pipeline_error} "
          f"CACHED={summary.n_cached} N={summary.n_cases}")
    print(f"copied from: {source_run_dir}")
    print(f"run dir: {paths.run_dir(run_name)}")

    if summary.n_pipeline_error > 0 and not args.tolerate_errors:
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    import sys
    sys.exit(main())
