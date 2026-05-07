#!/usr/bin/env python3
"""Smoke-test the DSPy + Ollama cancer-extraction stack on a few reports.

Purpose
-------
Before committing to a multi-hour full run, verify end-to-end that:
  * Ollama is reachable and the requested model key resolves to a pulled model
  * DSPy signatures still bind to the pipeline code
  * Reports under ``{experiment_root}/data/{dataset}/reports/{organ_n}/``
    are discoverable
  * JSON predictions are written where the full-run script would write them

Unlike the full driver, this script fails loudly: the process exits non-zero
if *any* sampled case raises a pipeline exception. A green smoke run is the
go/no-go for the real run.

Usage
-----
    python scripts/run_dspy_ollama_smoke.py \\
        --model gptoss \\
        --folder dummy \\
        --dataset tcga \\
        [--n 3] [--seed 0] [--organs 1 2]

``--model`` must be one of: gptoss, gemma3, gemma4, qwen3_5, medgemmalarge,
medgemmasmall. Each alias auto-loads ``configs/dspy_ollama_{alias}.yaml`` for
decoding overrides and smoke defaults (``smoke.n``, ``smoke.seed``). CLI
``--n`` / ``--seed`` win when provided; otherwise the config's smoke
section is used.

Output
------
    {experiment_root}/results/predictions/{dataset}/llm/{model_slug}/_smoke_{date}/
        {organ_n}/<case_id>.json   prediction (one per sampled case)
        _log.jsonl                  one row per case
        _summary.json               aggregate
        _run.log                    full-verbosity log

The output dir starts with ``_`` so downstream eval globs that filter
``not name.startswith('_')`` skip it — smoke runs never pollute real sweeps.
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))  # for _config_loader, _run_id
sys.path.insert(0, str(REPO_ROOT / "scripts" / "pipeline"))  # for sibling imports

# Reuse the low-level helpers from the full-run script so the two stay in
# lockstep w/r/t output format and atomic-write semantics.
from _config_loader import (  # noqa: E402
    load_model_config,
    resolve_folder,
    split_decoding_overrides,
)
from run_dspy_ollama_single import (  # noqa: E402
    DATASETS,
    PIPELINE_LOGGER_NAME,
    UNIFIED_MODELS,
    _atomic_write_json,
    discover_organs,
    model_slug,
    process_case,
)

from digital_registrar_research.models.common import (  # noqa: E402
    load_model,
    model_list,
)
from digital_registrar_research.pipeline import setup_pipeline  # noqa: E402
from digital_registrar_research.util.logger import setup_logger  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model", required=True, choices=UNIFIED_MODELS,
                    help="Model alias: one of " + ", ".join(UNIFIED_MODELS) +
                         ".")
    ap.add_argument("--folder", dest="experiment_root", required=True,
                    type=resolve_folder,
                    help="Experiment root. Shorthand 'dummy' or 'workspace' "
                         "resolves against the repo root.")
    ap.add_argument("--dataset", required=True, choices=DATASETS,
                    help="Dataset name under data/ (cmuh or tcga).")
    # --n / --seed default to None so the per-model config's smoke section
    # can supply them; CLI always wins when provided.
    ap.add_argument("--n", type=int, default=None,
                    help="Number of random cases to run "
                         "(default: from config smoke.n, else 3).")
    ap.add_argument("--seed", type=int, default=None,
                    help="Sampling seed for reproducibility "
                         "(default: from config smoke.seed, else 0).")
    ap.add_argument("--organs", nargs="*", default=None,
                    help="Restrict sampling to these numeric organ dirs.")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args(argv)


def run_with_args(
    args: argparse.Namespace, overrides: dict | None = None,
) -> int:
    """Execute the smoke loop given a prebuilt argparse.Namespace and an
    optional decoding-overrides dict. Factored out of ``main`` so the
    per-(model, tree) YAML wrappers in ``scripts/run_dspy_ollama_smoke_*.py``
    can reuse the same body without re-invoking argparse."""
    if args.model not in model_list:
        print(f"error: unknown --model {args.model!r}. "
              f"Valid keys: {', '.join(model_list.keys())}", file=sys.stderr)
        return 2

    reports_root = args.experiment_root / "data" / args.dataset / "reports"
    if not reports_root.is_dir():
        print(f"error: reports not found at {reports_root}", file=sys.stderr)
        return 2

    organs = discover_organs(reports_root, args.organs)
    if not organs:
        suffix = f" matching {args.organs}" if args.organs else ""
        print(f"error: no organ dirs with *.txt found under {reports_root}"
              f"{suffix}", file=sys.stderr)
        return 2

    # Flatten to (organ_n, report_path) pairs so the sample carries the
    # organ alongside the file — we need it to build {organ}/{case_id}.json.
    all_reports: list[tuple[str, Path]] = [
        (organ_n, p)
        for organ_n, organ_dir in organs
        for p in sorted(organ_dir.glob("*.txt"))
    ]
    if len(all_reports) < args.n:
        print(f"error: only {len(all_reports)} reports under {reports_root}, "
              f"cannot sample {args.n}", file=sys.stderr)
        return 2

    sampled = random.Random(args.seed).sample(all_reports, args.n)

    slug = model_slug(args.model)
    date_str = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (args.experiment_root / "results" / "predictions"
               / args.dataset / "llm" / slug / f"_smoke_{date_str}")
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(
        name=PIPELINE_LOGGER_NAME,
        level=logging.DEBUG if args.verbose else logging.INFO,
        log_file=str(out_dir / "_run.log"),
        json_format=False,
    )
    logger.info("smoke dir: %s", out_dir)
    logger.info("dataset: %s", args.dataset)
    logger.info("model: %s (%s) → slug=%s",
                args.model, model_list[args.model], slug)
    logger.info("sampled %d / %d reports (seed=%d)",
                len(sampled), len(all_reports), args.seed)
    for organ_n, p in sampled:
        logger.info("  - %s (organ=%s)", p.name, organ_n)

    setup_pipeline(args.model, overrides=overrides)
    lm = load_model(args.model, overrides=overrides)
    lm_seed = lm.kwargs.get("seed")
    if overrides:
        logger.info("decoding overrides: %s", overrides)

    run_name = out_dir.name  # e.g. "_smoke_20260423_101530"

    summary = {
        "run": run_name,
        "model": slug,
        "seed": lm_seed,
        "dataset": args.dataset,
        "n_cases": 0, "n_ok": 0, "n_pipeline_error": 0,
        "cancer_positive": 0, "sampled": [],
    }
    with (out_dir / "_log.jsonl").open("a", encoding="utf-8") as log_fh:
        for organ_n, report_path in sampled:
            row = process_case(
                report_path,
                organ=organ_n,
                run_name=run_name,
                seed=lm_seed,
                out_dir=out_dir,
                log_fh=log_fh,
                logger=logger,
                overwrite=True,  # smoke runs should never silently skip
            )
            summary["n_cases"] += 1
            summary["sampled"].append({
                "case_id": row["case_id"], "organ": row["organ"],
                "status": row["status"], "latency_s": row["latency_s"],
            })
            if row["status"] == "ok":
                summary["n_ok"] += 1
                if row.get("is_cancer"):
                    summary["cancer_positive"] += 1
            elif row["status"] == "pipeline_error":
                summary["n_pipeline_error"] += 1

    _atomic_write_json(out_dir / "_summary.json", summary)
    summary_line = (
        f"SMOKE OK={summary['n_ok']} ERR={summary['n_pipeline_error']} "
        f"N={summary['n_cases']}"
    )
    logger.info(summary_line)
    print(summary_line)
    print(f"smoke dir: {out_dir}")

    return 0 if summary["n_pipeline_error"] == 0 and summary["n_ok"] == args.n else 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_model_config(args.model)
    overrides = split_decoding_overrides(cfg.get("decoding"))
    # Fill --n / --seed from the config's smoke: section when CLI omitted them.
    smoke = cfg.get("smoke") or {}
    if args.n is None:
        args.n = int(smoke.get("n", 3))
    if args.seed is None:
        args.seed = int(smoke.get("seed", 0))
    return run_with_args(args, overrides)


if __name__ == "__main__":
    sys.exit(main())
