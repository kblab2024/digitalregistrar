#!/usr/bin/env python3
"""Rule-based extractor — canonical-layout batch runner.

Iterates every ``<case_id>.txt`` under
``{folder}/data/{dataset}/reports/{organ_n}/`` and writes one prediction
JSON per report to
``{folder}/results/predictions/{dataset}/rule_based/{organ_n}/{case_id}.json``.

The rule baseline is deterministic (no model, no run_id), so the
prediction tree is flat — no per-model or per-run subdirectories. Side
files (``_summary.json``, ``_log.jsonl``, ``_run.log``,
``_run_meta.json``) are written at the ``rule_based/`` directory level.

Mirrors the CLI shape of ``scripts/pipeline/run_dspy_ollama_single.py``
so the rule, BERT, and LLM runners feel uniform.

Usage
-----
    python scripts/baselines/run_rule.py \\
        --folder dummy --dataset tcga \\
        [--organs 1 2] [--limit N] [--overwrite] \\
        [--tolerate-errors] [-v]

Output tree
-----------
    {folder}/results/predictions/{dataset}/rule_based/
        _summary.json         run-level totals
        _log.jsonl            one row per case
        _run.log              full-verbosity log
        _run_meta.json        provenance (git sha, host, argv, ...)
        {organ_n}/
            <case_id>.json    prediction JSON
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import platform
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from _config_loader import resolve_folder  # noqa: E402

from digital_registrar_research.benchmarks.baselines.rules import (  # noqa: E402
    classify_organ,
    extract_for_organ,
)

DATASETS = ("cmuh", "tcga")
DEFAULT_PREDICT_DATASETS = ("tcga",)  # Privacy-aligned: TCGA is the LLM-comparable
                                       # evaluation corpus. Override with
                                       # --datasets cmuh for intra-corpus ablations.
LOGGER_NAME = "scripts.baselines.run_rule"


# --- IO helpers --------------------------------------------------------------


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _valid_existing(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return False
    return isinstance(data, dict) and not data.get("_pipeline_error")


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_sha(repo_root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except Exception:
        pass
    return None


# --- Discovery ---------------------------------------------------------------


def discover_organs(
    reports_root: Path, organ_filter: list[str] | None,
) -> list[tuple[str, Path]]:
    """Sorted ``(organ_n, organ_dir)`` for every numeric reports subdir
    that contains at least one ``*.txt``."""
    if not reports_root.is_dir():
        return []
    picked: list[tuple[str, Path]] = []
    for child in sorted(reports_root.iterdir(), key=lambda p: p.name):
        if not child.is_dir() or child.name.startswith("_"):
            continue
        if organ_filter and child.name not in organ_filter:
            continue
        if not any(child.glob("*.txt")):
            continue
        picked.append((child.name, child))
    return picked


def discover_cases(organ_dir: Path, limit: int | None) -> list[Path]:
    cases = sorted(organ_dir.glob("*.txt"), key=lambda p: p.name)
    if limit is not None:
        cases = cases[:limit]
    return cases


# --- Per-case work -----------------------------------------------------------


def _read_gold_organ(annotations_dir: Path, organ_n: str, case_id: str) -> str | None:
    """Best-effort lookup of cancer_category from gold annotation, if present.

    Returns None when no gold file exists or it lacks the field. Used as a
    fallback when the rule classifier disagrees with the lexicon — but the
    primary classification is still self-driven (rules.classify_organ).
    """
    p = annotations_dir / "gold" / organ_n / f"{case_id}.json"
    if not p.is_file():
        return None
    try:
        with p.open(encoding="utf-8") as f:
            return json.load(f).get("cancer_category")
    except Exception:
        return None


def process_case(
    report_path: Path,
    organ_n: str,
    out_dir: Path,
    log_fh,
    logger: logging.Logger,
    *,
    overwrite: bool,
) -> dict[str, Any]:
    """Extract one case and write its prediction JSON.

    The rule baseline classifies the organ from the report itself (no
    gold leakage). On classifier failure, the prediction has empty
    ``cancer_data`` and ``cancer_category=null``.
    """
    case_id = report_path.stem
    organ_out_dir = out_dir / organ_n
    out_path = organ_out_dir / f"{case_id}.json"
    started_at = _utc_now_iso()

    if not overwrite and _valid_existing(out_path):
        row = {
            "case_id": case_id, "organ": organ_n, "method": "rule_based",
            "status": "cached", "latency_s": 0.0, "parse_success": True,
            "is_cancer": None, "cancer_category": None, "error": None,
            "started_at": started_at,
        }
        log_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        log_fh.flush()
        logger.info("[%s/%s] cached — skipped", organ_n, case_id)
        return row

    report_text = report_path.read_text(encoding="utf-8")
    t0 = time.perf_counter()
    try:
        # Self-classify; do not read gold to keep the floor honest.
        classified = classify_organ(report_text)
        prediction = extract_for_organ(report_text, classified)
        latency_s = round(time.perf_counter() - t0, 4)
        _atomic_write_json(out_path, prediction)
        row = {
            "case_id": case_id, "organ": organ_n, "method": "rule_based",
            "status": "ok", "latency_s": latency_s, "parse_success": True,
            "is_cancer": bool(prediction.get("cancer_excision_report")),
            "cancer_category": prediction.get("cancer_category"),
            "error": None, "started_at": started_at,
        }
        logger.info(
            "[%s/%s] ok (%.4fs, cancer=%s, category=%s)",
            organ_n, case_id, latency_s,
            row["is_cancer"], row["cancer_category"],
        )
    except Exception as exc:
        latency_s = round(time.perf_counter() - t0, 4)
        sentinel = {
            "_pipeline_error": True,
            "reason": type(exc).__name__,
            "message": str(exc)[:2000],
        }
        _atomic_write_json(out_path, sentinel)
        row = {
            "case_id": case_id, "organ": organ_n, "method": "rule_based",
            "status": "pipeline_error", "latency_s": latency_s,
            "parse_success": False, "is_cancer": None, "cancer_category": None,
            "error": f"{type(exc).__name__}: {exc}", "started_at": started_at,
        }
        logger.error("[%s/%s] pipeline error: %s", organ_n, case_id, exc)

    log_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    log_fh.flush()
    return row


# --- Run driver --------------------------------------------------------------


def run(
    out_dir: Path,
    organs: list[tuple[str, Path]],
    annotations_dir: Path,
    logger: logging.Logger,
    args: argparse.Namespace,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "method": "rule_based",
        "dataset": args.dataset,
        "n_cases": 0, "n_ok": 0, "n_pipeline_error": 0, "n_cached": 0,
        "cancer_positive": 0,
        "per_organ": {},
        "wall_time_s": 0.0,
        "created_at": _utc_now_iso(),
    }
    t_run = time.perf_counter()
    log_path = out_dir / "_log.jsonl"
    with log_path.open("a", encoding="utf-8") as log_fh:
        for organ_n, organ_dir in organs:
            cases = discover_cases(organ_dir, args.limit)
            logger.info("organ %s: %d cases", organ_n, len(cases))
            per = summary["per_organ"].setdefault(organ_n, {
                "n_cases": 0, "n_ok": 0, "n_pipeline_error": 0,
                "n_cached": 0, "cancer_positive": 0,
            })
            for report_path in cases:
                row = process_case(
                    report_path, organ_n, out_dir, log_fh, logger,
                    overwrite=args.overwrite,
                )
                summary["n_cases"] += 1
                per["n_cases"] += 1
                status = row["status"]
                if status == "ok":
                    summary["n_ok"] += 1
                    per["n_ok"] += 1
                    if row.get("is_cancer"):
                        summary["cancer_positive"] += 1
                        per["cancer_positive"] += 1
                elif status == "cached":
                    summary["n_cached"] += 1
                    per["n_cached"] += 1
                elif status == "pipeline_error":
                    summary["n_pipeline_error"] += 1
                    per["n_pipeline_error"] += 1

    summary["wall_time_s"] = round(time.perf_counter() - t_run, 1)
    summary["parse_error_rate"] = (summary["n_pipeline_error"]
                                   / max(summary["n_cases"], 1))
    _atomic_write_json(out_dir / "_summary.json", summary)
    return summary


def _setup_logging(out_dir: Path, verbose: bool) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers.clear()
    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    console.setLevel(level)
    logger.addHandler(console)
    fh = logging.FileHandler(out_dir / "_run.log", mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.propagate = False
    return logger


# --- Entry point -------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--folder", dest="experiment_root", default="workspace",
                    type=resolve_folder,
                    help="Experiment root containing data/ and results/. "
                         "Default: workspace. Shorthand 'dummy' or absolute "
                         "path also accepted.")
    ap.add_argument("--datasets", nargs="+", default=list(DEFAULT_PREDICT_DATASETS),
                    choices=DATASETS,
                    help="Dataset name(s) to predict on (default: tcga only — "
                         "the LLM-comparable evaluation corpus). Pass 'cmuh tcga' "
                         "for both, or 'cmuh' for intra-corpus ablation.")
    ap.add_argument("--organs", nargs="*", default=None,
                    help="Only run these numeric organ directories, e.g. 1 2 "
                         "(default: every organ subdir of reports/).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap cases per organ (debugging).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Reprocess cases even if a valid output exists.")
    ap.add_argument("--tolerate-errors", action="store_true",
                    help="Always exit 0 if the script completes, even when "
                         "some cases failed.")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="Set console log level to DEBUG.")
    return ap.parse_args(argv)


def _run_one_dataset(
    dataset: str, args: argparse.Namespace,
) -> tuple[int, dict | None]:
    """Predict on one dataset. Returns (exit_code, summary or None)."""
    reports_root = args.experiment_root / "data" / dataset / "reports"
    annotations_root = args.experiment_root / "data" / dataset / "annotations"
    out_dir = (args.experiment_root / "results" / "predictions"
               / dataset / "rule_based")
    # Set up the logger up-front so missing-data errors land in _run.log
    # alongside everything else, instead of vanishing into stderr.
    logger = _setup_logging(out_dir, args.verbose)

    if not reports_root.is_dir():
        logger.error("[%s] reports not found at %s — workspace not "
                     "populated?", dataset, reports_root)
        return 0, None

    organs = discover_organs(reports_root, args.organs)
    if not organs:
        suffix = f" matching {args.organs}" if args.organs else ""
        logger.error("[%s] no organ dirs with *.txt found under %s%s",
                     dataset, reports_root, suffix)
        return 0, None

    logger.info("experiment_root: %s", args.experiment_root)
    logger.info("dataset: %s", dataset)
    logger.info("method: rule_based (deterministic; no model, no run_id)")
    logger.info("output: %s", out_dir)
    logger.info("organs: %s", [o[0] for o in organs])

    # Inject args.dataset for back-compat with run() which reads it.
    per_ds_args = argparse.Namespace(**{**vars(args), "dataset": dataset})

    started_at = _utc_now_iso()
    t_run = time.perf_counter()
    summary = run(out_dir, organs, annotations_root, logger, per_ds_args)
    finished_at = _utc_now_iso()

    _atomic_write_json(out_dir / "_run_meta.json", {
        "method": "rule_based",
        "dataset": dataset,
        "experiment_root": str(args.experiment_root.resolve()),
        "organs": [o[0] for o in organs],
        "started_at": started_at,
        "finished_at": finished_at,
        "git_sha": _git_sha(REPO_ROOT),
        "python": platform.python_version(),
        "host": socket.gethostname(),
        "argv": sys.argv,
    })

    wall_s = int(time.perf_counter() - t_run)
    wall_fmt = (f"{wall_s // 3600:02d}:{(wall_s % 3600) // 60:02d}:"
                f"{wall_s % 60:02d}")
    summary_line = (
        f"[{dataset}] OK={summary['n_ok']} ERR={summary['n_pipeline_error']} "
        f"CACHED={summary['n_cached']} N={summary['n_cases']} WALL={wall_fmt}"
    )
    logger.info(summary_line)
    print(summary_line)
    print(f"[{dataset}] out dir: {out_dir}")

    code = 1 if (summary["n_pipeline_error"] > 0 and not args.tolerate_errors) else 0
    return code, summary


def run_with_args(args: argparse.Namespace) -> int:
    overall = 0
    n_run = 0
    for dataset in args.datasets:
        code, summary = _run_one_dataset(dataset, args)
        if summary is not None:
            n_run += 1
        overall |= code
    if n_run == 0:
        print(f"error: no datasets in {args.datasets} had reports under "
              f"{args.experiment_root}/data/", file=sys.stderr)
        return 2
    return overall


def main(argv: list[str] | None = None) -> int:
    return run_with_args(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
