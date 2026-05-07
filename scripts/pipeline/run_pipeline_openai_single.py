#!/usr/bin/env python3
"""Single-run OpenAI cancer-extraction experiment using the loose pipeline.

OpenAI counterpart of ``run_dspy_strict_ollama_single.py``. Routes the
canonical :func:`digital_registrar_research.pipeline.run_cancer_pipeline`
(loose JSON, the same one shipped at ``src/digital_registrar_research/
pipeline.py``) at an OpenAI-hosted model via DSPy / LiteLLM. The
extraction signatures, prompts, and post-processing are held constant —
only the LM backend swaps — so the resulting predictions are
apples-to-apples with the local Ollama runs in the cascade comparator.

Why this exists
---------------
The reviewer asked for a "meaningful baseline" against a top-class
hosted model. Instead of hand-rolling a separate baseline script (which
would change prompts, parsing, and fall victim to the apples-to-oranges
critique), we drive the *same* pipeline through OpenAI. Every per-organ
``dspy.Predict`` call goes through ``openai/<model>`` via LiteLLM, with
the API key loaded from ``~/.config/digital-registrar/.env`` (out of
repo).

Layout, run-slot rotation, manifest format, and per-case JSON output are
byte-for-byte identical to the dspy_strict and structured siblings so
the eval pipeline ingests this run without modification.

Usage
-----
    python scripts/pipeline/run_pipeline_openai_single.py \\
        --model gpt5_4_mini \\
        --folder workspace \\
        --dataset tcga \\
        [--run run01] [--organs 1 2] [--limit N] [--overwrite] \\
        [--tolerate-errors] [-v]

``--model`` must resolve to a ``model_list`` alias whose target starts
with ``openai/``.  ``--folder`` accepts the shorthands ``dummy`` and
``workspace`` (resolved against the repo root) or any absolute /
relative path.

Output tree
-----------
    {experiment_root}/results/predictions/{dataset}/llm/{model_slug}/
        _manifest.yaml            aggregated across all runs (updated in place)
        {run}/                    e.g. run01
            _summary.json         run-level totals
            _log.jsonl            one row per case
            _run.log              full-verbosity log
            _run_meta.json        provenance for this run
            _cost_ledger.json     per-case wall-time (proxy for spend)
            {organ_n}/
                <case_id>.json    prediction or {"_pipeline_error": true, ...}
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import os
import platform
import re
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
# Make the in-tree package importable without requiring `pip install -e .`.
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))  # for _config_loader, _run_id

from _config_loader import resolve_folder  # noqa: E402
from _run_id import format_run_id, machine_slug  # noqa: E402

from digital_registrar_research.models.common import (  # noqa: E402
    compute_lm_kwargs,
    model_list,
)
from digital_registrar_research.pipeline import (  # noqa: E402
    run_cancer_pipeline,
    setup_pipeline,
)
from digital_registrar_research.util.logger import setup_logger  # noqa: E402
from digital_registrar_research.util.secrets import load_openai_key  # noqa: E402

PIPELINE_LOGGER_NAME = "experiment_logger"  # fixed by pipeline.run_pipeline

DATASETS = ("cmuh", "tcga")
MAX_RUN_SLOTS = 10  # memory: "run01..run10"

# Aliases offered from the CLI. Restricted to OpenAI-prefixed entries in
# model_list so a stray alias choice does not silently route an Ollama
# model through this runner (which would still work but defeats its
# purpose and skips the OPENAI_API_KEY load).
def _openai_aliases() -> tuple[str, ...]:
    return tuple(k for k, v in model_list.items() if v.startswith("openai/"))


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


def _atomic_write_yaml(path: Path, payload: dict[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False)
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


def _split_report_rows(report_text: str) -> list[str]:
    rows = report_text.split("\n\n")
    return [row.strip() for row in rows if row.strip()]


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


# --- Naming ------------------------------------------------------------------


def model_slug(model_key: str) -> str:
    """Canonical folder name for an OpenAI run.

    No suffix (unlike the ``_dspy_strict`` variant) because the loose
    pipeline.py runner is the *only* OpenAI driver in the codebase — no
    collision to disambiguate.

    ``openai/gpt-5.4-mini`` → ``gpt_5_4_mini``;
    ``openai/gpt-4o``       → ``gpt_4o``.
    """
    full = model_list[model_key]
    tail = full.split("/", 1)[-1]
    return re.sub(r"[-:./]", "_", tail)


# --- Discovery ---------------------------------------------------------------


def discover_organs(
    reports_root: Path, organ_filter: list[str] | None,
) -> list[tuple[str, Path]]:
    if not reports_root.is_dir():
        return []
    picked: list[tuple[str, Path]] = []
    for child in sorted(reports_root.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        if child.name.startswith("_"):
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


def pick_next_run(model_dir: Path) -> str:
    slug = machine_slug()
    for k in range(1, MAX_RUN_SLOTS + 1):
        name = format_run_id(k, padded=True)
        if not (model_dir / name / "_summary.json").exists():
            return name
    suffix_msg = f" (machine={slug!r})" if slug else ""
    raise RuntimeError(
        f"all {MAX_RUN_SLOTS} run slots are populated under {model_dir}{suffix_msg}; "
        f"pass --run runNN explicitly to re-run one (use --overwrite to clobber cases)."
    )


# --- Per-case work -----------------------------------------------------------


def process_case(
    report_path: Path,
    organ: str,
    run_name: str,
    seed: Any,
    out_dir: Path,
    log_fh,
    cost_ledger: list[dict[str, Any]],
    logger: logging.Logger,
    *,
    overwrite: bool,
) -> dict[str, Any]:
    case_id = report_path.stem
    organ_out_dir = out_dir / organ
    out_path = organ_out_dir / f"{case_id}.json"
    started_at = _utc_now_iso()

    if not overwrite and _valid_existing(out_path):
        row = {
            "case_id": case_id, "organ": organ, "run": run_name, "seed": seed,
            "status": "cached", "latency_s": 0.0, "parse_success": True,
            "is_cancer": None, "cancer_category": None, "error": None,
            "started_at": started_at,
        }
        log_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        log_fh.flush()
        logger.info("[%s/%s/%s] cached — skipped", run_name, organ, case_id)
        return row

    report = report_path.read_text(encoding="utf-8")
    report_rows = _split_report_rows(report)
    t0 = time.perf_counter()
    try:
        output, _elapsed_s = run_cancer_pipeline(report=report_rows, fname=case_id)
        latency_s = round(time.perf_counter() - t0, 3)
        _atomic_write_json(out_path, output)
        row = {
            "case_id": case_id, "organ": organ, "run": run_name, "seed": seed,
            "status": "ok", "latency_s": latency_s, "parse_success": True,
            "is_cancer": bool(output.get("cancer_excision_report")),
            "cancer_category": output.get("cancer_category"),
            "error": None, "started_at": started_at,
        }
        logger.info(
            "[%s/%s/%s] ok (%.2fs, cancer=%s, category=%s)",
            run_name, organ, case_id, latency_s,
            row["is_cancer"], row["cancer_category"],
        )
    except Exception as exc:
        latency_s = round(time.perf_counter() - t0, 3)
        sentinel = {
            "_pipeline_error": True,
            "reason": type(exc).__name__,
            "message": str(exc)[:2000],
        }
        _atomic_write_json(out_path, sentinel)
        row = {
            "case_id": case_id, "organ": organ, "run": run_name, "seed": seed,
            "status": "pipeline_error", "latency_s": latency_s,
            "parse_success": False, "is_cancer": None, "cancer_category": None,
            "error": f"{type(exc).__name__}: {exc}", "started_at": started_at,
        }
        logger.error("[%s/%s/%s] pipeline error: %s",
                     run_name, organ, case_id, exc)

    log_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    log_fh.flush()
    cost_ledger.append({
        "case_id": case_id, "organ": organ, "elapsed_s": row["latency_s"],
        "status": row["status"],
    })
    return row


# --- Manifest ----------------------------------------------------------------


def _config_hash(model_id: str, kwargs: dict) -> str:
    payload = json.dumps(
        {"model": model_id, "decoding": {k: kwargs.get(k)
                                         for k in ("temperature", "top_p",
                                                   "max_tokens", "num_ctx",
                                                   "seed")}},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def update_model_manifest(
    model_dir: Path, dataset: str, model_slug_: str, model_id: str,
    run_name: str, run_summary: dict, lm_kwargs: dict,
) -> None:
    import yaml

    manifest_path = model_dir / "_manifest.yaml"
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as f:
            manifest = yaml.safe_load(f) or {}
    else:
        manifest = {}

    manifest.setdefault("experiment_id",
                        f"multirun_{model_slug_}_{dataset}_v1")
    manifest["dataset"] = dataset
    manifest["model"] = model_slug_
    manifest.setdefault("created_at",
                        dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d"))
    manifest["config_hash"] = _config_hash(model_id, lm_kwargs)

    runs = list(manifest.get("runs") or [])
    entry = {
        "run": run_name,
        "seed": lm_kwargs.get("seed"),
        "valid": run_summary["n_pipeline_error"] == 0
                 and run_summary["n_cases"] > 0,
        "parse_error_rate": (run_summary["n_pipeline_error"]
                             / max(run_summary["n_cases"], 1)),
    }
    replaced = False
    for i, r in enumerate(runs):
        if r.get("run") == run_name:
            runs[i] = entry
            replaced = True
            break
    if not replaced:
        runs.append(entry)
    runs.sort(key=lambda r: r.get("run", ""))
    manifest["runs"] = runs
    manifest["k"] = len(runs)

    _atomic_write_yaml(manifest_path, manifest)


# --- Run driver --------------------------------------------------------------


def run_single(
    run_dir: Path, run_name: str, organs: list[tuple[str, Path]],
    seed: Any, logger: logging.Logger, args: argparse.Namespace,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "run": run_name,
        "model": model_slug(args.model),
        "seed": seed,
        "dataset": args.dataset,
        "n_cases": 0, "n_ok": 0, "n_pipeline_error": 0, "n_cached": 0,
        "cancer_positive": 0,
        "per_organ": {},
        "wall_time_s": 0.0,
        "created_at": _utc_now_iso(),
    }
    cost_ledger: list[dict[str, Any]] = []
    t_run = time.perf_counter()
    log_path = run_dir / "_log.jsonl"
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
                    report_path, organ_n, run_name, seed, run_dir, log_fh,
                    cost_ledger, logger, overwrite=args.overwrite,
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
    _atomic_write_json(run_dir / "_summary.json", summary)
    _atomic_write_json(run_dir / "_cost_ledger.json", {
        "model_slug": model_slug(args.model),
        "model_id": model_list[args.model],
        "run": run_name,
        "seed": seed,
        "total_wall_s": summary["wall_time_s"],
        "n_cases": summary["n_cases"],
        "per_case": cost_ledger,
    })
    return summary


# --- Entry point -------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    aliases = _openai_aliases()
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model", required=True, choices=aliases or None,
                    help="OpenAI model alias from models.common.model_list "
                         "(must resolve to 'openai/...'). Available: "
                         + (", ".join(aliases) if aliases else "(none)"))
    ap.add_argument("--folder", dest="experiment_root", required=False,
                    default=None, type=resolve_folder,
                    help="Experiment root containing data/ and results/. "
                         "Shorthand 'dummy', 'workspace', or 'obfustrated' "
                         "resolves against the repo root; absolute paths or "
                         "other relative paths are accepted too.")
    ap.add_argument("--dataset", required=True, choices=DATASETS,
                    help="Dataset name under data/ (cmuh or tcga). The "
                         "rebuttal sweep is tcga (public corpus).")
    ap.add_argument("--run", default=None,
                    help="Run slot name, e.g. run01..run10 "
                         "(default: next free slot under the model dir).")
    ap.add_argument("--organs", nargs="*", default=None,
                    help="Only run these numeric organ directories, e.g. 1 2 "
                         "(default: every organ subdir of reports/).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap cases per organ (debugging).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Reprocess cases even if a valid output exists.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Decoding seed forwarded to dspy.LM. Default: the "
                         "value from MODEL_PROFILES / _BASE_KWARGS (10).")
    ap.add_argument("--tolerate-errors", action="store_true",
                    help="Always exit 0 if the script completes, even when "
                         "some cases failed. Default: non-zero on any error.")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="Set console log level to DEBUG.")
    return ap.parse_args(argv)


def run_with_args(
    args: argparse.Namespace, overrides: dict | None = None,
) -> int:
    if args.model not in model_list:
        print(f"error: unknown --model {args.model!r}. "
              f"Valid keys: {', '.join(model_list.keys())}", file=sys.stderr)
        return 2

    model_id = model_list[args.model]
    if not model_id.startswith("openai/"):
        print(f"error: --model {args.model!r} resolves to {model_id!r}, "
              f"which is not an OpenAI-hosted model. Use the Ollama runner "
              f"for local models.", file=sys.stderr)
        return 2

    if args.experiment_root is None:
        print("error: --folder is required (use 'dummy', 'workspace', "
              "'workspace_obfustrated', or an absolute path).",
              file=sys.stderr)
        return 2

    # Fail fast if the API key cannot be located, before discovering cases.
    try:
        load_openai_key()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
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

    slug = model_slug(args.model)
    model_dir = (args.experiment_root / "results" / "predictions"
                 / args.dataset / "llm" / slug)

    if args.run:
        if not re.fullmatch(r"run\d{2}(-[a-z0-9][a-z0-9-]*)?", args.run):
            print(f"error: --run must look like 'run01' or 'run01-<machine>' "
                  f"(got {args.run!r})", file=sys.stderr)
            return 2
        run_name = args.run
    else:
        try:
            run_name = pick_next_run(model_dir)
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    run_dir = model_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(
        name=PIPELINE_LOGGER_NAME,
        level=logging.DEBUG if args.verbose else logging.INFO,
        log_file=str(run_dir / "_run.log"),
        json_format=False,
    )
    logger.info("experiment_root: %s", args.experiment_root)
    logger.info("dataset: %s", args.dataset)
    logger.info("model: %s (%s) → slug=%s", args.model, model_id, slug)
    logger.info("run: %s", run_name)
    logger.info("run dir: %s", run_dir)
    logger.info("organs: %s", [o[0] for o in organs])

    # Merge --seed (CLI) into overrides without clobbering wrapper-supplied
    # values.
    effective_overrides = dict(overrides or {})
    if args.seed is not None:
        effective_overrides["seed"] = int(args.seed)
    if effective_overrides:
        logger.info("decoding overrides: %s", effective_overrides)

    setup_pipeline(args.model, overrides=effective_overrides)
    lm_kwargs = compute_lm_kwargs(args.model, overrides=effective_overrides)

    started_at = _utc_now_iso()
    t_run = time.perf_counter()
    try:
        summary = run_single(
            run_dir, run_name, organs, lm_kwargs.get("seed"), logger, args,
        )
    finally:
        finished_at = _utc_now_iso()

    update_model_manifest(
        model_dir, args.dataset, slug, model_id,
        run_name, summary, lm_kwargs,
    )

    _atomic_write_json(run_dir / "_run_meta.json", {
        "run": run_name,
        "model_key": args.model,
        "model_id": model_id,
        "model_slug": slug,
        "dataset": args.dataset,
        "experiment_root": str(args.experiment_root.resolve()),
        "organs": [o[0] for o in organs],
        "decoding_mode": "openai_loose_json",
        "pipeline": "loose",  # pipeline.py, not pipeline_dspy_strict
        "started_at": started_at,
        "finished_at": finished_at,
        "dspy_lm_kwargs": lm_kwargs,
        "git_sha": _git_sha(REPO_ROOT),
        "python": platform.python_version(),
        "host": socket.gethostname(),
        "argv": sys.argv,
    })

    wall_s = int(time.perf_counter() - t_run)
    wall_fmt = (f"{wall_s // 3600:02d}:{(wall_s % 3600) // 60:02d}:"
                f"{wall_s % 60:02d}")
    summary_line = (
        f"OK={summary['n_ok']} ERR={summary['n_pipeline_error']} "
        f"CACHED={summary['n_cached']} N={summary['n_cases']} "
        f"WALL={wall_fmt}"
    )
    logger.info(summary_line)
    print(summary_line)
    print(f"run dir: {run_dir}")
    print(f"manifest: {model_dir / '_manifest.yaml'}")

    if summary["n_pipeline_error"] > 0 and not args.tolerate_errors:
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_with_args(args, overrides=None)


if __name__ == "__main__":
    sys.exit(main())
