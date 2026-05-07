#!/usr/bin/env python3
"""Single-run DSPy-strict Ollama cancer-extraction experiment.

Third sibling of ``run_dspy_ollama_single.py`` and
``run_structured_ollama_single.py``. Wraps every per-organ ``dspy.Predict``
call (the decomposition the structured pipeline collapsed into one giant
schema) in :class:`dspy.ChainOfThought` and routes the LM call through
DSPy's :class:`JSONAdapter`, which posts
``response_format={"type": "json_schema", "strict": true}`` to Ollama's
OpenAI-compat endpoint. Per-organ focus + token-level schema enforcement
+ a ``reasoning`` slot before the structured body.

Layout, run-slot rotation, manifest format, and per-case JSON output are
byte-for-byte identical to the other two variants so the eval pipeline
ingests all three without modification. The only structural difference
is that the model_slug carries a ``_dspy_strict`` suffix so the
prediction tree does not collide with the existing DSPy / structured
runs.

Iterates every ``<case_id>.txt`` under
``{experiment_root}/data/{dataset}/reports/{organ_n}/`` and writes one
prediction JSON per report to
``{experiment_root}/results/predictions/{dataset}/llm/{model_slug}_dspy_strict/{run}/{organ_n}/<case_id>.json``.

Requires Ollama >=0.5 for OpenAI-compat structured outputs and DSPy >=3.

Usage
-----
    python scripts/pipeline/run_dspy_strict_ollama_single.py \\
        --model gptoss \\
        --folder dummy \\
        --dataset tcga \\
        [--run run01] [--organs 1 2] [--limit N] [--overwrite] \\
        [--tolerate-errors] [-v]

``--model`` must be one of: gptoss, gemma3, gemma4, qwen3_5, medgemmalarge,
medgemmasmall. Each alias auto-loads ``configs/dspy_ollama_{alias}.yaml``
for decoding overrides (temperature, top_p, num_ctx, max_tokens, ...).
The same config files are reused across DSPy, structured, and dspy_strict
runs by design — the ``decoding`` block is model-agnostic.

``--folder`` accepts the shorthands ``dummy`` and ``workspace`` (resolved
against the repo root) or any absolute / relative path.

Output tree
-----------
    {experiment_root}/results/predictions/{dataset}/llm/{model_slug}_dspy_strict/
        _manifest.yaml            aggregated across all runs (updated in place)
        {run}/                    e.g. run01
            _summary.json         run-level totals
            _log.jsonl            one row per case
            _run.log              full-verbosity log
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

from _config_loader import (  # noqa: E402
    load_model_config,
    resolve_folder,
    split_decoding_overrides,
)
from _run_id import format_run_id, machine_slug  # noqa: E402

from digital_registrar_research.models.common import (  # noqa: E402
    localaddr,
    model_list,
)
from digital_registrar_research.pipeline_dspy_strict import (  # noqa: E402
    load_decoding_kwargs,
    run_cancer_pipeline_dspy_strict as run_cancer_pipeline,
    setup_pipeline_dspy_strict as setup_pipeline,
)
from digital_registrar_research.util.logger import setup_logger  # noqa: E402

PIPELINE_LOGGER_NAME = "experiment_logger"  # fixed by pipeline.run_pipeline

DATASETS = ("cmuh", "tcga")
MAX_RUN_SLOTS = 10  # memory: "run01..run10"

# Unified --model aliases consumed by the consolidated runners. The legacy
# model_list keys (gpt, gemma27b, qwen30b, ...) still resolve but are not
# offered from the CLI to keep the surface small and self-documenting.
UNIFIED_MODELS = (
    "gptoss", "gemma3", "gemma4", "qwen3_5", "medgemmalarge", "medgemmasmall", "gemma4large", "qwen3_6"
)


# --- IO helpers --------------------------------------------------------------


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON via a tmp file + rename so a killed run never leaves a
    half-written ``<case_id>.json`` that would later be mistaken for a valid
    cached prediction."""
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
    import yaml  # lazy so the rest of the script imports without pyyaml

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
    """Normalize raw report text into non-empty paragraph rows."""
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
    """Canonical folder name for a dspy_strict run.

    Suffixed with ``_dspy_strict`` so the prediction tree does not collide
    with the existing DSPy / structured runs (both of which write to
    ``ollama_chat/gpt-oss:20b`` → ``gpt_oss_20b``).

    ``ollama_chat/gpt-oss:20b`` → ``gpt_oss_20b_dspy_strict``;
    ``ollama_chat/qwen3:30b``  → ``qwen3_30b_dspy_strict``.
    """
    full = model_list[model_key]
    tail = full.split("/", 1)[-1]  # drop the backend prefix
    base = re.sub(r"[-:./]", "_", tail)
    return f"{base}_dspy_strict"


# --- Discovery ---------------------------------------------------------------


def discover_organs(
    reports_root: Path, organ_filter: list[str] | None,
) -> list[tuple[str, Path]]:
    """Return sorted ``(organ_n, organ_dir)`` pairs for every numeric subdir
    under ``reports_root`` that contains at least one ``*.txt``.
    ``organ_filter`` matches against the directory *name* (e.g. "1", "2")."""
    if not reports_root.is_dir():
        return []
    picked: list[tuple[str, Path]] = []
    for child in sorted(reports_root.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        if child.name.startswith("_"):  # sidecar convention
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
    """Return the first run-id in 01..10 without a `_summary.json`.

    Names follow ``run{NN}[-{machine_slug}]`` (see ``scripts/_run_id.py``).
    The slot space is per-machine: with ``DRR_MACHINE_ID=alpha`` the
    function only ever sees/returns ``run01-alpha..run10-alpha``, so two
    machines with distinct slugs each get an independent 10-slot budget.
    Partial-but-not-finalised runs are treated as free (to allow resumption)."""
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
    logger: logging.Logger,
    *,
    overwrite: bool,
) -> dict[str, Any]:
    """Predict for one case.

    ``out_dir`` is the per-run directory whose contents are written as
    ``{out_dir}/{organ}/<case_id>.json`` — i.e. the organ subdir is
    preserved inside the run, matching the canonical predictions layout."""
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
    except Exception as exc:  # OpenAI / Ollama / parse errors surface here
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
    """Read-modify-write the model-level ``_manifest.yaml`` so each new run
    appends/updates its own entry without clobbering prior runs."""
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
    """Execute one full pass over all discovered cases. Returns the
    ``_summary.json`` payload."""
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
                    logger, overwrite=args.overwrite,
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
    return summary


# --- Entry point -------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model", required=True, choices=UNIFIED_MODELS,
                    help="Model alias: one of " + ", ".join(UNIFIED_MODELS) +
                         ". Each alias auto-loads "
                         "configs/dspy_ollama_{alias}.yaml (if present) for "
                         "decoding overrides — the same config tree as the "
                         "DSPy / structured variants; the decoding block is "
                         "model-agnostic.")
    ap.add_argument("--folder", dest="experiment_root", required=False, default=None,
                    type=resolve_folder,
                    help="Experiment root containing data/ and results/. "
                         "Shorthand 'dummy', 'workspace', or 'obfustrated' "
                         "resolves against the repo root; absolute paths or other "
                         "relative paths are accepted too. Required unless "
                         "--obfustrated is set.")
    ap.add_argument("--obfustrated", action="store_true",
                    help="Shortcut for --folder obfustrated (= workspace_obfustrated/). "
                         "Use to debug pipeline against synthetic PHI-free data; "
                         "explicit --folder always wins.")
    ap.add_argument("--dataset", required=True, choices=DATASETS,
                    help="Dataset name under data/ (cmuh or tcga).")
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
    ap.add_argument("--tolerate-errors", action="store_true",
                    help="Always exit 0 if the script completes, even when "
                         "some cases failed. Default: non-zero on any error.")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="Set console log level to DEBUG.")
    return ap.parse_args(argv)


def run_with_args(
    args: argparse.Namespace, overrides: dict | None = None,
) -> int:
    """Execute a full single-run given a prebuilt argparse.Namespace and an
    optional decoding-overrides dict. Factored out of ``main`` so the
    multirun wrapper can reuse the same body without re-invoking argparse."""
    if args.model not in model_list:
        print(f"error: unknown --model {args.model!r}. "
              f"Valid keys: {', '.join(model_list.keys())}", file=sys.stderr)
        return 2

    # Resolve --obfustrated shortcut into args.experiment_root if --folder absent.
    if args.experiment_root is None:
        if getattr(args, "obfustrated", False):
            args.experiment_root = resolve_folder("obfustrated")
        else:
            print("error: --folder is required (use 'dummy', 'workspace', "
                  "'workspace_obfustrated', or an absolute path), or pass "
                  "--obfustrated for the synthetic workspace.", file=sys.stderr)
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
    logger.info("model: %s (%s) → slug=%s",
                args.model, model_list[args.model], slug)
    logger.info("run: %s", run_name)
    logger.info("run dir: %s", run_dir)
    logger.info("organs: %s", [o[0] for o in organs])
    if overrides:
        logger.info("decoding overrides: %s", overrides)

    setup_pipeline(args.model, overrides=overrides)  # DSPy LM + JSONAdapter
    lm_kwargs = load_decoding_kwargs(args.model, overrides=overrides)

    started_at = _utc_now_iso()
    t_run = time.perf_counter()
    try:
        summary = run_single(
            run_dir, run_name, organs, lm_kwargs.get("seed"), logger, args,
        )
    finally:
        finished_at = _utc_now_iso()

    # Update the model-level manifest idempotently
    update_model_manifest(
        model_dir, args.dataset, slug, model_list[args.model],
        run_name, summary, lm_kwargs,
    )

    # Stamp some single-run-only provenance next to the summary so it's
    # recoverable after the fact. Kept under a leading underscore so it
    # doesn't get picked up as a prediction.
    _atomic_write_json(run_dir / "_run_meta.json", {
        "run": run_name,
        "model_key": args.model,
        "model_id": model_list[args.model],
        "model_slug": slug,
        "dataset": args.dataset,
        "experiment_root": str(args.experiment_root.resolve()),
        "organs": [o[0] for o in organs],
        "ollama_endpoint": localaddr,
        "decoding_mode": "dspy_strict_json_schema",
        "pipeline": "dspy_strict",
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
    cfg = load_model_config(args.model)
    overrides = split_decoding_overrides(cfg.get("decoding"))
    return run_with_args(args, overrides)


if __name__ == "__main__":
    sys.exit(main())
