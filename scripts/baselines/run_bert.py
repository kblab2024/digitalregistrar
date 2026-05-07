#!/usr/bin/env python3
"""ClinicalBERT — canonical-layout batch runner.

Runs the CLS head (categorical / boolean fields), the QA head (numeric
span fields), and merges them into a single per-case prediction. All
three outputs land under the canonical prediction tree::

    {folder}/results/predictions/{dataset}/clinicalbert/cls/{organ_n}/<case_id>.json
    {folder}/results/predictions/{dataset}/clinicalbert/qa/{organ_n}/<case_id>.json
    {folder}/results/predictions/{dataset}/clinicalbert/merged/{organ_n}/<case_id>.json

Defaults are aligned to the cross-corpus baseline contract: BERT is
trained on CMUH (only) and predicted on the full TCGA corpus, since
TCGA is held out from training. The leakage guard in
``clinicalbert_*.predict`` refuses to run if any of the predict
datasets is in the checkpoint's training-set metadata — so the only
supported flow is "predict on a corpus disjoint from the training
corpus."

Usage
-----
    # Default: predict on TCGA (held out from CMUH training).
    python scripts/baselines/run_bert.py \\
        [--ckpt-cls ckpts/clinicalbert_cls.pt] \\
        [--ckpt-qa  ckpts/clinicalbert_qa] \\
        [--heads cls qa merged]

Output side files (under each head dir, e.g. ``clinicalbert/cls/``):
    _summary.json   per-head totals
    _run.log        per-head full-verbosity log
    _run_meta.json  provenance (git sha, ckpt path, host, argv, ...)
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
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from _config_loader import resolve_folder  # noqa: E402

from digital_registrar_research.benchmarks import organs as _organs  # noqa: E402

DATASETS = tuple(_organs.all_datasets())
DEFAULT_PREDICT_DATASETS = ("tcga",)  # Cross-corpus default: BERT trains on CMUH,
                                       # predicts on TCGA (held-out) for fair LLM
                                       # comparison via OpenAI API.
DEFAULT_HEADS = ("cls", "qa", "merged")
# Cross-corpus organ scope = TCGA ∩ CMUH (= TCGA's set, which is a subset of
# CMUH's). Sourced from configs/organ_code.yaml — do not duplicate the literal.
DEFAULT_ORGANS = _organs.common_organs("cmuh", "tcga")
LOGGER_NAME = "scripts.baselines.run_bert"


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

    # Fan out to the package logger so library modules' getLogger(__name__)
    # calls (e.g. _data._LOGGER) reach the same console + _run.log. Without
    # this, a "no reports under …" error from _walk_reports would land in
    # the root logger's default handler (or be lost) and never make it to
    # _run.log — exactly the silent-fail mode that hid the 0-cases bug.
    pkg_logger = logging.getLogger("digital_registrar_research")
    pkg_logger.handlers.clear()
    pkg_logger.setLevel(logging.DEBUG)
    pkg_logger.addHandler(console)
    pkg_logger.addHandler(fh)
    pkg_logger.propagate = False

    return logger


def _device_pick() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# --- Per-head runners --------------------------------------------------------


def run_cls(
    args: argparse.Namespace, out_dir: Path, logger: logging.Logger,
) -> dict[str, Any]:
    """Run the ClinicalBERT-CLS head; writes to <out_dir>/<organ_n>/<case_id>.json."""
    from digital_registrar_research.benchmarks.baselines import clinicalbert_cls

    if not Path(args.ckpt_cls).exists():
        raise SystemExit(
            f"CLS checkpoint missing: {args.ckpt_cls}. "
            f"Train it first via scripts/baselines/train_bert.py --head cls."
        )

    device = _device_pick()
    logger.info("[cls] device=%s ckpt=%s", device, args.ckpt_cls)
    clinicalbert_cls.DEVICE = device

    out_dir.mkdir(parents=True, exist_ok=True)
    predict_args = SimpleNamespace(
        ckpt=str(args.ckpt_cls),
        out=str(out_dir),
        organs=",".join(args.organs),
        dataset=args.dataset,
        datasets=args.dataset,
        data_root=str(args.experiment_root),
    )
    t0 = time.perf_counter()
    clinicalbert_cls.predict(predict_args)
    return {
        "head": "cls",
        "ckpt": str(args.ckpt_cls),
        "wall_time_s": round(time.perf_counter() - t0, 1),
    }


def run_qa(
    args: argparse.Namespace, out_dir: Path, logger: logging.Logger,
) -> dict[str, Any]:
    """Run the ClinicalBERT-QA head; writes to <out_dir>/<organ_n>/<case_id>.json."""
    from digital_registrar_research.benchmarks.baselines import clinicalbert_qa

    if not Path(args.ckpt_qa).exists():
        raise SystemExit(
            f"QA checkpoint missing: {args.ckpt_qa}. "
            f"Train it first via scripts/baselines/train_bert.py --head qa."
        )

    device = _device_pick()
    logger.info("[qa] device=%s ckpt=%s", device, args.ckpt_qa)
    clinicalbert_qa.DEVICE = device

    out_dir.mkdir(parents=True, exist_ok=True)
    predict_args = SimpleNamespace(
        ckpt=str(args.ckpt_qa),
        out=str(out_dir),
        organs=",".join(args.organs),
        dataset=args.dataset,
        datasets=args.dataset,
        data_root=str(args.experiment_root),
    )
    t0 = time.perf_counter()
    clinicalbert_qa.predict(predict_args)
    return {
        "head": "qa",
        "ckpt": str(args.ckpt_qa),
        "wall_time_s": round(time.perf_counter() - t0, 1),
    }


def merge_heads(cls_dir: Path, qa_dir: Path, out_dir: Path,
                logger: logging.Logger) -> dict[str, Any]:
    """Merge CLS + QA per-case predictions into ``out_dir``.

    The CLS prediction is the base (it carries cancer_category +
    cancer_excision_report). The QA cancer_data scalars overlay onto
    the CLS cancer_data (CLS wins on collisions, since CLS is
    authoritative for categorical / boolean fields).
    """
    if not cls_dir.is_dir():
        raise SystemExit(f"merge: CLS dir missing: {cls_dir}")
    if not qa_dir.is_dir():
        logger.warning("merge: QA dir missing: %s — proceeding with CLS only", qa_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    n_qa = 0
    t0 = time.perf_counter()
    for cls_file in sorted(cls_dir.rglob("*.json")):
        if cls_file.name.startswith("_"):
            continue
        rel = cls_file.relative_to(cls_dir)  # e.g. "1/tcga1_10.json"
        with cls_file.open(encoding="utf-8") as f:
            merged = json.load(f)
        qa_file = qa_dir / rel
        if qa_file.is_file():
            with qa_file.open(encoding="utf-8") as f:
                qa = json.load(f)
            merged_data = merged.setdefault("cancer_data", {})
            for k, v in (qa.get("cancer_data") or {}).items():
                merged_data.setdefault(k, v)
            n_qa += 1
        out_path = out_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        n += 1

    logger.info(
        "[merged] %d cases (%d picked up QA augment); wrote to %s",
        n, n_qa, out_dir,
    )
    return {
        "head": "merged",
        "n_cases": n, "n_qa_merged": n_qa,
        "wall_time_s": round(time.perf_counter() - t0, 1),
    }


# --- Entry point -------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--folder", dest="experiment_root", default="workspace",
                    type=resolve_folder,
                    help="Experiment root (default: workspace; dummy / abs path).")
    ap.add_argument("--datasets", nargs="+", default=list(DEFAULT_PREDICT_DATASETS),
                    choices=DATASETS,
                    help="Dataset name(s) to predict on (default: tcga — held "
                         "out from CMUH-only training). The leakage guard "
                         "refuses to predict on a dataset that was in the "
                         "checkpoint's training set.")
    ap.add_argument("--heads", nargs="+", default=list(DEFAULT_HEADS),
                    choices=("cls", "qa", "merged"),
                    help="Which heads to run. 'merged' requires both cls and "
                         "qa outputs to exist (or be run in the same call).")
    ap.add_argument("--ckpt-cls", default="ckpts/clinicalbert_cls.pt",
                    help="ClinicalBERT-CLS checkpoint path.")
    ap.add_argument("--ckpt-qa", default="ckpts/clinicalbert_qa",
                    help="ClinicalBERT-QA checkpoint dir.")
    ap.add_argument("--organs", nargs="*", default=list(DEFAULT_ORGANS),
                    help="Cancer-category names (breast, lung, ...) to keep.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Reprocess cases even if a valid output exists.")
    ap.add_argument("--run-id", default=None,
                    help="If set (e.g. 'run01'), predictions land under "
                         "{head}/{run_id}/{organ}/{case}.json instead of the "
                         "single-run {head}/{organ}/{case}.json layout. Used "
                         "by train_bert_multirun.py to write seeds into the "
                         "runNN/ shape that cascade auto-discovers.")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="Set console log level to DEBUG.")
    return ap.parse_args(argv)


def _run_one_dataset(dataset: str, args: argparse.Namespace) -> int:
    """Run cls/qa/merged for one dataset under canonical paths.

    With ``--run-id <slot>``, per-head outputs are written to
    ``{pred_root}/{head}/{slot}/{organ}/{case}.json`` so that
    ``scripts.eval.cascade.run_cascade`` can auto-discover the run via
    ``Paths.discover_runs(model={head}, method='clinicalbert')`` after
    the multi-run unlock. Without ``--run-id`` the legacy single-run
    layout is preserved.
    """
    pred_root = (args.experiment_root / "results" / "predictions"
                 / dataset / "clinicalbert")
    run_slot = args.run_id  # may be None
    if run_slot:
        cls_dir = pred_root / "cls" / run_slot
        qa_dir = pred_root / "qa" / run_slot
        merged_dir = pred_root / "merged" / run_slot
        # Side files live under the merged/ run dir so each seed has
        # its own _summary / _run_meta / _run.log without clobbering
        # siblings. merged is the canonical headline head.
        side_dir = merged_dir
    else:
        cls_dir = pred_root / "cls"
        qa_dir = pred_root / "qa"
        merged_dir = pred_root / "merged"
        side_dir = pred_root

    logger = _setup_logging(side_dir, args.verbose)
    logger.info("experiment_root: %s", args.experiment_root)
    logger.info("dataset: %s", dataset)
    logger.info("heads: %s", args.heads)
    logger.info("organs: %s", args.organs)

    # Per-dataset args view (the inner predict() functions accept a
    # single dataset).
    per_ds = argparse.Namespace(
        **{**vars(args), "dataset": dataset, "datasets": dataset},
    )

    started_at = _utc_now_iso()
    overall: dict[str, Any] = {
        "method": "clinicalbert",
        "dataset": dataset,
        "heads": list(args.heads),
        "started_at": started_at,
        "head_summaries": {},
    }
    t_run = time.perf_counter()

    if "cls" in args.heads:
        overall["head_summaries"]["cls"] = run_cls(per_ds, cls_dir, logger)
    if "qa" in args.heads:
        overall["head_summaries"]["qa"] = run_qa(per_ds, qa_dir, logger)
    if "merged" in args.heads:
        overall["head_summaries"]["merged"] = merge_heads(
            cls_dir, qa_dir, merged_dir, logger,
        )

    overall["finished_at"] = _utc_now_iso()
    overall["wall_time_s"] = round(time.perf_counter() - t_run, 1)
    if run_slot:
        overall["run_id"] = run_slot
    _atomic_write_json(side_dir / "_summary.json", overall)
    _atomic_write_json(side_dir / "_run_meta.json", {
        "method": "clinicalbert",
        "dataset": dataset,
        "experiment_root": str(args.experiment_root.resolve()),
        "heads": list(args.heads),
        "ckpt_cls": str(args.ckpt_cls),
        "ckpt_qa": str(args.ckpt_qa),
        "organs": list(args.organs),
        "run_id": run_slot,
        "started_at": started_at,
        "finished_at": overall["finished_at"],
        "git_sha": _git_sha(REPO_ROOT),
        "python": platform.python_version(),
        "host": socket.gethostname(),
        "argv": sys.argv,
    })

    print(f"[{dataset}] out dir: {side_dir}")
    return 0


def run_with_args(args: argparse.Namespace) -> int:
    overall = 0
    for dataset in args.datasets:
        overall |= _run_one_dataset(dataset, args)
    return overall


def main(argv: list[str] | None = None) -> int:
    return run_with_args(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
