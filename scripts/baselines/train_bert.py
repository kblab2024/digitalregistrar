#!/usr/bin/env python3
"""ClinicalBERT — canonical-layout trainer for CLS and QA heads.

Trains the CLS and/or QA head on **every CMUH gold case** by default.
TCGA is held out so it can serve as the cross-corpus evaluation set
against LLMs (which, via the OpenAI API, can only see TCGA for privacy
reasons). There is no train/test split inside a corpus — disjointness
is guaranteed by the CMUH/TCGA dataset boundary, and the predict step
enforces it via the ``datasets`` field on the saved checkpoint.

Output: checkpoints under ``ckpts/`` (default), or wherever ``--ckpt-*``
points. The predict step (``run_bert.py``) loads from these paths.

Usage
-----
    # Default canonical training: every CMUH gold case (15 epochs each head).
    python scripts/baselines/train_bert.py \\
        --heads cls qa --seed 42

    # Pooled training (ablation only — destroys TCGA's held-out status):
    python scripts/baselines/train_bert.py \\
        --datasets cmuh tcga --heads cls qa

    # Smoke train on dummy data:
    python scripts/baselines/train_bert.py \\
        --folder dummy --datasets cmuh \\
        --heads cls qa --epochs-cls 1 --epochs-qa 1

Mac note: MPS (Apple Silicon) is auto-detected; otherwise CUDA → CPU.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# MPS fallback — silenced on non-Apple platforms.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from _config_loader import resolve_folder  # noqa: E402

from digital_registrar_research.benchmarks import organs as _organs  # noqa: E402
from digital_registrar_research.benchmarks.baselines._data import (  # noqa: E402
    load_cases, per_dataset_counts,
)

DATASETS = tuple(_organs.all_datasets())
DEFAULT_TRAIN_DATASETS = ("cmuh",)  # Privacy-driven: TCGA stays held-out for the
                                     # cross-corpus baseline against OpenAI-API LLMs.
DEFAULT_HEADS = ("cls", "qa")
# Cross-corpus organ scope = TCGA ∩ CMUH. Both folds train/predict on these
# 5 organs so the BERT vs LLM cross-corpus comparison is apples-to-apples.
DEFAULT_ORGANS = _organs.common_organs("cmuh", "tcga")
LOGGER_NAME = "scripts.baselines.train_bert"


def _device_pick() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def train_cls(args: argparse.Namespace, logger: logging.Logger) -> None:
    from digital_registrar_research.benchmarks.baselines import clinicalbert_cls

    device = _device_pick()
    clinicalbert_cls.DEVICE = device
    logger.info("[cls] device=%s ckpt=%s", device, args.ckpt_cls)

    Path(args.ckpt_cls).parent.mkdir(parents=True, exist_ok=True)
    train_args = SimpleNamespace(
        ckpt=str(args.ckpt_cls),
        epochs=args.epochs_cls,
        organs=",".join(args.organs),
        datasets=",".join(args.datasets),
        data_root=str(args.experiment_root),
        included_only=args.included_only,
        seed=args.seed,
    )
    clinicalbert_cls.train(train_args)


def train_qa(args: argparse.Namespace, logger: logging.Logger) -> None:
    from digital_registrar_research.benchmarks.baselines import clinicalbert_qa

    device = _device_pick()
    clinicalbert_qa.DEVICE = device
    logger.info("[qa] device=%s ckpt=%s", device, args.ckpt_qa)

    Path(args.ckpt_qa).mkdir(parents=True, exist_ok=True)
    train_args = SimpleNamespace(
        ckpt=str(args.ckpt_qa),
        epochs=args.epochs_qa,
        organs=",".join(args.organs),
        datasets=",".join(args.datasets),
        data_root=str(args.experiment_root),
        seed=args.seed,
    )
    clinicalbert_qa.train(train_args)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--folder", dest="experiment_root", default="workspace",
                    type=resolve_folder,
                    help="Experiment root (default: workspace; dummy / abs path).")
    ap.add_argument("--datasets", nargs="+", default=list(DEFAULT_TRAIN_DATASETS),
                    choices=DATASETS,
                    help="Datasets to train on (default: cmuh only — TCGA is "
                         "held out for cross-corpus evaluation against the LLM, "
                         "which can only see TCGA via OpenAI API). Pass "
                         "'cmuh tcga' for pooled training (ablation).")
    ap.add_argument("--heads", nargs="+", default=list(DEFAULT_HEADS),
                    choices=("cls", "qa"),
                    help="Heads to train (default: both).")
    ap.add_argument("--organs", nargs="*", default=list(DEFAULT_ORGANS),
                    help="Cancer-category names to keep.")
    ap.add_argument("--ckpt-cls", default="ckpts/clinicalbert_cls.pt",
                    help="Output path for the CLS checkpoint.")
    ap.add_argument("--ckpt-qa", default="ckpts/clinicalbert_qa",
                    help="Output dir for the QA checkpoint.")
    ap.add_argument("--epochs-cls", type=int, default=15)
    ap.add_argument("--epochs-qa", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for head init + data-shuffle order. "
                         "Drives BERT training-noise multirun: each seed produces "
                         "a different checkpoint. See train_bert_multirun.py.")
    ap.add_argument("--included-only", action="store_true",
                    help="CLS only: drop cases where cancer_excision_report=False.")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args(argv)


def _print_corpus_summary(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Pre-flight: log per-dataset / per-organ training-corpus counts.

    Walks the gold-annotation tree under ``{folder}/data/{dataset}/`` and
    prints how many cases the encoder will see, broken down by organ_n.
    Fails fast if any requested dataset has no gold annotations.
    """
    logger.info("=" * 60)
    logger.info("Training corpus summary (cross-corpus: train CMUH, predict TCGA)")
    logger.info("=" * 60)
    organs = set(args.organs)
    total = 0
    for ds in args.datasets:
        cases = load_cases(
            datasets=[ds],
            root=args.experiment_root,
            organs=organs,
        )
        n = len(cases)
        total += n
        if n == 0:
            raise SystemExit(
                f"refusing to train: no gold annotations under "
                f"{args.experiment_root / 'data' / ds / 'annotations' / 'gold'}. "
                f"Populate the dataset (or use --folder dummy after "
                f"`python scripts/data/gen_dummy_skeleton.py --out dummy --clean`)."
            )
        logger.info("[%s] cases=%d", ds, n)
        per_organ: dict[str, int] = {}
        for c in cases:
            per_organ[c["organ_n"]] = per_organ.get(c["organ_n"], 0) + 1
        for organ_n in sorted(per_organ):
            logger.info("  organ %s: cases=%d", organ_n, per_organ[organ_n])
    logger.info("=" * 60)
    logger.info("pooled training cases: %d", total)
    logger.info("=" * 60)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(LOGGER_NAME)
    logger.info("experiment_root: %s", args.experiment_root)
    logger.info("datasets: %s", args.datasets)
    logger.info("heads: %s", args.heads)

    _print_corpus_summary(args, logger)

    if "cls" in args.heads:
        train_cls(args, logger)
    if "qa" in args.heads:
        train_qa(args, logger)
    return 0


if __name__ == "__main__":
    sys.exit(main())
