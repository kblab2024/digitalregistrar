#!/usr/bin/env python3
"""Train K ClinicalBERT seeds and predict each into a runNN/ slot.

Mirrors the LLM K-sweep template (``scripts/pipeline/run_dspy_strict_ollama_multirun.py``):
each iteration draws a fresh 31-bit seed, trains both heads, runs
inference into the canonical ``runNN/`` slot, then deletes the
checkpoint to keep transient disk to ~1× the per-seed footprint.

Why this exists
---------------
LLM variance is *inference-time* sampling on fixed weights. BERT
variance is *training-time* (random head init, data-shuffle order,
dropout). To put BERT on the same statistical footing as the LLM
K-sweep, we repeat training K times with different seeds and feed each
seed's predictions into the cascade pipeline as a separate run_id.
The cascade ``chapter_multirun_reliability`` reductions and the
``cli compare`` paired-bootstrap / McNemar / FDR machinery treat the
two variance sources identically as a run-level random effect.

Layout produced
---------------
::

    {folder}/results/predictions/{dataset}/clinicalbert/
        cls/run01..runK/{organ}/{case}.json
        qa/run01..runK/{organ}/{case}.json
        merged/run01..runK/{organ}/{case}.json
        merged/_manifest.yaml             # one entry per seed

The ``merged`` head is the canonical headline view; the cls/qa
sibling trees are kept for diagnostic completeness. Cascade discovery
(``Paths.discover_runs(model='merged', method='clinicalbert')``)
auto-finds run01..runK after the multi-run unlock in
``scripts/eval/_common/paths.py`` and ``scripts/eval/cascade/run_cascade.py``.

Usage
-----
::

    # Smoke (2 seeds × 1 epoch each, dummy data — ~3 min total)
    python scripts/baselines/train_bert_multirun.py \\
        --folder dummy --num-runs 2 --master-seed 1234 \\
        --epochs-cls 1 --epochs-qa 1 --datasets tcga

    # Full K=10 × 15 epochs on workspace (~6.5–7 hr on A6000 Ada)
    python scripts/baselines/train_bert_multirun.py \\
        --folder workspace --num-runs 10 --master-seed 42 --datasets tcga

``--master-seed`` makes the *sequence* of per-iteration seeds
reproducible: same master_seed → same K seeds → same K checkpoints
(modulo CUDA non-determinism, which is intentional). Without it each
seed is drawn from ``secrets.randbelow(2**31)``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import random
import secrets
import shutil
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

from baselines import run_bert as _run_bert  # noqa: E402
from baselines import train_bert as _train_bert  # noqa: E402

SEED_RANGE = 2**31  # match _config_loader's "random" token / LLM multirun
DEFAULT_DATASETS = ("tcga",)
DEFAULT_NUM_RUNS = 10
LOGGER_NAME = "scripts.baselines.train_bert_multirun"

logger = logging.getLogger(LOGGER_NAME)


def draw_seeds(n: int, master_seed: int | None) -> list[int]:
    """Return a list of ``n`` 31-bit seeds. Mirrors the LLM multirun helper.

    With ``master_seed`` set, the sequence is fully deterministic
    (``random.Random(master_seed).randrange``). Without it each seed is
    drawn from ``secrets.randbelow`` so unrelated invocations don't
    collide.
    """
    if n <= 0:
        return []
    if master_seed is None:
        return [secrets.randbelow(SEED_RANGE) for _ in range(n)]
    rng = random.Random(master_seed)
    return [rng.randrange(SEED_RANGE) for _ in range(n)]


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except Exception:
        pass
    return None


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


def update_manifest(
    merged_dir: Path, dataset: str, master_seed: int | None,
    run_name: str, seed: int, run_summary: dict[str, Any],
    epochs_cls: int, epochs_qa: int,
) -> None:
    """Append/replace one entry in the K-seed manifest under merged/.

    Idempotent: re-running the same run_name overwrites in place.
    """
    import yaml

    manifest_path = merged_dir / "_manifest.yaml"
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as f:
            manifest = yaml.safe_load(f) or {}
    else:
        manifest = {}

    manifest.setdefault("experiment_id", f"bert_multirun_{dataset}_v1")
    manifest["dataset"] = dataset
    manifest["model"] = "clinicalbert"
    manifest["head"] = "merged"
    manifest.setdefault(
        "created_at",
        dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d"),
    )
    manifest["master_seed"] = master_seed
    manifest["epochs_cls"] = epochs_cls
    manifest["epochs_qa"] = epochs_qa
    manifest["git_sha"] = _git_sha()

    runs = list(manifest.get("runs") or [])
    entry = {
        "run": run_name,
        "seed": int(seed),
        "wall_time_s": run_summary.get("wall_time_s"),
        "n_cases": run_summary.get("n_cases"),
    }
    for i, r in enumerate(runs):
        if r.get("run") == run_name:
            runs[i] = entry
            break
    else:
        runs.append(entry)
    runs.sort(key=lambda r: r.get("run", ""))
    manifest["runs"] = runs
    manifest["k"] = len(runs)

    _atomic_write_yaml(manifest_path, manifest)


def _train_one(
    seed: int, ckpt_root: Path, args: argparse.Namespace,
) -> None:
    ckpt_root.mkdir(parents=True, exist_ok=True)
    train_argv = [
        "--folder", str(args.experiment_root),
        "--datasets", *args.train_datasets,
        "--heads", "cls", "qa",
        "--seed", str(seed),
        "--epochs-cls", str(args.epochs_cls),
        "--epochs-qa", str(args.epochs_qa),
        "--ckpt-cls", str(ckpt_root / "cls.pt"),
        "--ckpt-qa", str(ckpt_root / "qa"),
    ]
    if args.organs:
        train_argv += ["--organs", *args.organs]
    if args.included_only:
        train_argv += ["--included-only"]
    if args.verbose:
        train_argv += ["-v"]
    rc = _train_bert.main(train_argv)
    if rc != 0:
        raise SystemExit(f"train_bert returned rc={rc} for seed={seed}")


def _predict_one(
    run_name: str, ckpt_root: Path, args: argparse.Namespace,
) -> None:
    predict_argv = [
        "--folder", str(args.experiment_root),
        "--datasets", *args.datasets,
        "--heads", "cls", "qa", "merged",
        "--ckpt-cls", str(ckpt_root / "cls.pt"),
        "--ckpt-qa", str(ckpt_root / "qa"),
        "--run-id", run_name,
    ]
    if args.organs:
        predict_argv += ["--organs", *args.organs]
    if args.overwrite:
        predict_argv += ["--overwrite"]
    if args.verbose:
        predict_argv += ["-v"]
    rc = _run_bert.main(predict_argv)
    if rc != 0:
        raise SystemExit(f"run_bert returned rc={rc} for {run_name}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--folder", dest="experiment_root", required=True,
                    type=resolve_folder,
                    help="Experiment root containing data/ and results/. "
                         "Shorthand 'dummy' or 'workspace' resolves against "
                         "the repo root.")
    ap.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS),
                    help="Dataset(s) to predict on (default: tcga — held "
                         "out from CMUH-only training).")
    ap.add_argument("--train-datasets", nargs="+", default=["cmuh"],
                    help="Dataset(s) to train on (default: cmuh; cross-corpus "
                         "contract). Forwarded to train_bert.py --datasets.")
    ap.add_argument("--num-runs", type=int, default=DEFAULT_NUM_RUNS,
                    help="Number of seeds (default: 10).")
    ap.add_argument("--master-seed", type=int, default=None,
                    help="If set, the K per-iteration seeds are drawn from "
                         "random.Random(master_seed) so the whole sweep is "
                         "reproducible. Default: secrets.randbelow per seed.")
    ap.add_argument("--epochs-cls", type=int, default=15)
    ap.add_argument("--epochs-qa", type=int, default=15)
    ap.add_argument("--organs", nargs="*", default=None,
                    help="Forwarded to train_bert.py and run_bert.py.")
    ap.add_argument("--included-only", action="store_true",
                    help="CLS only: drop cases where cancer_excision_report=False.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Reprocess cases even if a valid output exists.")
    ap.add_argument("--ckpt-root", type=Path, default=None,
                    help="Where to stage per-seed checkpoints "
                         "(default: <experiment_root>/ckpts_multirun). "
                         "Each seed's checkpoint is deleted after inference.")
    ap.add_argument("--keep-checkpoints", action="store_true",
                    help="Don't delete per-seed checkpoints after inference. "
                         "Default behaviour deletes them — predictions are the "
                         "durable artifact and seeds are reproducible from "
                         "master_seed + git_sha.")
    ap.add_argument("--tolerate-errors", action="store_true",
                    help="Continue the sweep even if a seed fails. "
                         "Default: stop on the first failure.")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.num_runs <= 0:
        logger.error("--num-runs must be >= 1 (got %d)", args.num_runs)
        return 2

    seeds = draw_seeds(args.num_runs, args.master_seed)
    logger.info("multirun: folder=%s train_datasets=%s predict_datasets=%s",
                args.experiment_root, args.train_datasets, args.datasets)
    logger.info("multirun: K=%d master_seed=%s seeds=%s",
                args.num_runs, args.master_seed, seeds)

    ckpt_root_base = args.ckpt_root or (args.experiment_root / "ckpts_multirun")

    iterations: list[dict[str, Any]] = []
    overall_t0 = time.perf_counter()
    last_rc = 0

    for i, seed in enumerate(seeds, start=1):
        run_name = f"run{i:02d}"
        ckpt_root = ckpt_root_base / run_name
        logger.info("=== iter %d/%d run=%s seed=%d ===",
                    i, args.num_runs, run_name, seed)
        t0 = time.perf_counter()
        try:
            _train_one(seed, ckpt_root, args)
            _predict_one(run_name, ckpt_root, args)
            rc = 0
        except SystemExit as exc:
            if exc.code is None:
                rc = 0
            elif isinstance(exc.code, int):
                rc = exc.code
            else:
                # Non-int code is a message string; surface it.
                logger.error("iter %d inner exit: %s", i, exc.code)
                rc = 1
        except Exception as exc:
            logger.exception("iter %d raised %s: %s",
                             i, type(exc).__name__, exc)
            rc = 1
        wall_s = round(time.perf_counter() - t0, 1)
        iterations.append(
            {"i": i, "run": run_name, "seed": int(seed),
             "rc": rc, "wall_s": wall_s},
        )
        last_rc = rc

        if rc == 0:
            # Update the manifest under each predict-dataset's merged dir.
            for ds in args.datasets:
                merged_dir = (
                    args.experiment_root / "results" / "predictions"
                    / ds / "clinicalbert" / "merged"
                )
                update_manifest(
                    merged_dir, dataset=ds, master_seed=args.master_seed,
                    run_name=run_name, seed=seed,
                    run_summary={"wall_time_s": wall_s},
                    epochs_cls=args.epochs_cls, epochs_qa=args.epochs_qa,
                )

        if not args.keep_checkpoints:
            shutil.rmtree(ckpt_root, ignore_errors=True)

        logger.info("=== iter %d/%d done rc=%d wall=%.1fs ===",
                    i, args.num_runs, rc, wall_s)
        if rc != 0 and not args.tolerate_errors:
            logger.error("stopping early (iter %d rc=%d). "
                         "pass --tolerate-errors to keep going.", i, rc)
            break

    overall_wall = round(time.perf_counter() - overall_t0, 1)
    ok = sum(1 for it in iterations if it["rc"] == 0)
    failed = len(iterations) - ok
    logger.info("MULTIRUN K=%d/%d OK=%d FAILED=%d WALL=%.1fs",
                len(iterations), args.num_runs, ok, failed, overall_wall)
    for it in iterations:
        logger.info("  iter %2d: run=%s seed=%-11d rc=%d wall=%.1fs",
                    it["i"], it["run"], it["seed"], it["rc"], it["wall_s"])

    if failed and not args.tolerate_errors:
        return last_rc or 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
