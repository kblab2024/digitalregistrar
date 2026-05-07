#!/usr/bin/env python3
"""K-seed multirun wrapper for the OpenAI cancer-extraction runner.

Sibling of ``run_dspy_strict_ollama_multirun.py``; only difference is
the inner single-run script — this one drives the loose-JSON
``pipeline.py`` against an OpenAI-hosted model. Each iteration draws a
fresh 31-bit seed, lets the inner script auto-pick the next free
``runNN`` slot, and writes its prediction tree under the canonical
``{experiment_root}/results/predictions/{dataset}/llm/{model_slug}/
runNN/`` layout. The model-level ``_manifest.yaml`` is updated
atomically by each inner call, so the K-sweep accumulates cleanly into a
single manifest with one entry per run (each carrying its own seed and
parse-error rate).

Why this exists
---------------
For the rebuttal vs reviewer (a) we treat ``gpt-5.4-mini`` as "yet
another top-class model" and want the same N=10 stochastic-sweep shape
the local Ollama baselines and the K-seed ClinicalBERT baseline use, so
ICC, Cronbach α, accuracy_flip_rate, and paired bootstrap are
comparable across all four model families.

Usage
-----
    python scripts/pipeline/run_pipeline_openai_multirun.py \\
        --model gpt5_4_mini \\
        --folder workspace \\
        --dataset tcga \\
        --n 10 [--master-seed 42] \\
        [--organs 1 2] [--limit N] [--overwrite] \\
        [--tolerate-errors] [-v]

``--master-seed`` makes the *sequence* of per-iteration seeds
reproducible: the same master seed always produces the same N seeds.
Without it, each iteration's seed is drawn from
``secrets.randbelow(2**31)``.
"""
from __future__ import annotations

import argparse
import random
import secrets
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "pipeline"))

from _config_loader import resolve_folder  # noqa: E402

from run_pipeline_openai_single import (  # noqa: E402
    DATASETS,
    MAX_RUN_SLOTS,
    _openai_aliases,
    run_with_args,
)

SEED_RANGE = 2**31


def draw_seeds(n: int, master_seed: int | None) -> list[int]:
    if n <= 0:
        return []
    if master_seed is None:
        return [secrets.randbelow(SEED_RANGE) for _ in range(n)]
    rng = random.Random(master_seed)
    return [rng.randrange(SEED_RANGE) for _ in range(n)]


def build_inner_args(args: argparse.Namespace) -> argparse.Namespace:
    """Project the wrapper's Namespace onto the field shape that
    ``run_pipeline_openai_single.run_with_args`` expects. ``run`` is
    left unset so ``pick_next_run`` rotates through free slots; ``seed``
    is left unset because the wrapper supplies it via ``overrides``.
    """
    return argparse.Namespace(
        model=args.model,
        experiment_root=args.experiment_root,
        dataset=args.dataset,
        run=None,
        organs=args.organs,
        limit=args.limit,
        overwrite=args.overwrite,
        seed=None,
        tolerate_errors=args.tolerate_errors,
        verbose=args.verbose,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    aliases = _openai_aliases()
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model", required=True, choices=aliases or None,
                    help="OpenAI alias from models.common.model_list. "
                         "Forwarded verbatim to run_pipeline_openai_single. "
                         "Available: "
                         + (", ".join(aliases) if aliases else "(none)"))
    ap.add_argument("--folder", dest="experiment_root", required=True,
                    type=resolve_folder,
                    help="Experiment root containing data/ and results/.")
    ap.add_argument("--dataset", required=True, choices=DATASETS,
                    help="Dataset name under data/ (cmuh or tcga).")
    ap.add_argument("--n", type=int, default=10,
                    help="Number of consecutive runs to execute "
                         "(default: 10; capped to MAX_RUN_SLOTS=%d slots)."
                         % MAX_RUN_SLOTS)
    ap.add_argument("--master-seed", type=int, default=None,
                    help="If set, the sequence of per-iteration seeds is "
                         "drawn from random.Random(master_seed) so the whole "
                         "sweep is reproducible. Default: each iteration "
                         "draws a fresh seed via secrets.randbelow.")
    ap.add_argument("--organs", nargs="*", default=None,
                    help="Forwarded to run_pipeline_openai_single.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap cases per organ (debugging).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Reprocess cases even if a valid output exists.")
    ap.add_argument("--tolerate-errors", action="store_true",
                    help="Continue the sweep even if an iteration returns "
                         "non-zero. Default: stop on the first failure.")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.n <= 0:
        print(f"error: --n must be >= 1 (got {args.n})", file=sys.stderr)
        return 2
    if args.n > MAX_RUN_SLOTS:
        print(f"error: --n={args.n} exceeds MAX_RUN_SLOTS={MAX_RUN_SLOTS}; "
              f"the inner script only allocates run01..run{MAX_RUN_SLOTS:02d}.",
              file=sys.stderr)
        return 2

    seeds = draw_seeds(args.n, args.master_seed)

    print(f"multirun: model={args.model} dataset={args.dataset} "
          f"folder={args.experiment_root}")
    print(f"multirun: n={args.n} master_seed={args.master_seed} "
          f"seeds={seeds}")

    iterations: list[dict] = []
    overall_t0 = time.perf_counter()
    last_rc = 0
    for i, seed in enumerate(seeds, start=1):
        overrides = {"seed": int(seed)}
        inner_args = build_inner_args(args)
        print(f"\n=== multirun iteration {i}/{args.n} (seed={seed}) ===")
        t0 = time.perf_counter()
        try:
            rc = run_with_args(inner_args, overrides=overrides)
        except SystemExit as exc:
            rc = int(exc.code) if exc.code is not None else 1
        except Exception as exc:
            print(f"multirun: iteration {i} raised "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)
            rc = 1
        wall_s = round(time.perf_counter() - t0, 1)
        iterations.append({"i": i, "seed": int(seed), "rc": rc,
                           "wall_s": wall_s})
        last_rc = rc
        print(f"=== multirun iteration {i}/{args.n} done "
              f"rc={rc} wall={wall_s}s ===")
        if rc != 0 and not args.tolerate_errors:
            print(f"multirun: stopping early (iteration {i} rc={rc}; "
                  f"pass --tolerate-errors to keep going).", file=sys.stderr)
            break

    overall_wall = round(time.perf_counter() - overall_t0, 1)
    ok_runs = sum(1 for it in iterations if it["rc"] == 0)
    failed_runs = len(iterations) - ok_runs
    print(f"\nMULTIRUN K={len(iterations)}/{args.n} OK_RUNS={ok_runs} "
          f"FAILED_RUNS={failed_runs} WALL={overall_wall}s")
    for it in iterations:
        print(f"  iter {it['i']:>2}: seed={it['seed']:<11} "
              f"rc={it['rc']} wall={it['wall_s']}s")

    if failed_runs and not args.tolerate_errors:
        return last_rc or 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
