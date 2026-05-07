#!/usr/bin/env python3
"""Chain N consecutive ``run_dspy_ollama_single`` runs with random seeds.

Each iteration draws a fresh 31-bit seed, lets the single-run script
auto-pick the next free ``runNN`` slot, and writes its prediction tree
under the canonical ``{experiment_root}/results/predictions/{dataset}/
llm/{model_slug}/runNN/`` layout. The model-level ``_manifest.yaml`` is
updated atomically by each inner call, so the K-sweep accumulates
cleanly into a single manifest with one entry per run (each carrying its
own seed, parse-error rate, and validity flag).

Why this exists
---------------
``run_dspy_ollama_single`` does one pass with the seed baked into
``configs/dspy_ollama_<alias>.yaml`` (or the base default ``10``).
Running a stochastic K-sweep used to mean a shell loop that mutated the
config file between iterations. This wrapper replaces that loop.

Usage
-----
    python scripts/run_dspy_ollama_multirun.py \\
        --model gptoss \\
        --folder dummy \\
        --dataset tcga \\
        --n 5 [--master-seed 1234] \\
        [--organs 1 2] [--limit N] [--overwrite] \\
        [--tolerate-errors] [-v]

``--master-seed`` makes the *sequence* of per-iteration seeds
reproducible: the same master seed always produces the same N seeds, so
"redo last week's K-sweep" is one flag. Without it, each iteration's
seed is drawn from ``secrets.randbelow(2**31)`` (matching the
``decoding.seed: random`` token already supported by
``scripts/_config_loader.py``).
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
sys.path.insert(0, str(REPO_ROOT / "scripts"))  # for _config_loader, _run_id
sys.path.insert(0, str(REPO_ROOT / "scripts" / "pipeline"))  # for sibling imports

from _config_loader import (  # noqa: E402
    load_model_config,
    resolve_folder,
    split_decoding_overrides,
)
from run_dspy_ollama_single import (  # noqa: E402
    DATASETS,
    MAX_RUN_SLOTS,
    UNIFIED_MODELS,
    run_with_args,
)

SEED_RANGE = 2**31  # match _config_loader's "random" token


def draw_seeds(n: int, master_seed: int | None) -> list[int]:
    """Return a list of ``n`` 31-bit seeds.

    With ``master_seed`` set, the sequence is fully deterministic
    (``random.Random(master_seed).randrange``). Without it, each seed is
    drawn independently from ``secrets`` so two unrelated invocations do
    not collide. Kept top-level so callers can sanity-check the seed
    sequence without invoking the LM.
    """
    if n <= 0:
        return []
    if master_seed is None:
        return [secrets.randbelow(SEED_RANGE) for _ in range(n)]
    rng = random.Random(master_seed)
    return [rng.randrange(SEED_RANGE) for _ in range(n)]


def build_inner_args(args: argparse.Namespace) -> argparse.Namespace:
    """Project the wrapper's Namespace onto the field shape that
    ``run_dspy_ollama_single.run_with_args`` expects. ``run`` is left
    unset so ``pick_next_run`` rotates through free slots."""
    return argparse.Namespace(
        model=args.model,
        experiment_root=args.experiment_root,
        dataset=args.dataset,
        run=None,
        organs=args.organs,
        limit=args.limit,
        overwrite=args.overwrite,
        tolerate_errors=args.tolerate_errors,
        verbose=args.verbose,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model", required=True, choices=UNIFIED_MODELS,
                    help="Model alias: one of " + ", ".join(UNIFIED_MODELS) +
                         ". Forwarded verbatim to run_dspy_ollama_single.")
    ap.add_argument("--folder", dest="experiment_root", required=True,
                    type=resolve_folder,
                    help="Experiment root containing data/ and results/. "
                         "Shorthand 'dummy' or 'workspace' resolves against "
                         "the repo root.")
    ap.add_argument("--dataset", required=True, choices=DATASETS,
                    help="Dataset name under data/ (cmuh or tcga).")
    ap.add_argument("--n", type=int, default=5,
                    help="Number of consecutive runs to execute "
                         "(default: 5; capped to MAX_RUN_SLOTS=%d slots)."
                         % MAX_RUN_SLOTS)
    ap.add_argument("--master-seed", type=int, default=None,
                    help="If set, the sequence of per-iteration seeds is "
                         "drawn from random.Random(master_seed) so the whole "
                         "sweep is reproducible. Default: each iteration "
                         "draws a fresh seed via secrets.randbelow.")
    ap.add_argument("--organs", nargs="*", default=None,
                    help="Forwarded to run_dspy_ollama_single.")
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

    cfg = load_model_config(args.model)
    base_overrides = split_decoding_overrides(cfg.get("decoding"))
    seeds = draw_seeds(args.n, args.master_seed)

    print(f"multirun: model={args.model} dataset={args.dataset} "
          f"folder={args.experiment_root}")
    print(f"multirun: n={args.n} master_seed={args.master_seed} "
          f"seeds={seeds}")

    iterations: list[dict] = []
    overall_t0 = time.perf_counter()
    last_rc = 0
    for i, seed in enumerate(seeds, start=1):
        overrides = {**base_overrides, "seed": int(seed)}
        inner_args = build_inner_args(args)
        print(f"\n=== multirun iteration {i}/{args.n} (seed={seed}) ===")
        t0 = time.perf_counter()
        try:
            rc = run_with_args(inner_args, overrides)
        except SystemExit as exc:
            rc = int(exc.code) if exc.code is not None else 1
        except Exception as exc:  # don't let one bad run kill the sweep
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
