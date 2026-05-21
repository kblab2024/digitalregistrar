#!/usr/bin/env python3
"""Smoke-test the FACTORY pipeline (v2) end-to-end on a few reports.

Same green/red semantics as ``legacy/run_dspy_ollama_smoke.py``: the
process exits non-zero if any sampled case raises a pipeline exception.
A clean smoke run is the go/no-go for a real factory sweep.

Usage
-----
    python scripts/pipeline/run_factory_ollama_smoke.py \\
        --model gptoss --folder dummy --dataset tcga \\
        [--n 3] [--seed 0] [--organs 1 2] \\
        [--decomposition auto] [--jsonize|--no-jsonize]

Output goes under ``_smoke_<date>/`` (leading underscore so eval globs
that filter ``not name.startswith('_')`` skip it). Smoke runs never
pollute real sweeps.
"""
from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "pipeline" / "legacy"))

import run_dspy_ollama_smoke as legacy  # noqa: E402
import run_dspy_ollama_single as legacy_single  # noqa: E402

from digital_registrar.pipeline_factory import (  # noqa: E402
    run_cancer_pipeline_v2,
    setup_pipeline_v2,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    extras = argparse.ArgumentParser(add_help=False)
    extras.add_argument("--decomposition", choices=["per_group", "monolithic", "auto"], default="auto")
    extras.add_argument("--jsonize", dest="jsonize", action="store_true")
    extras.add_argument("--no-jsonize", dest="jsonize", action="store_false")
    extras.set_defaults(jsonize=False)

    extras_ns, remaining = extras.parse_known_args(argv)
    legacy_ns = legacy.parse_args(remaining)
    for k, v in vars(extras_ns).items():
        setattr(legacy_ns, k, v)
    return legacy_ns


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Patch BOTH the smoke module AND the single-run module that smoke imports
    # from — they each hold their own reference to ``run_cancer_pipeline``.
    for mod in (legacy, legacy_single):
        mod.setup_pipeline = setup_pipeline_v2
        mod.run_cancer_pipeline = partial(
            run_cancer_pipeline_v2,
            decomposition=args.decomposition,
            jsonize_enabled=args.jsonize,
        )

    return legacy.run_with_args(args) if hasattr(legacy, "run_with_args") else legacy.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
