#!/usr/bin/env python3
"""Pre-flight smoke for the FACTORY pipeline (v2).

Drop-in replacement for ``legacy/run_inference_smoke.py``: same green/red
semantics — fail loudly if any sampled report breaks the v2 pipeline.

Usage
-----
    python scripts/pipeline/run_factory_inference_smoke.py \\
        --model gptoss \\
        --experiment-root dummy \\
        [--decomposition auto] [--jsonize|--no-jsonize]
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

import run_inference_smoke as legacy  # noqa: E402

from digital_registrar_research.pipeline_factory import (  # noqa: E402
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
    legacy.run_cancer_pipeline = partial(
        run_cancer_pipeline_v2,
        decomposition=args.decomposition,
        jsonize_enabled=args.jsonize,
    )
    if hasattr(legacy, "setup_pipeline"):
        legacy.setup_pipeline = setup_pipeline_v2
    return legacy.main(argv if argv is not None else sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
