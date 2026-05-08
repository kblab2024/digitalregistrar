#!/usr/bin/env python3
"""Single-run cancer-extraction with the FACTORY pipeline (v2) over OpenAI.

Drop-in replacement for ``legacy/run_pipeline_openai_single.py``: same CLI
shape and output tree. Differences mirror ``run_factory_ollama_single.py``:
  * pipeline import is ``digital_registrar_research.pipeline_factory``;
  * extra flags ``--decomposition`` and ``--jsonize/--no-jsonize``;
  * manifest is tagged ``"pipeline": "factory"``.

Usage
-----
    python scripts/pipeline/run_factory_openai_single.py \\
        --model gpt5_4_mini --folder workspace --dataset tcga \\
        [--decomposition auto] [--jsonize|--no-jsonize] [--run run01]
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

import run_pipeline_openai_single as legacy  # noqa: E402

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

    legacy.setup_pipeline = setup_pipeline_v2
    legacy.run_cancer_pipeline = partial(
        run_cancer_pipeline_v2,
        decomposition=args.decomposition,
        jsonize_enabled=args.jsonize,
    )

    real_write = legacy._atomic_write_json

    def tagged_write(path, payload):
        if isinstance(payload, dict) and (
            path.name.endswith("_summary.json") or path.name == "_manifest.yaml"
        ) and payload.get("pipeline") != "factory":
            payload = {**payload, "pipeline": "factory"}
        real_write(path, payload)

    legacy._atomic_write_json = tagged_write
    return legacy.run_with_args(args)


if __name__ == "__main__":
    raise SystemExit(main())
