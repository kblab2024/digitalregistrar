#!/usr/bin/env python3
"""Single-run cancer-extraction with the FACTORY pipeline (v2) over Ollama.

Drop-in replacement for ``legacy/run_dspy_ollama_single.py``: same CLI shape,
same output tree, same manifest fields. Differences:
  * pipeline import is :mod:`digital_registrar_research.pipeline_factory`;
  * two extra CLI flags expose the factory's decomposition + jsonize knobs:
      ``--decomposition {per_group,monolithic,auto}``  (default: ``auto``)
      ``--jsonize`` / ``--no-jsonize``                  (default: ``--no-jsonize``)
  * the per-case manifest tags ``"pipeline": "factory"`` so downstream eval
    can tell which engine produced each prediction tree.

Usage
-----
    python scripts/pipeline/run_factory_ollama_single.py \\
        --model gptoss --folder dummy --dataset tcga \\
        [--run run01] [--organs 1 2] [--limit N] \\
        [--decomposition auto] [--jsonize|--no-jsonize] \\
        [--tolerate-errors] [-v]
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

# Reuse the heavy machinery from the legacy single-run script.
import run_dspy_ollama_single as legacy  # noqa: E402

from digital_registrar_research.pipeline_factory import (  # noqa: E402
    run_cancer_pipeline_v2,
    setup_pipeline_v2,
)


_FACTORY_HELP_BANNER = (
    "[factory wrapper] extra flags: "
    "--decomposition {per_group,monolithic,auto} (default: auto), "
    "--jsonize / --no-jsonize (default: --no-jsonize). "
    "Legacy flags listed below.\n"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the legacy CLI plus our two factory-only flags."""
    extras = argparse.ArgumentParser(add_help=False)
    extras.add_argument(
        "--decomposition",
        choices=["per_group", "monolithic", "auto"],
        default="auto",
    )
    extras.add_argument("--jsonize", dest="jsonize", action="store_true")
    extras.add_argument("--no-jsonize", dest="jsonize", action="store_false")
    extras.set_defaults(jsonize=False)

    if argv is None:
        argv = sys.argv[1:]
    if "-h" in argv or "--help" in argv:
        print(_FACTORY_HELP_BANNER)
        # Delegate to legacy --help so users see the full surface.
    extras_ns, remaining = extras.parse_known_args(argv)
    legacy_ns = legacy.parse_args(remaining)
    for k, v in vars(extras_ns).items():
        setattr(legacy_ns, k, v)
    return legacy_ns


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Swap the legacy module's pipeline symbols. ``process_case`` calls these
    # by name; rebinding here flows through to every per-case invocation.
    legacy.setup_pipeline = setup_pipeline_v2
    legacy.run_cancer_pipeline = partial(
        run_cancer_pipeline_v2,
        decomposition=args.decomposition,
        jsonize_enabled=args.jsonize,
    )

    # Tag manifests/summaries with the engine identity so eval can distinguish.
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
