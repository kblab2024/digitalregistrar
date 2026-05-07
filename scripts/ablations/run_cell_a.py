#!/usr/bin/env python3
"""Cell-A wrapper — copy modular-DSPy predictions into the canonical ablations tree.

Cell A is the parent project's pipeline run end-to-end with the modular
per-organ DSPy signatures. The runner copies the per-case JSONs from
the canonical pipeline output tree
(``{folder}/results/predictions/{dataset}/llm/{model_slug}/{run}/...``)
into the canonical ablations tree
(``{folder}/results/ablations/{dataset}/dspy_modular/{model_slug}/{run}/...``)
so the aggregator can score them alongside the other cells.

Usage::

    python scripts/ablations/run_cell_a.py --folder dummy --dataset tcga --model gptoss
    python scripts/ablations/run_cell_a.py --folder workspace --dataset tcga \\
        --model gptoss --source-run run01
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO_ROOT  # noqa: E402,F401

from digital_registrar_research.ablations.runners.reuse_baseline import (  # noqa: E402
    main as runner_main,
)

if __name__ == "__main__":
    sys.exit(runner_main())
