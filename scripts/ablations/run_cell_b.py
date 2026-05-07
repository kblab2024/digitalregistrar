#!/usr/bin/env python3
"""Cell-B wrapper — DSPy + monolithic single-signature per organ.

Backed by ``digital_registrar_research.ablations.runners.dspy_monolithic``.

Usage::

    python scripts/ablations/run_cell_b.py --folder dummy --dataset tcga --model gptoss
    python scripts/ablations/run_cell_b.py --folder dummy --dataset tcga \\
        --model gptoss --skip-jsonize
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO_ROOT  # noqa: E402,F401

from digital_registrar_research.ablations.runners.dspy_monolithic import (  # noqa: E402
    main as runner_main,
)

if __name__ == "__main__":
    sys.exit(runner_main())
