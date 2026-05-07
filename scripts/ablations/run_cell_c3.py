#!/usr/bin/env python3
"""Cell C3 wrapper — Monolithic DSPy with 5 in-context demos per organ.

Backed by ``digital_registrar_research.ablations.runners.fewshot_demos``
with ``--n-shots 5`` injected.

Usage::

    python scripts/ablations/run_cell_c3.py --folder dummy --dataset tcga --model gptoss
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO_ROOT  # noqa: E402,F401

from digital_registrar_research.ablations.runners.fewshot_demos import (  # noqa: E402
    main as runner_main,
)

if __name__ == "__main__":
    if "--n-shots" not in sys.argv:
        sys.argv.extend(["--n-shots", "5"])
    sys.exit(runner_main())
