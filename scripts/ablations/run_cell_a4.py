#!/usr/bin/env python3
"""Cell A4 wrapper — Monolithic DSPy without the is_cancer router.

Backed by ``digital_registrar_research.ablations.runners.no_router``.

The router is bypassed; the gold ``cancer_category`` field from each
annotation file is used as the assumed organ. See the runner's
docstring for why this is an upper-bound estimate of router value.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO_ROOT  # noqa: E402,F401

from digital_registrar_research.ablations.runners.no_router import (  # noqa: E402
    main as runner_main,
)

if __name__ == "__main__":
    runner_main()
