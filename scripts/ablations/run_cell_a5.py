#!/usr/bin/env python3
"""Cell A5 wrapper — Per-section decomposition.

Backed by ``digital_registrar_research.ablations.runners.per_section``.
Splits each report via heuristic section markers and runs a section-
specialised DSPy signature on each slice.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO_ROOT  # noqa: E402,F401

from digital_registrar_research.ablations.runners.per_section import (  # noqa: E402
    main as runner_main,
)

if __name__ == "__main__":
    runner_main()
