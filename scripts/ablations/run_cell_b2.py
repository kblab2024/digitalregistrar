#!/usr/bin/env python3
"""Cell B2 wrapper — DSPy with str outputs + post-hoc parser.

Backed by ``digital_registrar_research.ablations.runners.str_outputs``.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO_ROOT  # noqa: E402,F401

from digital_registrar_research.ablations.runners.str_outputs import (  # noqa: E402
    main as runner_main,
)

if __name__ == "__main__":
    runner_main()
