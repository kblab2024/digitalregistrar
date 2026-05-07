#!/usr/bin/env python3
"""Cell F3 wrapper — Raw JSON-mode with denested (flat) per-organ schema.

Backed by ``digital_registrar_research.ablations.runners.flat_schema``.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO_ROOT  # noqa: E402,F401

from digital_registrar_research.ablations.runners.flat_schema import (  # noqa: E402
    main as runner_main,
)

if __name__ == "__main__":
    runner_main()
