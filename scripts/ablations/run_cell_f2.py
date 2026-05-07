#!/usr/bin/env python3
"""Cell F2 wrapper — Raw JSON-mode against a single union schema.

Backed by ``digital_registrar_research.ablations.runners.union_schema``.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO_ROOT  # noqa: E402,F401

from digital_registrar_research.ablations.runners.union_schema import (  # noqa: E402
    main as runner_main,
)

if __name__ == "__main__":
    runner_main()
