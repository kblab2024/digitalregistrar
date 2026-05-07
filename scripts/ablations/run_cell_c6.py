#!/usr/bin/env python3
"""Cell C6 wrapper — Minimal raw prompt (degenerate prompting baseline).

Backed by ``digital_registrar_research.ablations.runners.minimal_prompt``.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO_ROOT  # noqa: E402,F401

from digital_registrar_research.ablations.runners.minimal_prompt import (  # noqa: E402
    main as runner_main,
)

if __name__ == "__main__":
    runner_main()
