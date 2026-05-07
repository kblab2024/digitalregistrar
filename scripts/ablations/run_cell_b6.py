#!/usr/bin/env python3
"""Cell B6 wrapper — Free-text generation + regex post-extractor (degenerate baseline).

Backed by ``digital_registrar_research.ablations.runners.free_text_regex``.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO_ROOT  # noqa: E402,F401

from digital_registrar_research.ablations.runners.free_text_regex import (  # noqa: E402
    main as runner_main,
)

if __name__ == "__main__":
    runner_main()
