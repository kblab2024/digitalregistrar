#!/usr/bin/env python
"""Thin wrapper so `python scripts/obfuscate_workspace.py` works without `pip install -e obfuscator/`.

Adds ``obfuscator/src`` to sys.path then delegates to ``obfuscator.cli:main``.
"""
from __future__ import annotations

import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
_OBFUSCATOR_SRC = _REPO_ROOT / "obfuscator" / "src"
if str(_OBFUSCATOR_SRC) not in sys.path:
    sys.path.insert(0, str(_OBFUSCATOR_SRC))

from obfuscator.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
