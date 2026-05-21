"""Shared pytest fixtures for the obfuscator test suite."""
from __future__ import annotations

import sys
from pathlib import Path

# Make ``obfuscator`` importable when running pytest from the obfuscator/ dir
# without an editable install.
_OBFUSCATOR_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_OBFUSCATOR_SRC) not in sys.path:
    sys.path.insert(0, str(_OBFUSCATOR_SRC))
