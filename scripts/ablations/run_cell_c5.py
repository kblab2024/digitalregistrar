#!/usr/bin/env python3
"""Cell C5 wrapper — Compiled DSPy program (BootstrapFewShotWithRandomSearch).

Backed by ``digital_registrar_research.ablations.runners.compiled_dspy``.
Requires a one-time compile step::

    python scripts/ablations/compile_dspy.py --model gpt \\
        --out workspace/compiled/dspy_compiled_gpt-oss.json

Then point this wrapper at the compiled artifact via ``--compiled``.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO_ROOT  # noqa: E402,F401

from digital_registrar_research.ablations.runners.compiled_dspy import (  # noqa: E402
    main as runner_main,
)

if __name__ == "__main__":
    runner_main()
