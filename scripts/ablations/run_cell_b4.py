#!/usr/bin/env python3
"""Cell B4 wrapper — Constrained decoding via outlines.

Backed by ``digital_registrar_research.ablations.runners.constrained_decoding``.
Requires ``outlines`` installed; pass ``--backend vllm`` (recommended)
or ``--backend hf`` / ``--backend openai`` (for Ollama, prompt-level
schema only).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO_ROOT  # noqa: E402,F401

from digital_registrar_research.ablations.runners.constrained_decoding import (  # noqa: E402
    main as runner_main,
)

if __name__ == "__main__":
    runner_main()
