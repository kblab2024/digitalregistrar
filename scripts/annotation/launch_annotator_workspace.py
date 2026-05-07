#!/usr/bin/env python
"""Launch the canonical-layout annotator UI targeting ``workspace/``.

Equivalent to the installed console script ``registrar-annotate-workspace``.
Sets ``REGISTRAR_ANNOTATE_BASE_DIR`` to the repo-root ``workspace/`` folder
and spawns ``streamlit run`` on ``app_canonical.py``.
"""
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
APP = REPO_ROOT / "src" / "digital_registrar_research" / "annotation" / "app_canonical.py"

if __name__ == "__main__":
    env = os.environ.copy()
    env.setdefault("REGISTRAR_ANNOTATE_BASE_DIR", str(REPO_ROOT / "workspace"))
    sys.exit(subprocess.call(
        [sys.executable, "-m", "streamlit", "run", str(APP), *sys.argv[1:]],
        env=env,
    ))
