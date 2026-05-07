#!/usr/bin/env python
"""Convenience wrapper to launch the annotation Streamlit UI.

Equivalent to `registrar-annotate` (the installed console script). Useful when
running from a checkout without `pip install`-ing the package.
"""
import subprocess
import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[2] / "src" / "digital_registrar_research" / "annotation" / "app.py"

if __name__ == "__main__":
    sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(APP), *sys.argv[1:]]))
