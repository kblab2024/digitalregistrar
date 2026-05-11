#!/usr/bin/env python
"""Build the Digital Registrar Annotator bundle for a target platform.

Replaces the matrix of `build_<os>_<mode>[_kpc].{bat,sh}` wrappers. Each
of those wrappers is now a thin shim that sets two env vars
(``PLATFORM`` and ``ANNOTATOR_SET``) and execs ``_build_common.ps1``
(Windows host) or ``_build_common.sh`` (Unix host). This script does the
same job from one place — preferred entry point for new callers.

Examples
--------
    python packaging/build.py --platform windows --annotators single
    python packaging/build.py --platform windows --annotators single_kpc
    python packaging/build.py --platform unix    --annotators multi
    python packaging/build.py --platform macos   --annotators single_kpc

The bundle output lands at ``packaging/dist/digital-registrar-annotator-<platform>-<annotators>.<ext>``
where ``<ext>`` is ``.zip`` on Windows and ``.tar.gz`` on Unix/macOS.
"""
from __future__ import annotations

import argparse
import os
import platform as _platform
import subprocess
import sys
from pathlib import Path

PACKAGING_DIR = Path(__file__).resolve().parent

PLATFORMS = ("windows", "unix", "macos")
ANNOTATORS = ("single", "multi", "single_kpc")


def _common_script_for(host_os: str) -> Path:
    """Choose the right helper based on the *host* (not target) OS."""
    if host_os == "windows":
        return PACKAGING_DIR / "_build_common.ps1"
    return PACKAGING_DIR / "_build_common.sh"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the annotator bundle.")
    ap.add_argument(
        "--platform", "--os", required=True, choices=PLATFORMS,
        help="Target platform for the bundle.")
    ap.add_argument(
        "--annotators", required=True, choices=ANNOTATORS,
        help="Annotator-set baked into the bundle's defaults.json.")
    args = ap.parse_args(argv)

    host = "windows" if _platform.system() == "Windows" else "unix"
    helper = _common_script_for(host)
    if not helper.is_file():
        ap.error(f"Helper script missing: {helper}")

    env = os.environ.copy()
    env["PLATFORM"] = args.platform
    env["ANNOTATOR_SET"] = args.annotators

    if host == "windows":
        cmd = [
            "powershell.exe", "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-File", str(helper),
        ]
    else:
        cmd = ["bash", str(helper)]

    return subprocess.call(cmd, env=env, cwd=str(PACKAGING_DIR))


if __name__ == "__main__":
    sys.exit(main())
