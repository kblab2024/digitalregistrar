"""Verbatim file copy for files known to be PHI-free (configs, .gitkeep, .placeholder)."""
from __future__ import annotations

import shutil
from pathlib import Path

from ..manifest import Manifests


def handle(src: Path, dst: Path, manifests: Manifests) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    manifests.record_output(src, dst, handler="passthrough")
