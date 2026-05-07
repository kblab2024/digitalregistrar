"""Default handler for unknown file types — record in skipped manifest, do not write."""
from __future__ import annotations

from pathlib import Path

from ..manifest import Manifests, sha256_of_file


def handle(src: Path, manifests: Manifests, reason: str = "unknown file type") -> None:
    try:
        sha = sha256_of_file(src)
    except OSError:
        sha = None
    manifests.record_skipped(src, reason, sha256=sha)
