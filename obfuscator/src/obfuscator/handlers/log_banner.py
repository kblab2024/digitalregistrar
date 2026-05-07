"""Discard log content, write a single-line banner.

Pipeline tracebacks sometimes embed report excerpts in errors — too risky to keep.
"""
from __future__ import annotations

from pathlib import Path

from ..manifest import Manifests, sha256_of_file

_BANNER = ("[obfustrated workspace — original log discarded; "
           "rerun pipeline against workspace_obfustrated/ to regenerate]\n")


def handle(src: Path, dst: Path, manifests: Manifests) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(_BANNER, encoding="utf-8")
    manifests.record_output(src, dst, handler="log_banner",
                            extra={"src_sha256": sha256_of_file(src),
                                   "src_bytes_discarded": src.stat().st_size})
