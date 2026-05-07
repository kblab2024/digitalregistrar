"""Strip model weight blobs (.pt/.bin/.safetensors) — write a .placeholder text file."""
from __future__ import annotations

from pathlib import Path

from ..manifest import Manifests, sha256_of_file


def handle(src: Path, dst: Path, manifests: Manifests) -> None:
    placeholder_path = dst.with_suffix(dst.suffix + ".placeholder")
    placeholder_path.parent.mkdir(parents=True, exist_ok=True)
    src_sha = sha256_of_file(src)
    text = (f"Original {src.name} stripped during obfuscation.\n"
            f"BERT/torch code paths will not run against workspace_obfustrated/.\n"
            f"src_sha256={src_sha}\nsrc_bytes={src.stat().st_size}\n")
    placeholder_path.write_text(text, encoding="utf-8")
    manifests.record_output(src, placeholder_path, handler="blob_to_placeholder",
                            extra={"src_sha256": src_sha,
                                   "src_bytes_discarded": src.stat().st_size})
    manifests.record_skipped(src, "model weight blob — replaced by .placeholder",
                             sha256=src_sha)
