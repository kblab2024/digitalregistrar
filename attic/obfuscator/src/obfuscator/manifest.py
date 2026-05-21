"""Outputs/skipped manifests written at the end of an obfuscation run."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Manifests:
    out_root: Path
    outputs: list[dict] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)

    def record_output(self, src: Path | None, dst: Path, handler: str,
                      extra: dict | None = None) -> None:
        entry = {
            "dst": str(dst.relative_to(self.out_root)),
            "handler": handler,
        }
        if src is not None:
            try:
                entry["src"] = str(src)
            except Exception:
                pass
        if extra:
            entry.update(extra)
        self.outputs.append(entry)

    def record_skipped(self, src: Path, reason: str, *,
                       sha256: str | None = None) -> None:
        entry = {
            "src": str(src),
            "reason": reason,
        }
        if sha256:
            entry["sha256"] = sha256
        self.skipped.append(entry)

    def write(self, master_seed: int) -> None:
        meta = {
            "obfuscator_version": "0.1.0",
            "master_seed": master_seed,
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "n_outputs": len(self.outputs),
            "n_skipped": len(self.skipped),
        }
        self.out_root.mkdir(parents=True, exist_ok=True)
        outputs_path = self.out_root / "_obfuscator_outputs.json"
        skipped_path = self.out_root / "_obfuscator_skipped.json"
        with outputs_path.open("w", encoding="utf-8") as f:
            json.dump({"meta": meta, "outputs": self.outputs}, f, indent=2)
        with skipped_path.open("w", encoding="utf-8") as f:
            json.dump({"meta": meta, "skipped": self.skipped}, f, indent=2)
