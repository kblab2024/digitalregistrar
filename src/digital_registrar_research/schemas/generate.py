"""Regenerate `data/<organ>.json` from the canonical Pydantic case-models.

Usage:
    python -m digital_registrar_research.schemas.generate          # write
    python -m digital_registrar_research.schemas.generate --check  # verify only; exit 1 on drift

CI runs `--check` on every PR so a change to a DSPy signature that would
change the generated schema can't silently diverge from the checked-in
JSON artifact that downstream tools (annotation UI, ablations raw-JSON
runner) consume.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pydantic import CASE_MODELS, IsCancerCase

SCHEMAS_DATA_DIR = Path(__file__).resolve().parent / "data"

# Per-organ models + the top-level routing case-model; everything but `bladder`,
# which remains a hand-curated annotation-UI schema (no DSPy signature pipeline yet).
ALL_MODELS: dict[str, type] = {**CASE_MODELS, "common": IsCancerCase}


def _render_model(model: type) -> str:
    """Return JSON-schema text for a Pydantic model (sorted keys + 2-space indent)."""
    schema = model.model_json_schema()
    return json.dumps(schema, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def _write_all() -> list[Path]:
    SCHEMAS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, model in sorted(ALL_MODELS.items()):
        out_path = SCHEMAS_DATA_DIR / f"{name}.json"
        out_path.write_text(_render_model(model), encoding="utf-8")
        written.append(out_path)
    return written


def _check_all() -> list[str]:
    """Return a list of names whose on-disk JSON doesn't match the current Pydantic models."""
    drifted: list[str] = []
    for name, model in sorted(ALL_MODELS.items()):
        on_disk_path = SCHEMAS_DATA_DIR / f"{name}.json"
        if not on_disk_path.exists():
            drifted.append(f"{name} (missing file)")
            continue
        on_disk = on_disk_path.read_text(encoding="utf-8")
        fresh = _render_model(model)
        if on_disk.strip() != fresh.strip():
            drifted.append(name)
    return drifted


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="Verify only; fail on drift.")
    args = ap.parse_args()

    if args.check:
        drifted = _check_all()
        if drifted:
            print(f"[schemas] DRIFT detected in: {', '.join(drifted)}", file=sys.stderr)
            print("[schemas] Run `python -m digital_registrar_research.schemas.generate` to regenerate.", file=sys.stderr)
            sys.exit(1)
        print(f"[schemas] OK — all {len(ALL_MODELS)} schemas match their Pydantic models.")
        return

    written = _write_all()
    print(f"[schemas] Wrote {len(written)} schemas to {SCHEMAS_DATA_DIR}:")
    for p in written:
        print(f"  - {p.name}")


if __name__ == "__main__":
    main()
