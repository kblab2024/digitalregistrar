"""Build-time: dump `{organ: [[section_name, [field_names]]]}` to JSON.

The standalone Windows bundle ships the output of this script next to
`annotation/parser.py` so the runtime doesn't need DSPy or the Pydantic
case-model builder — see parser._load_precomputed_groups.

Run from the repo root with the dev venv active:

    python packaging/precompute_section_groups.py <output_json_path>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: precompute_section_groups.py <output.json>", file=sys.stderr)
        sys.exit(2)
    out_path = Path(sys.argv[1])

    from digital_registrar_research.models.modellist import organmodels
    from digital_registrar_research.schemas.pydantic import _builder

    module_globals = sys.modules[_builder.__name__].__dict__

    result: dict[str, list[list]] = {}
    for organ, sig_names in organmodels.items():
        groups: list[list] = []
        seen: set[str] = set()
        for sig_name in sig_names:
            cls = module_globals.get(sig_name)
            if cls is None:
                raise RuntimeError(
                    f"Signature class {sig_name!r} missing from {_builder.__name__}; "
                    "check the wildcard imports in schemas.pydantic._builder."
                )
            field_names: list[str] = []
            for name, _type, _fi in _builder._iter_signature_output_fields(cls):
                if name in seen:
                    continue
                seen.add(name)
                field_names.append(name)
            if field_names:
                groups.append([sig_name, field_names])
        result[organ] = groups

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    total_fields = sum(len(f) for g in result.values() for _, f in g)
    print(
        f"Wrote {out_path} — {len(result)} organs, "
        f"{sum(len(g) for g in result.values())} sections, "
        f"{total_fields} fields"
    )


if __name__ == "__main__":
    main()
