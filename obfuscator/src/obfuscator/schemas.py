"""Loads per-organ JSON schemas from the inference repo and the organ_n -> organ_name mapping."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# obfuscator/src/obfuscator/schemas.py -> parents[3] = repo root
REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMAS_DIR = REPO_ROOT / "src" / "digital_registrar_research" / "schemas" / "data"
ORGAN_CODE_YAML = REPO_ROOT / "configs" / "organ_code.yaml"

CANCER_CATEGORIES = (
    "stomach", "colorectal", "breast", "esophagus", "lung",
    "prostate", "thyroid", "pancreas", "cervix", "liver", "others",
)


@lru_cache(maxsize=None)
def organ_code_map() -> dict[str, dict[int, str]]:
    """Return {dataset: {organ_n: organ_name}} from configs/organ_code.yaml."""
    with ORGAN_CODE_YAML.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    out: dict[str, dict[int, str]] = {}
    for dataset, mapping in raw["organ_code"].items():
        out[dataset] = {int(k): str(v) for k, v in mapping.items()}
    return out


def organ_name(dataset: str, organ_n: str | int) -> str | None:
    return organ_code_map().get(dataset, {}).get(int(organ_n))


@lru_cache(maxsize=None)
def load_schema(organ: str) -> dict[str, Any]:
    """Load the cancer_data schema for one organ (Draft 2020-12)."""
    path = SCHEMAS_DIR / f"{organ}.json"
    if not path.is_file():
        raise FileNotFoundError(f"No schema for organ {organ!r} at {path}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def list_available_organs() -> list[str]:
    return sorted(p.stem for p in SCHEMAS_DIR.glob("*.json") if p.stem != "common")


def validate_cancer_data(organ: str, cancer_data: dict) -> list[str]:
    """Return [] if valid, else list of error messages."""
    import jsonschema

    schema = load_schema(organ)
    errors: list[str] = []
    validator = jsonschema.Draft202012Validator(schema)
    for err in sorted(validator.iter_errors(cancer_data), key=lambda e: list(e.path)):
        errors.append(f"{list(err.path)}: {err.message}")
    return errors
