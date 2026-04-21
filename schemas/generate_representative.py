#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
schemas/generate_representative.py
Generates representative JSON Schema files from canonical Pydantic models.

Run from within the digitalregistrar/ directory:
    python schemas/generate_representative.py

Output: schemas/representative/<organ>.json for each organ + common.
Each JSON Schema file nests per-signature schemas under a key named after
the DSPy Signature class (e.g., "PancreasCancerNonnested"), with shared
$defs collected at the document root.
"""

import json
import sys
from pathlib import Path

# Ensure the digitalregistrar/ directory is on sys.path (same as pipeline.py)
HERE = Path(__file__).resolve().parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from schemas.canonical.common import IsCancerSchema
from schemas.canonical.breast import (
    BreastCancerNonnestedSchema, DCISSchema, BreastCancerGradingSchema,
    BreastCancerStagingSchema, BreastCancerMarginsSchema,
    BreastCancerLNSchema, BreastCancerBiomarkersSchema,
)
from schemas.canonical.lung import (
    LungCancerNonnestedSchema, LungCancerStagingSchema,
    LungCancerMarginsSchema, LungCancerLNSchema,
    LungCancerBiomarkersSchema, LungCancerOthernestedSchema,
)
from schemas.canonical.pancreas import (
    PancreasCancerNonnestedSchema, PancreasCancerStagingSchema,
    PancreasCancerMarginsSchema, PancreasCancerLNSchema,
)
from schemas.canonical.colon import (
    ColonCancerNonnestedSchema, ColonCancerStagingSchema,
    ColonCancerMarginsSchema, ColonCancerLNSchema, ColonCancerBiomarkersSchema,
)
from schemas.canonical.esophagus import (
    EsophagusCancerNonnestedSchema, EsophagusCancerStagingSchema,
    EsophagusCancerMarginsSchema, EsophagusCancerLNSchema,
)
from schemas.canonical.stomach import (
    StomachCancerNonnestedSchema, StomachCancerStagingSchema,
    StomachCancerMarginsSchema, StomachCancerLNSchema,
)
from schemas.canonical.prostate import (
    ProstateCancerNonnestedSchema, ProstateCancerStagingSchema,
    ProstateCancerMarginsSchema, ProstateCancerLNSchema,
)
from schemas.canonical.thyroid import (
    ThyroidCancerNonnestedSchema, ThyroidCancerStagingSchema,
    ThyroidCancerMarginsSchema, ThyroidCancerLNSchema,
)
from schemas.canonical.cervix import (
    CervixCancerNonnestedSchema, CervixCancerStagingSchema,
    CervixCancerMarginsSchema, CervixCancerLNSchema,
)
from schemas.canonical.liver import (
    LiverCancerNonnestedSchema, LiverCancerExtentSchema,
    LiverCancerVascularInvasionSchema, LiverCancerStagingSchema,
    LiverCancerMarginsSchema, LiverCancerLNSchema,
)
from schemas.canonical.bladder import BladderCancerNonnestedSchema

# Maps each organ to its signatures in pipeline execution order
ORGAN_SCHEMA_MAP: dict[str, list] = {
    "common": [IsCancerSchema],
    "breast": [
        BreastCancerNonnestedSchema, DCISSchema, BreastCancerGradingSchema,
        BreastCancerStagingSchema, BreastCancerMarginsSchema,
        BreastCancerLNSchema, BreastCancerBiomarkersSchema,
    ],
    "lung": [
        LungCancerNonnestedSchema, LungCancerStagingSchema,
        LungCancerMarginsSchema, LungCancerLNSchema,
        LungCancerBiomarkersSchema, LungCancerOthernestedSchema,
    ],
    "pancreas": [
        PancreasCancerNonnestedSchema, PancreasCancerStagingSchema,
        PancreasCancerMarginsSchema, PancreasCancerLNSchema,
    ],
    "colorectal": [
        ColonCancerNonnestedSchema, ColonCancerStagingSchema,
        ColonCancerMarginsSchema, ColonCancerLNSchema, ColonCancerBiomarkersSchema,
    ],
    "esophagus": [
        EsophagusCancerNonnestedSchema, EsophagusCancerStagingSchema,
        EsophagusCancerMarginsSchema, EsophagusCancerLNSchema,
    ],
    "stomach": [
        StomachCancerNonnestedSchema, StomachCancerStagingSchema,
        StomachCancerMarginsSchema, StomachCancerLNSchema,
    ],
    "prostate": [
        ProstateCancerNonnestedSchema, ProstateCancerStagingSchema,
        ProstateCancerMarginsSchema, ProstateCancerLNSchema,
    ],
    "thyroid": [
        ThyroidCancerNonnestedSchema, ThyroidCancerStagingSchema,
        ThyroidCancerMarginsSchema, ThyroidCancerLNSchema,
    ],
    "cervix": [
        CervixCancerNonnestedSchema, CervixCancerStagingSchema,
        CervixCancerMarginsSchema, CervixCancerLNSchema,
    ],
    "liver": [
        LiverCancerNonnestedSchema, LiverCancerExtentSchema,
        LiverCancerVascularInvasionSchema, LiverCancerStagingSchema,
        LiverCancerMarginsSchema, LiverCancerLNSchema,
    ],
    "bladder": [BladderCancerNonnestedSchema],
}


def generate_organ_schema(organ: str, schema_classes: list) -> dict:
    merged = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": f"{organ.capitalize()} Cancer Registry Data",
        "description": f"Combined JSON Schema for all {organ} cancer extraction signatures.",
        "type": "object",
        "properties": {},
        "$defs": {},
    }
    for cls in schema_classes:
        sig_name = cls.__name__.removesuffix("Schema")
        sig_schema = cls.model_json_schema()
        # Collect nested model $defs at the document root
        merged["$defs"].update(sig_schema.pop("$defs", {}))
        # Nest this signature's schema under its name key
        merged["properties"][sig_name] = {
            "title": sig_name,
            "type": "object",
            "properties": sig_schema.get("properties", {}),
        }
        if "required" in sig_schema:
            merged["properties"][sig_name]["required"] = sig_schema["required"]
    # Remove empty $defs to keep output clean
    if not merged["$defs"]:
        del merged["$defs"]
    return merged


def main():
    output_dir = Path(__file__).parent / "representative"
    output_dir.mkdir(exist_ok=True)

    for organ, schema_classes in ORGAN_SCHEMA_MAP.items():
        doc = generate_organ_schema(organ, schema_classes)
        out_path = output_dir / f"{organ}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
        print(f"Written: {out_path}")

    print(f"\nDone. {len(ORGAN_SCHEMA_MAP)} files written to {output_dir}")


if __name__ == "__main__":
    main()
