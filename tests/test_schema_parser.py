"""The annotation-side schema parser (consumes the JSON schemas from data/)."""
from digital_registrar_research.annotation.parser import (
    CANCER_CATEGORIES,
    CANCER_TO_FILE,
    parse_cancer_schema,
)


def test_cancer_categories_includes_to_file_keys():
    """CANCER_CATEGORIES is the UI dropdown (None + 'others' + every organ that has a schema)."""
    for organ in CANCER_TO_FILE:
        assert organ in CANCER_CATEGORIES, f"{organ!r} missing from CANCER_CATEGORIES"


def test_parse_lung_schema_returns_field_specs():
    """Parser groups fields under SectionSpecs; lung should expose key staging + nested fields."""
    sections = parse_cancer_schema("lung")
    assert sections
    field_names = {f.name for s in sections for f in (s.flat_fields or [])}
    for required in {"procedure", "pt_category"}:
        assert required in field_names, f"lung schema missing field {required!r}"


def test_parse_breast_includes_dcis_fields():
    """DCIS is a per-field set in breast (dcis_present / dcis_grade / ...) — pin we expose them."""
    sections = parse_cancer_schema("breast")
    assert sections
    field_names = {f.name for s in sections for f in (s.flat_fields or [])}
    assert any(n.startswith("dcis_") for n in field_names), sorted(field_names)
