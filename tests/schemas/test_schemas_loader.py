"""Public API of `digital_registrar_research.schemas`."""
import pytest

from digital_registrar_research.schemas import (
    CASE_MODELS,
    list_organs,
    load_json_schema,
    load_pydantic_model,
)


def test_list_organs_is_sorted_and_nonempty():
    organs = list_organs()
    assert organs == sorted(organs)
    assert len(organs) >= 10


def test_load_pydantic_model_round_trip():
    Lung = load_pydantic_model("lung")
    instance = Lung(procedure="lobectomy", grade=2)
    assert instance.procedure == "lobectomy"
    assert instance.grade == 2
    # Unspecified fields default to None.
    assert instance.histology is None


def test_load_pydantic_model_unknown_organ():
    with pytest.raises(KeyError, match="Unknown organ"):
        load_pydantic_model("does_not_exist")


def test_load_json_schema_returns_dict_with_properties():
    schema = load_json_schema("lung")
    assert isinstance(schema, dict)
    assert schema["type"] == "object"
    assert "properties" in schema
    assert "procedure" in schema["properties"]


def test_case_models_registry_keys_match_listing():
    assert sorted(CASE_MODELS) == list_organs()
