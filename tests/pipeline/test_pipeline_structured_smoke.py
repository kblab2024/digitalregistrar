"""Smoke test for :mod:`digital_registrar_research.pipeline_structured`.

Mocks the OpenAI client to:
  * confirm the structured pipeline returns the expected output shape
    (matching ``CancerPipeline.forward``);
  * confirm every chat call ships ``response_format`` of type
    ``json_schema`` with a non-empty schema body — guards against
    accidental regression to plain JSON mode (which is what produced
    the 21% parse-error rate this pipeline exists to fix).

No live LLM is invoked.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


@pytest.fixture
def captured_calls() -> list[dict]:
    """Holds every kwargs dict passed to ``client.chat.completions.create``."""
    return []


@pytest.fixture
def fake_client(captured_calls):
    """A mock OpenAI client that:

    * captures kwargs from each ``chat.completions.create`` call;
    * returns the next staged response per call (FIFO);
    * raises if the queue is empty (test bug: too few staged responses).
    """
    queue: list[str] = []

    def create(**kwargs):
        captured_calls.append(kwargs)
        if not queue:
            raise AssertionError(
                "fake_client received an unexpected extra chat call")
        content = queue.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=content))])

    client = MagicMock()
    client.chat.completions.create.side_effect = create
    client._queue = queue
    return client


def _enqueue(client, payload: dict) -> None:
    client._queue.append(json.dumps(payload))


def test_classify_only_when_not_cancer(fake_client, captured_calls):
    """A negative classify result short-circuits — no extract call should fire."""
    from digital_registrar_research import pipeline_structured as ps

    _enqueue(fake_client, {
        "cancer_excision_report": False,
        "cancer_category": None,
        "cancer_category_others_description": None,
    })

    with patch.object(ps, "OpenAI", return_value=fake_client):
        ps.setup_pipeline_structured("gptoss")
        output, elapsed = ps.run_cancer_pipeline_structured(
            report=["Benign biopsy. No tumor."], fname="case_neg")

    assert output["cancer_excision_report"] is False
    assert output["cancer_category"] is None
    assert output["cancer_data"] == {}
    assert "_parse_error" not in output
    assert elapsed >= 0
    assert len(captured_calls) == 1, "extract must not be called for non-cancer"


def test_positive_breast_runs_extract_and_returns_cancer_data(
    fake_client, captured_calls,
):
    """Positive classify + breast organ triggers an extract call with the
    breast schema; output keeps the canonical shape."""
    from digital_registrar_research import pipeline_structured as ps

    _enqueue(fake_client, {
        "cancer_excision_report": True,
        "cancer_category": "breast",
        "cancer_category_others_description": None,
    })
    _enqueue(fake_client, {
        "tumor_size_cm": 2.5,
        "histology": "invasive_ductal_carcinoma",
    })

    with patch.object(ps, "OpenAI", return_value=fake_client):
        ps.setup_pipeline_structured("gptoss")
        output, _ = ps.run_cancer_pipeline_structured(
            report=["Mastectomy. IDC. 2.5 cm."], fname="case_pos")

    assert output["cancer_excision_report"] is True
    assert output["cancer_category"] == "breast"
    assert isinstance(output["cancer_data"], dict)
    assert output["cancer_data"].get("tumor_size_cm") == 2.5
    assert "_parse_error" not in output
    assert len(captured_calls) == 2, "expected classify + extract calls"


def test_every_call_uses_json_schema_response_format(
    fake_client, captured_calls,
):
    """Guards against regression to plain JSON mode."""
    from digital_registrar_research import pipeline_structured as ps

    _enqueue(fake_client, {
        "cancer_excision_report": True,
        "cancer_category": "breast",
        "cancer_category_others_description": None,
    })
    _enqueue(fake_client, {})

    with patch.object(ps, "OpenAI", return_value=fake_client):
        ps.setup_pipeline_structured("gptoss")
        ps.run_cancer_pipeline_structured(
            report=["Some report."], fname="case_format")

    assert len(captured_calls) == 2
    for call in captured_calls:
        rf = call.get("response_format")
        assert rf is not None, "response_format missing — regressed to plain text?"
        assert rf.get("type") == "json_schema", (
            f"expected json_schema, got {rf.get('type')!r} — "
            "regressed to JSON mode")
        schema_body = rf.get("json_schema", {}).get("schema", {})
        assert schema_body.get("properties"), (
            "json_schema.schema.properties is empty — schema not wired through")


def test_extract_call_targets_organ_schema(fake_client, captured_calls):
    """The second call's schema must be the breast organ schema, not the
    classify schema."""
    from digital_registrar_research import pipeline_structured as ps

    _enqueue(fake_client, {
        "cancer_excision_report": True,
        "cancer_category": "breast",
        "cancer_category_others_description": None,
    })
    _enqueue(fake_client, {})

    with patch.object(ps, "OpenAI", return_value=fake_client):
        ps.setup_pipeline_structured("gptoss")
        ps.run_cancer_pipeline_structured(
            report=["Mastectomy report."], fname="case_schema")

    extract_call = captured_calls[1]
    schema_name = extract_call["response_format"]["json_schema"].get("name")
    assert schema_name == "breast", (
        f"extract call schema name should be the organ ({schema_name!r})")
    extract_props = extract_call["response_format"]["json_schema"]["schema"][
        "properties"]
    classify_props = captured_calls[0]["response_format"]["json_schema"][
        "schema"]["properties"]
    assert set(extract_props) != set(classify_props), (
        "extract schema must differ from classify schema")


def test_parse_error_surfaces_under_underscore_key(
    fake_client, captured_calls,
):
    """If the model still emits invalid JSON despite schema enforcement,
    the residual failure is surfaced under ``_parse_error`` so the eval
    pipeline's parse-error tracking still sees it."""
    from digital_registrar_research import pipeline_structured as ps

    fake_client._queue.append("not a json string {{{ ")

    with patch.object(ps, "OpenAI", return_value=fake_client):
        ps.setup_pipeline_structured("gptoss")
        output, _ = ps.run_cancer_pipeline_structured(
            report=["Unparseable case."], fname="case_garbled")

    assert "_parse_error" in output
    assert output["cancer_excision_report"] is False
