"""Parity tests for the v2 (factory) pipeline.

These tests do NOT call any LM — they verify the structural contracts
between Pydantic schemas, the signature factory, and the pipeline
construction. The optional LM-driven smoke test at the bottom is gated
on ``RUN_V2_SMOKE=1`` and a reachable Ollama daemon.
"""
from __future__ import annotations

import os
import typing as _t

import pytest

from digital_registrar.schemas.extraction import EXTRACTION_META
from digital_registrar.schemas.pydantic import CASE_MODELS
from digital_registrar.signatures import (
    ExtractionStep,
    build_extraction_signatures,
    build_jsonize_signature,
    build_router_signature,
)

ORGANS = sorted(CASE_MODELS)


def _build_extraction_signatures(organ: str, **kwargs):
    """Build extraction signatures by threading the organ's metadata through.

    Centralised so individual parity tests don't repeat the metadata
    lookup boilerplate.
    """
    schema = CASE_MODELS[organ]
    organ_meta = EXTRACTION_META[organ]
    return build_extraction_signatures(
        schema, organ_meta["fields"], organ_meta["groups"], **kwargs,
    )


def test_router_keys_match_registry():
    """``build_router_signature()`` Literal == ``set(CASE_MODELS) | {'others'}``."""
    sig = build_router_signature()
    ann = sig.model_fields["cancer_category"].annotation
    members: set[str] = set()
    for arg in _t.get_args(ann):
        if _t.get_origin(arg) is _t.Literal:
            members |= set(_t.get_args(arg))
    assert members == set(CASE_MODELS) | {"others"}


@pytest.mark.parametrize("organ", ORGANS)
def test_factory_covers_schema(organ):
    """Per-group decomposition: union of step output fields == case-model fields."""
    schema = CASE_MODELS[organ]
    steps = _build_extraction_signatures(organ, decomposition="per_group")
    assert all(isinstance(s, ExtractionStep) for s in steps)
    factory_fields: set[str] = set()
    for s in steps:
        factory_fields |= set(s.output_field_names)
    schema_fields = set(schema.model_fields)
    missing = schema_fields - factory_fields
    extra = factory_fields - schema_fields
    assert not missing and not extra, (
        f"{organ}: factory missed {sorted(missing)}; factory had extra {sorted(extra)}"
    )


@pytest.mark.parametrize("organ", ORGANS)
def test_factory_monolithic_covers_schema(organ):
    """Monolithic mode produces exactly one step covering every field."""
    schema = CASE_MODELS[organ]
    steps = _build_extraction_signatures(organ, decomposition="monolithic")
    assert len(steps) == 1
    assert set(steps[0].output_field_names) == set(schema.model_fields)
    assert steps[0].group == "<monolithic>"


@pytest.mark.parametrize("organ", ORGANS)
def test_factory_per_group_no_orphans(organ):
    """Every step's group tag must be non-empty and a valid string."""
    for step in _build_extraction_signatures(organ, decomposition="per_group"):
        assert isinstance(step.group, str) and step.group, (
            f"{organ}: step {step.name} has empty group tag"
        )


def test_jsonize_literal_excludes_others():
    """Legacy asymmetry preserved on the v2 jsonize signature."""
    sig = build_jsonize_signature()
    ann = sig.model_fields["cancer_category"].annotation
    members: set[str] = set()
    for arg in _t.get_args(ann):
        if _t.get_origin(arg) is _t.Literal:
            members |= set(_t.get_args(arg))
    assert "others" not in members
    assert members == set(CASE_MODELS)


@pytest.mark.parametrize("organ", ORGANS)
def test_each_step_signature_has_required_inputs(organ):
    """Every per-group signature must accept ``report`` and ``report_jsonized`` inputs."""
    for step in _build_extraction_signatures(organ, decomposition="per_group"):
        fields = step.signature.model_fields
        assert "report" in fields
        assert "report_jsonized" in fields


def test_pipeline_v2_constructs_without_lm():
    """``CancerPipelineV2`` should construct without a configured LM.

    The router is built eagerly; per-organ extractors are built lazily on
    first use, so construction must succeed even before
    ``setup_pipeline_v2(model_name)`` is called.
    """
    from digital_registrar.pipeline_factory import CancerPipelineV2

    p = CancerPipelineV2()
    assert p.router is not None
    assert p.jsonize is None  # default OFF
    p2 = CancerPipelineV2(jsonize_enabled=True)
    assert p2.jsonize is not None


# --- Optional LM-driven smoke ---------------------------------------------

@pytest.mark.skipif(
    os.environ.get("RUN_V2_SMOKE") != "1",
    reason="LM-driven smoke disabled (set RUN_V2_SMOKE=1 to enable)",
)
def test_v2_smoke_runs_on_fixture():
    """End-to-end smoke: configure Ollama gptoss, run a tiny benign report.

    Requires a reachable Ollama daemon with gpt-oss:20b pulled. Skipped by
    default to keep ``pytest -x`` cheap.
    """
    from digital_registrar.pipeline_factory import (
        run_cancer_pipeline_v2,
        setup_pipeline_v2,
    )
    setup_pipeline_v2("gptoss")
    out, elapsed = run_cancer_pipeline_v2(
        report=["Benign skin biopsy. No tumor."],
        fname="smoke_neg",
        decomposition="per_group",
        jsonize_enabled=False,
    )
    assert isinstance(out, dict)
    assert "cancer_excision_report" in out
    assert isinstance(elapsed, float) and elapsed > 0
