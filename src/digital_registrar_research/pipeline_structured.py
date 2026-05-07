"""
pipeline_structured.py
Schema-enforced sibling of :mod:`pipeline`. Drives the same two-call
extraction (classify → per-organ extract) but routes every LLM call
through Ollama's OpenAI-compatible endpoint with
``response_format={"type": "json_schema", "json_schema": {...}}``,
which constrains generation to the schema at the token level.

Public surface mirrors ``pipeline.py`` so the daily driver scripts
(``scripts/pipeline/run_structured_ollama_{single,multirun}.py``) can
swap imports and otherwise look line-for-line like the DSPy variants:

    setup_pipeline_structured(model, overrides=None)
    run_cancer_pipeline_structured(report, fname="") -> (dict, float)
    load_decoding_kwargs(model, overrides=None) -> dict

Returns the same ``{cancer_excision_report, cancer_category,
cancer_category_others_description, cancer_data}`` shape as
``CancerPipeline.forward()`` so the canonical ``*_output.json`` layout
is preserved and the eval pipeline ingests it without modification.

Requires Ollama >=0.5 for OpenAI-compat structured outputs. On older
daemons the API call surfaces a clear error rather than silently
falling back to JSON mode (which would defeat the experiment).
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from openai import OpenAI

from .ablations.runners._base import default_api_base, ollama_tag
from .ablations.utils.categories import CANCER_CATEGORIES
from .models.common import MODEL_PROFILES, _BASE_KWARGS, model_list
from .paths import SCHEMAS_DATA
from .schemas.builder import (
    flatten_schema_for_prompt,
    load_organ_schema,
    validate_cancer_data,
)

__version__ = "0.1.0"
__date__ = "2026-05-05"
__author__ = ["Kai-Po Chang"]
__copyright__ = "Copyright 2026, Med NLP Lab, China Medical University"
__license__ = "MIT"

# OpenAI-standard sampler keys forwarded directly on chat.completions.create.
_OPENAI_SAMPLER_KEYS = ("temperature", "top_p", "max_tokens", "seed")
# Ollama-only knobs relayed through extra_body (LiteLLM uses the same
# convention to pass these through OpenAI-compat).
_OLLAMA_EXTRA_KEYS = ("top_k", "num_ctx", "repeat_penalty", "keep_alive")


CLASSIFY_SYSTEM = (
    "You are a cancer registrar. Given a pathology report, decide "
    "whether it documents a PRIMARY cancer excision eligible for "
    "cancer-registry entry, and if so which organ the cancer arises "
    "from. If no viable tumor remains after excision, or the finding "
    "is carcinoma in situ / high-grade dysplasia only, the report is "
    "NOT registry-eligible. If the organ is not in the supported list, "
    "use ``cancer_category=\"others\"`` and put the organ name in "
    "``cancer_category_others_description``."
)
EXTRACT_SYSTEM_TEMPLATE = (
    "You are a cancer registrar. Extract every cancer-registry field "
    "from this {organ} cancer pathology excision report and emit a "
    "single JSON object that conforms to the supplied schema. Use "
    "null for any field not documented in the report. Do not invent "
    "values; do not paraphrase enum values."
)


# ``cancer_excision_report`` plus a nullable ``cancer_category`` enum
# and a free-text fallback. Mirrors the ``is_cancer`` DSPy signature
# field-for-field so the JSON output is shape-compatible with what the
# DSPy pipeline emits via ``CancerPipeline.forward``.
CLASSIFY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "title": "classify",
    "properties": {
        "cancer_excision_report": {"type": "boolean"},
        "cancer_category": {
            "anyOf": [
                {"type": "string", "enum": list(CANCER_CATEGORIES)},
                {"type": "null"},
            ],
        },
        "cancer_category_others_description": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
        },
    },
    "required": [
        "cancer_excision_report",
        "cancer_category",
        "cancer_category_others_description",
    ],
    "additionalProperties": False,
}


def load_decoding_kwargs(
    model_name: str, overrides: dict | None = None,
) -> dict[str, Any]:
    """Resolve the per-model decoding kwargs without going through DSPy.

    Returns the same dict shape that the DSPy driver scripts pull off
    ``lm.kwargs`` (keys: ``temperature, top_p, top_k, max_tokens,
    num_ctx, repeat_penalty, keep_alive, cache, seed``), so
    ``_run_meta.json`` / ``_summary.json`` / the model-level manifest
    can be stamped identically across DSPy and structured runs.
    """
    if model_name not in model_list:
        raise ValueError(
            f"Model {model_name} not found. Available: "
            f"{list(model_list.keys())}")
    model_id = model_list[model_name]
    kwargs = {**_BASE_KWARGS, **MODEL_PROFILES.get(model_id, {})}
    if overrides:
        kwargs.update({k: v for k, v in overrides.items() if v is not None})
    return kwargs


_PIPELINE_STATE: dict[str, Any] = {
    "client": None,
    "model_tag": None,
    "decoding": None,
    "api_base": None,
}


def setup_pipeline_structured(
    model_name: str, overrides: dict | None = None,
    api_base: str | None = None,
) -> None:
    """Build the OpenAI client + cache decoding params for the
    structured pipeline.

    Mirrors :func:`pipeline.setup_pipeline` (which calls ``autoconf_dspy``
    under the hood). Called once per process by the driver scripts.
    """
    base = api_base or default_api_base()
    api_key = os.environ.get("OPENAI_API_KEY") or "ollama"
    client = OpenAI(base_url=base, api_key=api_key)
    decoding = load_decoding_kwargs(model_name, overrides=overrides)
    _PIPELINE_STATE.update({
        "client": client,
        "model_tag": ollama_tag(model_name),
        "decoding": decoding,
        "api_base": base,
    })
    print(f"Loaded model: {model_name} (structured) with {decoding}")


def _require_state() -> tuple[OpenAI, str, dict]:
    """Return (client, model_tag, decoding) or raise if setup missed."""
    client = _PIPELINE_STATE.get("client")
    model_tag = _PIPELINE_STATE.get("model_tag")
    decoding = _PIPELINE_STATE.get("decoding")
    if client is None or model_tag is None:
        raise RuntimeError(
            "structured pipeline not initialised; call "
            "setup_pipeline_structured(model_name, overrides) first.")
    return client, model_tag, decoding or {}


def _split_response_format(decoding: dict) -> tuple[dict, dict]:
    """Split decoding kwargs into (top-level, extra_body)."""
    top: dict = {}
    extra: dict = {}
    for key in _OPENAI_SAMPLER_KEYS:
        if decoding.get(key) is not None:
            top[key] = decoding[key]
    for key in _OLLAMA_EXTRA_KEYS:
        if decoding.get(key) is not None:
            extra[key] = decoding[key]
    return top, extra


def _chat_json_schema(
    client: OpenAI, model_tag: str, decoding: dict,
    schema_name: str, schema: dict,
    system: str, user: str,
) -> dict:
    """Single chat call with schema-enforced JSON. Returns the parsed
    object, or a ``{"_parse_error": ...}`` sentinel on the rare case
    that the response is still not valid JSON."""
    top, extra_body = _split_response_format(decoding)
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": schema_name, "schema": schema},
    }
    kwargs: dict[str, Any] = {
        "model": model_tag,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": response_format,
        **top,
    }
    if extra_body:
        kwargs["extra_body"] = extra_body
    resp = client.chat.completions.create(**kwargs)
    text = resp.choices[0].message.content or "{}"
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"_parse_error": text[:500]}


def _join_paragraphs(report: str | list[str]) -> str:
    """Normalise the report to the single string form expected by chat.

    Accepts the same shapes ``CancerPipeline.forward`` accepts: a raw
    string, or a pre-split list of paragraph strings (the canonical
    form the driver scripts emit via ``_split_report_rows``)."""
    if isinstance(report, list):
        rows = [p.strip() for p in report if isinstance(p, str) and p.strip()]
    else:
        rows = [p.strip() for p in str(report).split("\n\n") if p.strip()]
    return "\n\n".join(rows)


class StructuredCancerPipeline:
    """Schema-enforced two-call extraction. Stateless wrt input; reads
    its client + model_tag + decoding from the module-level state set
    by :func:`setup_pipeline_structured`."""

    def __init__(self) -> None:
        self.client, self.model_tag, self.decoding = _require_state()

    def forward(
        self, report: str | list[str], logger: logging.Logger,
        fname: str = "",
    ) -> dict:
        report_text = _join_paragraphs(report)
        print(f"Processing report: {fname}")
        logger.info("Processing report: %s", fname)

        cls = _chat_json_schema(
            self.client, self.model_tag, self.decoding,
            "classify", CLASSIFY_SCHEMA,
            CLASSIFY_SYSTEM, report_text,
        )
        if "_parse_error" in cls:
            logger.error("classify parse error: %s",
                         cls["_parse_error"][:200])
            return {
                "cancer_excision_report": False,
                "cancer_category": None,
                "cancer_category_others_description": None,
                "cancer_data": {},
                "_parse_error": cls["_parse_error"],
            }

        if not cls.get("cancer_excision_report"):
            logger.info("This is NOT a cancer excision report.")
            return {
                "cancer_excision_report": False,
                "cancer_category": None,
                "cancer_category_others_description": None,
                "cancer_data": {},
            }

        organ = cls.get("cancer_category")
        out: dict[str, Any] = {
            "cancer_excision_report": True,
            "cancer_category": organ,
            "cancer_category_others_description":
                cls.get("cancer_category_others_description"),
            "cancer_data": {},
        }
        logger.info("This is a cancer excision report.")
        if organ == "others":
            logger.info("Cancer category is %s, currently not implemented.",
                        out["cancer_category_others_description"])
            return out
        if organ in (None, "") or not (SCHEMAS_DATA / f"{organ}.json").exists():
            logger.info("Unknown / unsupported organ: %r", organ)
            return out

        logger.info("Cancer category is %s.", organ)
        flat_schema = flatten_schema_for_prompt(load_organ_schema(organ))
        extract_system = EXTRACT_SYSTEM_TEMPLATE.format(organ=organ)
        cancer_data = _chat_json_schema(
            self.client, self.model_tag, self.decoding,
            organ, flat_schema,
            extract_system, report_text,
        )
        out["cancer_data"] = cancer_data
        if "_parse_error" in cancer_data:
            logger.error("extract parse error for %s", organ)
        else:
            errors = validate_cancer_data(organ, cancer_data)
            if errors:
                out["_schema_errors"] = errors[:20]
        return out


def run_cancer_pipeline_structured(
    report: str | list[str], fname: str = "",
) -> tuple[dict, float]:
    """Drop-in for :func:`pipeline.run_cancer_pipeline` — same return
    signature ``(output_dict, elapsed_seconds)``."""
    pipeline = StructuredCancerPipeline()
    logger = logging.getLogger("experiment_logger")
    t0 = time.perf_counter()
    output = pipeline(report=report, logger=logger, fname=fname)
    return output, time.perf_counter() - t0


if __name__ == "__main__":
    setup_pipeline_structured("gptoss")
    print("Structured pipeline is ready for processing pathology reports.")
