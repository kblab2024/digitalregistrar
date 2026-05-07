"""
pipeline_dspy_strict.py
Third sibling of :mod:`pipeline` and :mod:`pipeline_structured`. Restores
the DSPy per-organ decomposition (one ``Predict`` per signature, ~5-10
fields each) used by :class:`pipeline.CancerPipeline`, but routes every
LM call through Ollama's OpenAI-compatible endpoint with token-level
schema enforcement via DSPy 3.x ``JSONAdapter`` (which posts
``response_format={"type": "json_schema", "strict": true}``).

Why this exists
---------------
``pipeline_structured`` collapsed the 4-7 organ-specific signatures into
a single flattened schema and asked the model to fill 50+ fields in one
shot with no room for reasoning before the opening ``{``. That degrades
field coverage badly. This module recovers the per-signature focus while
keeping the schema enforcement.

Each signature is wrapped in :class:`dspy.ChainOfThought`, which adds a
``reasoning: str`` output that ``JSONAdapter`` emits as the first JSON
property — recovering the thinking-space that strict JSON otherwise
forbids. ``reasoning`` is stripped from each organ prediction before it
is merged into ``cancer_data`` so the canonical output shape stays
byte-compatible with :meth:`pipeline.CancerPipeline.forward` (and the
eval pipeline ingests it without modification).

Public surface mirrors ``pipeline_structured`` so the driver scripts are
line-for-line clones with three import swaps:

    setup_pipeline_dspy_strict(model, overrides=None, api_base=None)
    run_cancer_pipeline_dspy_strict(report, fname="") -> (dict, float)
    load_decoding_kwargs(model, overrides=None) -> dict   # re-export

Caveats
-------
* ``ReportJsonize.output: dict`` triggers JSONAdapter's
  ``_has_open_ended_mapping`` short-circuit and falls back to
  ``response_format={"type": "json_object"}`` (loose JSON-mode) for that
  one call. Acceptable by design — the dict is unconstrained on purpose.
  The per-organ extractors only declare scalar / Literal / list outputs,
  so they hit the strict ``json_schema`` path.

* LiteLLM's ``ollama_chat/`` provider returns
  ``supports_response_schema=False`` out of the box, which would silently
  downgrade JSONAdapter to JSON-mode. We override the flag at LM-load
  time via ``litellm.register_model({...: {"supports_response_schema":
  True}})`` so strict mode actually takes effect; verify by running with
  ``LITELLM_LOG=DEBUG`` and grepping for
  ``'response_format': {'type': 'json_schema'``.

* Requires Ollama >=0.5 for OpenAI-compat structured outputs.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import dspy
from dspy.adapters import JSONAdapter

from .ablations.runners._base import default_api_base
from .models.breast import *  # noqa: F401, F403
from .models.cervix import *  # noqa: F401, F403
from .models.colon import *  # noqa: F401, F403
from .models.common import is_cancer, load_model, model_list  # noqa: F401
from .models.common import ReportJsonize
from .models.esophagus import *  # noqa: F401, F403
from .models.liver import *  # noqa: F401, F403
from .models.lung import *  # noqa: F401, F403
from .models.modellist import organmodels
from .models.pancreas import *  # noqa: F401, F403
from .models.prostate import *  # noqa: F401, F403
from .models.stomach import *  # noqa: F401, F403
from .models.thyroid import *  # noqa: F401, F403
from .pipeline_structured import load_decoding_kwargs  # noqa: F401
from .util.predictiondump import dump_prediction_plain

__version__ = "0.1.0"
__date__ = "2026-05-05"
__author__ = ["Kai-Po Chang"]
__copyright__ = "Copyright 2026, Med NLP Lab, China Medical University"
__license__ = "MIT"


def _enable_strict_schema_for_ollama(model_id: str) -> None:
    """Flip LiteLLM's ``supports_response_schema`` flag on for this model
    so DSPy's JSONAdapter emits ``response_format={"type":"json_schema"}``
    instead of falling back to loose JSON-mode. See module docstring.

    LiteLLM's ``supports_response_schema`` strips the provider prefix via
    ``get_llm_provider`` before looking the model up in ``model_cost`` —
    so ``ollama_chat/gpt-oss:20b`` is normalised to ``gpt-oss:20b``. We
    must register under the bare key for the override to be visible to
    the lookup. Registering under the prefixed key is silently ignored.
    """
    import litellm

    bare = model_id.split("/", 1)[-1]  # "ollama_chat/gpt-oss:20b" -> "gpt-oss:20b"
    provider = model_id.split("/", 1)[0] if "/" in model_id else "ollama_chat"
    litellm.register_model({
        bare: {
            "supports_response_schema": True,
            "litellm_provider": provider,
            "mode": "chat",
        },
    })


def setup_pipeline_dspy_strict(
    model_name: str, overrides: dict | None = None,
    api_base: str | None = None,
) -> None:
    """Configure DSPy with a JSONAdapter-backed Ollama LM.

    Mirrors :func:`pipeline.setup_pipeline` and
    :func:`pipeline_structured.setup_pipeline_structured` so the driver
    scripts are interchangeable.
    """
    base = api_base or default_api_base()
    if model_name not in model_list:
        raise ValueError(
            f"Model {model_name} not found. Available: "
            f"{list(model_list.keys())}")
    model_id = model_list[model_name]
    _enable_strict_schema_for_ollama(model_id)
    lm = load_model(model_name, overrides=overrides)
    # ``load_model`` ignores ``api_base`` and uses ``localaddr`` from
    # models.common. If a non-default api_base was requested, rebuild the
    # LM with the explicit base so multi-host deployments still work.
    if api_base is not None:
        lm_kwargs = load_decoding_kwargs(model_name, overrides=overrides)
        lm = dspy.LM(
            model=model_id, api_base=base, api_key="",
            model_type="chat", **lm_kwargs,
        )
    dspy.configure(lm=lm, adapter=JSONAdapter())
    print(f"Loaded model: {model_name} (dspy_strict) with adapter=JSONAdapter")


class DspyStrictCancerPipeline(dspy.Module):
    """Per-organ decomposition with ChainOfThought + JSONAdapter.

    Structurally identical to :class:`pipeline.CancerPipeline` — same
    classify → jsonize → per-organ extract flow — but every Predict is
    wrapped in :class:`dspy.ChainOfThought` so the model gets a
    ``reasoning`` slot before the strict-schema body.
    """

    def __init__(self) -> None:
        super().__init__()
        self.analyzer_is_cancer = dspy.Predict(is_cancer)
        self.jsonize = dspy.Predict(ReportJsonize)

    def forward(
        self, report: str | list[str], logger: logging.Logger,
        fname: str = "",
    ) -> dict:
        print(f"Processing report: {fname}")
        logger.info(f"Processing report: {fname}")
        if isinstance(report, list):
            paragraphs = [p.strip() for p in report
                          if isinstance(p, str) and p.strip()]
        else:
            paragraphs = [p.strip() for p in report.split("\n\n") if p.strip()]

        context_response = self.analyzer_is_cancer(report=paragraphs)
        if not context_response.cancer_excision_report:
            logger.info("This is NOT a cancer excision report.")
            return {
                "cancer_excision_report": False,
                "cancer_category": None,
                "cancer_category_others_description": None,
                "cancer_data": {},
            }

        output_report: dict[str, Any] = {
            "cancer_excision_report": True,
            "cancer_category": context_response.cancer_category,
            "cancer_category_others_description":
                context_response.cancer_category_others_description,
            "cancer_data": {},
        }
        logger.info("This is a cancer excision report.")
        if context_response.cancer_category == "others":
            logger.info(
                "Cancer category is %s, currently not implemented.",
                context_response.cancer_category_others_description,
            )
        elif context_response.cancer_category:
            logger.info(
                "Cancer category is %s.", context_response.cancer_category,
            )

        try:
            json_response = self.jsonize(
                report=paragraphs,
                cancer_category=context_response.cancer_category,
            )
            json_report = json_response.output
        except Exception:
            json_report = {}

        for items in organmodels.get(context_response.cancer_category, []):
            cls = globals().get(items)
            if cls is None:
                logger.error(f"Model class {items} not found.")
                continue
            logger.info(
                "Processing organ-specific model: %s at %s for %s cancer for %s",
                cls.__name__, time.strftime("%Y-%m-%d %H:%M:%S"),
                context_response.cancer_category, fname,
            )
            organ_analyzer = dspy.Predict(cls)
            try:
                organ_response = organ_analyzer(
                    report=paragraphs, report_jsonized=json_report,
                )
                organ_data = dump_prediction_plain(organ_response)
                # CoT emits ``reasoning`` first; strip it so cancer_data
                # stays byte-compatible with CancerPipeline.forward()
                # output. Do not modify dump_prediction_plain — other
                # callers may legitimately want the reasoning trace.
                organ_data.pop("reasoning", None)
                output_report["cancer_data"].update(organ_data)
            except Exception as e:
                logger.error(f"Error processing {cls.__name__}: {e}")
                continue

        return output_report


def run_cancer_pipeline_dspy_strict(
    report: str | list[str], fname: str = "",
) -> tuple[dict, float]:
    """Drop-in for :func:`pipeline.run_cancer_pipeline` and
    :func:`pipeline_structured.run_cancer_pipeline_structured` — same
    return signature ``(output_dict, elapsed_seconds)``.
    """
    pipeline = DspyStrictCancerPipeline()
    logger = logging.getLogger("experiment_logger")
    t0 = time.perf_counter()
    output = pipeline(report=report, logger=logger, fname=fname)
    return output, time.perf_counter() - t0


if __name__ == "__main__":
    setup_pipeline_dspy_strict("gptoss")
    print("DSPy-strict pipeline is ready for processing pathology reports.")
