"""Schema-driven cancer extraction pipeline (v2).

Sits alongside :mod:`digital_registrar_research.pipeline` (legacy) without
replacing it. Selected via the ``--engine factory`` flag on
``registrar-pipeline``, or directly via :func:`run_cancer_pipeline_v2`.

Differences from legacy:
  - signatures are built dynamically from Pydantic case-models (see
    ``digital_registrar_research.signatures.factory``);
  - the ``cancer_category`` Literal in the router is auto-derived from
    the ``CASE_MODELS`` registry, so adding an organ updates the router
    automatically;
  - the optional ``ReportJsonize`` preprocessing step is OFF by default
    (modern local models extract directly from raw text reliably);
  - extracted output is validated against the case-model in degraded mode
    (Pydantic errors are warnings, not failures).
"""
import json
import logging
import time
from typing import Literal

import dspy
from pydantic import ValidationError

from .models.common import autoconf_dspy, model_list
from .schemas import CASE_MODELS
from .schemas.extraction import EXTRACTION_META
from .signatures.factory import (
    build_extraction_signatures,
    build_jsonize_signature,
    build_router_signature,
)
from .util.predictiondump import dump_prediction_plain


def setup_pipeline_v2(
    model_name: str,
    overrides: dict | None = None,
    *,
    quiet_typeguard: bool = True,
) -> None:
    """Configure DSPy for the v2 pipeline.

    Mirrors :func:`pipeline.setup_pipeline` but additionally silences
    DSPy 3.2's new field-mismatch warnings (which fire frequently for
    dynamically-built signatures even when the LM output is valid).
    """
    autoconf_dspy(model_name, overrides=overrides)
    if quiet_typeguard:
        try:
            dspy.configure(disable_typeguard_warnings=True)
        except TypeError:
            # Older or future dspy may not accept this kwarg.
            pass
    print("Pipeline v2 setup complete.")


class CancerPipelineV2(dspy.Module):
    """Schema-driven cancer extraction module.

    The router signature is built eagerly in ``__init__``; per-organ
    extraction signatures are built lazily on first use in :meth:`forward`
    so that ``decomposition="auto"`` can consult the live LM. The lazy
    construction also means :func:`setup_pipeline_v2` MUST be called
    before the first ``forward()`` invocation.
    """

    def __init__(
        self,
        *,
        decomposition: Literal["per_group", "monolithic", "auto"] = "auto",
        jsonize_enabled: bool = False,
        model_profile: str | None = None,
        validate_output: bool = True,
    ) -> None:
        super().__init__()
        self._decomposition = decomposition
        self._model_profile = model_profile
        self._jsonize_enabled = jsonize_enabled
        self._validate_output = validate_output

        self.router = dspy.Predict(build_router_signature(CASE_MODELS))
        self.jsonize = (
            dspy.Predict(build_jsonize_signature(CASE_MODELS))
            if jsonize_enabled
            else None
        )

        # {organ: [(ExtractionStep, dspy.Predict), ...]}
        self._extractor_cache: dict[str, list] = {}

    # ----- internal helpers ----------------------------------------------

    @staticmethod
    def _normalize_report(report: str | list[str]) -> list[str]:
        if isinstance(report, list):
            return [p.strip() for p in report if isinstance(p, str) and p.strip()]
        return [p.strip() for p in str(report).split("\n\n") if p.strip()]

    def _get_extractors(self, organ: str) -> list:
        cached = self._extractor_cache.get(organ)
        if cached is not None:
            return cached
        try:
            organ_meta = EXTRACTION_META[organ]
        except KeyError as e:
            raise KeyError(
                f"No extraction metadata registered for organ {organ!r}. "
                f"Add a module under schemas/extraction/ and register it "
                f"in EXTRACTION_META."
            ) from e
        steps = build_extraction_signatures(
            CASE_MODELS[organ],
            organ_meta["fields"],
            organ_meta["groups"],
            decomposition=self._decomposition,
            model_profile=self._model_profile,
        )
        self._extractor_cache[organ] = [(s, dspy.Predict(s.signature)) for s in steps]
        return self._extractor_cache[organ]

    def _validate(self, organ: str, data: dict, logger: logging.Logger) -> None:
        try:
            CASE_MODELS[organ].model_validate(data)
        except ValidationError as e:
            # Degraded-mode: don't crash, but surface the first few errors.
            errs = e.errors()[:3]
            logger.warning(
                "v2 output validation failed for %s (%d errors); first 3: %s",
                organ, len(e.errors()), errs,
            )

    # ----- forward -------------------------------------------------------

    def forward(
        self,
        report: str | list[str],
        logger: logging.Logger,
        fname: str = "",
    ) -> dict:
        print(f"Processing report: {fname}")
        logger.info("Processing report: %s", fname)
        paragraphs = self._normalize_report(report)

        rsp = self.router(report=paragraphs)
        if not rsp.cancer_excision_report:
            logger.info("This is NOT a cancer excision report.")
            out = {
                "cancer_excision_report": False,
                "cancer_category": None,
                "cancer_data": {},
            }
            print(json.dumps(out, indent=2, ensure_ascii=False))
            return out

        out = {
            "cancer_excision_report": True,
            "cancer_category": rsp.cancer_category,
            "cancer_category_others_description": rsp.cancer_category_others_description,
            "cancer_data": {},
        }
        logger.info("This is a cancer excision report.")

        if rsp.cancer_category == "others":
            logger.info(
                "Cancer category is %s (others), currently not implemented.",
                rsp.cancer_category_others_description,
            )
            return out
        if rsp.cancer_category in (None, "") or rsp.cancer_category not in CASE_MODELS:
            logger.info("Unknown / unsupported cancer category: %r", rsp.cancer_category)
            return out

        logger.info("Cancer category is %s.", rsp.cancer_category)

        # Optional rough-JSON preprocessing.
        json_report: dict = {}
        if self._jsonize_enabled and self.jsonize is not None:
            try:
                json_resp = self.jsonize(
                    report=paragraphs, cancer_category=rsp.cancer_category
                )
                json_report = json_resp.output if isinstance(json_resp.output, dict) else {}
            except Exception as e:
                logger.warning("jsonize failed: %s", e)

        # Per-group (or monolithic) extraction.
        extractors = self._get_extractors(rsp.cancer_category)
        for step, predictor in extractors:
            logger.info(
                "running extractor %s at %s for %s [%s]",
                step.name,
                time.strftime("%Y-%m-%d %H:%M:%S"),
                rsp.cancer_category,
                fname,
            )
            try:
                pred = predictor(report=paragraphs, report_jsonized=json_report)
                organ_data = dump_prediction_plain(pred)
                # Filter to this step's output fields so we don't accidentally
                # leak echoed input fields back into cancer_data.
                kept = {k: organ_data.get(k) for k in step.output_field_names if k in organ_data}
                out["cancer_data"].update(kept)
            except Exception as e:
                logger.error("extractor %s failed: %s", step.name, e)
                continue

        if self._validate_output:
            self._validate(rsp.cancer_category, out["cancer_data"], logger)

        return out


def _timeit(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        return result, time.perf_counter() - start
    return wrapper


@_timeit
def _run_pipeline(experiment_model: dspy.Module, **kwargs):
    response = experiment_model(
        **kwargs, logger=logging.getLogger("runner_logger")
    )
    return response


def run_cancer_pipeline_v2(
    report: str | list[str],
    fname: str = "",
    *,
    decomposition: Literal["per_group", "monolithic", "auto"] = "auto",
    jsonize_enabled: bool = False,
    model_profile: str | None = None,
    validate_output: bool = True,
) -> tuple[dict, float]:
    """Run the v2 pipeline on a single report.

    Drop-in replacement for :func:`pipeline.run_cancer_pipeline` — same
    return signature ``(output_dict, elapsed_seconds)``.
    """
    pipeline = CancerPipelineV2(
        decomposition=decomposition,
        jsonize_enabled=jsonize_enabled,
        model_profile=model_profile,
        validate_output=validate_output,
    )
    response, timing = _run_pipeline(pipeline, report=report, fname=fname)
    return response, timing


__all__ = [
    "CancerPipelineV2",
    "setup_pipeline_v2",
    "run_cancer_pipeline_v2",
]


if __name__ == "__main__":
    setup_pipeline_v2("gpt")
    print("Pipeline v2 is ready for processing pathology reports.")
