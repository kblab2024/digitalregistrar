"""
Cell B — DSPy + monolithic (one big signature per organ).

Runs the same top-level pipeline as the parent project — ``is_cancer``
routing, optional ``ReportJsonize``, then **one** organ-specific DSPy
signature (instead of the 5–7 in the modular baseline).

Canonical layout (see ``runners/_base.py``):

    --folder dummy --dataset tcga --model gptoss [--run runNN]

Output:
    {root}/results/ablations/{dataset}/dspy_monolithic/{model_slug}/{run}/{organ}/<case_id>.json
"""
from __future__ import annotations

import argparse
import logging

import dspy

from ...models.common import ReportJsonize, is_cancer
from ...models.modellist import organmodels
from ...util.predictiondump import dump_prediction_plain
from ..signatures.monolithic import (
    get_monolithic_signature,
    list_monolithic_fields,
)
from . import _base

CELL_ID = "dspy_monolithic"


class MonolithicPipeline(dspy.Module):
    """Drop-in replacement for ``CancerPipeline`` with per-organ
    signatures collapsed into a single monolithic signature."""

    def __init__(self, skip_jsonize: bool = False):
        super().__init__()
        self.skip_jsonize = skip_jsonize
        self.analyzer_is_cancer = dspy.Predict(is_cancer)
        self.jsonize = dspy.Predict(ReportJsonize)
        self._organ_predictors: dict[str, dspy.Predict] = {}

    def _get_organ_predictor(self, organ: str) -> dspy.Predict:
        if organ not in self._organ_predictors:
            sig = get_monolithic_signature(organ)
            self._organ_predictors[organ] = dspy.Predict(sig)
        return self._organ_predictors[organ]

    def forward(self, report: str, logger: logging.Logger,
                fname: str = "") -> dict:
        logger.debug("[monolithic] %s", fname)
        paragraphs = [p.strip() for p in report.split("\n\n") if p.strip()]

        context_response = self.analyzer_is_cancer(report=paragraphs)
        cer = bool(context_response.cancer_excision_report)
        organ = context_response.cancer_category
        logger.info("[%s] is_cancer -> excision=%s category=%r",
                    fname, cer, organ)
        if not cer:
            logger.warning("[%s] SKIP: is_cancer says not a cancer-excision "
                           "report", fname)
            return {"cancer_excision_report": False,
                    "cancer_category": None, "cancer_data": {},
                    "_skip_reason": "not_cancer"}

        out: dict = {
            "cancer_excision_report": True,
            "cancer_category": organ,
            "cancer_category_others_description":
                context_response.cancer_category_others_description,
            "cancer_data": {},
        }
        if organ not in organmodels:
            logger.warning(
                "[%s] SKIP: organ %r not in organmodels keys=%s — "
                "downstream predictor will NOT run",
                fname, organ, sorted(organmodels))
            out["_skip_reason"] = "unknown_organ"
            return out

        report_jsonized: dict = {}
        if not self.skip_jsonize:
            try:
                jr = self.jsonize(report=paragraphs, cancer_category=organ)
                report_jsonized = jr.output or {}
            except Exception as e:
                logger.warning("jsonize failed for %s: %s", fname, e)

        try:
            predictor = self._get_organ_predictor(organ)
            field_names = list_monolithic_fields(organ)
            logger.info("[%s] invoking %s predictor (signature=%s, n_fields=%d)",
                        fname, organ, predictor.signature.__name__,
                        len(field_names))
            organ_response = predictor(
                report=paragraphs, report_jsonized=report_jsonized)
            raw = dump_prediction_plain(organ_response)
            # Backfill: guarantee every signature field appears in the
            # output even if the model omitted it. Null-valued fields are
            # diagnostically useful; missing keys silently disappear in
            # downstream graders.
            backfilled = {name: None for name in field_names}
            backfilled.update(raw)
            out["cancer_data"] = backfilled
            n_non_null = sum(1 for v in raw.values() if v is not None)
            logger.info(
                "[%s] %s predictor returned %d non-null fields out of %d",
                fname, organ, n_non_null, len(field_names))
            out["_downstream_called"] = True
        except Exception as e:
            logger.error("monolithic %s failed for %s (signature=%s): %s",
                         organ, fname,
                         self._organ_predictors.get(organ).signature.__name__
                         if organ in self._organ_predictors else "?", e)
            out["_error"] = str(e)
        return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    _base.add_canonical_args(ap)
    ap.add_argument("--skip-jsonize", action="store_true",
                    help="ablation-of-ablation: also drop ReportJsonize")
    return ap.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    paths, organs, run_name = _base.resolve_run_paths(args, CELL_ID)
    overrides = _base.load_decoding_overrides(args.model)
    lm_kwargs = _base.setup_dspy_lm(args.model, overrides=overrides)

    logger = _base.make_logger("dspy_monolithic", paths.run_dir(run_name),
                               args.verbose)
    logger.info("cell=%s model=%s slug=%s run=%s organs=%s",
                CELL_ID, args.model, paths.model_slug, run_name,
                [o[0] for o in organs])

    pipe = MonolithicPipeline(skip_jsonize=args.skip_jsonize)

    def _predict(report_text: str, organ: str, case_id: str) -> dict:
        return pipe(report=report_text, logger=logger, fname=case_id)

    summary = _base.run_loop(
        paths, organs, run_name, model_alias=args.model,
        predict=_predict, args=args, logger=logger,
        decoding=lm_kwargs,
        manifest_extra={"skip_jsonize": args.skip_jsonize},
        extra_meta={"skip_jsonize": args.skip_jsonize,
                    "dspy_lm_kwargs": lm_kwargs},
    )

    print(f"OK={summary.n_ok} ERR={summary.n_pipeline_error} "
          f"CACHED={summary.n_cached} N={summary.n_cases} "
          f"NOT_CANCER={summary.n_skipped_not_cancer} "
          f"UNKNOWN_ORGAN={summary.n_skipped_unknown_organ} "
          f"DOWNSTREAM={summary.n_downstream_called} "
          f"WALL={summary.wall_time_s:.1f}s")
    print(f"run dir: {paths.run_dir(run_name)}")

    if summary.n_pipeline_error > 0 and not args.tolerate_errors:
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    import sys
    sys.exit(main())
