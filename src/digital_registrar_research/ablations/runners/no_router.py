"""
A4 — Monolithic DSPy without the ``is_cancer`` router.

Same as Cell B (monolithic) but skips the LLM routing call. The router
decides BOTH whether-to-extract AND which organ; the cleanest way to
remove it is to derive the organ from the report itself.

We use a **rule-based** classifier on report text
(:func:`ablations.utils.organ_classifier.classify_organ_from_text`) — NOT
the folder number alone. Folder names are a noisy signal because the
TCGA and CMUH datasets number organs differently (TCGA folder 1 = breast,
CMUH folder 1 = pancreas — see ``configs/organ_code.yaml``). The dataset-
aware folder lookup is used only as a last-resort fallback when no
keyword matches.

This is an upper-bound estimate of router contribution: the cell sees
the right organ for free, with the failure modes of organ-folder
misalignment surfaced explicitly via ``_organ_n_folder`` /
``_organ_folder_mismatch`` in the per-case JSON.

Canonical layout:
    --folder dummy --dataset tcga --model gptoss [--run runNN]
"""
from __future__ import annotations

import argparse
import logging

import dspy

from ...benchmarks import organs as _organs
from ...benchmarks.organs import organ_n_to_name
from ...util.predictiondump import dump_prediction_plain
from ..signatures.monolithic import (
    get_monolithic_signature,
    list_monolithic_fields,
)
from ..utils.organ_classifier import classify_organ_from_text
from . import _base

CELL_ID = "no_router"


def _organ_from_index(dataset: str, organ_n: str | int) -> str | None:
    """Map a numeric organ subdir name to the organ key used by the
    per-organ signatures, using the dataset's organ_code.yaml mapping.

    Returns None on miss (unknown organ_n for dataset). Used as the
    folder-fallback path and re-exported for the few-shot demo loader.
    """
    try:
        return _organs.organ_name(dataset, int(organ_n))
    except (KeyError, ValueError):
        return None


class NoRouterPipeline(dspy.Module):
    """Monolithic pipeline minus the is_cancer router stage."""

    def __init__(self, skip_jsonize: bool = True):
        super().__init__()
        self.skip_jsonize = skip_jsonize
        self._organ_predictors: dict[str, dspy.Predict] = {}
        if not skip_jsonize:
            from ...models.common import ReportJsonize
            self.jsonize = dspy.Predict(ReportJsonize)

    def _get_organ_predictor(self, organ: str) -> dspy.Predict:
        if organ not in self._organ_predictors:
            sig = get_monolithic_signature(organ)
            self._organ_predictors[organ] = dspy.Predict(sig)
        return self._organ_predictors[organ]

    def forward(self, report: str, organ: str, logger: logging.Logger,
                fname: str = "") -> dict:
        logger.debug("[no_router] %s organ=%s", fname, organ)
        paragraphs = [p.strip() for p in report.split("\n\n") if p.strip()]
        out: dict = {
            "cancer_excision_report": True,
            "cancer_category": organ,
            "cancer_data": {},
            "_router_skipped": True,
        }
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
            logger.info(
                "[%s] invoking %s predictor (router skipped, n_fields=%d)",
                fname, organ, len(field_names))
            organ_response = predictor(
                report=paragraphs, report_jsonized=report_jsonized)
            raw = dump_prediction_plain(organ_response)
            backfilled = {name: None for name in field_names}
            backfilled.update(raw)
            out["cancer_data"] = backfilled
            n_non_null = sum(1 for v in raw.values() if v is not None)
            logger.info(
                "[%s] %s predictor returned %d non-null fields out of %d",
                fname, organ, n_non_null, len(field_names))
            out["_downstream_called"] = True
        except Exception as e:
            logger.error("no_router %s failed for %s: %s", organ, fname, e)
            out["_error"] = str(e)
        return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    _base.add_canonical_args(ap)
    ap.add_argument("--include-jsonize", action="store_true",
                    help="keep the upstream ReportJsonize step (default: skip)")
    return ap.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    paths, organs, run_name = _base.resolve_run_paths(args, CELL_ID)
    overrides = _base.load_decoding_overrides(args.model)
    lm_kwargs = _base.setup_dspy_lm(args.model, overrides=overrides)

    logger = _base.make_logger("no_router", paths.run_dir(run_name),
                               args.verbose)
    logger.info("cell=%s model=%s slug=%s run=%s organs=%s",
                CELL_ID, args.model, paths.model_slug, run_name,
                [o[0] for o in organs])

    skip_jsonize = not args.include_jsonize
    pipe = NoRouterPipeline(skip_jsonize=skip_jsonize)
    dataset = args.dataset

    def _predict(report_text: str, organ_n: str, case_id: str) -> dict:
        organ = classify_organ_from_text(
            report_text, dataset=dataset, fallback_organ_n=organ_n)
        folder_organ = organ_n_to_name(dataset, organ_n)
        if organ is None:
            # Failure: surface as pipeline error so smoke contracts
            # actually flag this case (the previous behaviour silently
            # wrote a sentinel and let the run pass).
            raise RuntimeError(
                f"cannot infer organ for case={case_id} organ_n={organ_n!r}: "
                f"no rule-based match and no dataset fallback")
        if folder_organ and folder_organ != organ:
            logger.warning(
                "[%s] organ-folder mismatch: text says %r, folder %r maps to %r",
                case_id, organ, organ_n, folder_organ)
        payload = pipe(report=report_text, organ=organ,
                       logger=logger, fname=case_id)
        payload["_organ_n_folder"] = organ_n
        payload["_folder_organ"] = folder_organ
        if folder_organ and folder_organ != organ:
            payload["_organ_folder_mismatch"] = True
        return payload

    summary = _base.run_loop(
        paths, organs, run_name, model_alias=args.model,
        predict=_predict, args=args, logger=logger,
        decoding=lm_kwargs,
        manifest_extra={"router_skipped": True, "skip_jsonize": skip_jsonize},
        extra_meta={"router_skipped": True, "skip_jsonize": skip_jsonize,
                    "dspy_lm_kwargs": lm_kwargs},
    )

    print(f"OK={summary.n_ok} ERR={summary.n_pipeline_error} "
          f"CACHED={summary.n_cached} N={summary.n_cases} "
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
