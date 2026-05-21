"""
A5 — Per-section decomposition.

Splits the report into header / gross / micro / dx / comments via
:mod:`utils.section_splitter` and runs a section-specialised DSPy
signature on each slice. Outputs are merged with first-wins to mirror
the existing modular pipeline's ``cancer_data.update()`` semantics.

Different decomposition axis from Cell A: the modular signatures are
organised around *output structure*, while per-section is organised
around *input position*. Uses the gold organ derived from the report's
organ subdirectory.

Canonical layout:
    --folder dummy --dataset tcga --model gptoss [--run runNN]
"""
from __future__ import annotations

import argparse
import logging

import dspy

from ...benchmarks.organs import organ_n_to_name
from ...util.predictiondump import dump_prediction_plain
from ..signatures.per_section import get_section_signature, list_section_fields
from ..utils.organ_classifier import classify_organ_from_text
from ..utils.section_splitter import SECTION_NAMES, split_report
from . import _base

CELL_ID = "per_section"


class PerSectionPipeline(dspy.Module):
    def __init__(self):
        super().__init__()
        self._predictors: dict[tuple[str, str], dspy.Predict] = {}

    def _get_predictor(self, organ: str, section: str) -> dspy.Predict:
        key = (organ, section)
        if key not in self._predictors:
            sig = get_section_signature(organ, section)
            self._predictors[key] = dspy.Predict(sig)
        return self._predictors[key]

    def forward(self, report: str, organ: str, logger: logging.Logger,
                fname: str = "") -> dict:
        logger.debug("[per_section] %s organ=%s", fname, organ)
        out: dict = {
            "cancer_excision_report": True,
            "cancer_category": organ,
            "cancer_data": {},
            "_per_section_used": [],
        }
        sections = split_report(report)
        # Detect silent degradation: section splitter regex collapses to a
        # single ``dx`` section when headers are non-canonical.
        non_empty = [s for s in SECTION_NAMES
                     if (sections.get(s, "") or "").strip()]
        if len(non_empty) <= 1:
            logger.warning(
                "[%s] section split degraded: only %d non-empty section(s) "
                "found (%s) — per_section will collapse to monolithic "
                "extraction", fname, len(non_empty), non_empty)
            out["_section_split_degraded"] = True
        merged: dict = {}
        for section in SECTION_NAMES:
            slice_text = sections.get(section, "").strip()
            if not slice_text:
                continue
            if not list_section_fields(organ, section):
                continue
            try:
                predictor = self._get_predictor(organ, section)
                resp = predictor(report=slice_text)
                section_data = dump_prediction_plain(resp)
                for k, v in section_data.items():
                    if k not in merged and v is not None:
                        merged[k] = v
                out["_per_section_used"].append(section)
            except Exception as e:
                logger.warning("per_section %s/%s failed for %s: %s",
                               organ, section, fname, e)
                out.setdefault("_per_section_errors", {})[section] = str(e)
        out["cancer_data"] = merged
        out["_downstream_called"] = True
        return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    _base.add_canonical_args(ap)
    return ap.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    paths, organs, run_name = _base.resolve_run_paths(args, CELL_ID)
    overrides = _base.load_decoding_overrides(args.model)
    lm_kwargs = _base.setup_dspy_lm(args.model, overrides=overrides)

    logger = _base.make_logger("per_section", paths.run_dir(run_name),
                               args.verbose)
    logger.info("cell=%s model=%s slug=%s run=%s organs=%s",
                CELL_ID, args.model, paths.model_slug, run_name,
                [o[0] for o in organs])

    pipe = PerSectionPipeline()
    dataset = args.dataset

    def _predict(report_text: str, organ_n: str, case_id: str) -> dict:
        organ = classify_organ_from_text(
            report_text, dataset=dataset, fallback_organ_n=organ_n)
        folder_organ = organ_n_to_name(dataset, organ_n)
        if organ is None:
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
        manifest_extra={"decomposition": "per_section"},
        extra_meta={"decomposition": "per_section",
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
