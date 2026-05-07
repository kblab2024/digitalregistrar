"""
B4 — Constrained decoding via ``outlines``.

Bypasses DSPy. Uses ``outlines`` to enforce JSON-schema conformance at
generation time. Backend support is conditional on the user having
``outlines`` installed plus a vLLM/HF/openai-compatible endpoint.

If ``outlines`` is not installed or the backend cannot be constructed,
the runner exits non-zero with a clear message — fail-loud per the
smoke contract.

Canonical layout:
    --folder dummy --dataset tcga --model gptoss --backend vllm [--api-base ...] [--run runNN]
"""
from __future__ import annotations

import argparse
import os
import sys

from ...paths import SCHEMAS_DATA
from ...schemas.builder import flatten_schema_for_prompt, load_organ_schema
from ...schemas.union_builder import build_union_schema
from ..utils.categories import CANCER_CATEGORIES
from . import _base

CELL_ID = "constrained_decoding"

CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "cancer_excision_report": {"type": "boolean"},
        "cancer_category": {
            "type": "string",
            "enum": list(CANCER_CATEGORIES),
        },
    },
    "required": ["cancer_excision_report", "cancer_category"],
    "additionalProperties": False,
}

CLASSIFY_PROMPT = (
    "Classify this pathology report. Return whether it is a primary "
    "cancer excision report and which organ.\n\nReport:\n{report}"
)
EXTRACT_PROMPT = (
    "Extract every cancer-registry field listed in the schema from "
    "this {organ} cancer pathology report. Use null for fields not "
    "present.\n\nReport:\n{report}"
)


def _import_outlines():
    try:
        import outlines  # noqa: F401
        return __import__("outlines")
    except ImportError as exc:
        sys.exit(
            "outlines is not installed. Install with `pip install outlines`. "
            f"Underlying error: {exc}")


def _build_model(args, outlines):
    backend = args.backend
    if backend == "vllm":
        try:
            return outlines.models.vllm(
                _base.ollama_tag(args.model),
                base_url=args.api_base or os.environ.get("VLLM_BASE_URL"),
            )
        except (AttributeError, TypeError):
            sys.exit(
                "outlines.models.vllm not available; install >=0.0.46 "
                "or use --backend hf.")
    if backend == "hf":
        return outlines.models.transformers(_base.ollama_tag(args.model))
    if backend == "openai":
        return outlines.models.openai(
            _base.ollama_tag(args.model),
            base_url=args.api_base or _base.default_api_base(),
            api_key=os.environ.get("OPENAI_API_KEY") or "ollama",
        )
    sys.exit(f"Unknown --backend {backend!r}")


class ConstrainedRunner:
    def __init__(self, args):
        self.args = args
        self.outlines = _import_outlines()
        self.model = _build_model(args, self.outlines)
        self._extract_generators: dict[str, object] = {}
        self.classify_generator = self.outlines.generate.json(
            self.model, CLASSIFY_SCHEMA)
        if args.use_union_schema:
            self._extract_generators["__union__"] = self.outlines.generate.json(
                self.model, build_union_schema())

    def _extract_generator(self, organ: str):
        if self.args.use_union_schema:
            return self._extract_generators["__union__"]
        if organ not in self._extract_generators:
            schema = flatten_schema_for_prompt(load_organ_schema(organ))
            self._extract_generators[organ] = self.outlines.generate.json(
                self.model, schema)
        return self._extract_generators[organ]

    def run_case(self, report: str, organ_n: str | None = None) -> dict:
        cls = self.classify_generator(CLASSIFY_PROMPT.format(report=report))
        if not cls.get("cancer_excision_report"):
            return {"cancer_excision_report": False,
                    "cancer_category": None, "cancer_data": {},
                    "_skip_reason": "not_cancer"}
        organ = cls.get("cancer_category")
        if (organ in {"others", None}
                or not (SCHEMAS_DATA / f"{organ}.json").exists()):
            return {"cancer_excision_report": True,
                    "cancer_category": organ, "cancer_data": {},
                    "_skip_reason": "unknown_organ",
                    "_organ_n_folder": organ_n}
        gen = self._extract_generator(organ)
        cancer_data = gen(EXTRACT_PROMPT.format(report=report, organ=organ))
        return {
            "cancer_excision_report": True,
            "cancer_category": organ,
            "cancer_data": cancer_data,
            "_downstream_called": True,
            "_organ_n_folder": organ_n,
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    _base.add_canonical_args(ap)
    ap.add_argument("--backend", choices=["vllm", "hf", "openai"],
                    default="vllm")
    ap.add_argument("--api-base", default=None)
    ap.add_argument("--use-union-schema", action="store_true",
                    help="extract against the union schema in one call")
    return ap.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    paths, organs, run_name = _base.resolve_run_paths(args, CELL_ID)
    api_base = args.api_base
    model_tag = _base.ollama_tag(args.model)

    logger = _base.make_logger("constrained_decoding",
                               paths.run_dir(run_name), args.verbose)
    logger.info("cell=%s model=%s slug=%s tag=%s backend=%s run=%s",
                CELL_ID, args.model, paths.model_slug, model_tag,
                args.backend, run_name)

    runner = ConstrainedRunner(args)

    def _predict(report_text: str, organ_n: str, case_id: str) -> dict:
        return runner.run_case(report_text, organ_n=organ_n)

    summary = _base.run_loop(
        paths, organs, run_name, model_alias=args.model,
        predict=_predict, args=args, logger=logger,
        decoding={"temperature": 0.0, "api_base": api_base,
                  "tag": model_tag, "backend": args.backend,
                  "use_union_schema": args.use_union_schema},
        manifest_extra={"backend": args.backend,
                        "use_union_schema": args.use_union_schema},
        extra_meta={"backend": args.backend, "model_tag": model_tag,
                    "api_base": api_base,
                    "use_union_schema": args.use_union_schema},
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
    sys.exit(main())
