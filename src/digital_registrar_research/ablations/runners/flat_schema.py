"""
F3 — Raw JSON-mode against a denested (flat) per-organ schema.

Cell C asks the model to emit nested-list fields (``margins``,
``biomarkers``, ``regional_lymph_node``) as lists of dicts. F3
collapses each to a flat list of strings — the model writes one summary
string per item — and the runner re-parses those strings into the dict
shape via :mod:`extractors.flat_grader` so the existing
``match_nested_list`` grader works unchanged.

Canonical layout:
    --folder dummy --dataset tcga --model gptoss [--api-base ...] [--run runNN]
"""
from __future__ import annotations

import argparse
import copy
import os

from openai import OpenAI

from ...benchmarks.eval.metrics import NESTED_KEY
from ...paths import SCHEMAS_DATA
from ...schemas.builder import (
    describe_field_list,
    flatten_schema_for_prompt,
    load_organ_schema,
)
from ..extractors.flat_grader import renest_cancer_data
from ..utils.categories import CANCER_CATEGORIES
from . import _base
from .raw_json import (
    CLASSIFY_SYSTEM,
    _parse_json_best_effort,
)

CELL_ID = "flat_schema"

EXTRACT_SYSTEM_TEMPLATE = """\
You are a cancer registrar. Extract structured fields from the given
{organ} cancer excision report into a JSON object matching the field
list below.

Note: nested-list fields ({nested}) MUST be returned as flat lists of
plain English strings — one string per item — not as lists of objects.

Schema (field_name (type): description):
{field_list}
"""


def _renest_with_errors(raw: dict) -> tuple[dict, list[str]]:
    """Run :func:`renest_cancer_data` and report any nested-list field
    where re-parsing the model's free-text strings produced a different
    item count or shape. Lets us tell flat-schema accuracy loss apart
    from re-parser loss in the eval.
    """
    renested = renest_cancer_data(raw)
    errors: list[str] = []
    for field in NESTED_KEY:
        before = raw.get(field)
        after = renested.get(field)
        if isinstance(before, list) and isinstance(after, list):
            if len(before) != len(after):
                errors.append(
                    f"{field}: re-nest changed item count "
                    f"{len(before)}->{len(after)}")
        elif (before is None) != (after is None):
            errors.append(f"{field}: re-nest produced shape mismatch "
                          f"(raw={type(before).__name__}, "
                          f"renested={type(after).__name__})")
    return renested, errors


def _flatten_nested_to_string_arrays(schema: dict) -> dict:
    out = copy.deepcopy(schema)
    for field in NESTED_KEY:
        if field in out["properties"]:
            out["properties"][field] = {
                "type": "array",
                "items": {"type": "string"},
                "description": out["properties"][field].get(
                    "description",
                    f"List of {field} as plain strings."),
            }
    return out


class FlatRunner:
    def __init__(self, model_tag: str, api_key: str | None,
                 api_base: str | None):
        self.model = model_tag
        kwargs: dict = {"api_key": api_key or "EMPTY"}
        if api_base:
            kwargs["base_url"] = api_base
        self.client = OpenAI(**kwargs)

    def _chat(self, system: str, user: str) -> dict:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
        except Exception:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system",
                           "content": system + "\n\nReply with one JSON object only."},
                          {"role": "user", "content": user}],
                temperature=0.0,
            )
        return _parse_json_best_effort(resp.choices[0].message.content or "{}")

    def classify(self, report: str) -> dict:
        return self._chat(CLASSIFY_SYSTEM, report)

    def extract(self, report: str, organ: str) -> dict:
        flat = _flatten_nested_to_string_arrays(
            flatten_schema_for_prompt(load_organ_schema(organ)))
        nested_field_list = ", ".join(sorted(NESTED_KEY))
        system = EXTRACT_SYSTEM_TEMPLATE.format(
            organ=organ, nested=nested_field_list,
            field_list=describe_field_list(flat))
        return self._chat(system, report)

    def run_case(self, report: str) -> dict:
        cls = self.classify(report)
        if not cls.get("cancer_excision_report"):
            return {"cancer_excision_report": False,
                    "cancer_category": None, "cancer_data": {},
                    "_skip_reason": "not_cancer"}
        organ = cls.get("cancer_category")
        if (organ in {"others", None}
                or organ not in CANCER_CATEGORIES
                or not (SCHEMAS_DATA / f"{organ}.json").exists()):
            return {"cancer_excision_report": True,
                    "cancer_category": organ, "cancer_data": {},
                    "_skip_reason": "unknown_organ"}
        raw = self.extract(report, organ)
        renested, renest_errors = _renest_with_errors(raw)
        out = {
            "cancer_excision_report": True,
            "cancer_category": organ,
            "cancer_category_others_description":
                cls.get("cancer_category_others_description"),
            "cancer_data": renested,
            "_flat_raw": raw,
            "_downstream_called": True,
        }
        if renest_errors:
            out["_renest_errors"] = renest_errors
        return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    _base.add_canonical_args(ap)
    ap.add_argument("--api-base", default=None,
                    help=f"override Ollama endpoint (default: {_base.default_api_base()})")
    return ap.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    paths, organs, run_name = _base.resolve_run_paths(args, CELL_ID)
    api_base = args.api_base or _base.default_api_base()
    api_key = os.environ.get("OPENAI_API_KEY") or "ollama"
    model_tag = _base.ollama_tag(args.model)

    logger = _base.make_logger("flat_schema", paths.run_dir(run_name),
                               args.verbose)
    logger.info("cell=%s model=%s slug=%s tag=%s run=%s",
                CELL_ID, args.model, paths.model_slug, model_tag, run_name)

    runner = FlatRunner(model_tag, api_key, api_base)

    def _predict(report_text: str, organ_n: str, case_id: str) -> dict:
        return runner.run_case(report_text)

    summary = _base.run_loop(
        paths, organs, run_name, model_alias=args.model,
        predict=_predict, args=args, logger=logger,
        decoding={"temperature": 0.0, "api_base": api_base, "tag": model_tag},
        manifest_extra={"schema": "flat"},
        extra_meta={"api_base": api_base, "model_tag": model_tag,
                    "schema": "flat"},
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
