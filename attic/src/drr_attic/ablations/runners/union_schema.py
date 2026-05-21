"""
F2 — Raw JSON-mode against a single union schema across all organs.

Identical to Cell C in spirit, but instead of routing to a per-organ
schema we ask the model — in one call — to populate a flat schema that
spans every organ's fields.

Canonical layout:
    --folder dummy --dataset tcga --model gptoss [--api-base ...] [--run runNN]
"""
from __future__ import annotations

import argparse
import os

from openai import OpenAI

from ...schemas.builder import describe_field_list
from ...schemas.union_builder import build_union_schema
from . import _base
from .raw_json import _parse_json_best_effort

CELL_ID = "union_schema"

UNION_SYSTEM_TEMPLATE = """\
You are a cancer registrar. Extract structured fields from a cancer
pathology report into a single JSON object that follows the union
schema below — every field that exists across our supported organs.

For fields that do not apply to this organ, return null.

You must populate cancer_excision_report (true/false) and
cancer_category (one of the listed organs or "others"). If
cancer_excision_report is false, return only those two fields and set
the rest to null.

Schema (field_name (type): description):
{field_list}
"""


class UnionRunner:
    def __init__(self, model_tag: str, api_key: str | None,
                 api_base: str | None):
        self.model = model_tag
        kwargs: dict = {"api_key": api_key or "EMPTY"}
        if api_base:
            kwargs["base_url"] = api_base
        self.client = OpenAI(**kwargs)
        self.union_schema = build_union_schema()
        self.system = UNION_SYSTEM_TEMPLATE.format(
            field_list=describe_field_list(self.union_schema))

    def _chat(self, report: str) -> dict:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": self.system},
                          {"role": "user", "content": report}],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
        except Exception:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system",
                           "content": self.system + "\n\nReply with one JSON object only."},
                          {"role": "user", "content": report}],
                temperature=0.0,
            )
        return _parse_json_best_effort(resp.choices[0].message.content or "{}")

    def run_case(self, report: str) -> dict:
        raw = self._chat(report)
        cer = raw.pop("cancer_excision_report", None)
        organ = raw.pop("cancer_category", None)
        others_desc = raw.pop("cancer_category_others_description", None)
        out = {
            "cancer_excision_report": bool(cer),
            "cancer_category": organ,
            "cancer_category_others_description": others_desc,
            "cancer_data": raw,
        }
        if not bool(cer):
            out["_skip_reason"] = "not_cancer"
        elif organ in {"others", None}:
            out["_skip_reason"] = "unknown_organ"
        else:
            out["_downstream_called"] = True
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

    logger = _base.make_logger("union_schema", paths.run_dir(run_name),
                               args.verbose)
    logger.info("cell=%s model=%s slug=%s tag=%s run=%s",
                CELL_ID, args.model, paths.model_slug, model_tag, run_name)

    runner = UnionRunner(model_tag, api_key, api_base)

    def _predict(report_text: str, organ_n: str, case_id: str) -> dict:
        return runner.run_case(report_text)

    summary = _base.run_loop(
        paths, organs, run_name, model_alias=args.model,
        predict=_predict, args=args, logger=logger,
        decoding={"temperature": 0.0, "api_base": api_base, "tag": model_tag},
        manifest_extra={"schema": "union",
                        "field_count": len(runner.union_schema["properties"])},
        extra_meta={"api_base": api_base, "model_tag": model_tag,
                    "schema": "union",
                    "field_count": len(runner.union_schema["properties"])},
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
