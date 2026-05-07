"""
Cell C — DSPy-less, raw LLM with JSON mode.

The pipeline runs without DSPy. We make at most two raw LLM calls per
report via the OpenAI-compatible chat API:

    Call 1 — classify: {cancer_excision_report, cancer_category}
    Call 2 — extract:  flat cancer_data dict matching the organ schema

Both calls use ``response_format={"type": "json_object"}``. For the
extraction call the JSON schema is inlined in the system prompt.

Local models go through Ollama's OpenAI-compatible endpoint
(``http://localhost:11434/v1``) — derived from
:data:`models.common.localaddr` so the alias `--model gptoss` resolves
to ``gpt-oss:20b`` end-to-end.

Canonical layout:
    --folder dummy --dataset tcga --model gptoss [--run runNN]
"""
from __future__ import annotations

import argparse
import json
import os

from openai import OpenAI

from ...paths import SCHEMAS_DATA
from ...schemas.builder import (
    describe_field_list_strict,
    describe_skeleton,
    flatten_schema_for_prompt,
    load_organ_schema,
    validate_cancer_data,
)
from ..utils.categories import CANCER_CATEGORIES
from . import _base

CELL_ID = "raw_json"

CLASSIFY_SYSTEM = """\
You are a cancer registrar. Given a pathology report, decide:
1. whether the report documents a PRIMARY cancer excision eligible for
   cancer-registry entry (false if no viable tumor remains after
   excision, or if the finding is carcinoma in situ / high-grade
   dysplasia only); and
2. which organ the primary cancer arises from.

Return ONLY a JSON object with this exact shape:

  {
    "cancer_excision_report": true | false,
    "cancer_category": """ + " | ".join(f'"{c}"' for c in CANCER_CATEGORIES) + """ | null,
    "cancer_category_others_description": string | null
  }
"""

EXTRACT_SYSTEM_TEMPLATE = """\
You are a cancer registrar. Extract structured fields from the given
{organ} cancer excision report.

You MUST output a single JSON object whose keys are EXACTLY the field
names listed below. Do not invent keys. Do not omit keys. Do not nest
beyond the schema. Use null for any field not present in the report.

When a field has an "Allowed:" line, the value MUST be one of the listed
strings (or null). Do NOT translate, paraphrase, or invent new values.

Schema:
{field_list}

Skeleton (replace each null with the value from the report when
present, otherwise leave it as null):
{skeleton}
"""


class RawJSONRunner:
    def __init__(self, model_tag: str, api_key: str | None,
                 api_base: str | None):
        self.model = model_tag
        kwargs: dict = {"api_key": api_key or "EMPTY"}
        if api_base:
            kwargs["base_url"] = api_base
        self.client = OpenAI(**kwargs)
        self.validation_retries = 0

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

    def extract(self, report: str, organ: str) -> tuple[dict, list[str]]:
        schema = flatten_schema_for_prompt(load_organ_schema(organ))
        system = EXTRACT_SYSTEM_TEMPLATE.format(
            organ=organ,
            field_list=describe_field_list_strict(schema),
            skeleton=describe_skeleton(schema))
        data = self._chat(system, report)
        errors = validate_cancer_data(organ, data)
        if errors and len(errors) < 20:
            self.validation_retries += 1
            allowed_keys = set(schema.get("properties", {}).keys())
            observed_keys = set(data.keys()) if isinstance(data, dict) else set()
            missing = sorted(allowed_keys - observed_keys)
            extra = sorted(observed_keys - allowed_keys)
            extra_msgs = []
            if missing:
                extra_msgs.append(
                    f"Missing required keys: {', '.join(missing[:20])}")
            if extra:
                extra_msgs.append(
                    f"Unexpected keys (remove these): {', '.join(extra[:20])}")
            repair_user = (
                "The previous output had these schema errors:\n"
                + "\n".join(f"  - {e}" for e in errors[:20])
                + ("\n" + "\n".join(extra_msgs) if extra_msgs else "")
                + f"\n\nFix them and return the corrected JSON only.\n\n"
                f"Original report:\n{report}"
            )
            data = self._chat(system, repair_user)
            errors = validate_cancer_data(organ, data)
        return data, errors

    def run_case(self, report: str) -> dict:
        cls = self.classify(report)
        if not cls.get("cancer_excision_report"):
            return {"cancer_excision_report": False,
                    "cancer_category": None, "cancer_data": {},
                    "_skip_reason": "not_cancer"}
        organ = cls.get("cancer_category")
        out: dict = {
            "cancer_excision_report": True,
            "cancer_category": organ,
            "cancer_category_others_description":
                cls.get("cancer_category_others_description"),
            "cancer_data": {},
        }
        if organ in {"others", None} or not (SCHEMAS_DATA / f"{organ}.json").exists():
            out["_skip_reason"] = "unknown_organ"
            return out
        cancer_data, errors = self.extract(report, organ)
        out["cancer_data"] = cancer_data
        out["_downstream_called"] = True
        if errors:
            out["_schema_errors"] = errors[:20]
        return out


def _parse_json_best_effort(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rsplit("```", 1)[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return {"_parse_error": text[:500]}


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

    logger = _base.make_logger("raw_json", paths.run_dir(run_name),
                               args.verbose)
    logger.info("cell=%s model=%s slug=%s tag=%s run=%s organs=%s",
                CELL_ID, args.model, paths.model_slug, model_tag, run_name,
                [o[0] for o in organs])

    runner = RawJSONRunner(model_tag, api_key, api_base)

    def _predict(report_text: str, organ: str, case_id: str) -> dict:
        return runner.run_case(report_text)

    summary = _base.run_loop(
        paths, organs, run_name, model_alias=args.model,
        predict=_predict, args=args, logger=logger,
        decoding={"temperature": 0.0, "api_base": api_base, "tag": model_tag},
        manifest_extra={"validation_retries": runner.validation_retries},
        extra_meta={"validation_retries": runner.validation_retries,
                    "api_base": api_base, "model_tag": model_tag},
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
