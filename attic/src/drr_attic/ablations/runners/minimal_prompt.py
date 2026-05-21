"""
C6 — Minimal raw prompt (degenerate prompting baseline).

Bypasses DSPy. Reuses Cell C's OpenAI client. System prompt is a single
sentence asking for cancer-registry fields as JSON; no field list, no
allowed-value hints, no examples, no organ-specific routing.

Canonical layout:
    --folder dummy --dataset tcga --model gptoss [--api-base ...] [--run runNN]
"""
from __future__ import annotations

import argparse
import os

from openai import OpenAI

from . import _base
from .raw_json import _parse_json_best_effort

CELL_ID = "minimal_prompt"

MINIMAL_SYSTEM = "Extract cancer-registry fields from the report as JSON."


def _chat(client: OpenAI, model_tag: str, report: str) -> dict:
    try:
        resp = client.chat.completions.create(
            model=model_tag,
            messages=[{"role": "system", "content": MINIMAL_SYSTEM},
                      {"role": "user", "content": report}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
    except Exception:
        resp = client.chat.completions.create(
            model=model_tag,
            messages=[{"role": "system",
                       "content": MINIMAL_SYSTEM + " Reply with one JSON object only."},
                      {"role": "user", "content": report}],
            temperature=0.0,
        )
    return _parse_json_best_effort(resp.choices[0].message.content or "{}")


def _coerce_to_pipeline_shape(raw: dict) -> dict:
    if not isinstance(raw, dict):
        return {"cancer_excision_report": False, "cancer_data": {}}
    if "cancer_data" in raw or "cancer_category" in raw:
        out = {
            "cancer_excision_report": raw.get("cancer_excision_report", True),
            "cancer_category": raw.get("cancer_category"),
            "cancer_data": raw.get("cancer_data", {}) or {},
        }
        if isinstance(out["cancer_data"], dict):
            return out
    cd = {k: v for k, v in raw.items()
          if k not in {"cancer_excision_report", "cancer_category",
                       "cancer_category_others_description"}}
    return {"cancer_excision_report": True,
            "cancer_category": raw.get("cancer_category"),
            "cancer_data": cd}


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

    logger = _base.make_logger("minimal_prompt", paths.run_dir(run_name),
                               args.verbose)
    logger.info("cell=%s model=%s slug=%s tag=%s run=%s",
                CELL_ID, args.model, paths.model_slug, model_tag, run_name)

    client = OpenAI(api_key=api_key, base_url=api_base)

    def _predict(report_text: str, organ_n: str, case_id: str) -> dict:
        raw = _chat(client, model_tag, report_text)
        return _coerce_to_pipeline_shape(raw)

    summary = _base.run_loop(
        paths, organs, run_name, model_alias=args.model,
        predict=_predict, args=args, logger=logger,
        decoding={"temperature": 0.0, "api_base": api_base, "tag": model_tag},
        manifest_extra={"prompt_style": "minimal"},
        extra_meta={"api_base": api_base, "model_tag": model_tag,
                    "prompt_style": "minimal"},
    )

    print(f"OK={summary.n_ok} ERR={summary.n_pipeline_error} "
          f"CACHED={summary.n_cached} N={summary.n_cases} "
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
