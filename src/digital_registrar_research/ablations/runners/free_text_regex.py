"""
B6 — Free-text generation + regex post-extraction (degenerate baseline).

The LM is asked for a plain-English summary of the cancer report — no
JSON, no field list, no schema, no structuring hints — then a fixed
bank of regexes scrapes back the FAIR_SCOPE primary endpoints and a
fuzzy enum-match recovers secondary fields.

Canonical layout:
    --folder dummy --dataset tcga --model gptoss [--api-base ...] [--run runNN]
"""
from __future__ import annotations

import argparse
import os

from openai import OpenAI

from ..extractors.regex_per_field import RegexExtractor
from . import _base

CELL_ID = "free_text_regex"

SUMMARY_SYSTEM = (
    "You are a cancer registrar. Summarise this pathology report in "
    "plain English, in one paragraph. Mention the cancer site, the "
    "tumour size, histologic grade, TNM stage (pT/pN/pM), the presence "
    "or absence of lymphovascular and perineural invasion, surgical "
    "margins, lymph-node involvement, and any biomarker results "
    "(ER/PR/HER2 for breast, etc). Do NOT use JSON or bullet points — "
    "write it as natural prose."
)


def _free_text_chat(client: OpenAI, model_tag: str, report: str) -> str:
    resp = client.chat.completions.create(
        model=model_tag,
        messages=[{"role": "system", "content": SUMMARY_SYSTEM},
                  {"role": "user", "content": report}],
        temperature=0.0,
    )
    return resp.choices[0].message.content or ""


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

    logger = _base.make_logger("free_text_regex", paths.run_dir(run_name),
                               args.verbose)
    logger.info("cell=%s model=%s slug=%s tag=%s run=%s",
                CELL_ID, args.model, paths.model_slug, model_tag, run_name)

    client = OpenAI(api_key=api_key, base_url=api_base)
    extractor = RegexExtractor()

    def _predict(report_text: str, organ_n: str, case_id: str) -> dict:
        summary = _free_text_chat(client, model_tag, report_text)
        result = extractor.extract(summary)
        result["_freetext_summary"] = summary
        return result

    summary = _base.run_loop(
        paths, organs, run_name, model_alias=args.model,
        predict=_predict, args=args, logger=logger,
        decoding={"temperature": 0.0, "api_base": api_base, "tag": model_tag},
        manifest_extra={"extractor": "regex_per_field"},
        extra_meta={"api_base": api_base, "model_tag": model_tag,
                    "extractor": "regex_per_field"},
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
