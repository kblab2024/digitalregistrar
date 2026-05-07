"""
GPT-4 baseline — DSPy model swap.

Reuses the parent project's `CancerPipeline` verbatim; only swaps the
LM backend from local Ollama gpt-oss:20b to openai/gpt-4-turbo. This
isolates the contribution of model capacity from pipeline design: the
signatures, prompts, and post-processing are all held constant.

Cases are discovered the same way as the BERT baselines: walk
``<folder>/data/<dataset>/reports/<organ_n>/*.txt`` via
``benchmarks.baselines._data.load_predict_cases`` (gold annotations are
optional for predict-side runs). The cross-corpus contract is "predict
on TCGA, full corpus" — TCGA is held out from any model that's trained
on CMUH, so there is no in-corpus split to honor.

Requires:
    pip install dspy-ai openai
    set OPENAI_API_KEY=sk-...

Usage:
    python -m digital_registrar_research.benchmarks.baselines.gpt4 \\
        --folder workspace --datasets tcga \\
        --out workspace/results/benchmarks/gpt4_dspy
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import dspy

from ...paths import BENCHMARKS_RESULTS
from ...pipeline import CancerPipeline
from .. import organs as _organs
from ._data import load_predict_cases

DEFAULT_MODEL = "openai/gpt-4-turbo"  # swap to "openai/gpt-4o" if preferred
DEFAULT_DATASETS = ("tcga",)
DEFAULT_ORGANS = list(_organs.common_organs("cmuh", "tcga"))


def setup_gpt4(model_id: str = DEFAULT_MODEL) -> CancerPipeline:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Provision the key before running.")
    lm = dspy.LM(
        model=model_id,
        api_key=api_key,
        max_tokens=4000,
        temperature=0.0,
    )
    dspy.configure(lm=lm)
    return CancerPipeline()


def run_on_dataset(folder: Path, datasets: list[str], out_dir: Path,
                   limit: int | None = None,
                   model_id: str = DEFAULT_MODEL,
                   organs: set[str] | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = load_predict_cases(datasets=datasets, root=folder, organs=organs)
    if limit is not None:
        cases = cases[:limit]

    pipe = setup_gpt4(model_id)
    logger = logging.getLogger("gpt4_dspy")
    logger.setLevel(logging.INFO)

    cost_ledger = {"model": model_id, "runs": []}
    for case in cases:
        report = Path(case["report_path"]).read_text(encoding="utf-8")
        t0 = time.perf_counter()
        try:
            result = pipe(report=report, logger=logger, fname=case["id"])
        except Exception as e:
            logger.error(f"{case['id']}: {e}")
            result = {"error": str(e)}
        elapsed = time.perf_counter() - t0

        with (out_dir / f"{case['id']}.json").open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        cost_ledger["runs"].append({"id": case["id"], "elapsed_s": elapsed})
        print(f"  [{case['id']}] {elapsed:.1f}s")

    with (out_dir / "_cost_ledger.json").open("w", encoding="utf-8") as f:
        json.dump(cost_ledger, f, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="workspace", type=Path,
                    help="Experiment root containing data/<dataset>/ subtrees.")
    ap.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS),
                    help="Dataset name(s) to predict on (default: tcga).")
    ap.add_argument("--organs", nargs="*", default=DEFAULT_ORGANS,
                    help="Cancer-category names to keep.")
    ap.add_argument("--out", default=str(BENCHMARKS_RESULTS / "gpt4_dspy"),
                    help="Output directory (default: %(default)s).")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    out_dir = Path(args.out).resolve()
    organs = set(args.organs) if args.organs else None
    run_on_dataset(args.folder.resolve(), args.datasets, out_dir,
                   args.limit, args.model, organs)


if __name__ == "__main__":
    main()
