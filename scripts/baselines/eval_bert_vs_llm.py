#!/usr/bin/env python3
"""Compare ClinicalBERT vs LLM predictions side-by-side.

Runs ``scripts.eval.cli cascade`` for both methods (BERT defaults to
the merged head), then joins the outputs with ``scripts.eval.cli
compare`` into chapter1-5 comparison folders + pairwise
paired-bootstrap deltas.

Usage
-----
    python scripts/baselines/eval_bert_vs_llm.py \\
        --folder workspace --dataset tcga \\
        --bert-head merged \\
        --llm-model gpt_oss_20b --llm-runs run01 run02 \\
        --out workspace/results/eval/bert_vs_llm

Prerequisites: predictions for both methods must exist under
``{folder}/results/predictions/{dataset}/{clinicalbert/<head>,llm/<model>/<run>}/``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from baselines._eval_pipeline import (  # noqa: E402
    MethodSpec, add_common_args, run_pipeline,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--bert-head", default="merged",
                    choices=("cls", "qa", "merged"),
                    help="Which ClinicalBERT head to compare (default: merged).")
    ap.add_argument("--llm-model", required=True,
                    help="LLM model slug, e.g. gpt_oss_20b.")
    ap.add_argument("--llm-runs", nargs="*", default=None,
                    help="LLM run IDs (default: auto-discover).")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    specs = [
        MethodSpec(label=f"bert_{args.bert_head}",
                   method="clinicalbert", model=args.bert_head),
        MethodSpec(label=f"llm_{args.llm_model}",
                   method="llm", model=args.llm_model,
                   run_ids=args.llm_runs),
    ]
    return run_pipeline(specs, args)


if __name__ == "__main__":
    sys.exit(main())
