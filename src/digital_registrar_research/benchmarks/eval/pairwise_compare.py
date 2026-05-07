"""Deprecated — use ``scripts.eval.compare.run_compare`` instead.

The head-to-head comparison logic has moved to the canonical eval
surface. The new tool consumes per-method ``correctness_table.parquet``
files produced by ``scripts.eval.cli non_nested`` (so it works for any
method, not just BERT-vs-LLM) and emits wide.csv + per_field.csv +
pairwise.csv + headline.csv with paired-bootstrap deltas, Wilson CIs,
and McNemar p-values.

Migration::

    # Old
    python -m digital_registrar_research.benchmarks.eval.pairwise_compare \\
        --a clinicalbert_merged --b rule_based --scope bert \\
        --datasets tcga --n-boot 2000

    # New
    python -m scripts.eval.cli non_nested \\
        --root workspace --dataset tcga --method clinicalbert --model merged \\
        --annotator gold --out workspace/results/eval/non_nested_bert_merged
    python -m scripts.eval.cli non_nested \\
        --root workspace --dataset tcga --method rule_based \\
        --annotator gold --out workspace/results/eval/non_nested_rule
    python -m scripts.eval.compare.run_compare \\
        --inputs bert_merged:workspace/results/eval/non_nested_bert_merged \\
                 rule_based:workspace/results/eval/non_nested_rule \\
        --out workspace/results/eval/compare/bert_vs_rule \\
        --n-boot 2000

Or use the convenience wrapper::

    python scripts/baselines/eval_bert_vs_llm.py --folder workspace --dataset tcga ...
"""
from __future__ import annotations

import sys

_MIGRATION = """\
'benchmarks.eval.pairwise_compare' has been retired in favor of
'scripts.eval.compare.run_compare'. See docs/benchmarks/ or the module
docstring for the migration path.
"""


def main() -> int:
    print(_MIGRATION, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
