"""Deprecated — use the canonical eval pipeline instead.

The benchmark eval surface has migrated to the canonical layout under
``scripts/eval/cli.py``:

    # Per-method metrics
    python -m scripts.eval.cli non_nested \\
        --root workspace --dataset tcga \\
        --method rule_based --annotator gold \\
        --out workspace/results/eval/non_nested_rule

    # Side-by-side comparison
    python -m scripts.eval.compare.run_compare \\
        --inputs rule_based:.../non_nested_rule \\
                 clinicalbert:.../non_nested_bert \\
                 llm:.../non_nested_llm \\
        --out workspace/results/eval/compare

    # Convenience wrappers
    python scripts/baselines/eval_rule_vs_llm.py    --folder workspace --dataset tcga ...
    python scripts/baselines/eval_bert_vs_llm.py    --folder workspace --dataset tcga ...
    python scripts/baselines/eval_rule_bert_llm.py  --folder workspace --dataset tcga ...

This module remains as a stub so the ``registrar-benchmark`` console
entry point in ``pyproject.toml`` still resolves; invoking it prints the
new commands and exits non-zero.
"""
from __future__ import annotations

import sys

_NEW_TOOLS = """\
The 'benchmarks.eval.run_all' / 'registrar-benchmark' entry point has been
retired. Use the canonical eval pipeline instead:

  Per-method metrics:
    python -m scripts.eval.cli non_nested \\
        --root workspace --dataset tcga \\
        --method <rule_based|clinicalbert|llm> [--model <name>] \\
        --annotator gold --out <out_dir>

  Side-by-side comparison:
    python -m scripts.eval.compare.run_compare \\
        --inputs label1:<dir1> label2:<dir2> ... --out <out_dir>

  Convenience wrappers (run non_nested + compare in one go):
    python scripts/baselines/eval_rule_vs_llm.py
    python scripts/baselines/eval_bert_vs_llm.py
    python scripts/baselines/eval_rule_bert_llm.py

See docs/benchmarks/ for the full benchmark workflow.
"""


def main() -> int:
    print(_NEW_TOOLS, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
