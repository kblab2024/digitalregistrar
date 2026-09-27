"""Prediction-vs-annotation evaluation toolkit.

Promoted to core in the 2026 reorg from `digital_registrar_research.benchmarks.eval`.
Includes basic field-level metrics, nested-field metrics (lymph nodes,
margins), pairwise run comparison, completeness/missingness analysis,
and bootstrap confidence intervals. Folder-level scoring of
``registrar-pipeline`` output backs the ``registrar-eval`` CLI.

For paper-specific evaluation (IAA, preann effect, multirun statistical
analysis, BERT-specific eval, GPU-accelerated CIs), install the
`drr-attic` package and use `registrar-benchmark`.
"""
from . import ci, completeness, folders, metrics, nested_metrics, pairwise_compare, scope, scope_organs
from .ci import wilson_ci
from .completeness import aggregate_missingness, completeness_atomic, method_pair_deltas, out_of_vocab_rate
from .folders import load_pairs
from .metrics import aggregate_cases_to_df, field_correct, score_case, score_pairs, summarize_scores, summary_table
from .nested_metrics import score_lymph_nodes, score_margins
from .pairwise_compare import compare_runs

# Public aliases — preferred names for the top-level digital_registrar.* API.
# Modules themselves are also exposed so users can `from digital_registrar.eval import metrics`.
field_metrics = metrics
nested_field_metrics = nested_metrics

__all__ = [
    # Submodules
    "metrics", "nested_metrics", "pairwise_compare", "completeness", "ci", "scope", "scope_organs",
    "folders",
    # Aliases for the top-level API
    "field_metrics", "nested_field_metrics",
    # Most-used functions
    "score_case", "aggregate_cases_to_df", "summary_table", "field_correct",
    "score_lymph_nodes", "score_margins",
    "load_pairs", "score_pairs", "summarize_scores", "compare_runs",
    "aggregate_missingness", "completeness_atomic", "out_of_vocab_rate", "method_pair_deltas",
    "wilson_ci",
]
