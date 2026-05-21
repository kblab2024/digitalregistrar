"""Prediction-vs-annotation evaluation toolkit.

Promoted to core in the 2026 reorg from `digital_registrar_research.benchmarks.eval`.
Includes basic field-level metrics, nested-field metrics (lymph nodes,
margins), pairwise run comparison, completeness/missingness analysis,
and bootstrap confidence intervals.

For paper-specific evaluation (IAA, preann effect, multirun statistical
analysis, BERT-specific eval, GPU-accelerated CIs), install the
`drr-attic` package and use `registrar-benchmark`.
"""
from . import metrics, nested_metrics, pairwise_compare, completeness, ci, scope, scope_organs
from .metrics import score_case, aggregate_cases_to_df, summary_table, field_correct
from .nested_metrics import score_lymph_nodes, score_margins
from .completeness import aggregate_missingness, out_of_vocab_rate, method_pair_deltas
from .ci import wilson_ci

# Public aliases — preferred names for the top-level digital_registrar.* API.
# Modules themselves are also exposed so users can `from digital_registrar.eval import metrics`.
field_metrics = metrics
nested_field_metrics = nested_metrics

__all__ = [
    # Submodules
    "metrics", "nested_metrics", "pairwise_compare", "completeness", "ci", "scope", "scope_organs",
    # Aliases for the top-level API
    "field_metrics", "nested_field_metrics",
    # Most-used functions
    "score_case", "aggregate_cases_to_df", "summary_table", "field_correct",
    "score_lymph_nodes", "score_margins",
    "aggregate_missingness", "out_of_vocab_rate", "method_pair_deltas",
    "wilson_ci",
]
