"""Reviewer-grade statistical methods for the cascade evaluator.

This package is the **single import surface** for every statistic the
cascade pipeline reports. Reductions in ``scripts/eval/cascade/`` and
``src/digital_registrar_research/ablations/eval/canonical_stats.py`` should
import from here, never directly from ``ci.py`` or ``stats_extra.py``.

Module map
----------
    ci                  Confidence intervals: Wilson, Clopper-Pearson,
                        Agresti-Coull, jackknife, BCa bootstrap, ``pick_ci``.
    paired              Paired tests: McNemar, Cochran-Q, Stuart-Maxwell,
                        Bhapkar, Friedman, Nemenyi, paired-bootstrap delta.
    effect_size         Cohen's d / Cliff's delta / Cohen's kappa / weighted
                        kappa / Krippendorff alpha / MCC / balanced accuracy /
                        Brier score.
    multiple_comparisons    Holm, BH-FDR, BY-FDR, Sidak, permutation omnibus.
    reliability         ICC(2,1), ICC(3,k), Cronbach alpha, Spearman-Brown,
                        per-case run SD, accuracy/missing flip rates.
    heterogeneity       I-squared, Cochran's Q heterogeneity, interaction
                        tests, forest-plot CSV builder.
    concordance         Lin's CCC, Bland-Altman, MAPE, Spearman rho, count
                        MAE — for continuous endpoints (LN counts, tumor_size).
    cascade_diag        Cascade-specific diagnostics: funnel, conditional
                        accuracy grid, attrition propensity.
    thresholds          Pre-registered effect-size and significance constants.
    result              ``TestResult`` dataclass — common output schema.

Backwards compatibility: the legacy modules ``benchmarks.eval.ci``,
``scripts.eval._common.stats_extra``, and ``benchmarks.eval.multirun``
remain importable for code that has not migrated yet. They are not
deprecated, but new code should depend on this package surface.
"""
from __future__ import annotations

from .result import TestResult
from .thresholds import (
    HEADLINE_ACCURACY_DELTA_THRESHOLD,
    HEADLINE_F1_DELTA_THRESHOLD,
    KAPPA_SUBSTANTIAL,
    KAPPA_NEAR_PERFECT,
    FAMILYWISE_ALPHA,
    FDR_ALPHA,
    POWER_FLAG_THRESHOLD,
    LANDIS_KOCH_BANDS,
    kappa_band,
)

# Re-export the public surface so reductions can do `from
# benchmarks.eval.stats import wilson_ci, mcnemar, lin_ccc, ...`
from .ci import (
    BootstrapResult,
    agresti_coull_ci,
    bca_bootstrap_ci,
    clopper_pearson_ci,
    fisher_z_ci_for_corr,
    jackknife_ci,
    paired_bootstrap_diff,
    pick_ci,
    t_ci,
    two_source_bootstrap_ci,
    wilson_ci,
)
from .paired import (
    bhapkar,
    cochran_q,
    friedman,
    mcnemar,
    nemenyi_posthoc,
    paired_bootstrap_delta,
    stuart_maxwell,
)
from .effect_size import (
    balanced_accuracy,
    brier_score,
    cliffs_delta,
    cohens_d,
    cohens_kappa,
    krippendorff_alpha,
    matthews_corrcoef,
    weighted_kappa,
)
from .multiple_comparisons import (
    adjust_pvalues,
    bh_fdr,
    by_fdr,
    holm,
    permutation_omnibus,
    sidak,
)
from .reliability import (
    accuracy_flip_rate,
    cronbach_alpha,
    icc_2_1,
    icc_3_k,
    missing_flip_rate,
    per_case_run_sd,
    spearman_brown,
)
from .heterogeneity import (
    forest_plot_csv,
    i_squared,
    interaction_test,
    q_test_heterogeneity,
)
from .concordance import (
    bland_altman,
    count_mae,
    lin_ccc,
    mape,
    spearman_rho,
)
from .cascade_diag import (
    attrition_propensity,
    cascade_funnel,
    conditional_accuracy_grid,
)

__all__ = [
    # core
    "TestResult",
    # thresholds
    "HEADLINE_ACCURACY_DELTA_THRESHOLD",
    "HEADLINE_F1_DELTA_THRESHOLD",
    "KAPPA_SUBSTANTIAL",
    "KAPPA_NEAR_PERFECT",
    "FAMILYWISE_ALPHA",
    "FDR_ALPHA",
    "POWER_FLAG_THRESHOLD",
    "LANDIS_KOCH_BANDS",
    "kappa_band",
    # CI
    "BootstrapResult",
    "agresti_coull_ci",
    "bca_bootstrap_ci",
    "clopper_pearson_ci",
    "fisher_z_ci_for_corr",
    "jackknife_ci",
    "paired_bootstrap_diff",
    "pick_ci",
    "t_ci",
    "two_source_bootstrap_ci",
    "wilson_ci",
    # paired
    "bhapkar",
    "cochran_q",
    "friedman",
    "mcnemar",
    "nemenyi_posthoc",
    "paired_bootstrap_delta",
    "stuart_maxwell",
    # effect size
    "balanced_accuracy",
    "brier_score",
    "cliffs_delta",
    "cohens_d",
    "cohens_kappa",
    "krippendorff_alpha",
    "matthews_corrcoef",
    "weighted_kappa",
    # multiple comparisons
    "adjust_pvalues",
    "bh_fdr",
    "by_fdr",
    "holm",
    "permutation_omnibus",
    "sidak",
    # reliability
    "accuracy_flip_rate",
    "cronbach_alpha",
    "icc_2_1",
    "icc_3_k",
    "missing_flip_rate",
    "per_case_run_sd",
    "spearman_brown",
    # heterogeneity
    "forest_plot_csv",
    "i_squared",
    "interaction_test",
    "q_test_heterogeneity",
    # concordance
    "bland_altman",
    "count_mae",
    "lin_ccc",
    "mape",
    "spearman_rho",
    # cascade diagnostics
    "attrition_propensity",
    "cascade_funnel",
    "conditional_accuracy_grid",
]
