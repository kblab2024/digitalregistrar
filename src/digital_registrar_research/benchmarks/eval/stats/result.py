"""Common output schema for every statistical test in the cascade.

A ``TestResult`` is what every test function in ``paired``, ``heterogeneity``,
and ``cascade_diag`` returns. Reductions consume them uniformly so each
chapter's CSV layout is consistent regardless of which test produced
the row.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class TestResult:
    """Result of a single statistical test.

    Fields
    ------
    name
        Short identifier, e.g. ``"mcnemar"``, ``"stuart_maxwell"``.
    statistic
        Test statistic (chi-square, F, etc.). NaN when not applicable.
    df
        Degrees of freedom. None when not applicable (e.g. exact tests).
    p_raw
        Unadjusted p-value.
    p_adjusted_holm
        Holm-Bonferroni adjusted p-value within whatever family the
        caller declared. NaN when not part of a multi-test family.
    p_adjusted_bh
        Benjamini-Hochberg FDR adjusted p-value within the family.
    effect_size
        Effect-size estimate (Cohen's d, Cliff's delta, accuracy delta,
        kappa, Lin's CCC, etc.). NaN when not reported.
    effect_kind
        Descriptor of the effect size, e.g. ``"accuracy_delta"``,
        ``"cohens_kappa"``, ``"lin_ccc"``.
    effect_ci_lo, effect_ci_hi
        Bounds of the effect-size CI at the test's alpha level.
    n
        Sample size used in the test.
    low_power
        True when post-hoc power at alpha=0.05 and the per-domain
        minimum-detectable-effect is below ``thresholds.POWER_FLAG_THRESHOLD``.
        Surfaces "we didn't see a difference because n was tiny" cases.
    notes
        Free-text qualifications: degenerate input that triggered a
        fallback method, exact-binomial used because b+c<25, etc.
    extra
        Method-specific payload (e.g. contingency-table cells for
        Stuart-Maxwell). Keys are stable per ``name``.
    """

    name: str
    statistic: float = float("nan")
    df: int | None = None
    p_raw: float = float("nan")
    p_adjusted_holm: float = float("nan")
    p_adjusted_bh: float = float("nan")
    effect_size: float = float("nan")
    effect_kind: str = ""
    effect_ci_lo: float = float("nan")
    effect_ci_hi: float = float("nan")
    n: int = 0
    low_power: bool = False
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        """Flat dict suitable for DataFrame ingestion. ``extra`` is dropped."""
        d = asdict(self)
        d.pop("extra", None)
        return d


__all__ = ["TestResult"]
