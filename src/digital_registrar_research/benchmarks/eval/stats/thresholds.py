"""Pre-registered effect-size and significance thresholds.

These constants are fixed before any cascade run is interpreted, so the
analysis cannot be steered by the data. Any change here must be a
*pre-registration amendment* with explicit justification — do not nudge
to make a result significant.

References
----------
- Landis & Koch (1977), "The measurement of observer agreement for
  categorical data," for the kappa interpretation bands.
- Cohen (1988), "Statistical Power Analysis for the Behavioral Sciences,"
  for the d-band convention referenced by ``cohens_d`` callers.
"""
from __future__ import annotations

# --- Effect-size minimums for "clinically meaningful" claims ----------------
# Below these thresholds, a difference is reported with CI but not framed
# as a *difference* in the manuscript.

HEADLINE_ACCURACY_DELTA_THRESHOLD: float = 0.02
"""Per-field accuracy delta below which we do not claim a difference."""

HEADLINE_F1_DELTA_THRESHOLD: float = 0.03
"""Per-field F1 delta below which we do not claim a difference."""


# --- Kappa interpretation bands (Landis-Koch 1977) --------------------------

LANDIS_KOCH_BANDS: list[tuple[float, str]] = [
    (-1.0, "poor"),         # < 0.0
    (0.0,  "slight"),       # 0.00–0.20
    (0.20, "fair"),         # 0.21–0.40
    (0.40, "moderate"),     # 0.41–0.60
    (0.60, "substantial"),  # 0.61–0.80
    (0.80, "almost_perfect"),  # 0.81–1.00
]
"""Lower-bound thresholds (inclusive) for kappa-class labels."""

KAPPA_SUBSTANTIAL: float = 0.80
"""Below this, agreement is at most 'substantial' on Landis-Koch."""

KAPPA_NEAR_PERFECT: float = 0.90
"""Internal threshold for 'near-perfect' agreement claims in the manuscript."""


def kappa_band(value: float) -> str:
    """Return the Landis-Koch label for a kappa value.

    NaN passes through as ``"undefined"``.
    """
    if value is None or value != value:  # NaN check
        return "undefined"
    label = "poor"
    for threshold, name in LANDIS_KOCH_BANDS:
        if value >= threshold:
            label = name
    return label


# --- Significance --------------------------------------------------------

FAMILYWISE_ALPHA: float = 0.05
"""Alpha for Holm-corrected family-wise error rate within each chapter
primary endpoint family."""

FDR_ALPHA: float = 0.10
"""Alpha for FDR control on the per-field × per-organ exploratory grid.
Larger than FAMILYWISE_ALPHA because exploratory cells warrant more
discovery latitude."""


# --- Power flagging ------------------------------------------------------

POWER_FLAG_THRESHOLD: float = 0.30
"""Post-hoc power below this is flagged as ``low_power=True`` on the
test result. Cells that fail this bar are not removed from output —
they are surfaced honestly so reviewers see "we tested, didn't have
the cohort to detect."""


__all__ = [
    "HEADLINE_ACCURACY_DELTA_THRESHOLD",
    "HEADLINE_F1_DELTA_THRESHOLD",
    "LANDIS_KOCH_BANDS",
    "KAPPA_SUBSTANTIAL",
    "KAPPA_NEAR_PERFECT",
    "kappa_band",
    "FAMILYWISE_ALPHA",
    "FDR_ALPHA",
    "POWER_FLAG_THRESHOLD",
]
