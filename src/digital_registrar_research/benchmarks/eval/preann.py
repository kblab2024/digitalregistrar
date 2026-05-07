"""Pre-annotation effect: paired analyses on with/without preann cohort.

The ``without_preann`` annotation set is a strict subset of the
``with_preann`` set — the same patients re-annotated from scratch by
the same human. Every metric here is a paired comparison on this case
intersection so observed differences cannot be attributed to case mix.

Headline metrics:
    paired_delta_kappa      — κ(human, gold) with vs. without preann
    convergence_to_preann   — P(human=preann) when preann was visible
    anchoring_index         — P(human=preann | with) − P(human=preann | without)
    disagreement_reduction  — Δ in inter-annotator κ between modes
    edit_distance           — number of fields the human changed away from preann

Implementation notes:
    - Cohen's κ via ``sklearn.metrics.cohen_kappa_score`` (Pedregosa et
      al. 2011); see Cohen (1960, 1968) for the original method.
    - Paired bootstrap via ``ci.paired_bootstrap_diff`` (in-house).

Convention: the input ``cases`` list contains aligned per-case records
``{"case_id", "organ", "field", "with_value", "without_value",
"preann_value", "gold_value"}`` already filtered to one (field, organ)
slice. Caller materialises this slice from the file system; this
module is pure-data.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .ci import wilson_ci
from .metrics import normalize

logger = logging.getLogger(__name__)


# --- Data containers ---------------------------------------------------------


@dataclass(frozen=True)
class PairedRecord:
    """One paired observation across with/without preann modes."""

    case_id: str
    organ: str
    field: str
    with_value: object
    without_value: object
    preann_value: object
    gold_value: object


# --- Building blocks ---------------------------------------------------------


def _eq(a: object, b: object) -> bool:
    """Normalised equality for primitive values (strings, ints, bools)."""
    return normalize(a) == normalize(b)


def _kappa(y_true: Sequence, y_pred: Sequence) -> float:
    """Cohen's κ via sklearn. NaN if classes are degenerate."""
    if len(y_true) == 0:
        return float("nan")
    from sklearn.metrics import cohen_kappa_score
    try:
        return float(cohen_kappa_score(y_true, y_pred))
    except Exception:
        return float("nan")


def _binary_correctness(records: Sequence[PairedRecord], side: str) -> np.ndarray:
    """Build a 0/1 vector of human-vs-gold correctness on the chosen side.

    ``side`` ∈ {"with", "without"}. Missing values count as 0 (incorrect)
    so the comparison stays balanced — caller restricts to records where
    ``gold_value`` is non-null upstream.
    """
    out = np.empty(len(records), dtype=int)
    for i, r in enumerate(records):
        v = r.with_value if side == "with" else r.without_value
        out[i] = 1 if _eq(v, r.gold_value) else 0
    return out


# --- Δκ headline -------------------------------------------------------------


def paired_delta_kappa(
    records: Sequence[PairedRecord],
    *,
    n_boot: int = 2000,
    random_state: int = 0,
    device: str = "cpu",
) -> dict:
    """Paired Δκ = κ(with_preann, gold) − κ(without_preann, gold).

    Computes both κ's on the SAME case set, then a paired-bootstrap CI
    on the difference using paired case-level resampling.

    Returns ``{kappa_with, kappa_without, delta, delta_ci_lo,
    delta_ci_hi, n_paired_cases}``.

    The bootstrap is routed through
    :func:`ci_gpu.paired_kappa_delta_ci` (vectorized confusion-matrix κ;
    optional GPU). For a typical ~50-200 case slice, the per-resample
    cost drops from a Python loop calling sklearn ``cohen_kappa_score``
    twice to one batched gather + two κ-from-confusion computations.
    """
    if not records:
        return {
            "kappa_with": float("nan"), "kappa_without": float("nan"),
            "delta": float("nan"), "delta_ci_lo": float("nan"),
            "delta_ci_hi": float("nan"), "n_paired_cases": 0,
        }
    from . import ci_gpu

    gold = [normalize(r.gold_value) for r in records]
    with_vals = [normalize(r.with_value) for r in records]
    without_vals = [normalize(r.without_value) for r in records]
    k_with = _kappa(gold, with_vals)
    k_without = _kappa(gold, without_vals)
    n = len(records)

    boot = ci_gpu.paired_kappa_delta_ci(
        gold, with_vals, gold, without_vals,
        n_boot=n_boot, alpha=0.05,
        random_state=random_state, device=device,
    )
    delta = (k_with - k_without) if (not np.isnan(k_with) and not np.isnan(k_without)) else float("nan")
    return {
        "kappa_with": k_with,
        "kappa_without": k_without,
        "delta": delta,
        "delta_ci_lo": boot.lo,
        "delta_ci_hi": boot.hi,
        "n_paired_cases": int(n),
    }


# --- Convergence-to-preann --------------------------------------------------


def convergence_to_preann(records: Sequence[PairedRecord]) -> dict:
    """Of cases where preann produced a non-null value, how often does
    the with-preann human's answer match preann?

    Stratified into ``preann_correct`` (preann == gold) vs
    ``preann_incorrect`` (preann ≠ gold) so we can tell good from bad
    anchoring.

    Returns counts + Wilson 95% CI on each rate.
    """
    base_n = matched = 0
    correct_n = correct_matched = 0
    incorrect_n = incorrect_matched = 0
    for r in records:
        if r.preann_value is None:
            continue
        base_n += 1
        is_match = _eq(r.with_value, r.preann_value)
        if is_match:
            matched += 1
        if _eq(r.preann_value, r.gold_value):
            correct_n += 1
            if is_match:
                correct_matched += 1
        else:
            incorrect_n += 1
            if is_match:
                incorrect_matched += 1

    p_overall = matched / base_n if base_n else float("nan")
    p_correct = correct_matched / correct_n if correct_n else float("nan")
    p_incorrect = incorrect_matched / incorrect_n if incorrect_n else float("nan")
    return {
        "p_human_eq_preann": p_overall,
        "p_human_eq_preann_ci_lo": wilson_ci(matched, base_n)[0] if base_n else float("nan"),
        "p_human_eq_preann_ci_hi": wilson_ci(matched, base_n)[1] if base_n else float("nan"),
        "p_when_preann_correct": p_correct,
        "p_when_preann_incorrect": p_incorrect,
        "n_with_preann": int(base_n),
        "n_preann_correct": int(correct_n),
        "n_preann_incorrect": int(incorrect_n),
    }


# --- Anchoring index --------------------------------------------------------


def anchoring_index(records: Sequence[PairedRecord]) -> dict:
    """Anchoring index = P(human=preann | with) − P(human=preann | without).

    A non-zero AI means showing preann to the human shifted their
    answer toward preann (positive AI) or away from it (negative AI;
    rare). Stratified by whether preann was correct vs incorrect.

    Returns ``{ai_overall, ai_correct, ai_incorrect, n_with, n_without}``.
    """
    n = len(records)
    if n == 0:
        return {"ai_overall": float("nan"),
                "ai_correct": float("nan"),
                "ai_incorrect": float("nan"),
                "n": 0}

    eligible = [r for r in records if r.preann_value is not None]
    if not eligible:
        return {"ai_overall": float("nan"),
                "ai_correct": float("nan"),
                "ai_incorrect": float("nan"),
                "n": 0}

    def _share(side: str, subset: Sequence[PairedRecord]) -> float:
        if not subset:
            return float("nan")
        m = sum(1 for r in subset
                if _eq(r.with_value if side == "with" else r.without_value,
                       r.preann_value))
        return m / len(subset)

    overall = _share("with", eligible) - _share("without", eligible)

    correct_subset = [r for r in eligible if _eq(r.preann_value, r.gold_value)]
    incorrect_subset = [r for r in eligible if not _eq(r.preann_value, r.gold_value)]

    return {
        "ai_overall": overall,
        "ai_correct": _share("with", correct_subset) - _share("without", correct_subset),
        "ai_incorrect": _share("with", incorrect_subset) - _share("without", incorrect_subset),
        "n": len(eligible),
        "n_correct": len(correct_subset),
        "n_incorrect": len(incorrect_subset),
    }


# --- Disagreement reduction --------------------------------------------------


@dataclass(frozen=True)
class DualPairedRecord:
    """A single (case, field) observation across two annotators × two modes."""

    case_id: str
    organ: str
    field: str
    a_with: object
    a_without: object
    b_with: object
    b_without: object


def disagreement_reduction(
    records: Sequence[DualPairedRecord],
    *,
    n_boot: int = 2000,
    random_state: int = 0,
    device: str = "cpu",
) -> dict:
    """Δ-disagreement = (1 − κ(a_with, b_with)) − (1 − κ(a_without, b_without)).

    Negative Δ means preann reduced inter-annotator disagreement
    (annotators converge with preann). Positive Δ means preann
    increased disagreement.

    Paired bootstrap CI on Δ via case-level resampling, routed through
    :func:`ci_gpu.paired_kappa_delta_ci`. The CI is reported on the
    Δκ scale (k_with − k_without); ``delta_disagreement`` flips the sign.
    """
    if not records:
        return {"k_with": float("nan"), "k_without": float("nan"),
                "delta_kappa": float("nan"),
                "delta_disagreement": float("nan"),
                "delta_ci_lo": float("nan"), "delta_ci_hi": float("nan"),
                "n": 0}

    from . import ci_gpu

    a_with = [normalize(r.a_with) for r in records]
    b_with = [normalize(r.b_with) for r in records]
    a_without = [normalize(r.a_without) for r in records]
    b_without = [normalize(r.b_without) for r in records]

    k_with = _kappa(a_with, b_with)
    k_without = _kappa(a_without, b_without)
    delta_kappa = (k_with - k_without
                   if (not np.isnan(k_with) and not np.isnan(k_without))
                   else float("nan"))
    delta_disag = (-delta_kappa if not np.isnan(delta_kappa) else float("nan"))

    boot = ci_gpu.paired_kappa_delta_ci(
        a_with, b_with, a_without, b_without,
        n_boot=n_boot, alpha=0.05,
        random_state=random_state, device=device,
    )
    # The original returned the CI on (1 - k_w) - (1 - k_n) = k_n - k_w
    # = -(k_w - k_n). bootstrap is on Δκ = k_w - k_n; flip endpoints.
    if np.isnan(boot.lo) or np.isnan(boot.hi):
        lo, hi = float("nan"), float("nan")
    else:
        lo, hi = -boot.hi, -boot.lo

    return {
        "k_with": k_with, "k_without": k_without,
        "delta_kappa": delta_kappa,
        "delta_disagreement": delta_disag,
        "delta_ci_lo": lo, "delta_ci_hi": hi,
        "n": int(len(records)),
    }


# --- Edit-distance from preann ----------------------------------------------


def edit_distance_from_preann(
    case_records: Sequence[Sequence[PairedRecord]],
) -> dict:
    """Per-case edit distance between with-preann human and preann.

    ``case_records`` is a list of per-case lists of PairedRecords (one
    list per case, with one entry per field). Returns a summary dict
    with mean, median, max, and the share of fields edited per case.
    """
    n_changes_list: list[int] = []
    n_fields_list: list[int] = []
    edited_share_list: list[float] = []
    for fields in case_records:
        if not fields:
            continue
        n_changes = 0
        n_fields = 0
        for r in fields:
            if r.preann_value is None:
                continue
            n_fields += 1
            if not _eq(r.with_value, r.preann_value):
                n_changes += 1
        if n_fields == 0:
            continue
        n_changes_list.append(n_changes)
        n_fields_list.append(n_fields)
        edited_share_list.append(n_changes / n_fields)
    if not n_changes_list:
        return {"mean_changes": float("nan"), "median_changes": float("nan"),
                "max_changes": 0, "mean_share_edited": float("nan"),
                "n_cases": 0}
    arr = np.asarray(n_changes_list, dtype=float)
    return {
        "mean_changes": float(arr.mean()),
        "median_changes": float(np.median(arr)),
        "max_changes": int(arr.max()),
        "mean_share_edited": float(np.mean(edited_share_list)),
        "n_cases": len(n_changes_list),
    }


__all__ = [
    "PairedRecord",
    "DualPairedRecord",
    "paired_delta_kappa",
    "convergence_to_preann",
    "anchoring_index",
    "disagreement_reduction",
    "edit_distance_from_preann",
]
