"""Paired comparison of two runs scored against the same gold annotations.

Backs ``registrar-eval compare``. Input is a long-form table from
:func:`metrics.score_pairs` holding both runs (distinguished by the
``method`` column); output is one row per (stage, field, metric) with
each run's score on the cases where BOTH runs were scored, the paired
difference, a paired-bootstrap CI, and — for accuracy rows — McNemar's
test.

Pairing is on the intersection: a (case, field) row counts only if both
runs attempted it. Under cascade scoring a run that fails the
eligibility or organ gate for a case has no Stage-C rows for that case,
so ``n_attempted_a`` / ``n_attempted_b`` are reported alongside
``n_paired`` to make gating differences visible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .ci import mcnemar_test, paired_bootstrap_diff

KEYS = ["stage", "field", "metric"]

COLUMNS = [
    *KEYS, "method_a", "method_b", "n_attempted_a", "n_attempted_b",
    "n_paired", "score_a", "score_b", "delta", "delta_ci_lo", "delta_ci_hi",
    "mcnemar_b", "mcnemar_c", "mcnemar_p_value",
]


def _attempted_scores(atomic: pd.DataFrame, method: str) -> pd.DataFrame:
    sub = atomic[(atomic["method"] == method) & atomic["attempted"].astype(bool)]
    sub = sub[["case_id", *KEYS, "correct"]].copy()
    sub["score"] = pd.to_numeric(sub["correct"], errors="coerce").astype(float)
    return sub.drop(columns="correct").dropna(subset=["score"])


def _row(keys: dict, method_a: str, method_b: str, n_a: int, n_b: int,
         a: np.ndarray, b: np.ndarray, *, binary: bool, n_boot: int,
         seed: int) -> dict:
    boot = paired_bootstrap_diff(a, b, n_boot=n_boot, random_state=seed)
    row = {
        **keys, "method_a": method_a, "method_b": method_b,
        "n_attempted_a": n_a, "n_attempted_b": n_b, "n_paired": int(a.size),
        "score_a": float(a.mean()) if a.size else float("nan"),
        "score_b": float(b.mean()) if b.size else float("nan"),
        "delta": boot.point, "delta_ci_lo": boot.lo, "delta_ci_hi": boot.hi,
        "mcnemar_b": np.nan, "mcnemar_c": np.nan, "mcnemar_p_value": np.nan,
    }
    if binary and a.size:
        n_b_only = int(((a == 1) & (b == 0)).sum())
        n_c_only = int(((a == 0) & (b == 1)).sum())
        mc = mcnemar_test(n_b_only, n_c_only)
        row.update({"mcnemar_b": n_b_only, "mcnemar_c": n_c_only,
                    "mcnemar_p_value": mc["p_value"]})
    return row


def compare_runs(
    atomic: pd.DataFrame,
    method_a: str,
    method_b: str,
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Per-field paired comparison of ``method_a`` vs ``method_b``.

    ``delta = score_a - score_b`` (positive → A is better). ``score_*``
    is accuracy for ``metric == "accuracy"`` rows and mean per-case F1
    for ``metric == "f1"`` rows. The CI is a percentile paired bootstrap
    over cases (:func:`ci.paired_bootstrap_diff`); McNemar
    (:func:`ci.mcnemar_test`) is reported for accuracy rows only, with
    ``mcnemar_b`` = A right & B wrong and ``mcnemar_c`` = A wrong & B right.

    The final row (``field == "ALL"``) compares per-case mean accuracy
    over every paired accuracy row, bootstrapped over cases.
    """
    a = _attempted_scores(atomic, method_a)
    b = _attempted_scores(atomic, method_b)
    paired = a.merge(b, on=["case_id", *KEYS], suffixes=("_a", "_b"))

    n_a = a.groupby(KEYS).size()
    n_b = b.groupby(KEYS).size()
    all_keys = sorted(set(n_a.index) | set(n_b.index))
    by_key = {k: g for k, g in paired.groupby(KEYS)}

    rows: list[dict] = []
    for key in all_keys:
        grp = by_key.get(key)
        sa = grp["score_a"].to_numpy() if grp is not None else np.array([])
        sb = grp["score_b"].to_numpy() if grp is not None else np.array([])
        rows.append(_row(
            dict(zip(KEYS, key, strict=True)), method_a, method_b,
            int(n_a.get(key, 0)), int(n_b.get(key, 0)), sa, sb,
            binary=key[2] == "accuracy", n_boot=n_boot, seed=seed,
        ))

    acc = paired[paired["metric"] == "accuracy"]
    per_case = acc.groupby("case_id")[["score_a", "score_b"]].mean()
    rows.append(_row(
        {"stage": "ALL", "field": "ALL", "metric": "accuracy"},
        method_a, method_b,
        int(a.loc[a["metric"] == "accuracy", "case_id"].nunique()),
        int(b.loc[b["metric"] == "accuracy", "case_id"].nunique()),
        per_case["score_a"].to_numpy(), per_case["score_b"].to_numpy(),
        binary=False, n_boot=n_boot, seed=seed,
    ))
    return pd.DataFrame(rows, columns=COLUMNS)


__all__ = ["compare_runs"]
