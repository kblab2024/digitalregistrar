"""Markdown rendering for the iaa_pair subcommand.

Pure string templating from a HeadlineRow list and the per-field /
per-section / per-organ DataFrames produced by
:mod:`...benchmarks.eval.iaa_headline`. No file IO; the orchestrator
writes the returned string to disk.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import pandas as pd

from digital_registrar_research.benchmarks.eval.iaa_headline import HeadlineRow

_HEADLINE_LABEL = {
    "mean_per_field_kappa": "Mean per-field κ",
    "n_weighted_mean_per_field_kappa": "n-weighted mean per-field κ",
    "pooled_categorical_kappa": "Pooled categorical κ",
    "agree_disagree_pabak": "Agree/disagree PABAK",
    "case_exact_match_rate": "Case exact-match rate",
    "krippendorff_alpha_nominal": "Krippendorff α (nominal)",
    "krippendorff_alpha_ordinal": "Krippendorff α (ordinal)",
    "krippendorff_alpha_interval": "Krippendorff α (interval)",
}

_SECTION_LABEL = {
    "top_level": "Top level",
    "scalar_pathology": "Scalar pathology",
    "nested": "Nested lists",
}


def _fmt_float(v: float | None, *, places: int = 3) -> str:
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    if math.isnan(f):
        return "—"
    return f"{f:.{places}f}"


def _fmt_ci(lo: float, hi: float) -> str:
    if lo is None or hi is None:
        return "—"
    if math.isnan(float(lo)) or math.isnan(float(hi)):
        return "—"
    return f"[{float(lo):.3f}, {float(hi):.3f}]"


def _is_within_annotator_preann_pair(ann_a: str, ann_b: str) -> bool:
    """True when both sides share an `nhc_*` or `kpc_*` prefix.

    Such pairs measure preann effect within one annotator; the headline
    κ is descriptive of preann-induced *change*, not whether preann
    helps.
    """
    for prefix in ("nhc_", "kpc_"):
        if ann_a.startswith(prefix) and ann_b.startswith(prefix):
            return True
    return False


def render_summary(
    *,
    ann_a: str,
    ann_b: str,
    headline_rows: Sequence[HeadlineRow],
    per_field_df: pd.DataFrame,
    per_section_df: pd.DataFrame,
    per_organ_df: pd.DataFrame,
    confusion_files: Sequence[tuple[str, str]],
    n_cases: int,
    n_boot: int,
    top_k: int = 10,
) -> str:
    """Build the markdown body for ``pair_<a>_vs_<b>/summary.md``.

    ``confusion_files`` is a sequence of (field, relative_path) for the
    confusion matrices that were written. ``per_field_df`` is expected
    to be already-filtered to organ='ALL' κ rows.
    """
    lines: list[str] = []
    lines.append(f"# IAA: {ann_a} vs {ann_b}")
    n_fields = (
        int(per_field_df[per_field_df["organ"] == "ALL"]["field"].nunique())
        if not per_field_df.empty else 0
    )
    lines.append(
        f"n cases: {n_cases}    n fields scored: {n_fields}    "
        f"bootstrap: {n_boot} case-level"
    )
    lines.append("")

    if _is_within_annotator_preann_pair(ann_a, ann_b):
        lines.append(
            "> **Within-annotator preann pair.** κ here measures stability "
            "under preann, not whether preann helps. For causal Δκ-vs-gold "
            "and anchoring analyses, see the `iaa` subcommand's `preann/` "
            "outputs."
        )
        lines.append("")

    # --- Headline κ table ---------------------------------------------------
    lines.append("## Headline κ")
    lines.append("")
    lines.append("| Statistic | Estimate | 95% CI | n | Note |")
    lines.append("|---|---:|:---|---:|:---|")
    for r in headline_rows:
        label = _HEADLINE_LABEL.get(r.stat_name, r.stat_name)
        lines.append(
            f"| {label} | {_fmt_float(r.estimate)} | "
            f"{_fmt_ci(r.ci_lo, r.ci_hi)} | {r.n} | "
            f"{r.note or ''} |"
        )
    lines.append("")

    # --- Per-section roll-up -----------------------------------------------
    lines.append("## Per-section roll-up    → per_section.csv")
    lines.append("")
    if per_section_df.empty:
        lines.append("_No section data._")
    else:
        lines.append("| Section | Statistic | Estimate | n |")
        lines.append("|---|---|---:|---:|")
        for _i, r in per_section_df.iterrows():
            label = _SECTION_LABEL.get(str(r["section"]), str(r["section"]))
            lines.append(
                f"| {label} | {r['stat_name']} | "
                f"{_fmt_float(r['estimate'])} | {int(r['n'])} |"
            )
    lines.append("")

    # --- Per-organ roll-up --------------------------------------------------
    lines.append("## Per-organ roll-up      → per_organ.csv")
    lines.append("")
    if per_organ_df.empty:
        lines.append("_No organ data._")
    else:
        # Pivot into one row per organ × four κ stats for readability.
        try:
            wide = per_organ_df.pivot_table(
                index="organ", columns="stat_name", values="estimate",
                aggfunc="first",
            )
            n_cases_per_organ = (
                per_organ_df.drop_duplicates("organ").set_index("organ")["n_cases"]
            )
            lines.append(
                "| Organ | n cases | mean κ | n-weighted mean κ | "
                "pooled categorical κ | agree/disagree PABAK |"
            )
            lines.append("|---|---:|---:|---:|---:|---:|")
            for organ in wide.index:
                row = wide.loc[organ]
                lines.append(
                    f"| {organ} | {int(n_cases_per_organ.get(organ, 0))} | "
                    f"{_fmt_float(row.get('mean_per_field_kappa'))} | "
                    f"{_fmt_float(row.get('n_weighted_mean_per_field_kappa'))} | "
                    f"{_fmt_float(row.get('pooled_categorical_kappa'))} | "
                    f"{_fmt_float(row.get('agree_disagree_pabak'))} |"
                )
        except Exception:
            # Fall back to long-form dump on any pivot edge case.
            lines.append("| Organ | Statistic | Estimate | n cases |")
            lines.append("|---|---|---:|---:|")
            for _i, r in per_organ_df.iterrows():
                lines.append(
                    f"| {r['organ']} | {r['stat_name']} | "
                    f"{_fmt_float(r['estimate'])} | {int(r['n_cases'])} |"
                )
    lines.append("")

    # --- Top-K disagreements -----------------------------------------------
    lines.append(
        f"## Top {top_k} most disagreed-on fields    → per_field_kappa.csv"
    )
    lines.append("")
    if per_field_df.empty:
        lines.append("_No fields scored._")
    else:
        sub = per_field_df.copy()
        sub["estimate_num"] = pd.to_numeric(sub["estimate"], errors="coerce")
        sub = sub.dropna(subset=["estimate_num"]).sort_values("estimate_num")
        sub = sub.head(top_k)
        lines.append("| Rank | Field | Field type | κ | n | Observed agreement |")
        lines.append("|---:|---|---|---:|---:|---:|")
        for rank, (_i, r) in enumerate(sub.iterrows(), start=1):
            lines.append(
                f"| {rank} | {r['field']} | {r['field_type']} | "
                f"{_fmt_float(r['estimate_num'])} | "
                f"{int(r['n']) if not pd.isna(r['n']) else 0} | "
                f"{_fmt_float(r.get('observed_agreement'))} |"
            )
    lines.append("")

    # --- Confusion matrix index --------------------------------------------
    lines.append("## Confusion matrices")
    lines.append("")
    if not confusion_files:
        lines.append("_No categorical fields with disagreements._")
    else:
        for field, rel_path in confusion_files:
            lines.append(f"- **{field}** → [`{rel_path}`]({rel_path})")
    lines.append("")
    return "\n".join(lines)


__all__ = ["render_summary"]
