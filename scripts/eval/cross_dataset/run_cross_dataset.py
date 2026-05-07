"""Cross-dataset comparison.

Takes two ``--cascade-out`` directories (each containing
``cascade_atomic.parquet``) and emits per-field Δ + distribution-shift
indicators.

Operates on **Stage-C scalar** rows for the per-field accuracy delta.
The distribution-shift table considers gold values across all stages so
eligibility / organ shift is also surfaced (each row carries its
``cascade_stage``).

Output tree:
    manifest.json
    per_field_delta.csv         (Stage-C scalar)
    distribution_shift.csv      (per-field gold-distribution shift, all stages)
    transferability.csv
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from digital_registrar_research.benchmarks.eval import ci_gpu
from digital_registrar_research.benchmarks.eval.ci import paired_bootstrap_diff

from .._common.loaders import (
    cascade_scalar_only, coerce_cascade_bool, coerce_cascade_correct,
)
from .._common.reporting import setup_logging, write_csv, write_manifest
from .._common.stats_extra import (
    jensen_shannon, kl_divergence, wasserstein,
)

logger = logging.getLogger("scripts.eval.cross_dataset")


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "cross_dataset",
        help="Cross-dataset Δ accuracy + distribution-shift indicators.",
        description=__doc__,
    )
    parser.add_argument(
        "--left", type=Path, required=True,
        help="Left cascade output directory containing cascade_atomic.parquet "
             "(typically CMUH).",
    )
    parser.add_argument(
        "--right", type=Path, required=True,
        help="Right cascade output directory containing cascade_atomic.parquet "
             "(typically TCGA).",
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("workspace") / "results" / "eval" / "cross_dataset",
        help="Output directory (default: %(default)s).",
    )
    parser.add_argument(
        "--n-boot", type=int, default=2000,
    )
    parser.add_argument(
        "--seed", type=int, default=0,
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="cpu",
        help="Device for the per-field-Δ bootstrap. Default: cpu.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
    )
    parser.set_defaults(_handler=_main)


def _main(args: argparse.Namespace) -> int:
    setup_logging(args.verbose)
    args.out.mkdir(parents=True, exist_ok=True)

    requested_device = getattr(args, "device", "cpu")
    resolved_device = ci_gpu.pick_device(requested_device)
    logger.info("device: requested=%s resolved=%s",
                requested_device, resolved_device)

    left_full = _read_cascade_atomic(args.left)
    right_full = _read_cascade_atomic(args.right)
    left_atomic = cascade_scalar_only(left_full)
    right_atomic = cascade_scalar_only(right_full)
    left_label = _label(args.left)
    right_label = _label(args.right)
    logger.info(
        "loaded cascade atomic: left=%d full / %d Stage-C scalar; "
        "right=%d full / %d Stage-C scalar",
        len(left_full), len(left_atomic),
        len(right_full), len(right_atomic),
    )

    # --- Per-field Δ ---------------------------------------------------
    per_field_delta = _per_field_delta(
        left_atomic, right_atomic,
        left_label=left_label, right_label=right_label,
        n_boot=args.n_boot, seed=args.seed,
        device=resolved_device,
    )
    write_csv(per_field_delta, args.out / "per_field_delta.csv")

    # --- Transferability ------------------------------------------------
    trans = _transferability(per_field_delta)
    write_csv(trans, args.out / "transferability.csv")

    # --- Distribution shift --------------------------------------------
    # Use the FULL atomic so eligibility (Stage A) + organ (Stage B) shift
    # are surfaced too. The function stamps `cascade_stage` per row.
    shift = _distribution_shift(left_full, right_full,
                                left_label=left_label, right_label=right_label)
    write_csv(shift, args.out / "distribution_shift.csv")

    write_manifest(
        args.out, args, subcommand="cross_dataset",
        extra={
            "left": str(args.left), "right": str(args.right),
            "left_label": left_label, "right_label": right_label,
            "n_left_rows_total": int(len(left_full)),
            "n_right_rows_total": int(len(right_full)),
            "n_left_rows_stage_c_scalar": int(len(left_atomic)),
            "n_right_rows_stage_c_scalar": int(len(right_atomic)),
            "device_requested": requested_device,
            "device_resolved": resolved_device,
        },
    )
    logger.info("done. outputs in %s", args.out)
    return 0


def _read_cascade_atomic(path: Path) -> pd.DataFrame:
    """Read ``cascade_atomic.parquet`` from a cascade output directory.

    Coerces JSON-stringified ``correct`` / boolean columns back to native
    types — see :func:`scripts.eval._common.loaders.coerce_cascade_correct`.
    """
    parquet = path / "cascade_atomic.parquet"
    if not parquet.is_file():
        raise SystemExit(
            f"missing {parquet}; run `python -m scripts.eval.cli cascade` first.",
        )
    df = pd.read_parquet(parquet)
    if "correct" in df.columns:
        df["correct"] = coerce_cascade_correct(df["correct"])
    for col in ("attempted", "gold_present", "field_missing", "parse_error"):
        if col in df.columns:
            df[col] = coerce_cascade_bool(df[col])
    return df


def _label(path: Path) -> str:
    """Infer a short label from the manifest, falling back to the dir name."""
    manifest_path = path / "manifest.json"
    if manifest_path.is_file():
        import json
        try:
            with manifest_path.open(encoding="utf-8") as f:
                m = json.load(f)
            return str(m.get("args", {}).get("dataset") or path.name)
        except Exception:
            pass
    return path.name


def _per_field_delta(
    left: pd.DataFrame, right: pd.DataFrame,
    *, left_label: str, right_label: str,
    n_boot: int, seed: int,
    device: str = "cpu",
) -> pd.DataFrame:
    """Per (organ, field) Δ-accuracy with bootstrap CI on the difference.

    Note: cases are NOT paired across datasets (different patients), so
    we use independent bootstrap on each side and report the Δ of means
    with the unpaired-bootstrap CI on the difference.

    The bootstrap is routed through :func:`ci_gpu.independent_bootstrap_diff`
    which preserves the original draw stream on a CPU device (so output
    matches pre-change byte-for-byte at the same seed) and offloads the
    gather + reduction onto cuda/mps when requested.
    """
    rows: list[dict] = []
    fields = sorted(set(left["field"].dropna()) & set(right["field"].dropna()))
    organs = sorted(set(left["organ"].dropna()) & set(right["organ"].dropna())) + ["ALL"]
    for organ in organs:
        for field in fields:
            l = _accuracy_vec(left, organ=organ, field=field)
            r = _accuracy_vec(right, organ=organ, field=field)
            if l.size == 0 or r.size == 0:
                continue
            mean_l = float(l.mean())
            mean_r = float(r.mean())
            delta = mean_l - mean_r
            boot_result = ci_gpu.independent_bootstrap_diff(
                l, r, n_boot=n_boot, alpha=0.05,
                random_state=seed, device=device,
            )
            rows.append({
                "organ": organ, "field": field,
                "left_label": left_label, "right_label": right_label,
                "n_left": int(l.size), "n_right": int(r.size),
                "left_accuracy": mean_l,
                "right_accuracy": mean_r,
                "delta": delta,
                "delta_ci_lo": boot_result.lo, "delta_ci_hi": boot_result.hi,
            })
    return pd.DataFrame(rows)


def _accuracy_vec(df: pd.DataFrame, *, organ: str, field: str) -> np.ndarray:
    sub = df[(df["field"] == field) & df["attempted"]]
    if organ != "ALL":
        sub = sub[sub["organ"] == organ]
    return sub["correct"].astype(float).to_numpy()


def _transferability(per_field_delta: pd.DataFrame) -> pd.DataFrame:
    """Per-organ summary: median |Δ|, mean Δ, fraction of fields where
    one dataset is ahead."""
    rows: list[dict] = []
    for organ, sub in per_field_delta.groupby("organ"):
        rows.append({
            "organ": organ,
            "n_fields": int(len(sub)),
            "mean_delta": float(sub["delta"].mean()),
            "median_abs_delta": float(sub["delta"].abs().median()),
            "frac_left_ahead": float((sub["delta"] > 0).mean()),
            "frac_right_ahead": float((sub["delta"] < 0).mean()),
        })
    return pd.DataFrame(rows)


def _distribution_shift(
    left: pd.DataFrame, right: pd.DataFrame,
    *, left_label: str, right_label: str,
) -> pd.DataFrame:
    """Per-(stage, field) gold-class distribution shift indicators.

    For categorical fields: Jensen-Shannon distance, KL divergence,
    chi-square p-value. For continuous fields: Wasserstein-1 distance.
    Each output row carries its ``cascade_stage`` so a reader can
    subset to eligibility (A), organ (B), or field-extraction (C) shift.
    Stage-C nested-list rows are excluded — their gold side is a list of
    dicts, not a scalar enum/number.
    """
    rows: list[dict] = []
    # Bucket each side by (cascade_stage, field). For Stage C, drop
    # nested-list rows so we don't choke on list/dict gold values.
    def _stages_present(df: pd.DataFrame) -> list[str]:
        if "cascade_stage" not in df.columns:
            return [""]
        return sorted(df["cascade_stage"].dropna().unique().tolist())

    stages = sorted(set(_stages_present(left)) & set(_stages_present(right)))
    for stage in stages:
        l_stage = left if not stage else left[left["cascade_stage"] == stage]
        r_stage = right if not stage else right[right["cascade_stage"] == stage]
        if stage == "C" and "field_kind" in l_stage.columns:
            l_stage = l_stage[l_stage["field_kind"] != "nested_list"]
        if stage == "C" and "field_kind" in r_stage.columns:
            r_stage = r_stage[r_stage["field_kind"] != "nested_list"]
        fields = sorted(
            set(l_stage["field"].dropna())
            & set(r_stage["field"].dropna()),
        )
        for field in fields:
            l = l_stage[
                (l_stage["field"] == field) & l_stage["gold_present"].fillna(False)
            ]["gold_value"].astype(str)
            r = r_stage[
                (r_stage["field"] == field) & r_stage["gold_present"].fillna(False)
            ]["gold_value"].astype(str)
            if l.empty or r.empty:
                continue
            # Try numeric first.
            try:
                l_num = l.astype(float).to_numpy()
                r_num = r.astype(float).to_numpy()
                wd = wasserstein(l_num, r_num)
                rows.append({
                    "cascade_stage": stage,
                    "field": field, "kind": "continuous",
                    "left_label": left_label, "right_label": right_label,
                    "n_left": int(l.size), "n_right": int(r.size),
                    "wasserstein": wd,
                    "js_distance": float("nan"), "kl": float("nan"),
                    "chi2_p": float("nan"),
                })
                continue
            except (ValueError, TypeError):
                pass
            # Categorical
            cats = sorted(set(l) | set(r))
            l_counts = np.array([float((l == c).sum()) for c in cats])
            r_counts = np.array([float((r == c).sum()) for c in cats])
            l_p = l_counts / l_counts.sum()
            r_p = r_counts / r_counts.sum()
            # Avoid zeros for KL.
            eps = 1e-9
            kl = float(kl_divergence(l_p + eps, r_p + eps))
            js = jensen_shannon(l_p, r_p)
            from scipy.stats import chi2_contingency
            try:
                _, p_chi, _, _ = chi2_contingency([l_counts, r_counts])
                p_val = float(p_chi)
            except Exception:
                p_val = float("nan")
            rows.append({
                "cascade_stage": stage,
                "field": field, "kind": "categorical",
                "left_label": left_label, "right_label": right_label,
                "n_left": int(l.size), "n_right": int(r.size),
                "js_distance": js, "kl": kl, "chi2_p": p_val,
                "wasserstein": float("nan"),
            })
    return pd.DataFrame(rows)


__all__ = ["register"]
