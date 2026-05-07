#!/usr/bin/env python3
"""Build the canonical statistics suite over one or more cascade atomic
parquets.

Reads ``cascade_atomic.parquet`` from each ``--cascade-out LABEL=PATH``
entry, stamps ``method = LABEL``, concatenates, and runs
``ablations.eval.canonical_stats.run_canonical_stats`` to emit eight
canonical CSVs plus a markdown run report.

Each label becomes the row's ``method`` value. The ``--modular-method``
argument names the label to compare against (Δ / McNemar / OR
baseline). Multiple labels are supported, including a mix of LLM
runs and ablation cells — both produce the same cascade_atomic schema
so unification is mechanical.

Output layout::

    {folder}/results/canonical/{dataset}/
        master_atomic.parquet
        headline.csv
        failure_modes.csv
        per_field.csv
        per_organ.csv
        seed_consistency.csv          (only when multi-run)
        modularity_advantage.csv
        low_performer_diagnostics.csv
        canonical_stats_report.md

Usage::

    python scripts/eval/canonical/make_paper_tables.py \\
        --folder workspace --dataset cmuh \\
        --modular-method dspy_modular_gpt_oss_20b \\
        --cascade-out dspy_modular_gpt_oss_20b=workspace/results/eval/cascade/cmuh_modular \\
        --cascade-out dspy_monolithic_gpt_oss_20b=workspace/results/eval/cascade/cmuh_monolithic \\
        [--out-dir PATH]

NOTE: the canonical 8-status schema (``null_value`` vs ``missing_key``,
``wrong_type`` vs ``wrong_value``) is finer-grained than what
cascade_atomic preserves. The adapter maps cascade rows to the closest
canonical status (``missing_key`` for ``field_missing``, ``wrong_value``
for ``correct=False``); rows that would be ``null_value`` or
``wrong_type`` under the old per-(cell, model) ablation aggregator
appear here as their coarser equivalents.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from digital_registrar_research.ablations.eval import canonical_stats  # noqa: E402


# ---------------------------------------------------------------------------
# Schema remap: non-ablation 5-status → canonical 8-status
# ---------------------------------------------------------------------------

_CANONICAL_COLUMNS = [
    "case_id", "organ", "method", "run", "field",
    "correct", "attempted", "gold_present",
    "case_status", "case_flags", "field_status", "field_error_detail",
]


def _remap_cascade_atomic(df: pd.DataFrame, *, method_label: str) -> pd.DataFrame:
    """Convert a ``cascade_atomic.parquet`` frame into the canonical schema.

    Stage A/B and Stage-C nested-list rows are dropped — the canonical
    suite reduces on Stage-C scalar correctness only. Stage-A/B
    accuracy is reported by the cascade chapter1/chapter2 outputs and
    by the ``compare`` subcommand; folding them in here would
    double-count cases.

    All retained rows are stamped with ``method = method_label``.

    Mapping:
      - ``parse_error == True``      → ``case_status="parse_error"``,
                                       ``field_status="unscoreable_due_to_case_error"``
      - ``gold_present == False``    → ``field_status="gold_missing"``
      - ``field_missing == True``    → ``field_status="missing_key"``
        (cascade can't distinguish ``missing_key`` from ``null_value``;
        the coarser bucket is used.)
      - ``correct == True``          → ``field_status="correct"``
      - ``correct == False``         → ``field_status="wrong_value"`` +
                                       ``gold=… pred=…`` detail
    """
    from scripts.eval._common.loaders import (
        coerce_cascade_bool, coerce_cascade_correct,
    )

    if df.empty:
        return pd.DataFrame(columns=_CANONICAL_COLUMNS)
    sub = df.copy()
    if "cascade_stage" in sub.columns:
        sub = sub[sub["cascade_stage"] == "C"]
    if "field_kind" in sub.columns:
        sub = sub[sub["field_kind"] != "nested_list"]
    if sub.empty:
        return pd.DataFrame(columns=_CANONICAL_COLUMNS)

    sub["correct"] = coerce_cascade_correct(sub["correct"])
    for col in ("attempted", "gold_present", "field_missing", "parse_error"):
        if col in sub.columns:
            sub[col] = coerce_cascade_bool(sub[col])

    sub["method"] = method_label
    if "run_id" in sub.columns and "run" not in sub.columns:
        sub = sub.rename(columns={"run_id": "run"})
    if "run" not in sub.columns:
        sub["run"] = ""

    def _row_status(r: pd.Series) -> tuple[str, str, str, str]:
        """(case_status, case_flags, field_status, field_error_detail)"""
        if bool(r.get("parse_error")):
            return ("parse_error", "parse_error",
                    "unscoreable_due_to_case_error", "")
        if not bool(r.get("gold_present", True)):
            return ("ok", "ok", "gold_missing", "")
        if bool(r.get("field_missing")):
            return ("ok", "ok", "missing_key", "")
        c = r.get("correct")
        if c == 1.0 or c is True:
            return ("ok", "ok", "correct", "")
        if c == 0.0 or c is False:
            detail = (
                f"gold={r.get('gold_value')!r} "
                f"pred={r.get('pred_value')!r}"
            )[:120]
            return ("ok", "ok", "wrong_value", detail)
        # Float in (0, 1) — should not appear after nested filter, but
        # be defensive.
        return ("ok", "ok", "wrong_value", "")

    statuses = sub.apply(_row_status, axis=1)
    sub["case_status"] = [s[0] for s in statuses]
    sub["case_flags"] = [s[1] for s in statuses]
    sub["field_status"] = [s[2] for s in statuses]
    sub["field_error_detail"] = [s[3] for s in statuses]

    keep = [c for c in _CANONICAL_COLUMNS if c in sub.columns]
    return sub[keep].copy()


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _resolve_folder(folder: str | Path) -> Path:
    """Resolve the folder shorthand (workspace / dummy / abs path)."""
    if folder is None:
        return None
    try:
        from _config_loader import resolve_folder  # noqa
        return resolve_folder(folder)
    except Exception:
        return Path(folder).resolve()


def _resolve_out_dir(args: argparse.Namespace) -> Path:
    """Resolve the output directory from --out-dir or --folder/--dataset."""
    if args.out_dir is not None:
        return Path(args.out_dir)
    folder = _resolve_folder(args.folder) if args.folder else None
    if folder and args.dataset:
        return folder / "results" / "canonical" / args.dataset
    raise SystemExit(
        "Must supply --out-dir or both --folder and --dataset.")


def _parse_cascade_out_specs(
    raw: list[str] | None,
) -> dict[str, Path]:
    """Parse a list of ``LABEL=PATH`` entries.

    Each PATH is a cascade output directory containing
    ``cascade_atomic.parquet``. Labels are arbitrary strings — they
    become the row's ``method`` value in the unified canonical
    atomic, and the user's ``--modular-method`` argument must match
    one of them to enable Δ / McNemar / OR computation.
    """
    out: dict[str, Path] = {}
    for spec in raw or []:
        if "=" not in spec:
            raise SystemExit(
                f"--cascade-out entry must be LABEL=PATH (got {spec!r}); "
                f"PATH should contain cascade_atomic.parquet."
            )
        label, raw_path = spec.split("=", 1)
        label = label.strip()
        if not label:
            raise SystemExit(f"empty label in --cascade-out {spec!r}")
        if label in out:
            raise SystemExit(f"duplicate label {label!r} in --cascade-out")
        out[label] = Path(raw_path.strip())
    return out


# ---------------------------------------------------------------------------
# IAA republish
# ---------------------------------------------------------------------------

def _publish_iaa_summary(folder: Path | None, dataset: str | None,
                         out_dir: Path) -> Path | None:
    """Locate an existing per-field IAA CSV under
    ``{folder}/results/iaa/{dataset}/`` and copy it to
    ``out_dir/iaa_summary.csv`` with canonical column names.

    Returns the destination path, or None if no source IAA CSV was
    found.
    """
    if folder is None or dataset is None:
        return None
    iaa_root = folder / "results" / "iaa" / dataset
    if not iaa_root.is_dir():
        return None
    # The IAA module's canonical per-field output. Best-effort lookup.
    candidates = [
        iaa_root / "iaa_per_field.csv",
        iaa_root / "per_field.csv",
        iaa_root / "iaa_summary.csv",
    ]
    src = next((p for p in candidates if p.is_file()), None)
    if src is None:
        return None
    df = pd.read_csv(src)
    dest = out_dir / "iaa_summary.csv"
    df.to_csv(dest, index=False)
    print(f"Wrote {dest} (republished from {src})")
    return dest


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--folder", default=None,
                    help="Experiment root shortcut (workspace / dummy / "
                         "absolute path). Used to locate default inputs.")
    ap.add_argument("--dataset", default=None,
                    help="Dataset name under data/ (e.g. cmuh, tcga). "
                         "Used to locate default inputs.")
    ap.add_argument("--modular-method", required=True,
                    help="The 'method' label to use as the comparator "
                         "for Δ / McNemar / OR computations. Must match "
                         "one of the LABEL values supplied via "
                         "--cascade-out (e.g. 'dspy_modular_gpt_oss_20b').")
    ap.add_argument("--cascade-out", dest="cascade_outs", action="append",
                    metavar="LABEL=PATH", default=None,
                    help="Cascade output directory to include. Each "
                         "entry contributes its cascade_atomic.parquet "
                         "to the master atomic, stamped with method=LABEL. "
                         "Repeat to include multiple methods / cells. "
                         "At least one --cascade-out is required.")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output directory. Default: "
                         "{folder}/results/canonical/{dataset}/.")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"),
                    default="cpu",
                    help="Device for the per-(method, field) bootstrap "
                         "tables. Default: cpu.")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = _resolve_out_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    cascade_outs = _parse_cascade_out_specs(args.cascade_outs)
    if not cascade_outs:
        raise SystemExit(
            "no --cascade-out entries supplied; pass at least one "
            "LABEL=PATH (PATH = cascade output directory containing "
            "cascade_atomic.parquet).",
        )

    frames: list[pd.DataFrame] = []
    for label, path in cascade_outs.items():
        atomic_path = path / "cascade_atomic.parquet"
        if not atomic_path.is_file():
            print(
                f"[warn] {atomic_path} missing for label={label!r}; skipping.",
                file=sys.stderr,
            )
            continue
        df = pd.read_parquet(atomic_path)
        adapted = _remap_cascade_atomic(df, method_label=label)
        frames.append(adapted)
        print(f"Loaded {label}: {atomic_path} ({len(df)} rows total, "
              f"{len(adapted)} Stage-C scalar rows kept)")

    if not frames:
        print("[warn] No input atomics found. Writing empty scaffolding.",
              file=sys.stderr)
        master = pd.DataFrame(columns=_CANONICAL_COLUMNS)
    else:
        master = pd.concat(frames, ignore_index=True)
    master_path = out_dir / "master_atomic.parquet"
    try:
        master.to_parquet(master_path)
    except Exception as exc:
        print(f"[warn] could not write {master_path}: {exc!r}",
              file=sys.stderr)
    print(f"Master atomic: {len(master)} rows. Wrote {master_path}")

    folder = _resolve_folder(args.folder) if args.folder else None
    from digital_registrar_research.benchmarks.eval.ci_gpu import pick_device
    resolved_device = pick_device(getattr(args, "device", "cpu"))
    print(f"device: requested={args.device} resolved={resolved_device}")
    canonical_stats.run_canonical_stats(
        master, modular_method=args.modular_method, out_dir=out_dir,
        command_line=" ".join(sys.argv),
        device=resolved_device)
    _publish_iaa_summary(folder, args.dataset, out_dir)
    print(f"\nCanonical paper tables: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
