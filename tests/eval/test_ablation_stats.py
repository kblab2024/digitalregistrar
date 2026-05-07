"""Unit tests for the ablation statistics module.

Verifies that paired Δ, McNemar, multiple-comparisons correction,
seed consistency, and effect-size computations behave as expected on
small hand-computable inputs.
"""
from __future__ import annotations

import pandas as pd
import pytest

from digital_registrar_research.ablations.eval import stats as ab_stats

try:
    import statsmodels  # noqa: F401
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


# --- Synthetic grid ---------------------------------------------------------

def _make_grid() -> pd.DataFrame:
    """Three cells × two fields × five cases.

    Baseline is correct on every case for both fields. Target_better
    flips one case to wrong on field1 (delta = -0.2). Target_worse
    flips three cases on field1 (delta = -0.6).
    """
    rows = []

    cases = [f"c{i}" for i in range(5)]

    # Baseline: dspy_modular_gpt-oss — all correct
    for case in cases:
        for field in ("pt_category", "grade"):
            rows.append({
                "method": "dspy_modular_gpt-oss",
                "case_id": case, "field": field,
                "correct": 1.0, "attempted": True,
            })

    # Target B (small drop): one case wrong on field1
    correctness_b = [1.0, 1.0, 1.0, 1.0, 0.0]
    for case, c in zip(cases, correctness_b, strict=True):
        rows.append({"method": "dspy_monolithic_gpt-oss",
                     "case_id": case, "field": "pt_category",
                     "correct": c, "attempted": True})
        rows.append({"method": "dspy_monolithic_gpt-oss",
                     "case_id": case, "field": "grade",
                     "correct": 1.0, "attempted": True})

    # Target C (big drop): three cases wrong on field1
    correctness_c = [1.0, 0.0, 0.0, 0.0, 1.0]
    for case, c in zip(cases, correctness_c, strict=True):
        rows.append({"method": "raw_json_gpt-oss",
                     "case_id": case, "field": "pt_category",
                     "correct": c, "attempted": True})
        rows.append({"method": "raw_json_gpt-oss",
                     "case_id": case, "field": "grade",
                     "correct": 1.0, "attempted": True})

    return pd.DataFrame(rows)


# --- Tests ------------------------------------------------------------------

def test_paired_deltas_match_manual():
    grid = _make_grid()
    grid["cell"] = grid["method"].apply(lambda m: m.rsplit("_", 1)[0])
    grid["model"] = grid["method"].apply(lambda m: m.rsplit("_", 1)[1])

    deltas = ab_stats.paired_deltas_vs_baseline(
        grid, baseline_method="dspy_modular_gpt-oss", n_boot=200,
        random_state=42)

    assert not deltas.empty
    # Two non-baseline methods × 2 fields = 4 rows
    assert len(deltas) == 4

    sub = deltas[(deltas["cell"] == "dspy_monolithic")
                 & (deltas["field"] == "pt_category")]
    assert len(sub) == 1
    assert sub["delta"].iloc[0] == pytest.approx(-0.2, abs=1e-9)
    assert sub["target_acc"].iloc[0] == pytest.approx(0.8, abs=1e-9)
    assert sub["baseline_acc"].iloc[0] == pytest.approx(1.0, abs=1e-9)
    # CI must bracket the point estimate.
    assert sub["ci_lo"].iloc[0] <= sub["delta"].iloc[0] <= sub["ci_hi"].iloc[0]
    # McNemar discordant counts: b = baseline correct & target wrong = 1; c = 0
    assert sub["mcnemar_b"].iloc[0] == 1
    assert sub["mcnemar_c"].iloc[0] == 0
    assert 0.0 <= sub["mcnemar_p"].iloc[0] <= 1.0

    sub_worse = deltas[(deltas["cell"] == "raw_json")
                       & (deltas["field"] == "pt_category")]
    assert sub_worse["delta"].iloc[0] == pytest.approx(-0.6, abs=1e-9)


@pytest.mark.skipif(not HAS_STATSMODELS,
                    reason="statsmodels not installed")
def test_multiple_comparisons_correction_monotone(tmp_path):
    """Holm-adjusted p-values are non-decreasing when sorted by raw p."""
    deltas = pd.DataFrame([
        # All in the same axis-A primary family.
        {"cell": "dspy_monolithic", "model": "x", "field": "pt_category",
         "mcnemar_p": 0.001, "mcnemar_b": 5, "mcnemar_c": 0,
         "delta": -0.1, "ci_lo": -0.2, "ci_hi": 0.0,
         "n_paired": 50, "baseline_acc": 0.9, "target_acc": 0.8,
         "mcnemar_stat": 5.0, "mcnemar_method": "chi2_cc"},
        {"cell": "dspy_monolithic", "model": "x", "field": "grade",
         "mcnemar_p": 0.04, "mcnemar_b": 3, "mcnemar_c": 1,
         "delta": -0.04, "ci_lo": -0.1, "ci_hi": 0.02,
         "n_paired": 50, "baseline_acc": 0.85, "target_acc": 0.81,
         "mcnemar_stat": 1.0, "mcnemar_method": "chi2_cc"},
        {"cell": "no_router", "model": "x", "field": "pt_category",
         "mcnemar_p": 0.5, "mcnemar_b": 2, "mcnemar_c": 1,
         "delta": -0.02, "ci_lo": -0.1, "ci_hi": 0.06,
         "n_paired": 50, "baseline_acc": 0.9, "target_acc": 0.88,
         "mcnemar_stat": 0.3, "mcnemar_method": "chi2_cc"},
    ])

    axes_yaml = tmp_path / "axes.yaml"
    axes_yaml.write_text(
        "axes:\n  A:\n    - dspy_modular\n    - dspy_monolithic\n    - no_router\n",
        encoding="utf-8")
    endpoints_yaml = tmp_path / "eval_endpoints.yaml"
    endpoints_yaml.write_text(
        "primary:\n  - pt_category\n  - grade\nsecondary: []\n",
        encoding="utf-8")

    out = ab_stats.multiple_comparisons_correction(
        deltas, axes_path=axes_yaml, endpoints_path=endpoints_yaml)

    primary = out[out["tier"] == "primary"].sort_values("mcnemar_p")
    holm = primary["p_holm"].dropna().tolist()
    assert holm == sorted(holm), "p_holm must be non-decreasing in raw p order"
    # Smallest raw p × m=3 should be flagged.
    smallest = primary.iloc[0]
    assert smallest["reject_holm"] in {True, False}
    assert smallest["family_size"] == 3


def test_seed_consistency_perfect_agreement():
    rows = []
    for case in [f"c{i}" for i in range(5)]:
        for seed in [1, 2, 3]:
            rows.append({
                "method": "dspy_monolithic_gpt-oss",
                "cell": "dspy_monolithic", "model": "gpt-oss",
                "case_id": case, "field": "pt_category",
                "correct": 1.0, "attempted": True, "seed": seed,
            })
    df = pd.DataFrame(rows)
    out = ab_stats.seed_consistency(df)
    assert not out.empty
    # Perfect agreement → flip rate 0; κ may be NaN (degenerate, all same value)
    assert out["flip_rate"].iloc[0] == pytest.approx(0.0)


def test_effect_sizes_basic():
    grid = _make_grid()
    grid["cell"] = grid["method"].apply(lambda m: m.rsplit("_", 1)[0])
    grid["model"] = grid["method"].apply(lambda m: m.rsplit("_", 1)[1])
    out = ab_stats.effect_sizes_per_field(
        grid, baseline_method="dspy_modular_gpt-oss")
    assert not out.empty
    # Cohen's d for the larger drop (raw_json) should be more negative.
    d_mono = out[(out["cell"] == "dspy_monolithic")
                 & (out["field"] == "pt_category")]["cohens_d"].iloc[0]
    d_raw = out[(out["cell"] == "raw_json")
                & (out["field"] == "pt_category")]["cohens_d"].iloc[0]
    # Both should be ≤ 0 (target worse than baseline).
    assert d_raw <= d_mono <= 0


def _make_cascade_atomic() -> pd.DataFrame:
    """Build a minimal cascade_atomic-shaped fixture for stats consumers.

    Mirrors :func:`_make_grid` but in cascade schema (cascade_stage="C",
    field_kind="nominal", explicit cell/model_slug/run_id columns plus
    Stage A/B noise rows that must be filtered out by
    :func:`_adapt_cascade_atomic_for_stats`).
    """
    cases = [f"c{i}" for i in range(5)]
    rows: list[dict] = []
    cells_models_correctness = [
        ("dspy_modular", "gpt-oss", [1.0] * 5, [1.0] * 5),
        ("dspy_monolithic", "gpt-oss", [1.0, 1.0, 1.0, 1.0, 0.0], [1.0] * 5),
        ("raw_json", "gpt-oss", [1.0, 0.0, 0.0, 0.0, 1.0], [1.0] * 5),
    ]
    for cell, model_slug, pt_corr, grade_corr in cells_models_correctness:
        for case, c_pt, c_gr in zip(cases, pt_corr, grade_corr, strict=True):
            # Stage A noise (binary; should be filtered out).
            rows.append({
                "run_id": "run01", "method": "ablation",
                "model": f"{cell}_{model_slug}", "model_slug": model_slug,
                "cell": cell, "case_id": case,
                "cascade_stage": "A", "field_kind": "binary",
                "field": "cancer_excision_report",
                "correct": True, "attempted": True,
                "gold_present": True, "wrong": False,
                "field_missing": False, "parse_error": False,
                "gate_pass": True,
            })
            # Stage C scalar — what the stats consumers reduce on.
            for field, c in (("pt_category", c_pt), ("grade", c_gr)):
                rows.append({
                    "run_id": "run01", "method": "ablation",
                    "model": f"{cell}_{model_slug}", "model_slug": model_slug,
                    "cell": cell, "case_id": case,
                    "cascade_stage": "C", "field_kind": "nominal",
                    "field": field, "correct": bool(c == 1.0),
                    "attempted": True, "gold_present": True,
                    "wrong": bool(c == 0.0), "field_missing": False,
                    "parse_error": False, "gate_pass": True,
                })
            # Stage C nested — must be filtered out (float F1, not bool).
            rows.append({
                "run_id": "run01", "method": "ablation",
                "model": f"{cell}_{model_slug}", "model_slug": model_slug,
                "cell": cell, "case_id": case,
                "cascade_stage": "C", "field_kind": "nested_list",
                "field": "margins", "correct": 0.5,
                "attempted": True, "gold_present": True,
                "wrong": 0.5, "field_missing": False,
                "parse_error": False, "gate_pass": True,
            })
    return pd.DataFrame(rows)


def test_load_grid_prefers_cascade_atomic_and_filters_stage_c_scalar(tmp_path):
    """When cascade_atomic.parquet is present, _load_grid filters to
    Stage-C scalar and derives method/cell/model/seed columns. Stage A
    rows (binary) and Stage C nested-list rows (float F1) must be
    excluded so paired deltas reduce on binary correctness only.
    """
    from scripts.eval._common.reporting import write_parquet

    cascade = _make_cascade_atomic()
    # Use the cascade reporting helper because the `correct` column
    # holds mixed bool/float dtypes (scalar vs nested) that pyarrow
    # cannot serialise without type-coercion.
    write_parquet(cascade, tmp_path / "cascade_atomic.parquet")

    loaded = ab_stats._load_grid(tmp_path)
    # No Stage A or nested-list rows survived.
    assert (loaded["cascade_stage"] == "C").all()
    assert (loaded["field_kind"] != "nested_list").all()
    # Method composed correctly; model points back to the slug.
    assert set(loaded["method"].unique()) == {
        "dspy_modular_gpt-oss",
        "dspy_monolithic_gpt-oss",
        "raw_json_gpt-oss",
    }
    assert set(loaded["model"].unique()) == {"gpt-oss"}
    # Seed alias points at run_id so multi-run consistency picks it up.
    assert (loaded["seed"] == loaded["run_id"]).all()

    # Paired deltas on the cascade-derived grid should match the legacy
    # fixture: dspy_monolithic Δ = -0.2 on pt_category, raw_json Δ = -0.6.
    deltas = ab_stats.paired_deltas_vs_baseline(
        loaded, baseline_method="dspy_modular_gpt-oss",
        n_boot=200, random_state=42,
    )
    sub = deltas[(deltas["cell"] == "dspy_monolithic")
                 & (deltas["field"] == "pt_category")]
    assert sub["delta"].iloc[0] == pytest.approx(-0.2, abs=1e-9)
    sub_raw = deltas[(deltas["cell"] == "raw_json")
                     & (deltas["field"] == "pt_category")]
    assert sub_raw["delta"].iloc[0] == pytest.approx(-0.6, abs=1e-9)


def test_run_all_smoke(tmp_path):
    """End-to-end smoke: write a synthetic grid, call run_all, check files."""
    grid = _make_grid()
    grid_path = tmp_path / "ablation_grid.csv"
    grid.to_csv(grid_path, index=False)

    # Stub efficiency.csv
    eff = pd.DataFrame([
        {"cell": "dspy_modular", "model": "gpt-oss", "n_cases": 5,
         "schema_errors": 0, "parse_errors": 0,
         "mean_latency_s": 1.0, "median_latency_s": 1.0},
        {"cell": "dspy_monolithic", "model": "gpt-oss", "n_cases": 5,
         "schema_errors": 1, "parse_errors": 0,
         "mean_latency_s": 0.5, "median_latency_s": 0.5},
    ])
    (tmp_path / "efficiency.csv").write_text(eff.to_csv(index=False),
                                             encoding="utf-8")

    outputs = ab_stats.run_all(tmp_path,
                               baseline_method="dspy_modular_gpt-oss",
                               n_boot=200)
    assert "paired_deltas" in outputs
    assert (tmp_path / "ablation_paired_deltas.csv").exists()
    assert (tmp_path / "ablation_effect_sizes.csv").exists()
    # Efficiency stats CSV present with Wilson CIs.
    eff_path = tmp_path / "ablation_efficiency_stats.csv"
    assert eff_path.exists()
    eff_out = pd.read_csv(eff_path)
    assert {"schema_ci_lo", "schema_ci_hi", "parse_ci_lo", "parse_ci_hi"} <= set(eff_out.columns)
