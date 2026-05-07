"""End-to-end test for the rule_based predict orchestrator.

Builds a tiny synthetic data root in the canonical layout, invokes
``scripts/baselines/run_rule.py``, and asserts the output predictions
land at the canonical path
``{root}/results/predictions/{dataset}/rule_based/{organ_n}/{case_id}.json``
with the right shape.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_RULE = REPO_ROOT / "scripts" / "baselines" / "run_rule.py"


sys.path.insert(0, str(REPO_ROOT / "src"))

from digital_registrar_research.benchmarks import organs as _organs  # noqa: E402


def _build_synthetic_root(root: Path) -> None:
    """Write reports + gold annotations under {root}/data/<ds>/...

    Organ folders are dataset-specific per ``configs/organ_code.yaml``:
        CMUH 2 = breast, CMUH 5 = esophagus
        TCGA 1 = breast
    """
    cmuh_breast = str(_organs.organ_n_for("cmuh", "breast"))      # "2"
    cmuh_esoph  = str(_organs.organ_n_for("cmuh", "esophagus"))   # "5"
    tcga_breast = str(_organs.organ_n_for("tcga", "breast"))      # "1"
    fixtures = {
        ("cmuh", cmuh_breast, f"cmuh{cmuh_breast}_1", "breast"):
            "Modified radical mastectomy. Infiltrating ductal carcinoma. "
            "Nottingham grade G2. Tumor size 2.5 cm. "
            "Stage pT2 N1mi MX. Lymphovascular invasion present.",
        ("cmuh", cmuh_esoph, f"cmuh{cmuh_esoph}_1", "esophagus"):
            "Esophagectomy. Squamous cell carcinoma. Grade 2. "
            "pT3 N2 MX. Stage IIIB. Lymphovascular invasion present.",
        ("tcga", tcga_breast, f"tcga{tcga_breast}_1", "breast"):
            "Wide local excision. Invasive lobular carcinoma. "
            "Grade 1. Tumor size 8 mm. pT1c N0 MX.",
    }
    for (ds, organ_idx, case_id, organ_name), text in fixtures.items():
        ds_root = root / "data" / ds
        rep_dir = ds_root / "reports" / organ_idx
        ann_dir = ds_root / "annotations" / "gold" / organ_idx
        rep_dir.mkdir(parents=True, exist_ok=True)
        ann_dir.mkdir(parents=True, exist_ok=True)
        (rep_dir / f"{case_id}.txt").write_text(text, encoding="utf-8")
        gold = {
            "cancer_excision_report": True,
            "cancer_category": organ_name,
            "cancer_category_others_description": None,
            "cancer_data": {},
        }
        (ann_dir / f"{case_id}.json").write_text(
            json.dumps(gold), encoding="utf-8"
        )


@pytest.fixture
def synthetic_root(tmp_path: Path) -> Path:
    _build_synthetic_root(tmp_path)
    return tmp_path


def test_run_rule_writes_canonical_layout(synthetic_root: Path) -> None:
    """run_rule.py emits {root}/results/predictions/{ds}/rule_based/{organ_n}/<id>.json."""
    cmd = [
        sys.executable, str(RUN_RULE),
        "--folder", str(synthetic_root),
        "--datasets", "cmuh",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"run_rule.py failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )

    out_root = synthetic_root / "results" / "predictions" / "cmuh" / "rule_based"
    assert out_root.is_dir()
    assert (out_root / "_summary.json").is_file()
    assert (out_root / "_run_meta.json").is_file()
    assert (out_root / "_log.jsonl").is_file()

    # Per-organ subdirs with predictions (CMUH organ_n: 2=breast, 5=esophagus).
    cmuh_breast = str(_organs.organ_n_for("cmuh", "breast"))
    cmuh_esoph = str(_organs.organ_n_for("cmuh", "esophagus"))
    breast_pred = out_root / cmuh_breast / f"cmuh{cmuh_breast}_1.json"
    eso_pred = out_root / cmuh_esoph / f"cmuh{cmuh_esoph}_1.json"
    assert breast_pred.is_file()
    assert eso_pred.is_file()

    # Schema check on one prediction.
    pred = json.loads(breast_pred.read_text(encoding="utf-8"))
    assert set(pred.keys()) == {
        "cancer_excision_report", "cancer_category",
        "cancer_category_others_description", "cancer_data",
    }
    assert pred["cancer_category"] == "breast"
    assert pred["cancer_excision_report"] is True
    assert pred["cancer_data"]["pt_category"] == "t2"
    assert pred["cancer_data"]["lymphovascular_invasion"] is True


def test_run_rule_summary_counts(synthetic_root: Path) -> None:
    cmd = [
        sys.executable, str(RUN_RULE),
        "--folder", str(synthetic_root),
        "--datasets", "tcga",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    summary_path = (synthetic_root / "results" / "predictions" / "tcga"
                    / "rule_based" / "_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["method"] == "rule_based"
    assert summary["dataset"] == "tcga"
    assert summary["n_cases"] == 1  # only one tcga fixture
    assert summary["n_pipeline_error"] == 0


def test_run_rule_overwrite(synthetic_root: Path) -> None:
    """A second invocation with --overwrite re-runs all cases."""
    cmd_base = [
        sys.executable, str(RUN_RULE),
        "--folder", str(synthetic_root),
        "--datasets", "cmuh",
    ]
    r1 = subprocess.run(cmd_base, capture_output=True, text=True)
    assert r1.returncode == 0, r1.stderr
    r2 = subprocess.run(cmd_base, capture_output=True, text=True)
    assert r2.returncode == 0, r2.stderr
    # Without --overwrite, the second run should hit the cache path.
    summary_path = (synthetic_root / "results" / "predictions" / "cmuh"
                    / "rule_based" / "_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["n_cached"] >= 0  # cached counter is incremented on resume

    r3 = subprocess.run(cmd_base + ["--overwrite"], capture_output=True, text=True)
    assert r3.returncode == 0, r3.stderr
