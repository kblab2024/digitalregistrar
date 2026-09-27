"""End-to-end tests for the ``registrar-eval`` CLI on folders of
registrar-pipeline style outputs (``<stem>_output.json``) vs gold
(``<stem>_annotation.json``)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from digital_registrar.eval.cli import main
from digital_registrar.eval.completeness import out_of_vocab_rate
from digital_registrar.eval.folders import case_id_from_path, index_folder, load_pairs

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "reference" / "preannotation" / "gpt_oss_20b"
STOMACH = FIXTURES / "4"
COLORECTAL = FIXTURES / "2"


def _copy_as(src: Path, dst: Path, suffix: str) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.glob("*.json")):
        shutil.copy(f, dst / f"{f.stem}{suffix}.json")


@pytest.fixture
def run_dirs(tmp_path: Path) -> tuple[Path, Path]:
    """Identical stomach cases as predictions and gold."""
    pred, gold = tmp_path / "pred", tmp_path / "gold"
    _copy_as(STOMACH, pred, "_output")
    _copy_as(STOMACH, gold, "_annotation")
    return pred, gold


def _perturb(pred: Path) -> None:
    """Change one pt_category and drop one prediction file."""
    path = pred / "tcga4_1_output.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["cancer_data"]["pt_category"] = "t4a" if record["cancer_data"]["pt_category"] != "t4a" else "t1a"
    path.write_text(json.dumps(record), encoding="utf-8")
    (pred / "tcga4_10_output.json").unlink()


def _summary_row(summary: pd.DataFrame, field: str) -> pd.Series:
    rows = summary[summary["field"] == field]
    assert len(rows) == 1, field
    return rows.iloc[0]


# --- Pairing ---------------------------------------------------------------


@pytest.mark.parametrize("name, expected", [
    ("tcga4_1_output.json", "tcga4_1"),
    ("tcga4_1_annotation.json", "tcga4_1"),
    ("tcga4_1.json", "tcga4_1"),
    ("tcga4_1.txt_output.json", "tcga4_1"),
    ("_output.json", "_output"),
])
def test_case_id_from_path(name, expected):
    assert case_id_from_path(Path("some/dir") / name) == expected


def test_index_folder_rejects_duplicate_ids(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "x_output.json").write_text("{}")
    (tmp_path / "a" / "x.json").write_text("{}")
    with pytest.raises(ValueError, match="duplicate case id 'x'"):
        index_folder(tmp_path)


def test_load_pairs_counts(run_dirs):
    pred, gold = run_dirs
    _perturb(pred)
    (pred / "tcga4_11_output.json").write_text("not json", encoding="utf-8")
    (pred / "extra_output.json").write_text("{}", encoding="utf-8")
    pairs, stats = load_pairs(pred, gold)
    assert len(pairs) == 42
    assert stats == {"n_gold": 42, "n_matched": 40, "n_missing_pred": 1,
                     "n_unreadable_pred": 1, "n_extra_pred": 1, "n_unreadable_gold": 0}


# --- metrics ---------------------------------------------------------------


def test_metrics_identical_is_perfect(run_dirs, tmp_path):
    pred, gold = run_dirs
    out = tmp_path / "out"
    assert main(["metrics", "--pred", str(pred), "--gold", str(gold), "--out", str(out)]) == 0

    atomic = pd.read_csv(out / "atomic.csv")
    summary = pd.read_csv(out / "summary.csv")
    assert atomic["case_id"].nunique() == 42
    assert set(summary["stage"]) == {"A", "B", "C"}
    assert {"margins", "regional_lymph_node.group_f1"} <= set(summary.loc[summary["metric"] == "f1", "field"])
    assert set(summary["method"]) == {"pred"}
    acc = summary["accuracy_attempted"].dropna()
    assert len(acc) == len(summary)
    assert (acc == 1.0).all()
    assert _summary_row(summary, "cancer_excision_report")["coverage"] == 1.0


def test_metrics_detects_errors_and_missing_predictions(run_dirs, tmp_path):
    pred, gold = run_dirs
    _perturb(pred)
    out = tmp_path / "out"
    assert main(["metrics", "--pred", str(pred), "--gold", str(gold),
                 "--out", str(out), "--label", "run1"]) == 0

    summary = pd.read_csv(out / "summary.csv")
    assert set(summary["method"]) == {"run1"}
    pt = _summary_row(summary, "pt_category")
    assert pt["accuracy_attempted"] == pytest.approx(40 / 41)
    eligibility = _summary_row(summary, "cancer_excision_report")
    assert eligibility["total"] == 42
    assert eligibility["attempted"] == 41
    assert eligibility["accuracy_attempted"] == 1.0


def test_metrics_fair_scope(run_dirs, tmp_path):
    pred, gold = run_dirs
    out = tmp_path / "out"
    assert main(["metrics", "--pred", str(pred), "--gold", str(gold),
                 "--out", str(out), "--scope", "fair"]) == 0
    summary = pd.read_csv(out / "summary.csv")
    assert "pt_category" in set(summary["field"])
    assert "margins" not in set(summary["field"])
    assert (summary["accuracy_attempted"].dropna() == 1.0).all()


def test_metrics_empty_nested_lists_are_not_scored(tmp_path):
    """Colorectal fixtures all have ``biomarkers: []`` — identical empty
    lists must not show up as F1 = 0."""
    pred, gold = tmp_path / "pred", tmp_path / "gold"
    _copy_as(COLORECTAL, pred, "_output")
    _copy_as(COLORECTAL, gold, "_annotation")
    out = tmp_path / "out"
    assert main(["metrics", "--pred", str(pred), "--gold", str(gold), "--out", str(out)]) == 0
    summary = pd.read_csv(out / "summary.csv")
    assert "biomarkers" not in set(summary["field"])
    assert (summary["accuracy_attempted"].dropna() == 1.0).all()
    assert summary["accuracy_attempted"].notna().all()


def test_metrics_nested_dataset_folders(tmp_path):
    """registrar-pipeline without --input writes <run>/<dataset>/<stem>_output.json."""
    pred, gold = tmp_path / "pred", tmp_path / "gold"
    _copy_as(STOMACH, pred / "tcga4", "_output")
    _copy_as(STOMACH, gold / "tcga4", "_annotation")
    out = tmp_path / "out"
    assert main(["metrics", "--pred", str(pred), "--gold", str(gold), "--out", str(out)]) == 0
    atomic = pd.read_csv(out / "atomic.csv")
    assert set(atomic["dataset"]) == {"tcga4"}
    assert atomic["case_id"].nunique() == 42


def test_metrics_missing_gold_folder_fails(tmp_path, capsys):
    pred = tmp_path / "pred"
    pred.mkdir()
    assert main(["metrics", "--pred", str(pred), "--gold", str(tmp_path / "nope"),
                 "--out", str(tmp_path / "out")]) == 1
    assert "not a directory" in capsys.readouterr().err


def test_metrics_empty_gold_folder_fails(tmp_path, capsys):
    pred, gold = tmp_path / "pred", tmp_path / "gold"
    pred.mkdir()
    gold.mkdir()
    assert main(["metrics", "--pred", str(pred), "--gold", str(gold),
                 "--out", str(tmp_path / "out")]) == 1
    assert "no gold annotations" in capsys.readouterr().err


# --- compare ---------------------------------------------------------------


def test_compare_identical_vs_perturbed(run_dirs, tmp_path):
    pred_a, gold = run_dirs
    pred_b = tmp_path / "pred_b"
    shutil.copytree(pred_a, pred_b)
    _perturb(pred_b)
    out = tmp_path / "out"
    assert main(["compare", "--pred-a", str(pred_a), "--pred-b", str(pred_b),
                 "--gold", str(gold), "--out", str(out), "--n-boot", "200"]) == 0

    for name in ("atomic.csv", "summary.csv", "compare.csv"):
        assert (out / name).is_file()
    cmp = pd.read_csv(out / "compare.csv")
    assert set(cmp["method_a"]) == {"pred"}
    assert set(cmp["method_b"]) == {"pred_b"}

    pt = cmp[cmp["field"] == "pt_category"].iloc[0]
    assert pt["n_paired"] == 41
    assert pt["delta"] == pytest.approx(1 / 41)
    assert pt["mcnemar_b"] == 1
    assert pt["mcnemar_c"] == 0

    others = cmp[~cmp["field"].isin(["pt_category", "ALL"])]
    assert (others["delta"] == 0).all()

    overall = cmp[cmp["field"] == "ALL"].iloc[0]
    assert overall["delta"] > 0

    summary = pd.read_csv(out / "summary.csv")
    assert set(summary["method"]) == {"pred", "pred_b"}


def test_compare_same_folder_names_get_distinct_labels(run_dirs, tmp_path):
    pred_a, gold = run_dirs
    pred_b = tmp_path / "other" / "pred"
    shutil.copytree(pred_a, pred_b)
    out = tmp_path / "out"
    assert main(["compare", "--pred-a", str(pred_a), "--pred-b", str(pred_b),
                 "--gold", str(gold), "--out", str(out), "--n-boot", "50"]) == 0
    cmp = pd.read_csv(out / "compare.csv")
    assert set(cmp["method_a"]) == {"a"}
    assert set(cmp["method_b"]) == {"b"}
    assert (cmp["delta"] == 0).all()


# --- completeness ----------------------------------------------------------


def test_completeness_identical(run_dirs, tmp_path):
    pred, gold = run_dirs
    out = tmp_path / "out"
    assert main(["completeness", "--pred", str(pred), "--gold", str(gold), "--out", str(out)]) == 0

    report = pd.read_csv(out / "completeness.csv")
    assert not report.empty
    assert (report["n_parse_error"] == 0).all()
    assert (report["n_total"] == 42).all()
    assert (report["attempted_accuracy"].dropna() == 1.0).all()
    assert (out / "refusal_calibration.csv").is_file()

    oov = pd.read_csv(out / "out_of_vocab.csv")
    assert not oov.empty
    assert set(oov["organ"]) == {"stomach"}
    assert "grade" in set(oov["field"])


def test_completeness_counts_missing_predictions(run_dirs, tmp_path):
    pred, gold = run_dirs
    _perturb(pred)
    out = tmp_path / "out"
    assert main(["completeness", "--pred", str(pred), "--gold", str(gold), "--out", str(out)]) == 0
    report = pd.read_csv(out / "completeness.csv")
    assert (report["n_parse_error"] == 1).all()
    pt = report[report["field"] == "pt_category"].iloc[0]
    assert pt["n_wrong"] == 1


def test_out_of_vocab_accepts_int_and_bool_values():
    """Allowed values are stringified ("1", "true"); raw JSON ints and
    bools from the pipeline must still count as in-vocabulary."""
    preds = [{"cancer_data": {"grade": 2, "perineural_invasion": True}},
             {"cancer_data": {"grade": 7, "perineural_invasion": False}}]
    grade = out_of_vocab_rate(preds, field="grade", organ="stomach")
    assert (grade["n_attempted"], grade["n_oov"]) == (2, 1)
    pni = out_of_vocab_rate(preds, field="perineural_invasion", organ="stomach")
    assert (pni["n_attempted"], pni["n_oov"]) == (2, 0)


# --- help ------------------------------------------------------------------


def test_help_mentions_file_naming(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    text = capsys.readouterr().out
    assert "_output.json" in text
    assert "_annotation.json" in text
