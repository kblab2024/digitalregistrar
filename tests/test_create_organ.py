"""Contract tests for the organ scaffolder (`schemas/create_organ.py`).

The scaffolder writes three files (pydantic / extraction / aliases) into
the live ``schemas/`` package by default. These tests use the
``base_dir`` parameter to redirect writes into a sandbox under
``tmp_path``, so we never touch the real tree.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from digital_registrar_research.schemas.create_organ import (
    CreatedOrganReport,
    _pascal_case,
    create_organ,
)


@pytest.fixture
def sandbox(tmp_path: Path) -> Path:
    """A bare-bones schemas/ tree with empty pydantic/extraction/aliases dirs."""
    (tmp_path / "pydantic").mkdir()
    (tmp_path / "extraction").mkdir()
    (tmp_path / "aliases").mkdir()
    return tmp_path


def _write_file(report: CreatedOrganReport, organ_key: str, base: Path) -> None:
    """Assert the three expected paths are present in the report and on disk."""
    expected_paths = {
        base / "pydantic" / f"{organ_key}.py",
        base / "extraction" / f"{organ_key}.py",
        base / "aliases" / f"{organ_key}.toml",
    }
    assert set(report.files_written) == expected_paths
    for p in expected_paths:
        assert p.exists(), f"{p} not written"


def test_pascal_case_helper():
    assert _pascal_case("skeletal") == "Skeletal"
    assert _pascal_case("head_neck") == "HeadNeck"
    assert _pascal_case("a") == "A"


def test_scaffolds_three_files(sandbox: Path):
    report = create_organ("skeletal", base_dir=sandbox, regen_json=False)
    _write_file(report, "skeletal", sandbox)
    assert report.organ_key == "skeletal"
    assert report.class_prefix == "Skeletal"
    assert report.json_regenerated is False


def test_generated_python_parses(sandbox: Path):
    report = create_organ("skeletal", base_dir=sandbox, regen_json=False)
    for p in report.files_written:
        if p.suffix == ".py":
            text = p.read_text(encoding="utf-8")
            compile(text, str(p), "exec")  # raises SyntaxError on failure


def test_generated_toml_parses(sandbox: Path):
    report = create_organ("skeletal", base_dir=sandbox, regen_json=False)
    toml_path = next(p for p in report.files_written if p.suffix == ".toml")
    with toml_path.open("rb") as f:
        data = tomllib.load(f)
    # The stub has no tables — every line is a comment — so the parse
    # result is the empty dict.
    assert data == {}


def test_substitution_happened(sandbox: Path):
    create_organ("skeletal", base_dir=sandbox, regen_json=False)
    pyd_text = (sandbox / "pydantic" / "skeletal.py").read_text(encoding="utf-8")
    ext_text = (sandbox / "extraction" / "skeletal.py").read_text(encoding="utf-8")
    toml_text = (sandbox / "aliases" / "skeletal.toml").read_text(encoding="utf-8")

    # Placeholders are gone.
    for text in (pyd_text, ext_text, toml_text):
        assert "__ORGAN_KEY__" not in text
        assert "__CLASS_PREFIX__" not in text

    # Substituted values are present.
    assert "SkeletalCancerCase" in pyd_text
    assert "SkeletalProcedure" in pyd_text
    assert "skeletal cancer" in ext_text
    assert "skeletal" in toml_text


def test_multi_word_organ_key(sandbox: Path):
    report = create_organ("head_neck", base_dir=sandbox, regen_json=False)
    assert report.class_prefix == "HeadNeck"
    pyd_text = (sandbox / "pydantic" / "head_neck.py").read_text(encoding="utf-8")
    assert "HeadNeckCancerCase" in pyd_text
    assert "HeadNeckProcedure" in pyd_text


def test_refuses_existing_files(sandbox: Path):
    create_organ("skeletal", base_dir=sandbox, regen_json=False)
    # After the first call, `skeletal` is registered in the sandbox tree;
    # the validator fires first and surfaces the conceptual error
    # ("already registered") rather than the mechanical one ("file exists").
    with pytest.raises(ValueError, match="already registered"):
        create_organ("skeletal", base_dir=sandbox, regen_json=False)


def test_refuses_file_clash_without_registration_collision(sandbox: Path):
    """Even if the registry check passes (somehow), file-exists is still caught."""
    # Pre-create just the extraction file with no corresponding pydantic file.
    # The registry walker only scans pydantic/, so `weirdcase` isn't "registered",
    # but the file-existence check should still refuse.
    (sandbox / "extraction" / "weirdcase.py").write_text("# stray\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        create_organ("weirdcase", base_dir=sandbox, regen_json=False)


def test_overwrite_clobbers(sandbox: Path):
    create_organ("skeletal", base_dir=sandbox, regen_json=False)
    # Modify one of the generated files; verify overwrite restores it.
    pyd_path = sandbox / "pydantic" / "skeletal.py"
    pyd_path.write_text("# manually edited\n", encoding="utf-8")
    report = create_organ(
        "skeletal", base_dir=sandbox, regen_json=False, overwrite=True
    )
    assert pyd_path.read_text(encoding="utf-8") != "# manually edited\n"
    assert "SkeletalCancerCase" in pyd_path.read_text(encoding="utf-8")
    _write_file(report, "skeletal", sandbox)


def test_explicit_class_prefix(sandbox: Path):
    """`class_prefix` overrides the PascalCase derivation."""
    report = create_organ(
        "skin", class_prefix="Cutaneous", base_dir=sandbox, regen_json=False
    )
    assert report.class_prefix == "Cutaneous"
    pyd_text = (sandbox / "pydantic" / "skin.py").read_text(encoding="utf-8")
    assert "CutaneousCancerCase" in pyd_text
    assert "SkinCancerCase" not in pyd_text


@pytest.mark.parametrize(
    "bad_key",
    [
        "",            # empty
        "1foo",        # starts with digit
        "Foo",         # uppercase
        "foo bar",     # space
        "foo-bar",     # hyphen
        "_foo",        # leading underscore
        "foo.bar",     # dot
    ],
)
def test_rejects_invalid_organ_keys(sandbox: Path, bad_key: str):
    with pytest.raises(ValueError):
        create_organ(bad_key, base_dir=sandbox, regen_json=False)


def test_rejects_already_registered_key(sandbox: Path):
    """If an organ already exists in the sandbox, refuse without --overwrite."""
    # Seed an existing organ in the sandbox.
    (sandbox / "pydantic" / "existing.py").write_text("# placeholder\n", encoding="utf-8")
    with pytest.raises(ValueError, match="already registered"):
        create_organ("existing", base_dir=sandbox, regen_json=False)
