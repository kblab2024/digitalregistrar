"""End-to-end determinism + PHI-absence tests against the dummy fixture."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from obfuscator.cli import main


REPO_ROOT = Path(__file__).resolve().parents[2]
DUMMY = REPO_ROOT / "dummy"


@pytest.fixture
def tmp_out(tmp_path) -> Path:
    return tmp_path / "obfustrated"


def _run_obfuscator(src: Path, out: Path, seed: int) -> int:
    return main(["--src", str(src), "--out", str(out),
                 "--seed", str(seed), "--force"])


def test_runs_against_dummy(tmp_out):
    rc = _run_obfuscator(DUMMY, tmp_out, 42)
    assert rc == 0
    assert (tmp_out / "_obfuscator_outputs.json").is_file()
    assert (tmp_out / "_obfuscator_skipped.json").is_file()


def test_determinism_same_seed(tmp_path):
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    _run_obfuscator(DUMMY, out_a, 42)
    _run_obfuscator(DUMMY, out_b, 42)
    sample = "data/cmuh/annotations/gold/2/cmuh2_5.json"
    if (out_a / sample).is_file() and (out_b / sample).is_file():
        assert (out_a / sample).read_bytes() == (out_b / sample).read_bytes()


def test_diverges_on_different_seeds(tmp_path):
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    _run_obfuscator(DUMMY, out_a, 42)
    _run_obfuscator(DUMMY, out_b, 99)
    # Sample at least one cancer case (skip non-cancer cases — all-empty JSONs).
    different = 0
    for organ in ("1", "2", "3"):
        for case in ("cmuh%s_5" % organ, "cmuh%s_11" % organ, "cmuh%s_22" % organ):
            p_a = out_a / "data" / "cmuh" / "annotations" / "gold" / organ / f"{case}.json"
            p_b = out_b / "data" / "cmuh" / "annotations" / "gold" / organ / f"{case}.json"
            if p_a.is_file() and p_b.is_file() and p_a.read_bytes() != p_b.read_bytes():
                different += 1
    assert different >= 1, "different seeds produced identical outputs across all sampled cases"


def test_phi_absence_no_real_tcga_ids(tmp_out):
    """Generated reports must not echo the watchlist of real TCGA case IDs."""
    _run_obfuscator(DUMMY, tmp_out, 42)
    watchlist = ("TCGA-A2-A04N", "TCGA-A6-2670", "TCGA-A2-A04Y")
    for txt in (tmp_out / "data").rglob("*.txt"):
        text = txt.read_text(encoding="utf-8")
        for needle in watchlist:
            assert needle not in text, f"PHI watchlist hit: {needle} in {txt}"
