"""id_mapper produces stable, format-correct synthetic IDs."""
from __future__ import annotations

import re

from obfuscator.id_mapper import synth_patient_filename, synth_tcga_case_id


_PF_RE = re.compile(r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}\.[0-9A-F-]+$")
_CID_RE = re.compile(r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}$")


def test_synth_patient_filename_format():
    s = synth_patient_filename(42, "tcga1_5")
    assert _PF_RE.match(s), s


def test_synth_patient_filename_deterministic():
    a = synth_patient_filename(42, "tcga1_5")
    b = synth_patient_filename(42, "tcga1_5")
    assert a == b


def test_synth_patient_filename_diverges_per_case():
    a = synth_patient_filename(42, "tcga1_5")
    b = synth_patient_filename(42, "tcga1_6")
    assert a != b


def test_synth_patient_filename_diverges_per_seed():
    a = synth_patient_filename(42, "tcga1_5")
    b = synth_patient_filename(43, "tcga1_5")
    assert a != b


def test_synth_tcga_case_id_format():
    s = synth_tcga_case_id(42, "TCGA-A2-A04N")
    assert _CID_RE.match(s), s
