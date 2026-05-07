"""seeding.derive must be deterministic across processes (no PYTHONHASHSEED salting)."""
from __future__ import annotations

import subprocess
import sys

from obfuscator.seeding import derive, derive_int


def test_derive_is_deterministic_same_inputs():
    a = derive(42, "case", "tcga", "1", "5").random()
    b = derive(42, "case", "tcga", "1", "5").random()
    assert a == b


def test_derive_diverges_on_different_master_seeds():
    a = derive(42, "case", "tcga", "1", "5").random()
    b = derive(43, "case", "tcga", "1", "5").random()
    assert a != b


def test_derive_diverges_on_different_parts():
    a = derive(42, "case", "tcga", "1", "5").random()
    b = derive(42, "case", "tcga", "1", "6").random()
    assert a != b


def test_derive_int_in_range():
    n = derive_int(42, "x", "y", bits=16)
    assert 0 <= n < 2**16


def test_derive_survives_subprocess():
    """blake2b-based derive must NOT depend on PYTHONHASHSEED."""
    import os
    from pathlib import Path
    src_path = str((Path(__file__).resolve().parents[1] / "src").resolve())
    code = (
        f"import sys; sys.path.insert(0, r'{src_path}'); "
        "from obfuscator.seeding import derive; "
        "print(derive(42, 'case', 'tcga', '1', '5').random())"
    )
    env_base = dict(os.environ)
    env_a = {**env_base, "PYTHONHASHSEED": "0"}
    env_b = {**env_base, "PYTHONHASHSEED": "12345"}
    out1 = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env_a)
    out2 = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env_b)
    assert out1.returncode == 0 and out2.returncode == 0, (out1.stderr, out2.stderr)
    assert out1.stdout.strip() == out2.stdout.strip()
