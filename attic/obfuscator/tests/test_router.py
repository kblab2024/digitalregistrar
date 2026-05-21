"""Router dispatches files to the right handler and records skipped ones."""
from __future__ import annotations

import json
from pathlib import Path

from obfuscator.manifest import Manifests
from obfuscator.router import route


def test_yaml_config_passthrough(tmp_path: Path):
    src_root = tmp_path / "src"
    out_root = tmp_path / "out"
    cfg = src_root / "configs" / "models" / "x.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("name: x\n", encoding="utf-8")
    m = Manifests(out_root=out_root)
    route(cfg, src_root, out_root, m, master_seed=42, id_map={})
    assert (out_root / "configs" / "models" / "x.yaml").read_text(encoding="utf-8") == "name: x\n"
    assert any(o["handler"] == "passthrough" for o in m.outputs)


def test_log_banner(tmp_path: Path):
    src_root = tmp_path / "src"
    out_root = tmp_path / "out"
    log = src_root / "results" / "predictions" / "tcga" / "llm" / "x" / "run01" / "_run.log"
    log.parent.mkdir(parents=True)
    log.write_text("PHI-laden traceback containing TCGA-A2-A04N\n", encoding="utf-8")
    m = Manifests(out_root=out_root)
    route(log, src_root, out_root, m, master_seed=42, id_map={})
    out = (out_root / log.relative_to(src_root)).read_text(encoding="utf-8")
    assert "obfustrated" in out
    assert "TCGA-A2-A04N" not in out
    assert any(o["handler"] == "log_banner" for o in m.outputs)


def test_blob_to_placeholder(tmp_path: Path):
    src_root = tmp_path / "src"
    out_root = tmp_path / "out"
    blob = src_root / "models" / "clinicalbert" / "v1" / "checkpoint.pt"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"\x00" * 1024)
    m = Manifests(out_root=out_root)
    route(blob, src_root, out_root, m, master_seed=42, id_map={})
    placeholder = out_root / "models" / "clinicalbert" / "v1" / "checkpoint.pt.placeholder"
    assert placeholder.is_file()
    assert "stripped" in placeholder.read_text(encoding="utf-8")
    # And recorded in the skipped manifest.
    assert any("model weight blob" in s["reason"] for s in m.skipped)


def test_unknown_file_skipped_not_copied(tmp_path: Path):
    src_root = tmp_path / "src"
    out_root = tmp_path / "out"
    weird = src_root / "weird" / "thing.parquet"
    weird.parent.mkdir(parents=True)
    weird.write_bytes(b"\x00\x01\x02")
    m = Manifests(out_root=out_root)
    route(weird, src_root, out_root, m, master_seed=42, id_map={})
    assert not (out_root / "weird" / "thing.parquet").exists()
    assert any("unrouted" in s["reason"] for s in m.skipped)
