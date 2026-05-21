# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the annotator app."""
from pathlib import Path

REPO_ROOT = Path(SPECPATH).resolve().parents[1]

a = Analysis(
    [str(REPO_ROOT / "apps" / "annotator" / "src" / "digital_registrar_annotator" / "app_canonical.py")],
    pathex=[
        str(REPO_ROOT / "src"),
        str(REPO_ROOT / "apps" / "annotator" / "src"),
    ],
    datas=[
        (str(REPO_ROOT / "src" / "digital_registrar" / "schemas" / "data"),
         "digital_registrar/schemas/data"),
    ],
    hiddenimports=[
        "digital_registrar.schemas", "digital_registrar_annotator",
        "streamlit",
    ],
    excludes=["drr_attic", "torch", "transformers"],
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True,
          name="registrar-annotate", console=True)
coll = COLLECT(exe, a.binaries, a.datas, name="registrar-annotate")
