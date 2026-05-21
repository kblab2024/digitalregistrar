# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the schema editor GUI."""
from pathlib import Path

REPO_ROOT = Path(SPECPATH).resolve().parents[1]

a = Analysis(
    [str(REPO_ROOT / "apps" / "schema-editor" / "src" / "digital_registrar_schema_editor" / "app.py")],
    pathex=[
        str(REPO_ROOT / "src"),
        str(REPO_ROOT / "apps" / "schema-editor" / "src"),
    ],
    datas=[
        (str(REPO_ROOT / "src" / "digital_registrar" / "schemas"),
         "digital_registrar/schemas"),
    ],
    hiddenimports=[
        "digital_registrar.schemas", "digital_registrar_schema_editor",
        "streamlit", "pandas",
    ],
    excludes=["drr_attic", "torch", "transformers"],
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True,
          name="registrar-schema-gui", console=True)
coll = COLLECT(exe, a.binaries, a.datas, name="registrar-schema-gui")
