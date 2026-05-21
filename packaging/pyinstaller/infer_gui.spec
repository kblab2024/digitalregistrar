# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the inference GUI.

Build: from repo root, `pyinstaller packaging/pyinstaller/infer_gui.spec`.

Output: dist/registrar-infer-gui/ containing the executable + bundled
Python runtime + all transitive dependencies. Wrap with create-dmg
(macOS) / nsis (Windows) / appimage-builder (Linux) for end-user
installers.
"""
from pathlib import Path

REPO_ROOT = Path(SPECPATH).resolve().parents[1]

a = Analysis(
    [str(REPO_ROOT / "apps" / "infer-gui" / "src" / "digital_registrar_gui" / "app.py")],
    pathex=[
        str(REPO_ROOT / "src"),
        str(REPO_ROOT / "apps" / "infer-gui" / "src"),
    ],
    binaries=[],
    datas=[
        # Schemas + staging bundle ship as package data
        (str(REPO_ROOT / "src" / "digital_registrar" / "schemas" / "data"),
         "digital_registrar/schemas/data"),
        (str(REPO_ROOT / "src" / "digital_registrar" / "staging" / "data"),
         "digital_registrar/staging/data"),
    ],
    hiddenimports=[
        "digital_registrar",
        "digital_registrar.pipeline",
        "digital_registrar.pipeline_factory",
        "digital_registrar.schemas",
        "digital_registrar.signatures",
        "digital_registrar.models",
        "digital_registrar.staging",
        "digital_registrar_gui",
        "streamlit",
        "dspy",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        "drr_attic", "torch", "transformers", "sklearn", "matplotlib", "seaborn",
    ],
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="registrar-infer-gui",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="registrar-infer-gui",
)
