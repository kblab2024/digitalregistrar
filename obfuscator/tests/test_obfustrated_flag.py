"""--obfustrated flag is wired into the eval/ablation/pipeline arg helpers."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

# Make scripts/ importable so we can poke at the eval args helper.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO_ROOT / "scripts"
_SRC = _REPO_ROOT / "src"
for p in (_SCRIPTS, _SRC):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)


def test_eval_add_common_args_registers_flag():
    from scripts.eval._common.args import add_common_args, apply_obfustrated_defaults
    ap = argparse.ArgumentParser()
    add_common_args(ap, subcommand="non_nested")
    args = ap.parse_args(["--obfustrated", "--dataset", "tcga"])
    apply_obfustrated_defaults(args)
    assert args.obfustrated is True
    assert args.root == "workspace_obfustrated"
    assert "workspace_obfustrated" in str(args.out)
    assert "non_nested" in str(args.out)


def test_eval_explicit_root_overrides_obfustrated():
    from scripts.eval._common.args import add_common_args, apply_obfustrated_defaults
    ap = argparse.ArgumentParser()
    add_common_args(ap, subcommand="non_nested")
    args = ap.parse_args(["--obfustrated", "--root", "dummy", "--dataset", "tcga"])
    apply_obfustrated_defaults(args)
    assert args.root == "dummy"


def test_eval_neither_root_nor_obfustrated_errors():
    from scripts.eval._common.args import add_common_args, apply_obfustrated_defaults
    ap = argparse.ArgumentParser()
    add_common_args(ap, subcommand="non_nested")
    args = ap.parse_args(["--dataset", "tcga"])
    with pytest.raises(SystemExit):
        apply_obfustrated_defaults(args)


def test_config_loader_obfustrated_shorthand():
    from _config_loader import resolve_folder
    p = resolve_folder("obfustrated")
    assert p.name == "workspace_obfustrated"
    assert p.is_absolute()


def test_config_loader_dummy_and_workspace_unchanged():
    """Existing shorthands must keep working exactly as before."""
    from _config_loader import resolve_folder, REPO_ROOT
    assert resolve_folder("dummy") == (REPO_ROOT / "dummy").resolve()
    assert resolve_folder("workspace") == (REPO_ROOT / "workspace").resolve()


def test_ablations_apply_obfustrated_default():
    from digital_registrar_research.ablations.runners._base import (
        apply_obfustrated_default,
    )
    ns = argparse.Namespace(experiment_root=None, obfustrated=True)
    apply_obfustrated_default(ns)
    assert ns.experiment_root.name == "workspace_obfustrated"


def test_ablations_default_idempotent_when_folder_set():
    from digital_registrar_research.ablations.runners._base import (
        apply_obfustrated_default,
    )
    explicit = Path("/tmp/whatever")
    ns = argparse.Namespace(experiment_root=explicit, obfustrated=True)
    apply_obfustrated_default(ns)
    assert ns.experiment_root == explicit
