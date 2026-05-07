"""Auto-loader for per-model decoding-override YAMLs.

The consolidated runners ``scripts/pipeline/run_dspy_ollama_{single,smoke}.py``
read an optional YAML of the form ``configs/dspy_ollama_{model}.yaml`` so
decoding params (``temperature``, ``top_p``, ``num_ctx``, ...) and smoke
defaults (``smoke.n``, ``smoke.seed``) can be tuned without editing code. If
the config file is absent, or every override is null, the per-model defaults
in ``models.common.MODEL_PROFILES`` are used unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"
LOCAL_CONFIGS_DIR = CONFIGS_DIR / "local"

DECODING_KEYS = (
    "temperature", "top_p", "top_k", "num_ctx", "max_tokens",
    "repeat_penalty", "keep_alive", "cache", "seed",
)


def model_config_path(model_alias: str) -> Path:
    """Return the config-file path for a model alias, preferring local/ override.

    File-level resolution (used when callers want a single canonical path,
    e.g. for logging). Per-key merging is handled in ``load_model_config``."""
    local = LOCAL_CONFIGS_DIR / f"dspy_ollama_{model_alias}.yaml"
    if local.is_file():
        return local
    return CONFIGS_DIR / f"dspy_ollama_{model_alias}.yaml"


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *over* on top of *base*. Nested dicts merge key-by-key;
    everything else (lists, scalars) is replaced wholesale by *over*."""
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    import yaml
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_model_config(model_alias: str) -> dict[str, Any]:
    """Load the config for *model_alias* with per-key local-over-shared merge.

    Loads ``configs/dspy_ollama_{alias}.yaml`` as the base and overlays
    ``configs/local/dspy_ollama_{alias}.yaml`` on top. Each key is resolved
    independently, so a local file can hold a partial override (e.g. just
    ``decoding.num_ctx``) without redeclaring every other knob. Returns
    ``{}`` if neither file exists."""
    shared = _load_yaml(CONFIGS_DIR / f"dspy_ollama_{model_alias}.yaml")
    local = _load_yaml(LOCAL_CONFIGS_DIR / f"dspy_ollama_{model_alias}.yaml")
    return _deep_merge(shared, local)


def split_decoding_overrides(decoding_section: dict[str, Any] | None) -> dict[str, Any]:
    """Filter a config ``decoding:`` block down to the non-null overrides.

    Special value for ``seed``: the string ``"random"`` (or ``"auto"``) is
    replaced with a freshly drawn 31-bit integer. The chosen seed is still
    baked into the LM kwargs, so it gets logged into ``_run_meta.json`` and
    the model-level ``_manifest.yaml`` — runs remain individually
    reproducible, only the starting point is non-deterministic.
    """
    if not decoding_section:
        return {}
    out = {k: decoding_section[k] for k in DECODING_KEYS
           if k in decoding_section and decoding_section[k] is not None}
    seed = out.get("seed")
    if isinstance(seed, str):
        token = seed.strip().lower()
        if token in ("random", "auto"):
            import secrets
            out["seed"] = secrets.randbelow(2**31)
        else:
            raise ValueError(
                f"decoding.seed must be an int, null, or 'random'; got {seed!r}"
            )
    return out


def resolve_folder(raw: str | Path) -> Path:
    """Resolve a ``--folder`` CLI arg against the repo root when relative.

    "dummy" and "workspace" are plain relative paths — they resolve to
    ``<repo_root>/dummy`` and ``<repo_root>/workspace`` respectively.

    "obfustrated" is a shortcut for ``<repo_root>/workspace_obfustrated`` —
    the schema-conformant synthetic copy produced by the obfuscator
    (``scripts/obfuscate_workspace.py`` / ``obfuscate-workspace`` CLI). Use
    this when debugging eval/ablation/inference paths without touching PHI.

    "reference" is a virtual experiment root: it resolves to
    ``<repo_root>/reference/_staged``, a symlink tree built on demand from
    the canonical TCGA test data at ``reference/reports/<organ_n>/*.txt``.
    The staging tree mirrors the canonical
    ``data/{dataset}/reports/{organ_n}/*.txt`` layout that runners expect,
    so they need no special handling. Predict-side runs work from this
    tree directly; gold annotations are not staged (the reference checkout
    is reports-only).

    Any absolute path is used verbatim; any other relative path is resolved
    against the repo root (not the caller's cwd) for reproducibility."""
    if str(raw) == "reference":
        return _ensure_reference_staged()
    if str(raw) == "obfustrated":
        return (REPO_ROOT / "workspace_obfustrated").resolve()
    p = Path(raw)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p.resolve()


_REFERENCE_DIR = REPO_ROOT / "reference"
_REFERENCE_STAGED = _REFERENCE_DIR / "_staged"
# Datasets whose reports live under the canonical flat layout
# ``reference/reports/<organ_n>/*.txt``. The reference checkout is
# reports-only — gold annotations are not staged. Add a tuple here when a
# new dataset gets a reference subset, and split `reports/` per-dataset
# (e.g. ``reference/reports/<dataset>/<organ_n>/...``) if/when the
# reports tree stops being TCGA-only.
_REFERENCE_DATASETS: tuple[str, ...] = ("tcga",)


def _ensure_reference_staged() -> Path:
    """Build (or refresh) the canonical-layout symlink tree under
    ``reference/_staged/``. Idempotent: re-uses existing symlinks when the
    target paths haven't changed.

    Source layout (reports-only):
        reference/reports/<organ_n>/<case_id>.txt

    Staged layout (what runners read via ``--folder reference``):
        reference/_staged/data/<dataset>/reports/<organ_n>/<case_id>.txt
    """
    if not _REFERENCE_DIR.is_dir():
        raise FileNotFoundError(
            f"--folder reference requires {_REFERENCE_DIR} to exist")
    src_reports = _REFERENCE_DIR / "reports"
    if not src_reports.is_dir():
        raise FileNotFoundError(
            f"--folder reference requires {src_reports} (canonical "
            f"reports/<organ_n>/*.txt layout) to exist")
    _REFERENCE_STAGED.mkdir(exist_ok=True)
    (_REFERENCE_STAGED / "results").mkdir(exist_ok=True)
    for dataset in _REFERENCE_DATASETS:
        _stage_reports(dataset, src_reports)
    return _REFERENCE_STAGED.resolve()


def _stage_reports(dataset: str, src_root: Path) -> None:
    """Symlink ``reference/reports/<organ_n>/<case_id>.txt`` to
    ``reference/_staged/data/<dataset>/reports/<organ_n>/<case_id>.txt``.

    The source layout is already canonical (numeric ``<organ_n>`` dirs,
    ``<case_id>.txt`` filenames), so this is a 1:1 file-level symlink
    pass with no path rewriting.
    """
    dst_root = _REFERENCE_STAGED / "data" / dataset / "reports"
    dst_root.mkdir(parents=True, exist_ok=True)
    for src_organ in sorted(src_root.iterdir()):
        if not src_organ.is_dir() or not src_organ.name.isdigit():
            continue
        dst_organ = dst_root / src_organ.name
        dst_organ.mkdir(exist_ok=True)
        for txt in src_organ.glob("*.txt"):
            link = dst_organ / txt.name
            _refresh_symlink(link, txt)


def _refresh_symlink(link: Path, target: Path) -> None:
    """Create or update a symlink at ``link`` pointing to ``target``.
    No-op when the existing symlink already points at ``target``."""
    if link.is_symlink():
        try:
            if link.resolve() == target.resolve():
                return
        except OSError:
            pass
        link.unlink()
    elif link.exists():
        # Concrete file in the way; leave alone (don't clobber real data).
        return
    link.symlink_to(target)
